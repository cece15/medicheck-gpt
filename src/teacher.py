"""
Step 1b of the flow (Kacper's recommendation, 3 Nov): an LLM teacher writes the reference outputs,
and my small GPT learns to reproduce them. This is knowledge distillation from a large model to a small one.

Task: structured record  ->  one plain-English sentence a non-specialist can read.

Guarding the teacher. An LLM writing about numbers is itself a source of silent errors, so:
  1. Python computes every number (Medicare share, charge-to-payment ratio) before the LLM sees it.
     The LLM only phrases; it never calculates.
  2. Every number in the LLM's sentence is checked against the allowed set. A sentence with a number
     that isn't in the record is rejected, and the rejection rate is reported.
"""
import re

from .data import parse

DEFAULT_TEACHER = 'Qwen/Qwen2.5-1.5B-Instruct'   # small open instruction model: runs on a free Colab T4, no API key

SYSTEM = ('You explain Medicare hospital billing records to a non-specialist in one sentence of at most '
          '45 words. Use only the facts given. Copy numbers exactly as written. Do not add any other number.')


def facts(f):
    """Everything the teacher may say, computed here so the arithmetic is right."""
    c, t, m = int(f['c']), int(f['t']), int(f['m'])
    return {
        'hospital': f['name'], 'place': f"{f['city']}, {f['state']}",
        'condition': f['desc'].lower(), 'drg': f['drg'], 'discharges': f['n'],
        'charges': f'${c:,}', 'total_paid': f'${t:,}', 'medicare_paid': f'${m:,}',
        'medicare_share': f'{round(100 * m / t)}%' if t else 'n/a',          # how much of the payment Medicare covered
        'paid_vs_charged': f'{round(100 * t / c)}%' if c else 'n/a',         # how much of the bill was actually paid
    }


def prompt(fx):
    lines = '\n'.join(f'{k}: {v}' for k, v in fx.items())
    return f'Facts:\n{lines}\n\nWrite the sentence.'


def numbers_in(text):
    """Every number in a piece of text as a value, so '057', '57' and '$57.00' all count as 57."""
    return {float(n.replace(',', '')) for n in re.findall(r'\d[\d,]*(?:\.\d+)?', text)}


def allowed_numbers(fx):
    """The only numbers a correct sentence can contain."""
    return numbers_in(' '.join(str(v) for v in fx.values()))


def check(sentence, fx):
    """Reject a sentence that invents a number, runs to several sentences, or forgets the condition."""
    if not numbers_in(sentence) <= allowed_numbers(fx):
        return 'invented_number'
    if '\n' in sentence or ' || ' in sentence:
        return 'format'
    if len(sentence.split()) > 60:
        return 'too_long'
    return None


def write_descriptions(records, model_name=DEFAULT_TEACHER, batch=32, device='cuda'):
    """Run the teacher over records. Returns (pairs, rejection counts)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side='left')   # left padding for batched generation
    try:                                                          # newer transformers call it `dtype`, older `torch_dtype`
        llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    llm = llm.to(device).eval()
    pairs, rejected = [], {'invented_number': 0, 'format': 0, 'too_long': 0}
    parsed = [(r, parse(r)) for r in records]
    parsed = [(r, f) for r, f in parsed if f]
    for i in range(0, len(parsed), batch):
        chunk = parsed[i:i + batch]
        fxs = [facts(f) for _, f in chunk]
        texts = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                                          {'role': 'user', 'content': prompt(fx)}],
                                         tokenize=False, add_generation_prompt=True) for fx in fxs]
        enc = tok(texts, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            out = llm.generate(**enc, max_new_tokens=90, do_sample=False)      # greedy: same references every run
        for (rec, _), fx, ids in zip(chunk, fxs, out[:, enc['input_ids'].shape[1]:]):
            s = tok.decode(ids, skip_special_tokens=True).strip()
            why = check(s, fx)
            if why:
                rejected[why] += 1                                            # the teacher's own silent errors
            else:
                pairs.append(f'{rec} || {s}')                                 # one training example per line
        print(f'{min(i + batch, len(parsed))}/{len(parsed)} written, {sum(rejected.values())} rejected')
    return pairs, rejected
