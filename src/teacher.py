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

SYSTEM = ('You explain Medicare hospital billing records to a non-specialist in one sentence of at most 40 words. '
          'Use only the facts given and keep each number with the words it comes with. Do not add any other number.\n'
          'Example of the style (invented hospital): Lakeview Hospital in Elm City, MI treated 20 Medicare patients for '
          'heart failure; it billed $40,000 on average, was paid $10,000 in total, and Medicare paid $8,500 of that, '
          'which is 85% of the payment.')


def facts(f):
    """Everything the teacher may say, computed here so the arithmetic is right."""
    c, t, m = int(f['c']), int(f['t']), int(f['m'])
    # each fact is a full phrase, so the LLM copies the number together with what it means
    return {
        'hospital': f['name'], 'place': f"{f['city']}, {f['state']}",
        'condition': f['desc'].lower(), 'patients': f"{f['n']} Medicare patients",
        'billed': f'billed ${c:,} on average', 'paid_total': f'was paid ${t:,} in total',
        'medicare_paid': f'Medicare paid ${m:,}',
        'medicare_share': f'{round(100 * m / t)}% of the payment came from Medicare' if t else '',
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


ROLE_WORDS = {'billed': ('bill', 'charge'), 'paid_total': ('total', 'paid', 'payment'),
              'medicare_paid': ('medicare',), 'medicare_share': ('medicare', 'payment')}


def wrong_label(sentence, fx):
    """Every number must sit next to a word that says what it is. Catches 'received $15,466 from Medicare'
    when $15,466 is the total, and '$12,681 billed' when $12,681 is the payment."""
    words = re.findall(r"[\w$%,.']+", sentence.lower())
    for key, keep in ROLE_WORDS.items():
        if not fx.get(key):
            continue
        value = max(numbers_in(fx[key]))                              # the number this fact carries
        for i, w in enumerate(words):
            if numbers_in(w) and max(numbers_in(w)) == value:
                window = ' '.join(words[max(0, i - 8):i + 4])          # eight words before, three after
                if not any(k in window for k in keep):
                    return True
                after = ' '.join(words[i + 1:i + 3])                  # the two words right after the number
                if key != 'billed' and any(k in after for k in ('bill', 'charge')):
                    return True                                       # a payment described as the bill
                if key == 'billed' and 'medicare' in after:
                    return True                                       # the bill described as Medicare's payment
                if key == 'medicare_share' and any(k in ' '.join(words[i + 1:i + 5]) for k in ('charge', 'bill')):
                    return True                                       # "95% of the charges": share of the wrong thing
                if key == 'paid_total' and value != max(numbers_in(fx['medicare_paid'])) and \
                        f'{w} from medicare' in ' '.join(words[i:i + 3]):
                    return True                                       # the total, credited to Medicare
    return False


def check(sentence, fx):
    """Reject a sentence that invents a number, mislabels one, or runs on."""
    if not numbers_in(sentence) <= allowed_numbers(fx):
        return 'invented_number'
    if wrong_label(sentence, fx):
        return 'wrong_label'
    if '\n' in sentence or ' || ' in sentence:
        return 'format'
    if len(sentence.split()) > 50:
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
    pairs, rejected = [], {'invented_number': 0, 'wrong_label': 0, 'format': 0, 'too_long': 0}
    parsed = [(r, parse(r)) for r in records]
    parsed = [(r, f) for r, f in parsed if f]
    for i in range(0, len(parsed), batch):
        chunk = parsed[i:i + batch]
        fxs = [facts(f) for _, f in chunk]
        texts = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                                          {'role': 'user', 'content': prompt(fx)}],
                                         tokenize=False, add_generation_prompt=True) for fx in fxs]
        enc = tok(texts, return_tensors='pt', padding=True).to(device)
        n_in = enc['input_ids'].shape[1]
        with torch.no_grad():
            first = llm.generate(**enc, max_new_tokens=80, do_sample=False)  # greedy: same answer every run
        sents, redo = {}, []
        for j, fx in enumerate(fxs):
            s = tok.decode(first[j, n_in:], skip_special_tokens=True).strip()
            why = check(s, fx)
            if why:
                rejected[why] += 1                                    # the teacher's own silent error, counted
                redo.append(j)
            else:
                sents[j] = s
        if redo:                                                      # one second chance, only for the rejected
            torch.manual_seed(i)
            sub = {k: v[redo] for k, v in enc.items()}
            with torch.no_grad():
                again = llm.generate(**sub, max_new_tokens=80, do_sample=True, temperature=0.7)
            for row, j in enumerate(redo):
                s = tok.decode(again[row, n_in:], skip_special_tokens=True).strip()
                if not check(s, fxs[j]):
                    sents[j] = s                                      # wrong twice: the record is left out
        for j, (rec, _) in enumerate(chunk):
            if j in sents:
                pairs.append(f'{rec} || {sents[j]}')                  # one training example per line
        print(f'{min(i + batch, len(parsed))}/{len(parsed)} done, {len(pairs)} kept, first attempts rejected: {rejected}')
    return pairs, rejected
