"""
Step 5 of the flow: evaluation. Kacper's four (Lecture 6-7, slides 2-4) plus one of my own.

  perplexity        how surprised the model is by unseen hospitals' records (lower is better)
  token accuracy    how often its top guess for the next character is right
  ROUGE-L / BLEU    overlap between a completed record and the real one (a sanity check, not a goal)
  BERTScore         semantic similarity, optional because it downloads a BERT model
  record validity   mine: do generated records obey the rules real records obey?

ROUGE-L is computed here directly (longest common subsequence) so the core evaluation runs with no
extra downloads; BLEU and BERTScore use Hugging Face `evaluate` when it is installed.
"""
import math

import torch
import torch.nn.functional as F

from .data import parse
from .generate import complete


@torch.no_grad()
def char_surprise(model, tok, records, device, batch=64):
    """For each record, the model's surprise (negative log-likelihood) at every character, as a 1-D tensor.
    Position k is the surprise at character k of the record (plus the closing newline).
    This is the engine of the error flagger."""
    model.eval()
    nl, ctx, out = tok.stoi['\n'], model.context_size, []
    for i in range(0, len(records), batch):
        seqs = [tok.encode('\n' + r + '\n') for r in records[i:i + batch]]
        if ctx > 1:
            seqs = [s[:ctx + 1] for s in seqs]                   # a record longer than the window is cut at the end
        T = max(len(s) for s in seqs) - 1
        x = torch.full((len(seqs), T), nl, dtype=torch.long)      # pad with newlines...
        y = torch.full((len(seqs), T), -100, dtype=torch.long)    # ...and mark padding -100 so it never counts
        for j, s in enumerate(seqs):
            x[j, :len(s) - 1] = torch.tensor(s[:-1])
            y[j, :len(s) - 1] = torch.tensor(s[1:])
        logits, _ = model(x.to(device))
        nll = F.cross_entropy(logits.transpose(1, 2), y.to(device), ignore_index=-100, reduction='none').cpu()
        for j, s in enumerate(seqs):
            out.append(nll[j, :len(s) - 1])
    return out


def peak(v, width=8):
    """Highest average surprise over any `width` consecutive characters: one wrong field looks like this."""
    return (v.unfold(0, width, 1).mean(1) if len(v) >= width else v).max().item()


def after_name(record):
    """Index where the hospital name ends. Everything before it is skipped when flagging: an unseen
    hospital's name is new to the model, and new is not the same as wrong."""
    return record.index(' in ') if ' in ' in record else 0


@torch.no_grad()
def perplexity_and_accuracy(model, tok, text, device, ctx):
    """Walk through the whole test text in non-overlapping windows. Perplexity = exp(mean cross-entropy)."""
    model.eval()
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    total_nll, correct, n = 0.0, 0, 0
    for i in range(0, len(data) - 1, ctx):
        x = data[i:i + ctx][None].to(device)
        y = data[i + 1:i + ctx + 1][None].to(device)
        x = x[:, :y.shape[1]]                                     # last window may be shorter
        logits, _ = model(x)
        total_nll += F.cross_entropy(logits[0], y[0], reduction='sum').item()
        correct += (logits[0].argmax(-1) == y[0]).sum().item()    # was the top guess the true next character?
        n += y.shape[1]
    return math.exp(total_nll / n), correct / n


def rouge_l(pred, ref):
    """ROUGE-L F1 on words: length of the longest common subsequence, as precision/recall against the reference."""
    p, r = pred.split(), ref.split()
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def completion_eval(model, tok, test_records, device, n=100, temperature=0.8, use_hf=False):
    """Give the model the start of a real record (hospital, city, state, 'billed DRG 123 ') and let it
    finish. Compare what it wrote with what the hospital actually reported."""
    preds, refs = [], []
    for rec in test_records[:n]:
        cut = rec.index(' billed DRG ') + len(' billed DRG ') + 4 # prompt ends just after the 3-digit code
        prompt, ref = rec[:cut], rec[cut:]
        preds.append(complete(model, tok, prompt, temperature=temperature, device=device)[cut:])
        refs.append(ref)
    out = {'rougeL': sum(rouge_l(p, r) for p, r in zip(preds, refs)) / len(refs)}
    if use_hf:                                                    # optional extras from Hugging Face
        import evaluate
        out['bleu'] = evaluate.load('bleu').compute(predictions=preds, references=[[r] for r in refs])['bleu']
        bs = evaluate.load('bertscore').compute(predictions=preds, references=refs, lang='en',
                                                rescale_with_baseline=True)   # raw BERTScore crowds into 0.85-1.0
        out['bertscore_f1'] = sum(bs['f1']) / len(bs['f1'])
    return out, list(zip(refs, preds))


def describe_eval(model, tok, pairs, device, n=100, temperature=0.8, use_hf=False):
    """Kacper's task: give the student a record, let it write the description, compare with the teacher's.
    Here BLEU / ROUGE / BERTScore have a real reference to compare against."""
    preds, refs, recs = [], [], []
    for p in pairs[:n]:
        rec, ref = p.split(' || ', 1)
        recs.append(rec)
        prompt = rec + ' || '
        preds.append(complete(model, tok, prompt, max_new=400, temperature=temperature, device=device)[len(prompt):])
        refs.append(ref)
    out = {'rougeL': sum(rouge_l(a, b) for a, b in zip(preds, refs)) / len(refs)}
    if use_hf:
        import evaluate
        out['bleu'] = evaluate.load('bleu').compute(predictions=preds, references=[[r] for r in refs])['bleu']
        bs = evaluate.load('bertscore').compute(predictions=preds, references=refs, lang='en',
                                                rescale_with_baseline=True)   # raw BERTScore crowds into 0.85-1.0
        out['bertscore_f1'] = sum(bs['f1']) / len(bs['f1'])
    return out, list(zip(recs, refs, preds))


def numbers_faithful(pairs_out):
    """Share of student descriptions whose numbers all appear in the record it was given: did it copy or invent?"""
    from .teacher import numbers_in
    ok = 0
    for rec, _, pred in pairs_out:
        ok += numbers_in(pred) <= numbers_in(rec) | _derived(rec)
    return ok / len(pairs_out)


def _derived(rec):
    """Percentages the teacher was allowed to use, recomputed from the record."""
    f = parse(rec)
    if not f or not int(f['t']) or not int(f['c']):
        return set()
    return {float(round(100 * int(f['m']) / int(f['t']))), float(round(100 * int(f['t']) / int(f['c'])))}


def known_facts(train_records):
    """What the training data says is true: which description goes with each DRG code, where each hospital is."""
    drg, place = {}, {}
    for r in train_records:
        f = parse(r)
        if f:
            drg[f['drg']] = f['desc']
            place[f['name']] = (f['city'], f['state'])
    return drg, place


def validity(generated, drg_map, place_map):
    """Share of generated records that pass each rule real records pass. This is the domain test:
    a model that writes fluent but impossible records is exactly the 'silent error' this project is about."""
    checks = {'parses': 0, 'drg_matches_description': 0, 'medicare_le_total_payment': 0,
              'total_payment_le_charges': 0}
    known, located = 0, 0
    for g in generated:
        f = parse(g)
        if not f:
            continue
        checks['parses'] += 1
        checks['drg_matches_description'] += drg_map.get(f['drg']) == f['desc']
        checks['medicare_le_total_payment'] += int(f['m']) <= int(f['t'])
        checks['total_payment_le_charges'] += int(f['t']) <= int(f['c'])
        if f['name'] in place_map:                                # location can only be judged for hospitals we know
            known += 1
            located += place_map[f['name']] == (f['city'], f['state'])
    out = {k: round(v / len(generated), 3) for k, v in checks.items()}
    out['known_hospital_in_right_place'] = round(located / known, 3) if known else None
    return out
