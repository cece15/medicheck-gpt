"""
Step 6 of the flow: the function the whole project exists for. Flag silent errors.

A silent error is a record that is well-formed and looks plausible but is wrong: no null, no crash,
no failed type check, so a pipeline passes it through. The idea: a language model trained on clean
records learns what a normal record looks like, so a wrong record should surprise it (high loss).

Test: take records from hospitals the model never saw, plant four kinds of silent error, and ask
whether the model's surprise ranks corrupted records above clean ones. Measured as ROC AUC
(0.5 = coin flip, 1.0 = every corrupted record ranked above every clean one; below 0.5 = backwards).

Three scores, each fixing a weakness of the one before (first run, 14 Nov):
  raw     mean surprise over the whole record
  cond    peak surprise after the hospital name. The name of an unseen hospital dominated every
          record's peak, so clean and corrupted records tied.
  ratio   peak of (GPT surprise - bigram surprise) after the name. Raw surprise rewards common values:
          a wrong but common city (Columbus) looked more normal than a true but rare one (Grayling).
          Subtracting a simple background model cancels plain rarity and leaves what the GPT knows
          about context. Likelihood ratio, after Ren et al., "Likelihood Ratios for Out-of-Distribution
          Detection", NeurIPS 2019.
"""
import random

import random as _r

from .data import parse
from .metrics import after_name, char_surprise, peak


def rebuild(f):
    """Write a parsed record back as a sentence, in exactly the training format."""
    return (f"{f['name']} in {f['city']}, {f['state']} billed DRG {f['drg']} ({f['desc']}) for {f['n']} "
            f"discharges: covered charges ${f['c']}, total payment ${f['t']}, Medicare payment ${f['m']}.")


def corrupt(record, kind, drg_map, cities_by_state, rng):
    """Plant one silent error. Returns None when this record can't carry this error type."""
    f = parse(record)
    if f is None:
        return None
    f = dict(f)
    if kind == 'drg_mismatch':                                    # code says one diagnosis, text says another
        f['desc'] = drg_map[rng.choice([d for d in drg_map if d != f['drg']])]
    elif kind == 'wrong_city':                                    # hospital moved to another town in the same state
        others = [c for c in cities_by_state.get(f['state'], []) if c != f['city']]
        if not others:
            return None
        f['city'] = rng.choice(others)
    elif kind == 'payments_swapped':                              # Medicare share now larger than the total paid
        if int(f['m']) >= int(f['t']):
            return None
        f['t'], f['m'] = f['m'], f['t']
    elif kind == 'extra_digit':                                   # a classic keying error: charges x10
        f['c'] = f['c'] + '0'
    return rebuild(f)


KINDS = ['drg_mismatch', 'wrong_city', 'payments_swapped', 'extra_digit']


def auc(pos, neg):
    """ROC AUC as a probability: pick one corrupted and one clean record; how often does the corrupted
    one score higher? (Mann-Whitney U. Ties count half.)"""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_ci(pos, neg, reps=200, seed=0):
    """95% bootstrap interval: resample both groups, recompute AUC, keep the middle 95%."""
    rng, vals = _r.Random(seed), []
    for _ in range(reps):
        p = [rng.choice(pos) for _ in pos]
        n = [rng.choice(neg) for _ in neg]
        vals.append(auc(p, n))
    vals.sort()
    return round(vals[int(0.025 * reps)], 3), round(vals[int(0.975 * reps) - 1], 3)


def scores(model, tok, recs, device, background=None):
    """All three scores for each record."""
    g = char_surprise(model, tok, recs, device)
    b = char_surprise(background, tok, recs, device) if background is not None else None
    out = []
    for i, r in enumerate(recs):
        k = after_name(r)                                         # characters after the hospital name
        row = {'raw': g[i].mean().item(), 'cond': peak(g[i][k:])}
        if b is not None:
            row['ratio'] = peak(g[i][k:] - b[i][k:])              # surprise beyond what plain rarity explains
        out.append(row)
    return out


def run(model, tok, test_records, train_records, device, seed=1337, n=500, background=None):
    """Corrupt, score, compare. Returns AUC (with a 95% interval for the best score) per error type."""
    rng = random.Random(seed)                                     # same corruptions every run
    drg_map, cities = {}, {}
    for r in train_records:                                       # replacement values come from training data only
        f = parse(r)
        if f:
            drg_map[f['drg']] = f['desc']
            cities.setdefault(f['state'], set()).add(f['city'])
    cities = {s: sorted(c) for s, c in cities.items()}

    clean = [r for r in test_records if parse(r)][:n]
    clean_s = scores(model, tok, clean, device, background)
    keys = list(clean_s[0])
    results, examples = {}, {}
    for kind in KINDS:
        bad = [b for b in (corrupt(r, kind, drg_map, cities, rng) for r in clean) if b]
        bad_s = scores(model, tok, bad, device, background)
        res = {'n': len(bad)}
        for k in keys:
            res[f'auc_{k}'] = round(auc([s[k] for s in bad_s], [s[k] for s in clean_s]), 3)
        best = keys[-1]                                           # the most corrected score available
        res[f'ci_{best}'] = auc_ci([s[best] for s in bad_s], [s[best] for s in clean_s])
        results[kind] = res
        examples[kind] = bad[0] if bad else None
    return results, examples


def flag_file(model, tok, records, device, top=10, background=None):
    """Real use: score every record in a file and return the most surprising ones for a human to check."""
    s = scores(model, tok, records, device, background)
    key = 'ratio' if background is not None else 'cond'
    ranked = sorted(zip(s, records), key=lambda t: -t[0][key])    # worst first
    return [(round(sc[key], 2), r) for sc, r in ranked[:top]]
