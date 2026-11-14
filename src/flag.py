"""
Step 6 of the flow: the function the whole project exists for. Flag silent errors.

A silent error is a record that is well-formed and looks plausible but is wrong: no null, no crash,
no failed type check, so a pipeline passes it through. The idea: a language model trained on clean
records learns what a normal record looks like, so a wrong record should surprise it (high loss).

Test: take records from hospitals the model never saw, plant four kinds of silent error, and ask
whether the model's surprise ranks corrupted records above clean ones. Measured as ROC AUC
(0.5 = coin flip, 1.0 = every corrupted record ranked above every clean one).
"""
import random

from .data import parse
from .metrics import record_scores


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


def run(model, tok, test_records, train_records, device, seed=1337, n=500):
    """Corrupt, score, compare. Returns one AUC per error type, for both the mean and the peak score."""
    rng = random.Random(seed)                                     # same corruptions every run
    drg_map, cities = {}, {}
    for r in train_records:                                       # replacement values come from training data only
        f = parse(r)
        if f:
            drg_map[f['drg']] = f['desc']
            cities.setdefault(f['state'], set()).add(f['city'])
    cities = {s: sorted(c) for s, c in cities.items()}

    clean = [r for r in test_records if parse(r)][:n]
    clean_scores = record_scores(model, tok, clean, device)
    results, examples = {}, {}
    for kind in KINDS:
        bad = [b for b in (corrupt(r, kind, drg_map, cities, rng) for r in clean) if b]
        bad_scores = record_scores(model, tok, bad, device)
        results[kind] = {'n': len(bad),
                         'auc_mean': round(auc([s[0] for s in bad_scores], [s[0] for s in clean_scores]), 3),
                         'auc_peak': round(auc([s[1] for s in bad_scores], [s[1] for s in clean_scores]), 3)}
        examples[kind] = bad[0] if bad else None
    return results, examples


def flag_file(model, tok, records, device, top=10):
    """Real use: score every record in a file and return the most surprising ones for a human to check."""
    scores = record_scores(model, tok, records, device)
    ranked = sorted(zip(scores, records), key=lambda t: -t[0][1])  # sort by peak surprise, worst first
    return [(round(s[1], 2), r) for s, r in ranked[:top]]
