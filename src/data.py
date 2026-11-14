"""
Step 1 of the flow: raw CMS table -> one sentence per row -> hospital-level split -> characters -> integers.

Source data: CMS "Medicare Inpatient Hospitals - by Provider and Service", data year 2024
(file MUP_INP_RY26_P03_V10_DY24_PrvSvc.CSV, data.cms.gov). One row = one hospital x one DRG.
"""
import json
import random
import re

import pandas as pd
import torch

# The 15 columns CMS publishes. Checked against the file in 01_data.ipynb: same names, different order.
COLUMNS = ['Rndrng_Prvdr_CCN', 'Rndrng_Prvdr_Org_Name', 'Rndrng_Prvdr_City', 'Rndrng_Prvdr_St',
           'Rndrng_Prvdr_State_FIPS', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_State_Abrvtn',
           'Rndrng_Prvdr_RUCA', 'Rndrng_Prvdr_RUCA_Desc', 'DRG_Cd', 'DRG_Desc', 'Tot_Dschrgs',
           'Avg_Submtd_Cvrd_Chrg', 'Avg_Tot_Pymt_Amt', 'Avg_Mdcr_Pymt_Amt']


def load_rows(csv_path, states=('MI', 'OH')):
    """Read the CSV and keep only the states we train on."""
    df = pd.read_csv(csv_path, dtype=str)                       # everything as text: keeps leading zeros in CCN, ZIP, DRG
    assert set(df.columns) == set(COLUMNS), 'unexpected columns: is this the CMS provider-and-service file?'
    return df[df['Rndrng_Prvdr_State_Abrvtn'].isin(states)].copy()   # MI + OH: home state plus a neighbour, ~9.6k rows


def to_sentence(row):
    """Turn one structured row into one line of English the model can learn from."""
    c = int(round(float(row['Avg_Submtd_Cvrd_Chrg'])))          # what the hospital billed, rounded to whole dollars
    t = int(round(float(row['Avg_Tot_Pymt_Amt'])))              # what was actually paid in total
    m = int(round(float(row['Avg_Mdcr_Pymt_Amt'])))             # the Medicare share of that payment
    return (f"{row['Rndrng_Prvdr_Org_Name']} in {row['Rndrng_Prvdr_City']}, "
            f"{row['Rndrng_Prvdr_State_Abrvtn']} billed DRG {row['DRG_Cd']} "
            f"({row['DRG_Desc']}) for {row['Tot_Dschrgs']} discharges: "
            f"covered charges ${c}, total payment ${t}, Medicare payment ${m}.")


def split_hospitals(ccns, seed=1337, frac=0.1):
    """Split by HOSPITAL, not by row, so the test set holds hospitals the model has never seen.
    A row-level split would leak: the same hospital name and address would sit in train and test."""
    ids = sorted(set(ccns))                                      # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(ids)                             # fixed seed: same split on every run
    n = max(1, round(frac * len(ids)))                           # 10% of hospitals each for val and test
    return set(ids[2 * n:]), set(ids[n:2 * n]), set(ids[:n])     # train, val, test


def prepare(csv_path, out_dir, states=('MI', 'OH')):
    """Write train.txt / val.txt / test.txt (one record per line) and remember which hospital went where."""
    df = load_rows(csv_path, states)
    df['text'] = df.apply(to_sentence, axis=1)
    tr, va, te = split_hospitals(df['Rndrng_Prvdr_CCN'])
    stats = {}
    for name, ids in [('train', tr), ('val', va), ('test', te)]:
        part = df[df['Rndrng_Prvdr_CCN'].isin(ids)]
        text = '\n' + '\n'.join(part['text']) + '\n'             # newline = "record starts / record ends" marker
        open(f'{out_dir}/{name}.txt', 'w').write(text)
        stats[name] = {'hospitals': len(ids), 'records': len(part), 'chars': len(text)}
    json.dump({k: sorted(v) for k, v in zip(['train', 'val', 'test'], [tr, va, te])},
              open(f'{out_dir}/split_ccns.json', 'w'))           # saved so anyone can audit that no hospital leaks
    return stats


# ---- Tokenizer: character level -------------------------------------------------------------
# Why characters and not BPE: records are short (longest ~270 chars), the vocabulary is tiny (~77),
# and every digit is its own token, which matters when the job is to notice a wrong number.

class CharTokenizer:
    def __init__(self, text=None, stoi=None):
        self.stoi = stoi if stoi is not None else {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.itos = {i: ch for ch, i in self.stoi.items()}       # reverse lookup for decoding

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]       # characters never seen in training are dropped

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

    def __len__(self):
        return len(self.stoi)


def get_batch(data, batch_size, context_size, device):
    """Sample random windows. x = a window of text, y = the same window shifted one character right."""
    ix = torch.randint(len(data) - context_size, (batch_size,))  # random start positions
    x = torch.stack([data[i:i + context_size] for i in ix])       # inputs
    y = torch.stack([data[i + 1:i + context_size + 1] for i in ix])   # targets: the next character at every position
    return x.to(device), y.to(device)


# ---- Parsing a record back into fields (used by the validity check and the error injector) ----
RECORD_RE = re.compile(
    r'^(?P<name>.+) in (?P<city>.+), (?P<state>[A-Z]{2}) billed DRG (?P<drg>\d{3}) '
    r'\((?P<desc>.+)\) for (?P<n>\d+) discharges: covered charges \$(?P<c>\d+), '
    r'total payment \$(?P<t>\d+), Medicare payment \$(?P<m>\d+)\.$')


def parse(record):
    """Return a dict of fields, or None if the line is not a well-formed record."""
    hit = RECORD_RE.match(record.strip())
    return hit.groupdict() if hit else None
