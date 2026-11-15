"""
Unit tests for the parts of the project that must never be wrong.
Run from the repo root:  python -m pytest tests -q
No GPU and no data needed; each test builds the little it uses.
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D                                           # noqa: E402
from src.flag import auc, corrupt, rule_hits                        # noqa: E402
from src.model import GPT, init_from                                # noqa: E402
from src.teacher import check, facts                                # noqa: E402

REC = ("Munson Healthcare Grayling Hospital in Grayling, MI billed DRG 177 (RESPIRATORY INFECTIONS AND "
       "INFLAMMATIONS WITH MCC) for 27 discharges: covered charges $37016, total payment $15466, "
       "Medicare payment $14681.")


# ---- data ----------------------------------------------------------------------------------------

def test_parse_reads_every_field():
    f = D.parse(REC)
    assert (f['name'], f['city'], f['drg'], f['n'], f['c'], f['t'], f['m']) == \
        ('Munson Healthcare Grayling Hospital', 'Grayling', '177', '27', '37016', '15466', '14681')


def test_split_has_no_hospital_in_two_sets():
    tr, va, te = D.split_hospitals([f'{i:06d}' for i in range(200)])
    assert not (tr & va or tr & te or va & te)
    assert (len(tr), len(va), len(te)) == (160, 20, 20)


def test_tokenizer_round_trip():
    tok = D.CharTokenizer(REC + '\n')
    assert tok.decode(tok.encode(REC)) == REC


def test_finetuning_batch_grades_only_the_explanation():
    line = REC + ' || It was paid $15,466 in total.'
    tok = D.CharTokenizer(line + '\n')
    (ids, given), = D.encode_examples([line], tok)
    x, y = D.get_sft_batch([(ids, given)], 1, 'cpu', tok.stoi['\n'])
    graded = ''.join(tok.itos[i] for i in y[0].tolist() if i != -100)
    assert graded == 'It was paid $15,466 in total.\n'           # the record itself is never graded


# ---- placeholders ----------------------------------------------------------------------------------

def test_placeholders_round_trip():
    s = ("Munson Healthcare Grayling Hospital in Grayling, MI treated 27 Medicare patients, averaging a bill of "
         "$37,016 but receiving $15,466 in total payments, with Medicare covering 95% of those payments.")
    d = D.delex(REC, s)
    assert '$' not in d and '{BILL}' in d and '{HOSPITAL}' in d and '{PATIENTS}' in d
    assert D.relex(REC, d) == s


# ---- the teacher's checks ------------------------------------------------------------------------

def test_teacher_check_accepts_correct_sentences():
    fx = facts(D.parse(REC))
    for s in ["Munson Healthcare Grayling Hospital treated 27 Medicare patients; it billed $37,016 on average, "
              "was paid $15,466 in total, and Medicare paid $14,681 of that, which is 95% of the payment.",
              "Grayling Hospital charged $37,016 on average, received $15,466 in total payments, and Medicare "
              "paid $14,681, or 95% of the payment."]:
        assert check(s, fx) is None, s


def test_teacher_check_catches_each_kind_of_mistake():
    fx = facts(D.parse(REC))
    assert check("Grayling Hospital billed $37,016 for 30 patients.", fx) == 'invented_number'
    assert check("Grayling Hospital billed $37,016 and received $15,466 from Medicare.", fx) == 'wrong_label'
    assert check("Medicare paid $14,681 out of $15,466 billed.", fx) == 'wrong_label'
    assert check("It billed $37,016, was paid $15,466 in total, and Medicare paid $14,681, "
                 "covering 95% of the charges.", fx) == 'wrong_label'


# ---- flagging ------------------------------------------------------------------------------------

def test_auc_matches_the_slow_definition():
    rng = random.Random(0)
    for _ in range(20):
        pos = [round(rng.random(), 1) for _ in range(40)]           # rounding forces ties
        neg = [round(rng.random(), 1) for _ in range(35)]
        slow = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        assert abs(auc(pos, neg) - slow) < 1e-12


def test_rules_catch_arithmetic_and_pass_a_clean_record():
    assert rule_hits(REC, ratio_limit=20) == []
    swapped = corrupt(REC, 'payments_swapped', {}, {}, random.Random(0))
    assert 'Medicare paid more than the total' in rule_hits(swapped, 20)
    tenfold = corrupt(REC, 'extra_digit', {}, {}, random.Random(0))
    assert any('x the payment' in h for h in rule_hits(tenfold, 20))


def test_planted_errors_change_exactly_one_field():
    drg = {'177': 'RESPIRATORY INFECTIONS AND INFLAMMATIONS WITH MCC', '602': 'CELLULITIS WITH MCC'}
    cities = {'MI': ['Grayling', 'Gaylord']}
    base = D.parse(REC)
    for kind, field in [('drg_mismatch', 'desc'), ('wrong_city', 'city'), ('extra_digit', 'c')]:
        changed = [k for k, v in D.parse(corrupt(REC, kind, drg, cities, random.Random(1))).items() if v != base[k]]
        assert changed == [field], (kind, changed)


# ---- model ---------------------------------------------------------------------------------------

def test_gpt_learns_a_tiny_text():
    torch.manual_seed(0)
    text = ('\n' + REC) * 20
    tok = D.CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    model = GPT(len(tok), n_embd=32, context_size=64, n_head=2, n_layer=1, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        x, y = D.get_batch(data, 8, 64, 'cpu')
        loss = model(x, y)[1]
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first / 2                                   # it is actually learning


def test_init_from_copies_matching_characters_and_positions():
    torch.manual_seed(0)
    small = GPT(5, n_embd=16, context_size=10, n_head=2, n_layer=1)
    ck = {'model_state': small.state_dict(), 'stoi': {c: i for i, c in enumerate('abcde')}}
    stoi = {c: i for i, c in enumerate('xabcdef')}                    # new characters, new order
    big = GPT(len(stoi), n_embd=16, context_size=20, n_head=2, n_layer=1)
    init_from(big, ck, stoi)
    assert torch.equal(big.token_embedding_table.weight[stoi['c']], small.token_embedding_table.weight[2])
    assert torch.equal(big.position_embedding_table.weight[:10], small.position_embedding_table.weight)
