"""
End-to-end test: the whole command-line flow on a small synthetic file (clearly fake hospitals),
with tiny models so it runs on a laptop CPU in a few minutes. It checks that every stage runs and
hands the next one what it needs; it says nothing about model quality, which the notebooks measure.
Run from the repo root:  python -m pytest tests -q
"""
import csv
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import COLUMNS                                         # noqa: E402


def fake_csv(path):
    r = random.Random(0)
    drgs = {f'{i:03d}': f'CONDITION NUMBER {i} WITH MCC' for i in range(1, 40)}
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for h in range(40):
            st = r.choice(['MI', 'OH'])
            for d in r.sample(sorted(drgs), 12):
                c = r.randint(20000, 200000); t = int(c * r.uniform(.15, .4)); m = int(t * r.uniform(.6, .95))
                w.writerow([f'{230000 + h:06d}', f'Test Hospital {h}', f'Town{h % 9}', '1 Main St', '26', '48000',
                            st, '1', 'Metro', d, drgs[d], str(r.randint(11, 90)), f'{c}.5', f'{t}.2', f'{m}.1'])


def run(*args):
    out = subprocess.run([sys.executable, str(ROOT / 'gpt.py'), *map(str, args)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_whole_flow(tmp_path):
    fake_csv(tmp_path / 'cms.csv')
    proc, ck = tmp_path / 'proc', tmp_path / 'ck'
    run('--input', tmp_path / 'cms.csv', '--prepare', proc)
    run('--input', proc, '--train', ck / 'bigram.pth', '--model', 'bigram', '--context_size', 8, '--steps', 50, '--eval_every', 25)
    run('--input', proc, '--train', ck / 'gpt.pth', '--n_layer', 1, '--n_embd', 32, '--steps', 40, '--eval_every', 20)
    assert 'billed' in run('--eval', ck / 'gpt.pth', '--n', 1, '--prompt', 'Test Hospital 3 in Town3, MI billed')
    assert '"test_perplexity"' in run('--input', proc, '--evaluate', ck / 'gpt.pth')
    assert 'paired_raw' in run('--input', proc, '--flag', ck / 'gpt.pth')

    # explainer with placeholders, on template sentences standing in for the LLM teacher
    from src.data import parse
    from src.teacher import facts
    desc = tmp_path / 'desc'; desc.mkdir()
    for split in ['train', 'val', 'test']:
        lines = []
        for rec in (proc / f'{split}.txt').read_text().split('\n'):
            f = parse(rec)
            if f:
                fx = facts(f)
                lines.append(f"{rec} || {fx['hospital']} treated {fx['patients']}; it {fx['billed']} and {fx['paid_total']}.")
        (desc / f'{split}.txt').write_text('\n' + '\n'.join(lines) + '\n')
    run('--input', desc, '--train', ck / 'explain.pth', '--init_from', ck / 'gpt.pth', '--delex', '--steps', 30, '--eval_every', 15)
    assert '"passes_teacher_checks"' in run('--input', desc, '--evaluate', ck / 'explain.pth')

    # the file flagger: a planted arithmetic error must be caught by the rules
    test_recs = [r for r in (proc / 'test.txt').read_text().split('\n') if parse(r)][:4]
    f = parse(test_recs[0])
    bad = test_recs[0].replace(f"total payment ${f['t']}, Medicare payment ${f['m']}",
                               f"total payment ${f['m']}, Medicare payment ${f['t']}")
    (tmp_path / 'check.txt').write_text('\n'.join(test_recs[1:] + [bad]) + '\n')
    out = run('--input', proc, '--flag', ck / 'gpt.pth', '--flag_file', tmp_path / 'check.txt')
    ranked = [line for line in out.split('\n') if 'RULE:' in line or 'model only' in line]
    assert ranked[0].strip().split('  ', 1)[1].startswith('RULE: Medicare paid more')   # caught, and ranked first
    assert sum('RULE:' in line for line in ranked) == 1                                  # clean records pass the rules
