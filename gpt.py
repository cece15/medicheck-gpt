#!/usr/bin/env python3
"""
MediCheck-GPT: small character-level GPTs trained on CMS Medicare inpatient records, used three ways:
  1. explain: turn a record into a plain-English sentence, learned from an LLM teacher (--teacher, then --train)
  2. write:   generate records in the CMS style (--eval)
  3. flag:    rank records by how surprising they are, to catch silent errors (--flag)

One command per stage of the flow:
  ./gpt.py --input data/raw/MUP_INP_RY26_P03_V10_DY24_PrvSvc.CSV --prepare data/processed
  ./gpt.py --input data/processed --train checkpoints/bigram.pth --model bigram --context_size 8 --lr 1e-2 --steps 3000
  ./gpt.py --input data/processed --train checkpoints/gpt.pth --steps 10000
  ./gpt.py --input data/processed --teacher data/described --limit 3000
  ./gpt.py --input data/described --train checkpoints/explain.pth --steps 10000
  ./gpt.py --eval checkpoints/explain.pth --prompt "<paste a record>"
  ./gpt.py --eval checkpoints/gpt.pth --prompt "Trinity Health Livonia Hospital in Livonia, MI billed DRG "
  ./gpt.py --eval checkpoints/gpt.pth --interactive
  ./gpt.py --input data/processed --evaluate checkpoints/gpt.pth
  ./gpt.py --input data/processed --flag checkpoints/gpt.pth
"""
import argparse
import json
import os

import torch

from src import data as D
from src import flag as FL
from src import metrics as M
from src.generate import complete, load
from src.model import build, count_params
from src.train import train


def pick_device():
    """GPU if there is one (Colab T4 / NVIDIA), Apple GPU on an M-series Mac, else CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def read_split(folder, name):
    return open(os.path.join(folder, f'{name}.txt')).read()


def records(text):
    return [r for r in text.split('\n') if r]                     # one record per non-empty line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # what to do (pick one)
    p.add_argument('--input', help='CMS CSV (with --prepare) or the processed folder (train/eval/flag)')
    p.add_argument('--prepare', metavar='OUT_DIR', help='CSV -> train/val/test text, split by hospital')
    p.add_argument('--teacher', metavar='OUT_DIR', help='LLM writes a plain-English description for each record')
    p.add_argument('--teacher_model', default=None, help='Hugging Face model id for the teacher')
    p.add_argument('--limit', type=int, default=3000, help='max training records sent to the teacher')
    p.add_argument('--train', metavar='CKPT', help='train a model and save the best checkpoint here')
    p.add_argument('--eval', metavar='CKPT', help='generate: complete --prompt, or chat with --interactive')
    p.add_argument('--evaluate', metavar='CKPT', help='perplexity, token accuracy, ROUGE-L, record validity')
    p.add_argument('--flag', metavar='CKPT', help='plant silent errors in test records and measure detection')
    p.add_argument('--flag_file', metavar='TXT', help='with --flag: rank the records in this file instead')
    # model and training hyperparameters (defaults = the configuration of my first GPT run, 14 Nov)
    p.add_argument('--model', default='gpt', choices=['gpt', 'bigram'])
    p.add_argument('--states', default='MI,OH', help='states to keep when preparing')
    p.add_argument('--n_embd', type=int, default=128)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--context_size', type=int, default=271)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--patience', type=int, default=4, help='evaluations without val improvement before stopping')
    p.add_argument('--seed', type=int, default=1337)
    # generation settings
    p.add_argument('--prompt', default='')
    p.add_argument('--n', type=int, default=3, help='how many records to generate')
    p.add_argument('--max_new', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=None)
    p.add_argument('--interactive', action='store_true')
    p.add_argument('--hf_metrics', action='store_true', help='also compute BLEU and BERTScore (downloads models)')
    args = p.parse_args()
    device = pick_device()
    print('device:', device)

    # ---- stage 1: data ------------------------------------------------------------------------
    if args.prepare:
        os.makedirs(args.prepare, exist_ok=True)
        stats = D.prepare(args.input, args.prepare, tuple(args.states.split(',')))
        print(json.dumps(stats, indent=1))
        return

    # ---- stage 1b: LLM teacher writes reference descriptions (Kacper's recommendation) -----------
    if args.teacher:
        from src.teacher import DEFAULT_TEACHER, write_descriptions
        os.makedirs(args.teacher, exist_ok=True)
        summary = {}
        for split in ['train', 'val', 'test']:
            recs = records(read_split(args.input, split))
            if split == 'train':
                recs = recs[:args.limit]                          # the teacher is the slow part; cap training examples
            pairs, rejected = write_descriptions(recs, args.teacher_model or DEFAULT_TEACHER, device=device)
            open(os.path.join(args.teacher, f'{split}.txt'), 'w').write('\n' + '\n'.join(pairs) + '\n')
            summary[split] = {'records': len(recs), 'kept': len(pairs), 'rejected': rejected}
        print(json.dumps(summary, indent=1))
        json.dump(summary, open(os.path.join(args.teacher, 'teacher_report.json'), 'w'), indent=1)
        print('example:', pairs[0])
        return

    # ---- stage 2-3: train ---------------------------------------------------------------------
    if args.train:
        train_text, val_text = read_split(args.input, 'train'), read_split(args.input, 'val')
        tok = D.CharTokenizer(train_text)                         # vocabulary comes from training data only
        train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
        val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
        task = 'explain' if ' || ' in train_text else 'record'    # pairs file -> explain task
        longest = max(len(r) for r in records(train_text)) + 2
        if args.model == 'gpt' and args.context_size < longest:     # the window must hold one whole example
            print(f'context_size {args.context_size} -> {longest} so one full example fits')
            args.context_size = longest
        config = {'vocab_size': len(tok)}
        if args.model == 'gpt':
            config.update(n_embd=args.n_embd, context_size=args.context_size, n_head=args.n_head,
                          n_layer=args.n_layer, dropout=args.dropout)
        model = build(args.model, **config).to(device)
        print(f'{args.model}: {count_params(model) / 1e6:.2f}M parameters | vocab {len(tok)} | '
              f'train {len(train_data):,} chars | val {len(val_data):,} chars')
        os.makedirs(os.path.dirname(args.train) or '.', exist_ok=True)
        val_recs = records(val_text)

        def on_eval(m):                                           # metrics reported from inside the training loop
            if task == 'explain':
                return {'val_rougeL': M.describe_eval(m, tok, val_recs, device, n=8)[0]['rougeL']}
            gens = [complete(m, tok, '', 300, 0.8, None, device) for _ in range(8)]
            return {'gen_valid': M.validity(gens, {}, {})['medicare_le_total_payment']}

        train(model, tok, train_data, val_data, args, config, args.train, device,
              log_path=args.train.replace('.pth', '_history.json'),  # bigram batches use --context_size
              task=task, on_eval=on_eval if args.model == 'gpt' else None)
        return

    # ---- stage 4: generate --------------------------------------------------------------------
    if args.eval:
        model, tok, ck = load(args.eval, device)
        explain = ck.get('task') == 'explain'

        def gen(prompt):
            if explain:                                           # record in, plain-English sentence out
                prompt = prompt.rstrip() + ' || '
                return complete(model, tok, prompt, 400, args.temperature, args.top_k, device)[len(prompt):]
            return complete(model, tok, prompt, args.max_new, args.temperature, args.top_k, device)

        if args.interactive:                                      # live demo mode
            print('Paste a record to explain (q = quit).' if explain else
                  'Type the start of a record (Enter = from scratch, q = quit).')
            while (prompt := input('\n> ')) != 'q':
                print(gen(prompt))
        else:
            for _ in range(args.n):
                print(gen(args.prompt), '\n')
        return

    # ---- stage 5: evaluate --------------------------------------------------------------------
    if args.evaluate:
        model, tok, ck = load(args.evaluate, device)
        test_text, train_text = read_split(args.input, 'test'), read_split(args.input, 'train')
        test_recs, train_recs = records(test_text), records(train_text)
        ctx = ck['config'].get('context_size', args.context_size)
        ppl, acc = M.perplexity_and_accuracy(model, tok, test_text, device, ctx)
        if ck.get('task') == 'explain':                           # student vs teacher on unseen hospitals
            scores, pairs = M.describe_eval(model, tok, test_recs, device, use_hf=args.hf_metrics)
            report = {'model': 'explain', 'test_perplexity': round(ppl, 3), 'test_token_accuracy': round(acc, 3),
                      **{k: round(v, 3) for k, v in scores.items()},
                      'numbers_faithful_to_record': round(M.numbers_faithful(pairs), 3)}
            print(json.dumps(report, indent=1))
            print('\nexample\n  record :', pairs[0][0], '\n  teacher:', pairs[0][1], '\n  student:', pairs[0][2])
            json.dump(report, open(args.evaluate.replace('.pth', '_eval.json'), 'w'), indent=1)
            return
        gen_scores, pairs = M.completion_eval(model, tok, test_recs, device, use_hf=args.hf_metrics)
        drg_map, place_map = M.known_facts(train_recs)
        torch.manual_seed(args.seed)
        free = [complete(model, tok, '', args.max_new, args.temperature, None, device) for _ in range(200)]
        report = {'model': ck['kind'], 'test_perplexity': round(ppl, 3), 'test_token_accuracy': round(acc, 3),
                  **{k: round(v, 3) for k, v in gen_scores.items()},
                  'validity_generated': M.validity(free, drg_map, place_map),
                  'validity_real_test_records': M.validity(test_recs, drg_map, place_map)}  # the ceiling to compare against
        print(json.dumps(report, indent=1))
        print('\nexample completion\n  real :', pairs[0][0], '\n  model:', pairs[0][1])
        json.dump(report, open(args.evaluate.replace('.pth', '_eval.json'), 'w'), indent=1)
        return

    # ---- stage 6: flag silent errors ----------------------------------------------------------
    if args.flag:
        model, tok, ck = load(args.flag, device)
        if args.flag_file:                                        # real use on any file of records
            for score, rec in FL.flag_file(model, tok, records(open(args.flag_file).read()), device):
                print(f'{score:6.2f}  {rec}')
            return
        res, ex = FL.run(model, tok, records(read_split(args.input, 'test')),
                         records(read_split(args.input, 'train')), device, args.seed)
        print(f"{'error type':<18}{'n':>5}{'AUC mean':>10}{'AUC peak':>10}")
        for k, v in res.items():
            print(f"{k:<18}{v['n']:>5}{v['auc_mean']:>10}{v['auc_peak']:>10}")
        print('\nexample planted errors:')
        for k, e in ex.items():
            print(f'  {k}: {e}')
        json.dump(res, open(args.flag.replace('.pth', '_flag.json'), 'w'), indent=1)
        return

    p.print_help()


if __name__ == '__main__':
    main()
