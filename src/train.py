"""
Step 3 of the flow: training. AdamW, cosine learning-rate decay with warmup, early stopping on
validation loss, and a checkpoint that always holds the best model seen so far (Lecture 6-7, slide 13).
"""
import json
import math
import time

import torch

@torch.no_grad()
def estimate_loss(model, batchers, iters=50):
    """Average loss over many random batches, so one lucky batch cannot fool us."""
    model.eval()                                                  # dropout off while measuring
    out = {}
    for split, next_batch in batchers.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = next_batch()
            losses[k] = model(xb, yb)[1].item()
        out[split] = losses.mean().item()
    model.train()                                                 # back to training mode
    return out


def lr_at(step, max_steps, lr, warmup):
    """Warm up linearly for `warmup` steps, then follow a cosine curve down to 10% of the peak."""
    if step < warmup:
        return lr * (step + 1) / warmup                           # small steps first while weights are random
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, tok, batchers, args, config, ckpt_path, log_path=None, task='record', on_eval=None):
    """Train, log every eval_every steps, keep the best checkpoint, stop when val stops improving.
    on_eval: optional function(model) -> dict of extra metrics, run at every evaluation (Kacper, 10 Nov:
    report evaluation metrics periodically from inside the training loop)."""
    torch.manual_seed(args.seed)                                  # reproducible batches and dropout
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, bad_evals, history, t0 = float('inf'), 0, [], time.time()

    for step in range(args.steps):
        for g in opt.param_groups:                                # set this step's learning rate
            g['lr'] = lr_at(step, args.steps, args.lr, args.warmup)

        if step % args.eval_every == 0 or step == args.steps - 1:
            l = estimate_loss(model, batchers)
            l['val_perplexity'] = math.exp(l['val'])                 # perplexity = exp(cross-entropy)
            if on_eval:
                l.update(on_eval(model))                              # e.g. ROUGE-L on a few validation examples
            history.append({'step': step, **l, 'seconds': round(time.time() - t0)})
            extra = ' | '.join(f'{k} {v:.3f}' for k, v in l.items() if k not in ('train', 'val'))
            print(f"step {step:>6} | train {l['train']:.4f} | val {l['val']:.4f} | {extra} | {time.time() - t0:.0f}s")
            if l['val'] < best:                                   # new best: save it, reset patience
                best, bad_evals = l['val'], 0
                save(ckpt_path, model, tok, config, args, history, task)
            else:                                                 # val got worse: one strike
                bad_evals += 1
                if bad_evals >= args.patience:                    # train keeps falling, val keeps rising = overfitting
                    print(f'early stop at step {step}: val has not improved for {args.patience} evaluations')
                    break

        xb, yb = batchers['train']()
        _, loss = model(xb, yb)                                   # forward: how wrong are we?
        opt.zero_grad(set_to_none=True)
        loss.backward()                                           # backward: which way should each weight move?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against one exploding batch
        opt.step()                                                # move them

    if log_path:
        json.dump(history, open(log_path, 'w'), indent=1)         # loss curve for the slides
    print(f'best val loss {best:.4f} saved to {ckpt_path}')
    return history


def save(path, model, tok, config, args, history, task='record'):
    """Everything needed to rebuild the model later: weights, architecture, tokenizer, training settings."""
    torch.save({'kind': args.model, 'task': task, 'model_state': model.state_dict(), 'config': config,
                'stoi': tok.stoi, 'train_args': vars(args), 'history': history}, path)
