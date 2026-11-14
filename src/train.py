"""
Step 3 of the flow: training. AdamW, cosine learning-rate decay with warmup, early stopping on
validation loss, and a checkpoint that always holds the best model seen so far (Lecture 6-7, slide 13).
"""
import json
import math
import time

import torch

from .data import get_batch


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_size, device, iters=50):
    """Average loss over many random batches, so one lucky batch cannot fool us."""
    model.eval()                                                  # dropout off while measuring
    out = {}
    for split, d in data.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(d, batch_size, context_size, device)
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


def train(model, tok, train_data, val_data, args, config, ckpt_path, device, log_path=None):
    """Train, log every eval_every steps, keep the best checkpoint, stop when val stops improving."""
    torch.manual_seed(args.seed)                                  # reproducible batches and dropout
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctx = config.get('context_size', args.context_size)
    best, bad_evals, history, t0 = float('inf'), 0, [], time.time()

    for step in range(args.steps):
        for g in opt.param_groups:                                # set this step's learning rate
            g['lr'] = lr_at(step, args.steps, args.lr, args.warmup)

        if step % args.eval_every == 0 or step == args.steps - 1:
            l = estimate_loss(model, {'train': train_data, 'val': val_data}, args.batch_size, ctx, device)
            history.append({'step': step, **l, 'seconds': round(time.time() - t0)})
            print(f"step {step:>6} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time() - t0:.0f}s")
            if l['val'] < best:                                   # new best: save it, reset patience
                best, bad_evals = l['val'], 0
                save(ckpt_path, model, tok, config, args, history)
            else:                                                 # val got worse: one strike
                bad_evals += 1
                if bad_evals >= args.patience:                    # train keeps falling, val keeps rising = overfitting
                    print(f'early stop at step {step}: val has not improved for {args.patience} evaluations')
                    break

        xb, yb = get_batch(train_data, args.batch_size, ctx, device)
        _, loss = model(xb, yb)                                   # forward: how wrong are we?
        opt.zero_grad(set_to_none=True)
        loss.backward()                                           # backward: which way should each weight move?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against one exploding batch
        opt.step()                                                # move them

    if log_path:
        json.dump(history, open(log_path, 'w'), indent=1)         # loss curve for the slides
    print(f'best val loss {best:.4f} saved to {ckpt_path}')
    return history


def save(path, model, tok, config, args, history):
    """Everything needed to rebuild the model later: weights, architecture, tokenizer, training settings."""
    torch.save({'kind': args.model, 'model_state': model.state_dict(), 'config': config,
                'stoi': tok.stoi, 'train_args': vars(args), 'history': history}, path)
