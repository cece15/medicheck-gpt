"""
Step 4 of the flow: inference. Load a checkpoint, complete a prompt one character at a time.
Temperature and top-k come from Lecture 6-7, slide 12.
"""
import torch
import torch.nn.functional as F

from .data import CharTokenizer
from .model import build


def load(path, device):
    """Rebuild the exact model that was saved: same architecture, same weights, same vocabulary."""
    ck = torch.load(path, map_location=device)
    model = build(ck['kind'], **ck['config']).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()                                                  # inference mode: dropout off
    return model, CharTokenizer(stoi=ck['stoi']), ck


@torch.no_grad()
def complete(model, tok, prompt='', max_new=300, temperature=1.0, top_k=None, device='cpu', stop_at_newline=True):
    """Feed '\\n' + prompt, then sample until the record ends (newline) or max_new characters."""
    idx = torch.tensor([tok.encode('\n' + prompt)], dtype=torch.long, device=device)   # newline = "a record starts here"
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.context_size:])           # only the last context_size chars fit in the window
        logits = logits[:, -1, :] / temperature                   # <1 sharper and safer, >1 flatter and more inventive
        if top_k:                                                 # keep only the k most likely next characters
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)     # sample, don't argmax: argmax loops on repeats
        if stop_at_newline and nxt.item() == tok.stoi['\n']:      # the model decided the record is finished
            break
        idx = torch.cat((idx, nxt), dim=1)
    return tok.decode(idx[0].tolist()).lstrip('\n')
