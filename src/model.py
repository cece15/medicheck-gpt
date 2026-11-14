"""
Step 2 of the flow: the models.

BigramLanguageModel and the GPT below are adapted from Kacper Raczy's baseline.ipynb and
transformer.ipynb (Open Avenues Build Fellowship, "Build, Train, and Deploy a Transformer Language
Model in PyTorch"), which follow Andrej Karpathy's nanoGPT lectures. My changes are marked.
Architecture: Vaswani et al., "Attention Is All You Need" (2017); pre-norm blocks as in GPT-2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Baseline: predicts the next character from the current character only. No memory, no attention.
    If the transformer cannot beat this, the transformer has learned nothing."""

    def __init__(self, vocab_size, **_):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   # row i = scores for "what follows char i"
        self.context_size = 1                                                # it only ever looks at one character

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)                             # (B, T, vocab)
        return logits, _loss(logits, targets)


class Head(nn.Module):
    """One head of causal self-attention: each character decides which earlier characters matter to it."""

    def __init__(self, head_size, n_embd, context_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)      # "what do I contain?"
        self.query = nn.Linear(n_embd, head_size, bias=False)    # "what am I looking for?"
        self.value = nn.Linear(n_embd, head_size, bias=False)    # "what do I pass on if chosen?"
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))   # causal mask
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)                                    # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5                  # change: scale by sqrt(head size) so softmax stays soft
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))         # no peeking at future characters
        wei = self.dropout(F.softmax(wei, dim=-1))                           # attention weights, each row sums to 1
        return wei @ self.value(x)                                           # weighted mix of earlier characters


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, each free to track a different pattern (e.g. one the DRG code, one the city)."""

    def __init__(self, num_heads, head_size, n_embd, context_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)                 # change: mix the heads back together
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    """After characters talk to each other (attention), each one thinks on its own (this MLP)."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),   # change: 4x wider hidden layer, as in GPT-2
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate, then compute. Residuals (x + ...) keep gradients flowing in deep stacks."""

    def __init__(self, n_embd, n_head, context_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, context_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)      # change: pre-norm for stable training

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only transformer. context_size 271 = the longest record (269 chars) plus its two newlines,
    so the model always sees a whole record at once."""

    def __init__(self, vocab_size, n_embd=128, context_size=271, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)        # what each character is
        self.position_embedding_table = nn.Embedding(context_size, n_embd)   # where it sits in the record
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, context_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)                         # back to one score per character

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)                             # change: device follows the input
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        logits = self.lm_head(self.ln_f(self.blocks(x)))                     # (B, T, vocab)
        return logits, _loss(logits, targets)


def _loss(logits, targets):
    """Cross-entropy: how surprised the model is by the true next character, averaged."""
    if targets is None:
        return None
    B, T, C = logits.shape
    return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))


MODELS = {'bigram': BigramLanguageModel, 'gpt': GPT}


def build(kind, **config):
    return MODELS[kind](**config)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
