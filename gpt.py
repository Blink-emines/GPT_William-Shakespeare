"""
gpt.py - a character-level, decoder-only Transformer (GPT) trained on Tiny Shakespeare.

This is the final model from Andrej Karpathy's "Let's build GPT: from scratch, in code,
spelled out." It follows "Attention Is All You Need" (Vaswani et al., 2017): token and
position embeddings, a stack of pre-norm Transformer blocks (masked multi-head
self-attention + feed-forward, both with residual connections), and a linear head that
predicts the next character.

Usage
-----
    python gpt.py train                         # train, save the best checkpoint, write samples
    python gpt.py sample                        # generate text from the saved checkpoint
    python gpt.py sample --prompt "ROMEO:" --tokens 1000
    python gpt.py train --target_loss 1.7       # stop early once val loss <= 1.7
    python gpt.py train --iters 200             # quick smoke test
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters (the same values as the final model in the lecture)
# ---------------------------------------------------------------------------
batch_size = 64        # B: independent sequences processed in parallel
block_size = 256       # T: maximum context length, in characters
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384           # C: embedding dimension
n_head = 6             # head_size = 384 / 6 = 64
n_layer = 6
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = 'input.txt'
ckpt_path = 'gpt.pt'
log_path = 'train_log.json'
samples_dir = 'samples'
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data and character-level tokenizer
# ---------------------------------------------------------------------------
def load_text(path=data_path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class CharTokenizer:
    """Maps each unique character to an integer and back (65 tokens for Tiny Shakespeare)."""

    def __init__(self, chars):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """One head of masked (causal) self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular mask: position t may only attend to positions <= t
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                       # (B, T, hs)
        q = self.query(x)                                     # (B, T, hs)
        # attention scores, scaled by 1/sqrt(head_size) so softmax stays diffuse at init
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # no peeking at the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                                     # (B, T, hs)
        return wei @ v                                        # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel; their outputs are concatenated and projected."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP: each token processes the information attention gathered."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # 4x expansion, as in the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # projection back into the residual stream
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP).
    Uses pre-norm residual connections: x + f(LayerNorm(x))."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # what the token is
        self.position_embedding_table = nn.Embedding(block_size, n_embd)   # where the token is
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                    # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb                                                        # (B, T, C)
        x = self.blocks(x)                                                           # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                     # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive sampling: predict a character, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # crop to the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                # only the last time step matters
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(iters=max_iters, target_loss=None):
    torch.manual_seed(1337)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True   # faster fp32 matmuls on Ampere+ GPUs
        torch.backends.cudnn.allow_tf32 = True
    # bf16 mixed precision on GPUs that support it: ~2x faster, same model and results
    use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()
    autocast = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16)

    text = load_text()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))                 # first 90% train, last 10% validation
    splits = {'train': data[:n], 'val': data[n:]}

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # targets = inputs shifted by one
        return x.to(device), y.to(device)

    model = GPTLanguageModel(tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | parameters: {n_params / 1e6:.2f}M | vocab: {tok.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ('train', 'val'):
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with autocast():
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    history = []
    best_val = float('inf')
    stop_reason = 'max_iters'
    t0 = time.time()
    try:
        for it in range(iters + 1):
            if it % eval_interval == 0 or it == iters:
                losses = estimate_loss()
                elapsed = time.time() - t0
                history.append({'step': it, 'train': losses['train'], 'val': losses['val'], 'time_s': round(elapsed, 1)})
                note = ''
                if losses['val'] < best_val:     # keep the checkpoint that generalizes best
                    best_val = losses['val']
                    torch.save({'model': model.state_dict(), 'chars': tok.chars,
                                'step': it, 'val_loss': best_val}, ckpt_path)
                    note = '  (saved)'
                print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} "
                      f"| {elapsed:6.0f}s{note}", flush=True)
                if target_loss is not None and losses['val'] <= target_loss:
                    stop_reason = f'reached target val loss {target_loss}'
                    break
            if it == iters:
                break

            xb, yb = get_batch('train')
            with autocast():
                _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:          # Ctrl+C: stop early but keep the best checkpoint and write the log
        stop_reason = 'interrupted'
    print(f"stopped: {stop_reason}")

    meta = {
        'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'precision': 'bf16 autocast' if use_bf16 else 'fp32',
        'parameters': n_params,
        'best_val_loss': best_val,
        'train_time_s': round(time.time() - t0, 1),
        'stop_reason': stop_reason,
        'hyperparameters': {
            'batch_size': batch_size, 'block_size': block_size, 'max_iters': iters,
            'learning_rate': learning_rate, 'n_embd': n_embd, 'n_head': n_head,
            'n_layer': n_layer, 'dropout': dropout,
        },
        'history': history,
    }
    with open(log_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nbest val loss {best_val:.4f} | checkpoint: {ckpt_path} | log: {log_path}\n")

    sample(tokens=500, save_as='sample_500.txt')
    sample(tokens=10000, save_as='sample_10k.txt', show=False)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def load_model(path=ckpt_path):
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint at {path}. Train first: python gpt.py train")
    ckpt = torch.load(path, map_location=device)
    tok = CharTokenizer(ckpt['chars'])
    model = GPTLanguageModel(tok.vocab_size).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, tok


def sample(prompt='\n', tokens=500, seed=None, save_as=None, show=True):
    if seed is not None:
        torch.manual_seed(seed)
    model, tok = load_model()
    start = tok.encode(prompt) or [0]
    idx = torch.tensor([start], dtype=torch.long, device=device)
    text = tok.decode(model.generate(idx, max_new_tokens=tokens)[0].tolist())
    if show:
        print(text)
    if save_as:
        os.makedirs(samples_dir, exist_ok=True)
        with open(os.path.join(samples_dir, save_as), 'w', encoding='utf-8') as f:
            f.write(text)
    return text


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Character-level GPT on Tiny Shakespeare')
    p.add_argument('mode', choices=['train', 'sample'])
    p.add_argument('--iters', type=int, default=max_iters, help='training steps')
    p.add_argument('--target_loss', type=float, default=None, help='stop early once val loss reaches this')
    p.add_argument('--prompt', default='\n', help='text to condition on (sample mode)')
    p.add_argument('--tokens', type=int, default=500, help='characters to generate (sample mode)')
    p.add_argument('--seed', type=int, default=None, help='random seed for sampling')
    p.add_argument('--out', default=None, help='also save the sample to samples/<out>')
    args = p.parse_args()

    if args.mode == 'train':
        train(args.iters, args.target_loss)
    else:
        sample(args.prompt, args.tokens, args.seed, args.out)
