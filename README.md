# nanoGPT — Character-Level Language Model on Shakespeare

A from-scratch implementation of a decoder-only transformer (GPT architecture) trained on the Tiny Shakespeare dataset. Built by typing and understanding every line, following Andrej Karpathy's "Let's build GPT" walkthrough.

---

## What this is

A minimal, readable implementation of the GPT architecture — multi-head causal self-attention, positional embeddings, layer norm, feedforward blocks with residual connections — trained character-by-character on Shakespeare's complete works. No HuggingFace, no abstractions. Every component is explicit.

---

## Architecture

| Component | Detail |
|---|---|
| Model type | Decoder-only transformer (GPT-style) |
| Attention | Multi-head causal self-attention with triangular mask |
| Heads | 6 attention heads |
| Embedding dim | 32 (`n_embd`) |
| Layers | 6 transformer blocks |
| Context window | 8 tokens (`block_size`) |
| FFN | Linear → ReLU → Linear with 4× expansion |
| Norm | Pre-norm LayerNorm (applied before attention and FFN) |
| Residual | `x = x + sublayer(norm(x))` throughout |
| Dropout | 0.2 |
| Optimizer | AdamW, lr=3e-4 |
| Tokenization | Character-level (83 unique chars from Shakespeare) |

---

## Training

- Dataset: Tiny Shakespeare (~1MB, ~1M characters)
- Train/val split: 90/10
- Batch size: 32, block size: 8
- Max iterations: 5000
- Hardware: NVIDIA RTX 5060 (CUDA)

### Loss curve

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 4.86 | 4.77 |
| 500 | 2.81 | 3.02 |
| 1000 | 2.60 | 2.83 |
| 1500 | 2.49 | 2.78 |
| 2000 | 2.43 | 2.71 |
| 2500 | 2.39 | 2.66 |
| 3000 | 2.35 | 2.61 |
| 3500 | 2.33 | 2.59 |
| 4000 | 2.30 | 2.55 |
| 4500 | 2.27 | 2.55 |

Starting loss ~4.86 is consistent with `ln(83) ≈ 4.42` for a randomly initialized model over a 83-token vocabulary. Steady convergence with mild train/val gap — expected for a model this small.

---

## Sample output (500 tokens, after 5000 steps)

```
    clER menty wavellvef
  Queak th mp fou Yandir yout 'ard,
   Yer, o my shillow taut man!
      Anthit coreeso enender Lishar, s youd, nco nk,
Shase us igseshy Q,
 TO mun ntrt te do pllear, iunthe hacr'of tere por
    Fither thy. ar foulfour br aden
   Whet toun the tou yoond yorntso
```

Not coherent English — but structurally Shakespeare-shaped: indentation matching play format, character-name casing, punctuation in plausible positions, English-length word patterns. Correct output for this model scale and training budget.

---

## Key implementation notes

**Why `y = data[i+1:i+block_size+1]`**: The model's task is next-token prediction at every position. A single (x, y) window of length 8 contains 8 stacked prediction problems — each position's label is the character that actually follows it in the real text.

**Why causal masking**: Training must match generation conditions. At generation time the model only has past tokens, never future ones. The triangular mask enforces this during training by setting future-position attention scores to `-inf` before softmax, making their weights exactly 0.

**Why `n_embd` must divide evenly by `n_head`**: Multi-head attention splits the embedding dimension across heads (`head_size = n_embd // n_head`), has each head attend independently, then concatenates back. If the split isn't exact, the concatenated output dimension won't match `n_embd`, breaking the projection layer (`nn.Linear(n_embd, n_embd)`).

**Pre-norm vs post-norm**: This implementation uses pre-norm (`x = x + sublayer(norm(x))`), which is more stable during training than the post-norm formulation in the original "Attention is All You Need" paper. All modern LLMs use pre-norm.

---

## Files

```
model.py        — full model: tokenizer, data loading, Head, MultiHeadAttention,
                  FeedForward, Block, BigramLanguageModule, training loop, generation
input.txt       — Tiny Shakespeare dataset (~1MB)
```

---

## Run

```bash
python model.py
```

Requires Python 3.10+, PyTorch with CUDA. Trains in ~5 minutes on an RTX 5060.

---

## Next

- [ ] Replace ReLU activation with SwiGLU (`x * sigmoid(x) * gate`) in `FeedForward`
- [ ] Replace learned positional embeddings with RoPE (Rotary Position Embeddings)
- [ ] BPE tokenizer (Karpathy's tokenizer video) — replacing character-level with sub-word vocab
- [ ] Scale up: `n_embd=384`, `n_head=6`, `n_layer=6`, `block_size=256`