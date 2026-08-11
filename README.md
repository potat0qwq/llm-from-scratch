# LLM from Scratch

> A compact PyTorch implementation of a **decoder-only Transformer**—including a byte-level BPE tokenizer, memory-mapped data pipeline, causal pretraining loop, and autoregressive text generation.

This repository follows the full path from raw TinyStories text to generated stories. The Transformer architecture, training loop, and decoding logic are implemented explicitly instead of relying on high-level language-model classes.

<table>
  <tr>
    <th>Training and validation loss</th>
    <th>Learning-rate schedule</th>
  </tr>
  <tr>
    <td><img src="artifacts/figures/loss_curves.png" alt="Training and validation loss curves" width="100%"></td>
    <td><img src="artifacts/figures/learning_rate.png" alt="Linear warmup and cosine learning-rate schedule" width="100%"></td>
  </tr>
</table>

## At a glance

| | Reference experiment |
|---|---|
| Data | TinyStories, with 5M training tokens and 250K validation tokens |
| Tokenizer | Custom byte-level BPE, 8,192-token vocabulary |
| Model | 13.77M-parameter decoder-only Transformer |
| Architecture | 6 layers, 384 hidden size, 6 heads, 1,024-dimensional FFN |
| Training | 1,500 steps, batch size 32, context length 256, AdamW |
| Result | Validation loss **2.3963**, perplexity **≈ 10.98** |
| Runtime | Automatically uses CUDA when available; mixed precision is CUDA-only |

## Why this project?

Large language-model libraries make experimentation convenient, but they can hide the mechanics that connect tokenization, attention, optimization, and decoding. This project keeps that path small enough to inspect end to end:

```text
TinyStories ──▶ byte-level BPE ──▶ uint16 token streams ──▶ causal Transformer
                                                               │
                                                               ▼
prompt ──▶ tokenizer ──▶ autoregressive decoding ◀── trained checkpoint
                            ├─ greedy
                            └─ temperature + top-k
```

The repository is useful for:

- reading a modern decoder-only Transformer as ordinary PyTorch modules;
- training a tokenizer and language model without a pretrained model stack;
- studying RMSNorm, RoPE, causal attention, SwiGLU, and weight tying;
- comparing greedy and stochastic decoding on the same checkpoint;
- extending a complete, tested baseline rather than an isolated model class.

## Quick start

### 1. Set up the environment

```bash
git clone https://github.com/potat0qwq/llm-from-scratch.git
cd llm-from-scratch
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required. For GPU training, install a CUDA-enabled PyTorch build that matches your system.

### 2. Prepare the dataset

The repository includes the trained 8K tokenizer. Use it to stream and tokenize TinyStories:

```bash
python scripts/prepare_data.py
```

This downloads the dataset and creates:

```text
artifacts/data/
├── train.bin    # 5,000,000 tokens
└── val.bin      #   250,000 tokens
```

To reproduce the tokenizer first, run:

```bash
python scripts/train_tokenizer.py
```

It trains on 50,000 TinyStories examples and overwrites `artifacts/tokenizer/tinystories_bpe_8k.json`.

### 3. Pretrain

```bash
python scripts/train.py
```

Checkpoints are written to `outputs/checkpoints/` at steps 500, 1,000, and 1,500. Prepared data and checkpoints are ignored by Git.

### 4. Generate text

```bash
python scripts/generate.py
```

The script loads `outputs/checkpoints/checkpoint_step_1500.pt` and compares greedy decoding with three temperature-based, top-k sampling settings.

> The repository does not include a pretrained checkpoint. Run pretraining before generation.

### 5. Run the tests

```bash
pytest -v
```

`scripts/benchmark_batch.py` can probe CUDA memory at batch sizes 8, 16, 32, and 64 before a full run. It requires an NVIDIA CUDA environment.

## Results

### Pretraining

Validation loss decreased throughout the 1,500-step reference run:

| Step | Validation loss | Perplexity |
|---:|---:|---:|
| 100 | 4.9747 | 144.71 |
| 500 | 2.8606 | 17.47 |
| 1,000 | 2.4956 | 12.13 |
| 1,500 | **2.3963** | **10.98** |

Perplexity is computed as `exp(validation loss)`. Validation uses 20 batches every 100 optimizer steps, so these values are a consistent training diagnostic rather than a full-corpus benchmark.

### Generation

Reference prompt:

```text
Once upon a time, there was a little girl named Lily who
```

| Strategy | Observed behavior |
|---|---|
| Greedy | Locally coherent, but increasingly repetitive |
| Temperature 0.7 + top-k 50 | Best coherence–diversity balance in this run |
| Temperature 1.0 + top-k 50 | More varied, with weaker logical consistency |
| Temperature 1.3 + top-k 50 | Unstable and stopped at EOS almost immediately |

One sampled continuation at temperature 0.7 begins:

> Once upon a time, there was a little girl named Lily who loved to draw. One day, she went to the beach to get a little bit of the water...

See [`artifacts/generation_samples.md`](artifacts/generation_samples.md) for the complete outputs. These samples illustrate decoding behavior; they are not a general language-quality benchmark.

## How it works

### Tokenizer and data pipeline

A byte-level BPE tokenizer is trained with an 8,192-token vocabulary and two special tokens: `<unk>` and `<eos>`. Each story receives a terminal `<eos>` token before its IDs are stored in a compact `uint16` stream.

Training reads these streams through `numpy.memmap`, so the complete corpus does not need to be loaded into memory. For a context length of 256, every example is a shifted next-token pair:

```text
input   [x₀, x₁, ..., x₂₅₅]
target  [x₁, x₂, ..., x₂₅₆]
```

### Model architecture

```text
token IDs
   │
   ▼
token embedding
   │
   ▼
┌────────────────────────────────────┐
│ Transformer block × 6              │
│                                    │
│ RMSNorm → causal attention + RoPE  │
│      └──────── residual ─────────┐  │
│ RMSNorm → SwiGLU MLP             │  │
│      └──────── residual ─────────┘  │
└────────────────────────────────────┘
   │
   ▼
final RMSNorm → tied LM head → next-token logits
```

| Component | Implementation |
|---|---|
| Normalization | Pre-Norm RMSNorm |
| Position information | Rotary Position Embeddings applied to queries and keys |
| Attention | Multi-head self-attention with a causal mask |
| Feed-forward network | SwiGLU with a 1,024-dimensional hidden layer |
| Output projection | Weight-tied with the token embedding matrix |
| Regularization | No dropout in the reference configuration |

For token sequence $x_1, \ldots, x_N$, training minimizes next-token cross-entropy:

$$
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p_\theta(x_i \mid x_{1:i-1}).
$$

### Optimization and decoding

Training uses AdamW, gradient-norm clipping at 1.0, and FP16 automatic mixed precision on CUDA. The learning rate warms up linearly for 100 steps, then follows cosine decay from `3e-4` to `3e-5`.

Generation repeatedly feeds the latest 256 tokens to the model and selects the next token using either argmax or sampling. Temperature rescales logits before softmax, while top-k sampling masks every token outside the 50 highest-scoring candidates. Generation stops after 200 new tokens or when `<eos>` is produced.

## Configuration

The reference experiment keeps its configuration in the entry-point scripts and dataclasses.

| Parameter | Default | Change in |
|---|---:|---|
| `vocab_size` | 8,192 | `scripts/train.py`, `scripts/generate.py` |
| `max_seq_len` | 256 | `src/llm/config.py`, entry-point scripts |
| `d_model` | 384 | `src/llm/config.py`, entry-point scripts |
| `n_layers` | 6 | `src/llm/config.py`, entry-point scripts |
| `n_heads` | 6 | `src/llm/config.py`, entry-point scripts |
| `d_ff` | 1,024 | `src/llm/config.py`, entry-point scripts |
| `batch_size` | 32 | `scripts/train.py` |
| `max_steps` | 1,500 | `scripts/train.py` |
| `learning_rate` | `3e-4` | `scripts/train.py` |
| `weight_decay` | `0.1` | `scripts/train.py` |
| `warmup_steps` | 100 | `scripts/train.py` |
| `max_new_tokens` | 200 | `scripts/generate.py` |

When loading a checkpoint, the model settings in `scripts/generate.py` must match those used during training. If you change the vocabulary or context length, regenerate the prepared data and keep the tokenizer, model, and checkpoint configuration aligned.

## Project structure

```text
llm-from-scratch/
├── artifacts/
│   ├── figures/                 # Reference training plots
│   ├── tokenizer/               # Trained byte-level BPE tokenizer
│   ├── generation_samples.md
│   └── results.md
├── scripts/
│   ├── benchmark_batch.py       # Probe CUDA memory by batch size
│   ├── generate.py              # Compare decoding strategies
│   ├── plot_training.py         # Render metrics from saved history
│   ├── prepare_data.py          # Build uint16 token streams
│   ├── smoke_test.py            # Tiny optimization sanity check
│   ├── train.py                 # Run causal LM pretraining
│   └── train_tokenizer.py       # Train the byte-level BPE tokenizer
├── src/llm/
│   ├── data/                    # Tokenizer wrappers and memmap dataset
│   ├── model/                   # Attention, blocks, GPT, MLP, norm, RoPE
│   ├── training/                # Trainer and optimization schedule
│   ├── config.py
│   └── generation.py
├── tests/                       # Unit tests for the core components
├── LICENSE
└── pyproject.toml
```

`artifacts/data/` and `outputs/` are created locally and ignored by Git. The plotting script reads `outputs/training_history.json`; the committed figures and results record the reference run.

## Reproducibility

The reported metrics and samples come from the included reference artifacts. The training entry point does not currently set a global random seed, so exact losses can vary with initialization, data-loader order, hardware, and PyTorch version.

For sampled generation, `scripts/generate.py` resets PyTorch's seed to 42 before each temperature comparison. Greedy decoding does not sample. Fully repeatable training would additionally require seeded data loading and deterministic backend settings.

## Scope and next steps

This is an educational 13.77M-parameter model, not a production LLM or a claim of competitive language quality. Its 256-token context, modest training budget, and lack of a KV cache make the trade-offs easy to study but limit long-range consistency and generation speed.

Natural extensions include:

- KV-cached autoregressive generation;
- PyTorch scaled dot-product or Flash Attention;
- top-p sampling and repetition controls;
- file-based experiment configuration and checkpoint resumption;
- larger context windows, datasets, and model sizes;
- controlled scaling experiments across parameters, tokens, and compute.

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), NeurIPS 2017.
- Zhang and Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467), NeurIPS 2019.
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202), 2020.
- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864), 2021.
- Eldan and Li, [*TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*](https://arxiv.org/abs/2305.07759), 2023.

## License

Released under the [MIT License](LICENSE).
