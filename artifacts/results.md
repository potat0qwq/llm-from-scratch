# TinyStories Pre-training Results

## Model

- Parameters: 13.77M
- Architecture: Decoder-only Transformer
- Layers: 6
- Hidden dimension: 384
- Attention heads: 6
- FFN dimension: 1024
- Context length: 256

## Tokenizer

- Type: Byte-level BPE
- Vocabulary size: 8,192
- Training corpus: TinyStories

## Training

- Training tokens: 5,000,000
- Validation tokens: 250,000
- Batch size: 32
- Sequence length: 256
- Tokens per optimizer step: 8,192
- Optimizer steps: 1,500
- Peak learning rate: 3e-4
- Minimum learning rate: 3e-5
- Warmup steps: 100
- Scheduler: Linear Warmup + Cosine Decay

## Results

- Validation loss @ step 100: 4.9747
- Validation loss @ step 500: 2.8606
- Validation loss @ step 1000: 2.4956
- Validation loss @ step 1500: 2.3963
- Final validation perplexity: ~10.98