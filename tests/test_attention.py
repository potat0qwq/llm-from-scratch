import torch

from llm.config import GPTConfig
from llm.model.attention import CausalSelfAttention


def test_attention_shape():
    config = GPTConfig(vocab_size=32000)

    attention = CausalSelfAttention(config)

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    y = attention(x)

    assert y.shape == x.shape


def test_attention_is_causal():
    torch.manual_seed(42)

    config = GPTConfig(
        vocab_size=32000,
        dropout=0.0,
    )

    attention = CausalSelfAttention(config)
    attention.eval()

    x = torch.randn(
        1,
        8,
        config.d_model,
    )

    x_modified = x.clone()

    prefix_len = 4

    x_modified[:, prefix_len:] = torch.randn_like(
        x_modified[:, prefix_len:]
    )

    y = attention(x)
    y_modified = attention(x_modified)

    assert torch.allclose(
        y[:, :prefix_len],
        y_modified[:, :prefix_len],
        atol=1e-6,
    )