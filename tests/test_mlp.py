import torch

from llm.config import GPTConfig
from llm.model.mlp import SwiGLU


def test_swiglu_shape():
    config = GPTConfig(vocab_size=32000)

    mlp = SwiGLU(config)

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    y = mlp(x)

    assert y.shape == x.shape


def test_swiglu_backward():
    config = GPTConfig(vocab_size=32000)

    mlp = SwiGLU(config)

    x = torch.randn(
        2,
        16,
        config.d_model,
        requires_grad=True,
    )

    y = mlp(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None