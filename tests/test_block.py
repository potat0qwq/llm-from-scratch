import torch

from llm.config import GPTConfig
from llm.model.block import TransformerBlock


def test_transformer_block_shape():
    config = GPTConfig(vocab_size=32000)

    block = TransformerBlock(config)

    x = torch.randn(
        2,
        16,
        config.d_model,
    )

    y = block(x)

    assert y.shape == x.shape


def test_transformer_block_backward():
    config = GPTConfig(vocab_size=32000)

    block = TransformerBlock(config)

    x = torch.randn(
        2,
        16,
        config.d_model,
        requires_grad=True,
    )

    y = block(x)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None