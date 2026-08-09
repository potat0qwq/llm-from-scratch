import torch

from llm.model.rope import RotaryEmbedding


def test_rope_shape():
    rope = RotaryEmbedding(
        head_dim=64,
        max_seq_len=256,
    )

    x = torch.randn(2, 6, 16, 64)

    y = rope(x)

    assert y.shape == x.shape


def test_rope_position_zero_unchanged():
    rope = RotaryEmbedding(
        head_dim=64,
        max_seq_len=256,
    )

    x = torch.randn(2, 6, 16, 64)

    y = rope(x)

    assert torch.allclose(
        y[:, :, 0],
        x[:, :, 0],
        atol=1e-6,
    )


def test_rope_preserves_norm():
    rope = RotaryEmbedding(
        head_dim=64,
        max_seq_len=256,
    )

    x = torch.randn(2, 6, 16, 64)

    y = rope(x)

    x_norm = x.norm(dim=-1)
    y_norm = y.norm(dim=-1)

    assert torch.allclose(
        x_norm,
        y_norm,
        atol=1e-5,
    )