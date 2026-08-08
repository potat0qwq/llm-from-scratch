import torch

from llm.model.norm import RMSNorm


def test_rms_norm_shape():
    norm = RMSNorm(d_model=384)

    x = torch.randn(2, 16, 384)
    y = norm(x)

    assert y.shape == x.shape


def test_rms_norm_unit_rms():
    norm = RMSNorm(d_model=384)

    x = torch.randn(2, 16, 384)
    y = norm(x)

    rms = y.pow(2).mean(dim=-1).sqrt()

    expected = torch.ones_like(rms)

    assert torch.allclose(rms, expected, atol=1e-4)

