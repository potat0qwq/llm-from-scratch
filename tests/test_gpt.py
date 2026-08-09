import torch

from llm.config import GPTConfig
from llm.model.gpt import GPT


def make_tiny_config():
    return GPTConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )


def test_gpt_logits_shape():
    config = make_tiny_config()
    model = GPT(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        16,
        config.vocab_size,
    )


def test_gpt_loss():
    config = make_tiny_config()
    model = GPT(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    logits, loss = model(
        input_ids,
        targets,
    )

    assert logits.shape == (
        2,
        16,
        config.vocab_size,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gpt_backward():
    config = make_tiny_config()
    model = GPT(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    _, loss = model(
        input_ids,
        targets,
    )

    loss.backward()

    assert model.token_embedding.weight.grad is not None


def test_gpt_weight_tying():
    config = make_tiny_config()
    model = GPT(config)

    assert (
        model.token_embedding.weight
        is model.lm_head.weight
    )