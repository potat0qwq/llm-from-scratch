import torch

from llm.config import GPTConfig
from llm.generation import generate
from llm.model.gpt import GPT


def make_tiny_model():
    config = GPTConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )

    return GPT(config)


def test_generation_adds_tokens():
    model = make_tiny_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,
    )

    assert output_ids.shape == (1, 13)


def test_generation_preserves_prompt():
    model = make_tiny_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,
    )

    assert torch.equal(
        output_ids[:, :8],
        input_ids,
    )