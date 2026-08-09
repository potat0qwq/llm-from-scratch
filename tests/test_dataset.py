import torch

from llm.data.dataset import CausalLMDataset


def test_dataset_shape():
    token_ids = torch.arange(100)

    dataset = CausalLMDataset(
        token_ids=token_ids,
        seq_len=16,
    )

    input_ids, targets = dataset[0]

    assert input_ids.shape == (16,)
    assert targets.shape == (16,)


def test_dataset_next_token_shift():
    token_ids = torch.arange(100)

    dataset = CausalLMDataset(
        token_ids=token_ids,
        seq_len=16,
    )

    input_ids, targets = dataset[0]

    assert torch.equal(
        input_ids[1:],
        targets[:-1],
    )

    assert targets[0] == input_ids[1]


def test_dataset_length():
    token_ids = torch.arange(101)

    dataset = CausalLMDataset(
        token_ids=token_ids,
        seq_len=10,
    )

    assert len(dataset) == 10