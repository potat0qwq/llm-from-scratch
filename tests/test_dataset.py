import numpy as np
import torch

from llm.data.dataset import (
    CausalLMDataset,
    MemmapCausalLMDataset,
)


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

def test_memmap_dataset(tmp_path):
    token_ids = np.arange(
        101,
        dtype=np.uint16,
    )

    data_path = tmp_path / "tokens.bin"

    token_ids.tofile(data_path)

    dataset = MemmapCausalLMDataset(
        data_path=data_path,
        seq_len=10,
    )

    input_ids, targets = dataset[0]

    assert input_ids.shape == (10,)
    assert targets.shape == (10,)

    assert torch.equal(
        input_ids[1:],
        targets[:-1],
    )