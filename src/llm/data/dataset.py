from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):
    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int,
    ):
        if token_ids.ndim != 1:
            raise ValueError(
                f"token_ids must be a 1D tensor, "
                f"got shape {tuple(token_ids.shape)}."
            )

        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        if len(token_ids) <= seq_len:
            raise ValueError(
                "token_ids must contain more than seq_len tokens."
            )

        self.token_ids = token_ids.long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.token_ids[start:end]
        targets = self.token_ids[start + 1:end + 1]

        return input_ids, targets


class MemmapCausalLMDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
    ):
        if seq_len <= 0:
            raise ValueError(
                "seq_len must be positive."
            )

        self.data = np.memmap(
            data_path,
            dtype=np.uint16,
            mode="r",
        )

        if len(self.data) <= seq_len:
            raise ValueError(
                "Dataset contains fewer tokens "
                "than required sequence length."
            )

        self.seq_len = seq_len

    def __len__(self) -> int:
        return (
            len(self.data) - 1
        ) // self.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = np.asarray(
            self.data[start:end],
            dtype=np.int64,
        ).copy()

        targets = np.asarray(
            self.data[start + 1:end + 1],
            dtype=np.int64,
        ).copy()

        return (
            torch.from_numpy(input_ids),
            torch.from_numpy(targets),
        )