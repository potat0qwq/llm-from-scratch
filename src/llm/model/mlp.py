import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import GPTConfig


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=config.bias,
        )

        self.up_proj = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=config.bias,
        )

        self.down_proj = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)

        return x