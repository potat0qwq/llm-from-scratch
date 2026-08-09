import torch
import torch.nn as nn

from llm.config import GPTConfig
from llm.model.attention import CausalSelfAttention
from llm.model.mlp import SwiGLU
from llm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.attn_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )

        self.attention = CausalSelfAttention(config)

        self.mlp_norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )

        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))

        return x