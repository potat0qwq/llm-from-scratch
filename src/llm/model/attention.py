import math

import torch
import torch.nn as nn

from llm.config import GPTConfig
from llm.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.o_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.bias,
        )

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.triu(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

        attn_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        out = self.o_proj(out)
        out = self.resid_dropout(out)

        return out