import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.config import GPTConfig
from llm.model.block import TransformerBlock
from llm.model.norm import RMSNorm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Initialize model weights before tying the embedding
        # and language-model head parameters.
        self.apply(self._init_weights)

        # Weight tying.
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B, T], "
                f"got {tuple(input_ids.shape)}."
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds "
                f"max_seq_len ({self.config.max_seq_len})."
            )

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits

        if targets.shape != input_ids.shape:
            raise ValueError(
                f"targets must have the same shape as input_ids, "
                f"got input_ids={tuple(input_ids.shape)}, "
                f"targets={tuple(targets.shape)}."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
        )

        return logits, loss