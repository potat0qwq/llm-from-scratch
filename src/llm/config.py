from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 256

    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1024

    dropout: float = 0.0

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    bias: bool = False

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible "
                f"by n_heads ({self.n_heads})."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads