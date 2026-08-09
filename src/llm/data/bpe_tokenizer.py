from pathlib import Path

from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path):
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        token_id = self.tokenizer.token_to_id("<eos>")

        if token_id is None:
            raise ValueError("Tokenizer does not contain <eos>.")

        return token_id

    def encode(
        self,
        text: str,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids

        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)