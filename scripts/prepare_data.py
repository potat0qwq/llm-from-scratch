from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from llm.data.bpe_tokenizer import BPETokenizer


TOKENIZER_PATH = Path(
    "artifacts/tokenizer/tinystories_bpe_8k.json"
)

OUTPUT_DIR = Path("artifacts/data")

TRAIN_TOKEN_LIMIT = 5_000_000
VAL_TOKEN_LIMIT = 250_000

BUFFER_SIZE = 100_000


def write_split(
    split: str,
    output_path: Path,
    token_limit: int,
    tokenizer: BPETokenizer,
):
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_tokens = 0
    buffer = []

    with open(output_path, "wb") as f:
        progress = tqdm(
            total=token_limit,
            desc=f"Tokenizing {split}",
            unit="tok",
        )

        for example in dataset:
            token_ids = tokenizer.encode(
                example["text"],
                add_eos=True,
            )

            remaining = token_limit - total_tokens

            if len(token_ids) > remaining:
                token_ids = token_ids[:remaining]

            buffer.extend(token_ids)
            total_tokens += len(token_ids)

            if len(buffer) >= BUFFER_SIZE:
                np.asarray(
                    buffer,
                    dtype=np.uint16,
                ).tofile(f)

                buffer.clear()

            progress.update(len(token_ids))

            if total_tokens >= token_limit:
                break

        if buffer:
            np.asarray(
                buffer,
                dtype=np.uint16,
            ).tofile(f)

        progress.close()

    print(
        f"{split}: {total_tokens:,} tokens "
        f"written to {output_path}"
    )


def main():
    tokenizer = BPETokenizer.from_file(
        TOKENIZER_PATH
    )

    print(
        f"Tokenizer vocab size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"<eos> token id: "
        f"{tokenizer.eos_token_id}"
    )

    write_split(
        split="train",
        output_path=OUTPUT_DIR / "train.bin",
        token_limit=TRAIN_TOKEN_LIMIT,
        tokenizer=tokenizer,
    )

    write_split(
        split="validation",
        output_path=OUTPUT_DIR / "val.bin",
        token_limit=VAL_TOKEN_LIMIT,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()