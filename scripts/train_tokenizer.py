from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers


VOCAB_SIZE = 8192
NUM_STORIES = 50_000

OUTPUT_PATH = Path(
    "artifacts/tokenizer/tinystories_bpe_8k.json"
)

SPECIAL_TOKENS = [
    "<unk>",
    "<eos>",
]


def main():
    print("Loading TinyStories in streaming mode...")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    dataset = dataset.take(NUM_STORIES)

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>",
        )
    )

    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False,
        )
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=(
            pre_tokenizers.ByteLevel.alphabet()
        ),
        show_progress=True,
    )

    def text_iterator():
        for example in dataset:
            yield example["text"]

    print(
        f"Training BPE tokenizer on "
        f"{NUM_STORIES:,} stories..."
    )

    tokenizer.train_from_iterator(
        text_iterator(),
        trainer=trainer,
        length=NUM_STORIES,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(OUTPUT_PATH))

    print(f"Tokenizer saved to: {OUTPUT_PATH}")
    print(
        f"Vocabulary size: "
        f"{tokenizer.get_vocab_size()}"
    )

    sample = (
        "Once upon a time, a little girl "
        "found a beautiful red flower."
    )

    encoding = tokenizer.encode(sample)

    print()
    print("Sample:")
    print(sample)

    print()
    print("Tokens:")
    print(encoding.tokens)

    print()
    print("Token IDs:")
    print(encoding.ids)

    print()
    print(
        "UTF-8 bytes:",
        len(sample.encode("utf-8")),
    )

    print(
        "BPE tokens:",
        len(encoding.ids),
    )

    decoded = tokenizer.decode(
        encoding.ids
    )

    print()
    print("Decoded:")
    print(decoded)

    assert decoded == sample


if __name__ == "__main__":
    main()