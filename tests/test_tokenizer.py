from llm.data.tokenizer import ByteTokenizer


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()

    text = "Hello, world!"

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_byte_tokenizer_vocab_range():
    tokenizer = ByteTokenizer()

    text = "Language model"

    token_ids = tokenizer.encode(text)

    assert all(
        0 <= token_id < tokenizer.vocab_size
        for token_id in token_ids
    )


def test_byte_tokenizer_unicode():
    tokenizer = ByteTokenizer()

    text = "Hello 世界"

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text