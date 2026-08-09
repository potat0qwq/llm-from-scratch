import torch

from llm.config import GPTConfig
from llm.data.dataset import CausalLMDataset
from llm.data.tokenizer import ByteTokenizer
from llm.model.gpt import GPT


def main():
    tokenizer = ByteTokenizer()

    text = (
        "The transformer is a neural network architecture "
        "designed for sequence modeling. "
    ) * 100

    token_ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    dataset = CausalLMDataset(
        token_ids=token_ids,
        seq_len=64,
    )

    input_ids, targets = dataset[0]

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    model = GPT(config)

    logits, loss = model(
        input_ids,
        targets,
    )

    print("input_ids:", input_ids.shape)
    print("targets:", targets.shape)
    print("logits:", logits.shape)
    print("loss:", loss.item())

    loss.backward()

    print("Backward pass successful.")


if __name__ == "__main__":
    main()