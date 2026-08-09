import torch

from llm.data.tokenizer import ByteTokenizer
from llm.generation import generate
from llm.model.gpt import GPT


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "outputs/checkpoints/tiny_overfit.pt",
        map_location=device,
        weights_only=False,
    )

    model_config = checkpoint["model_config"]

    model = GPT(model_config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    tokenizer = ByteTokenizer()

    prompt = "The transformer"

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=150,
        do_sample=False,
    )

    text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print(text)


if __name__ == "__main__":
    main()