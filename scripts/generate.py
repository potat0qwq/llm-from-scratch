import torch

from llm.config import GPTConfig
from llm.data.bpe_tokenizer import BPETokenizer
from llm.generation import generate
from llm.model.gpt import GPT


CHECKPOINT_PATH = (
    "outputs/checkpoints/checkpoint_step_1500.pt"
)

TOKENIZER_PATH = (
    "artifacts/tokenizer/tinystories_bpe_8k.json"
)


def load_model(device: torch.device) -> GPT:
    model_config = GPTConfig(
        vocab_size=8192,
        max_seq_len=256,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1024,
        dropout=0.0,
    )

    model = GPT(model_config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def run_generation(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
):
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=200,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(
        output_ids[0].tolist()
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = BPETokenizer.from_file(
        TOKENIZER_PATH
    )

    model = load_model(device)

    prompt = (
        "Once upon a time, there was "
        "a little girl named Lily who"
    )

    print("\n=== Greedy ===\n")

    print(
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            do_sample=False,
        )
    )

    print("\n=== Temperature 0.7 / Top-k 50 ===\n")

    torch.manual_seed(42)

    print(
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            temperature=0.7,
            top_k=50,
            do_sample=True,
        )
    )

    print("\n=== Temperature 1.0 / Top-k 50 ===\n")

    torch.manual_seed(42)

    print(
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            temperature=1.0,
            top_k=50,
            do_sample=True,
        )
    )

    print("\n=== Temperature 1.3 / Top-k 50 ===\n")

    torch.manual_seed(42)

    print(
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            temperature=1.3,
            top_k=50,
            do_sample=True,
        )
    )


if __name__ == "__main__":
    main()