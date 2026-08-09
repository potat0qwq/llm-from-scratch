import gc

import torch

from llm.config import GPTConfig
from llm.model.gpt import GPT


def test_batch_size(batch_size: int) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config = GPTConfig(
        vocab_size=8192,
        max_seq_len=256,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1024,
        dropout=0.0,
    )

    model = GPT(config).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scaler = torch.amp.GradScaler("cuda")

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, config.max_seq_len),
        device="cuda",
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (batch_size, config.max_seq_len),
        device="cuda",
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        _, loss = model(
            input_ids,
            targets,
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"batch_size={batch_size} | "
        f"loss={loss.item():.4f} | "
        f"peak allocated={peak_memory:.2f} GB"
    )

    del model
    del optimizer
    del scaler
    del input_ids
    del targets
    del loss

    gc.collect()
    torch.cuda.empty_cache()


def main():
    for batch_size in [8, 16, 32, 64]:
        try:
            test_batch_size(batch_size)

        except torch.OutOfMemoryError:
            print(
                f"batch_size={batch_size} | CUDA OOM"
            )

            gc.collect()
            torch.cuda.empty_cache()

            break


if __name__ == "__main__":
    main()