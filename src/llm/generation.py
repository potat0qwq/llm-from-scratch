import torch

from llm.model.gpt import GPT


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
) -> torch.Tensor:
    model.eval()

    for _ in range(max_new_tokens):
        context = input_ids[
            :, -model.config.max_seq_len:
        ]

        logits = model(context)

        next_token_logits = logits[:, -1, :]

        if do_sample:
            if temperature <= 0:
                raise ValueError(
                    "temperature must be positive "
                    "when sampling."
                )

            next_token_logits = (
                next_token_logits / temperature
            )

            if top_k is not None:
                k = min(
                    top_k,
                    next_token_logits.size(-1),
                )

                values, _ = torch.topk(
                    next_token_logits,
                    k,
                )

                threshold = values[:, -1].unsqueeze(-1)

                next_token_logits = (
                    next_token_logits.masked_fill(
                        next_token_logits < threshold,
                        float("-inf"),
                    )
                )

            probs = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        else:
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return input_ids