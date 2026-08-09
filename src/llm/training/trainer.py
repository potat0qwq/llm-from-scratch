from dataclasses import dataclass

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from llm.model.gpt import GPT


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    max_steps: int = 500
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    use_amp: bool = True

    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 10


class Trainer:
    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        config: TrainingConfig,
        val_loader: DataLoader | None = None,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.use_amp = (
            config.use_amp
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self._train_iterator = iter(self.train_loader)

    def _next_train_batch(self):
        try:
            batch = next(self._train_iterator)

        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            batch = next(self._train_iterator)

        return batch

    def train(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for step in range(1, self.config.max_steps + 1):
            total_loss = 0.0

            for _ in range(self.config.grad_accum_steps):
                input_ids, targets = self._next_train_batch()

                input_ids = input_ids.to(
                    self.device,
                    non_blocking=True,
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    _, loss = self.model(
                        input_ids,
                        targets,
                    )

                    loss = loss / self.config.grad_accum_steps

                self.scaler.scale(loss).backward()

                total_loss += loss.item()

            self.scaler.unscale_(self.optimizer)

            grad_norm = clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            if step % self.config.log_interval == 0:
                print(
                    f"step {step:5d} | "
                    f"loss {total_loss:.4f} | "
                    f"grad_norm {grad_norm:.4f}"
                )

            if (
                self.val_loader is not None
                and step % self.config.eval_interval == 0
            ):
                val_loss = self.evaluate()

                print(
                    f"step {step:5d} | "
                    f"val_loss {val_loss:.4f}"
                )

                self.model.train()

    @torch.inference_mode()
    def evaluate(self) -> float:
        if self.val_loader is None:
            raise ValueError("val_loader is not provided.")

        self.model.eval()

        losses = []

        for batch_idx, (input_ids, targets) in enumerate(
            self.val_loader
        ):
            if batch_idx >= self.config.eval_batches:
                break

            input_ids = input_ids.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                _, loss = self.model(
                    input_ids,
                    targets,
                )

            losses.append(loss.item())

        return sum(losses) / len(losses)