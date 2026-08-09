from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm.config import GPTConfig
from llm.data.dataset import MemmapCausalLMDataset
from llm.model.gpt import GPT
from llm.training.trainer import Trainer, TrainingConfig

def main():
    train_dataset = MemmapCausalLMDataset(
        "artifacts/data/train.bin",
        seq_len=256,
    )

    val_dataset = MemmapCausalLMDataset(
        "artifacts/data/val.bin",
        seq_len=256,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
    )

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

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{num_parameters / 1e6:.2f}M"
    )

    training_config = TrainingConfig(
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        weight_decay=0.1,

        max_steps=1500,
        warmup_steps=100,

        grad_accum_steps=1,
        max_grad_norm=1.0,

        log_interval=10,
        eval_interval=100,
        eval_batches=20,

        checkpoint_interval=500,
        checkpoint_dir="outputs/checkpoints",

        use_amp=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    trainer.train()

    trainer.save_checkpoint(
        training_config.max_steps
    )

if __name__ == "__main__":
    main()