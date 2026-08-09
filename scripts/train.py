import torch
from torch.utils.data import DataLoader

from pathlib import Path
from llm.config import GPTConfig
from llm.data.dataset import CausalLMDataset
from llm.data.tokenizer import ByteTokenizer
from llm.model.gpt import GPT
from llm.training.trainer import Trainer, TrainingConfig


def main():
    tokenizer = ByteTokenizer()

    text = (
        "The transformer is a neural network architecture "
        "designed for sequence modeling. "
        "Attention allows each token to interact with previous tokens. "
        "Language models predict the next token in a sequence. "
    ) * 200

    token_ids = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    dataset = CausalLMDataset(
        token_ids=token_ids,
        seq_len=64,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
    )

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
    )

    model = GPT(model_config)

    training_config = TrainingConfig(
        learning_rate=3e-4,
        max_steps=500,
        grad_accum_steps=1,
        log_interval=10,
        use_amp=True,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=training_config,
    )

    trainer.train()

    checkpoint_path = Path(
        "outputs/checkpoints/tiny_overfit.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()