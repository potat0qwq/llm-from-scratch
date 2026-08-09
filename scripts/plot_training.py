import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_PATH = Path(
    "outputs/training_history.json"
)

OUTPUT_DIR = Path(
    "artifacts/figures"
)


def load_history():
    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def plot_train_loss(history):
    steps = history["step"]
    train_loss = history["train_loss"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        train_loss,
        linewidth=1.5,
    )

    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "train_loss.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {path}")


def plot_validation_loss(history):
    steps = history["val_step"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        val_loss,
        marker="o",
        linewidth=1.5,
    )

    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Validation Loss")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "validation_loss.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {path}")


def plot_learning_rate(history):
    steps = history["step"]
    learning_rate = history["learning_rate"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        learning_rate,
        linewidth=1.5,
    )

    plt.xlabel("Optimizer Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "learning_rate.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {path}")

def plot_loss_comparison(history):
    train_steps = history["step"]
    train_loss = history["train_loss"]

    val_steps = history["val_step"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_steps,
        train_loss,
        label="Training Loss",
        linewidth=1.2,
    )

    plt.plot(
        val_steps,
        val_loss,
        label="Validation Loss",
        marker="o",
        linewidth=1.8,
    )

    plt.xlabel("Optimizer Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "loss_curves.png"

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {path}")    


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_history()

    plot_train_loss(history)
    plot_validation_loss(history)
    plot_learning_rate(history)
    plot_loss_comparison(history)


if __name__ == "__main__":
    main()