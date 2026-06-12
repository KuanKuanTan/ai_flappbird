from __future__ import annotations

from pathlib import Path

class TrainingPlotter:
    def __init__(self, output_dir: Path) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs: list[int] = []
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []
        self.plt.ion()
        self.figure, self.axes = self.plt.subplots(1, 2, figsize=(10, 4))

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
    ) -> None:
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[0].plot(self.epochs, self.train_loss, label="train")
        self.axes[0].plot(self.epochs, self.val_loss, label="val")
        self.axes[0].set_title("Loss")
        self.axes[0].legend()
        self.axes[1].plot(self.epochs, self.train_acc, label="train")
        self.axes[1].plot(self.epochs, self.val_acc, label="val")
        self.axes[1].set_title("Accuracy")
        self.axes[1].legend()
        self.figure.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.savefig(self.output_dir / "training_curves.png", dpi=150)
