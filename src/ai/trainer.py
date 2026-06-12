from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from .dataset import EpisodeDataset
from .model import FlappyCNN
from .visualization import TrainingPlotter
from ..game import AppConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingSummary:
    model_path: Path
    best_val_accuracy: float
    epochs: int
    seconds: float


def _run_epoch(
    model: FlappyCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
    scaler: torch.cuda.amp.GradScaler,
    mixed_precision: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for states, actions in loader:
        states = states.to(device)
        actions = actions.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=mixed_precision and device.type == "cuda",
        ):
            logits = model(states)
            loss = criterion(logits, actions)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.item()) * states.size(0)
        total_correct += int((logits.argmax(dim=1) == actions).sum().item())
        total_samples += int(states.size(0))

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_model(
    config: AppConfig,
    resume: bool = False,
) -> TrainingSummary:
    dataset = EpisodeDataset(config.dataset_dir)
    val_size = max(1, int(len(dataset) * config.training.validation_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.game.random_seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlappyCNN(input_channels=config.preprocessing.frame_stack).to(device)
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.training.mixed_precision and device.type == "cuda"
    )
    writer = None
    if config.training.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(config.logs_dir))
        except ModuleNotFoundError:
            LOGGER.warning("TensorBoard is not installed; skipping TensorBoard logging.")
    plotter = TrainingPlotter(config.logs_dir)

    checkpoint_path = config.checkpoints_dir / "cnn_latest.pth"
    start_epoch = 0
    best_val_accuracy = 0.0

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        LOGGER.info("Resumed training from epoch %s", start_epoch)

    start_time = time.perf_counter()
    for epoch in range(start_epoch, config.training.epochs):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            config.training.mixed_precision,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            scaler,
            config.training.mixed_precision,
        )
        plotter.update(epoch + 1, train_loss, val_loss, train_acc, val_acc)

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("accuracy/train", train_acc, epoch + 1)
            writer.add_scalar("accuracy/val", val_acc, epoch + 1)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "config_frame_stack": config.preprocessing.frame_stack,
                },
                config.models_dir / "flappy_cnn.pth",
            )

        if (epoch + 1) % config.training.checkpoint_interval == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

        LOGGER.info(
            "Epoch %s/%s train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
            epoch + 1,
            config.training.epochs,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )

    if writer is not None:
        writer.close()

    return TrainingSummary(
        model_path=config.models_dir / "flappy_cnn.pth",
        best_val_accuracy=best_val_accuracy,
        epochs=config.training.epochs,
        seconds=time.perf_counter() - start_time,
    )
