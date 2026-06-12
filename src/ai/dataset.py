from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
from torch.utils.data import Dataset

from ..env import FlappyEnv
from ..game import AppConfig, GameAssets


LOGGER = logging.getLogger(__name__)


@dataclass
class CollectionSummary:
    episodes: int
    frames: int
    avg_score: float
    output_dir: Path


class EpisodeDataset(Dataset):
    def __init__(self, dataset_dir: str | Path) -> None:
        base_dir = Path(dataset_dir)
        self.files = sorted(base_dir.glob("episode_*.npz")) + sorted(
            base_dir.glob("session_*.npz")
        )
        if not self.files:
            raise FileNotFoundError(f"No dataset files found in {dataset_dir}")
        self.index: list[tuple[Path, int]] = []
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
        for file in self.files:
            with np.load(file) as data:
                size = int(data["actions"].shape[0])
            for sample_index in range(size):
                self.index.append((file, sample_index))

    def __len__(self) -> int:
        return len(self.index)

    def _load_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        with np.load(path) as data:
            states = data["states"].astype(np.float32)
            actions = data["actions"].astype(np.int64)
        self.cache[path] = (states, actions)
        return self.cache[path]

    def __getitem__(self, idx: int):
        file, sample_index = self.index[idx]
        states, actions = self._load_file(file)
        return torch.from_numpy(states[sample_index]), torch.tensor(actions[sample_index])


def get_next_episode_index(dataset_dir: str | Path) -> int:
    output_dir = Path(dataset_dir)
    if not output_dir.exists():
        return 0
    indices: list[int] = []
    for path in output_dir.glob("episode_*.npz"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(indices) + 1) if indices else 0


def get_next_session_index(dataset_dir: str | Path) -> int:
    output_dir = Path(dataset_dir)
    if not output_dir.exists():
        return 0
    indices: list[int] = []
    for path in output_dir.glob("session_*.npz"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(indices) + 1) if indices else 0


def trim_episode_tail(
    states: list[np.ndarray],
    actions: list[int],
    frames: list[np.ndarray],
    trim_frames: int,
    min_frames: int,
) -> tuple[list[np.ndarray], list[int], list[np.ndarray], int]:
    if trim_frames <= 0:
        return states, actions, frames, 0

    trimmed = min(trim_frames, max(0, len(states) - min_frames))
    if trimmed <= 0:
        return states, actions, frames, 0

    keep = len(states) - trimmed
    return states[:keep], actions[:keep], frames[:keep], trimmed


def collect_dataset(
    config: AppConfig,
    assets: GameAssets,
    episodes: int,
    controller: Literal["expert", "human"] = "expert",
) -> CollectionSummary:
    dataset_dir = config.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    env = FlappyEnv(config, assets, seed=config.game.random_seed)
    total_frames = 0
    total_score = 0.0

    for episode in range(episodes):
        states: list[np.ndarray] = []
        actions: list[int] = []
        env.reset(seed=config.game.random_seed + episode)
        done = False
        steps = 0
        while not done and steps < config.training.max_episode_steps:
            pygame.event.pump()
            if controller == "expert":
                action = env.world.get_expert_action()
            else:
                action = 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0

            state = env.get_state().copy()
            next_state, _, done, info = env.step(action)
            del next_state
            states.append(state)
            actions.append(action)
            steps += 1

        total_frames += len(states)
        total_score += env.world.score
        file_path = dataset_dir / f"episode_{episode:05d}.npz"
        np.savez_compressed(
            file_path,
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            score=np.asarray([env.world.score], dtype=np.int64),
        )
        LOGGER.info(
            "Collected episode %s with %s frames and score %s",
            episode,
            len(states),
            env.world.score,
        )

    return CollectionSummary(
        episodes=episodes,
        frames=total_frames,
        avg_score=(total_score / max(episodes, 1)),
        output_dir=dataset_dir,
    )


def save_episode_record(
    dataset_dir: str | Path,
    episode_index: int,
    states: list[np.ndarray],
    actions: list[int],
    frames: list[np.ndarray],
    score: int,
) -> Path:
    output_dir = Path(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"episode_{episode_index:05d}.npz"
    np.savez_compressed(
        file_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        frames=np.asarray(frames, dtype=np.uint8),
        score=np.asarray([score], dtype=np.int64),
    )
    return file_path


def save_recording_session(
    dataset_dir: str | Path,
    session_index: int,
    states: list[np.ndarray],
    actions: list[int],
    frames: list[np.ndarray],
    segment_lengths: list[int],
    segment_scores: list[int],
) -> Path:
    output_dir = Path(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"session_{session_index:05d}.npz"
    np.savez_compressed(
        file_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        frames=np.asarray(frames, dtype=np.uint8),
        segment_lengths=np.asarray(segment_lengths, dtype=np.int64),
        segment_scores=np.asarray(segment_scores, dtype=np.int64),
    )
    return file_path


def inspect_episode_record(
    dataset_dir: str | Path,
    episode_index: int | None = None,
    frame_index: int = 0,
    show: bool = True,
) -> dict[str, object]:
    output_dir = Path(dataset_dir)
    files = sorted(output_dir.glob("episode_*.npz")) + sorted(
        output_dir.glob("session_*.npz")
    )
    if not files:
        raise FileNotFoundError(f"No dataset files found in {output_dir}")

    if episode_index is None:
        target = files[-1]
    else:
        episode_target = output_dir / f"episode_{episode_index:05d}.npz"
        session_target = output_dir / f"session_{episode_index:05d}.npz"
        if episode_target.exists():
            target = episode_target
        elif session_target.exists():
            target = session_target
        else:
            raise FileNotFoundError(
                f"Episode/session file not found: {episode_target} or {session_target}"
            )

    with np.load(target) as data:
        states = data["states"].astype(np.float32)
        actions = data["actions"].astype(np.int64)
        frames = data["frames"].astype(np.uint8) if "frames" in data else None
        score = int(data["score"][0]) if "score" in data else -1
        segment_lengths = (
            data["segment_lengths"].astype(np.int64)
            if "segment_lengths" in data
            else None
        )
        segment_scores = (
            data["segment_scores"].astype(np.int64)
            if "segment_scores" in data
            else None
        )

    if len(actions) == 0:
        raise RuntimeError(f"Episode file is empty: {target}")

    frame_index = max(0, min(frame_index, len(actions) - 1))
    info = {
        "path": target,
        "num_frames": int(len(actions)),
        "score": score,
        "frame_index": frame_index,
        "action": int(actions[frame_index]),
        "state_shape": tuple(states.shape),
        "has_frames": frames is not None,
        "segment_lengths": segment_lengths.tolist()
        if segment_lengths is not None
        else None,
        "segment_scores": segment_scores.tolist()
        if segment_scores is not None
        else None,
    }

    if show:
        columns = 2 if frames is not None else 1
        fig, axes = plt.subplots(1, columns, figsize=(10, 4))
        if columns == 1:
            axes = [axes]

        if frames is not None:
            axes[0].imshow(frames[frame_index])
            axes[0].set_title("Recorded RGB Frame")
            axes[0].axis("off")
            state_axis = axes[1]
        else:
            state_axis = axes[0]

        stacked = np.concatenate([states[frame_index][i] for i in range(states[frame_index].shape[0])], axis=1)
        state_axis.imshow(stacked, cmap="gray")
        state_axis.set_title("CNN State Stack")
        state_axis.axis("off")

        fig.suptitle(
            f"{target.name} | score={score} | frame={frame_index} | action={actions[frame_index]}"
        )
        fig.tight_layout()
        plt.show()

    return info
