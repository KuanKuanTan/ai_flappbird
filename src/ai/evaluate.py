from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..env import FlappyEnv
from ..game import AppConfig, GameAssets
from .model import FlappyCNN


@dataclass
class EvaluationSummary:
    episodes: int
    average_score: float
    best_score: int
    average_survival_time: float
    average_distance: float


def load_model(config: AppConfig, model_path: str | Path) -> FlappyCNN:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = FlappyCNN(input_channels=config.preprocessing.frame_stack)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_model(
    config: AppConfig,
    assets: GameAssets,
    model_path: str | Path,
    episodes: int,
) -> EvaluationSummary:
    model = load_model(config, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    env = FlappyEnv(config, assets, seed=config.game.random_seed)
    scores: list[int] = []
    times: list[float] = []
    distances: list[float] = []

    for episode in range(episodes):
        state = env.reset(seed=config.game.random_seed + episode)
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(state_tensor)
            action = int(torch.argmax(logits, dim=1).item())
            state, _, done, info = env.step(action)

        scores.append(info.score)
        times.append(info.survival_time)
        distances.append(info.distance_traveled)

    return EvaluationSummary(
        episodes=episodes,
        average_score=sum(scores) / max(len(scores), 1),
        best_score=max(scores) if scores else 0,
        average_survival_time=sum(times) / max(len(times), 1),
        average_distance=sum(distances) / max(len(distances), 1),
    )
