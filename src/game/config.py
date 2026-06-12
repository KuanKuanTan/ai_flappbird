from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict
import tomllib


ROOT_DIR = Path(__file__).resolve().parents[2]


def _update_dataclass(instance, values: Dict[str, Any]):
    valid = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key in valid:
            setattr(instance, key, value)
    return instance


@dataclass
class GameSettings:
    width: int = 288
    height: int = 512
    fps: int = 30
    render_scale: int = 2
    pipe_gap: int = 120
    pipe_speed: int = 5
    pipe_distance: int = 150
    initial_pipe_offset: int = 220
    gravity: float = 1.0
    flap_velocity: float = -9.0
    max_fall_speed: float = 10.0
    bird_x_ratio: float = 0.2
    viewport_ratio: float = 0.79
    random_seed: int = 7
    deterministic_reset: bool = True
    debug: bool = False
    background_index: int = 0
    player_index: int = 2
    pipe_index: int = 0

    @property
    def viewport_height(self) -> float:
        return self.height * self.viewport_ratio

    @property
    def floor_y(self) -> float:
        return self.viewport_height


@dataclass
class PreprocessSettings:
    frame_width: int = 84
    frame_height: int = 84
    frame_stack: int = 4


@dataclass
class TrainingSettings:
    dataset_dir: str = "data/datasets"
    models_dir: str = "models"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    batch_size: int = 64
    learning_rate: float = 5e-4
    epochs: int = 20
    validation_split: float = 0.1
    num_workers: int = 0
    mixed_precision: bool = True
    tensorboard: bool = True
    checkpoint_interval: int = 1
    collect_episodes: int = 100
    max_episode_steps: int = 5000
    evaluation_episodes: int = 10
    trim_death_frames: int = 8
    min_segment_frames: int = 12


@dataclass
class ExpertSettings:
    center_tolerance: float = 12.0
    emergency_descent_velocity: float = 5.0
    top_safe_margin: float = 26.0
    bottom_safe_margin: float = 36.0


@dataclass
class BattleSettings:
    countdown_seconds: int = 2


@dataclass
class AppConfig:
    game: GameSettings
    preprocessing: PreprocessSettings
    training: TrainingSettings
    expert: ExpertSettings
    battle: BattleSettings
    root_dir: Path = ROOT_DIR

    @property
    def dataset_dir(self) -> Path:
        return self.root_dir / self.training.dataset_dir

    @property
    def models_dir(self) -> Path:
        return self.root_dir / self.training.models_dir

    @property
    def checkpoints_dir(self) -> Path:
        return self.root_dir / self.training.checkpoints_dir

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / self.training.logs_dir

    def ensure_directories(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path

    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

    config = AppConfig(
        game=_update_dataclass(GameSettings(), raw.get("game", {})),
        preprocessing=_update_dataclass(
            PreprocessSettings(), raw.get("preprocessing", {})
        ),
        training=_update_dataclass(TrainingSettings(), raw.get("training", {})),
        expert=_update_dataclass(ExpertSettings(), raw.get("expert", {})),
        battle=_update_dataclass(BattleSettings(), raw.get("battle", {})),
    )
    config.ensure_directories()
    return config
