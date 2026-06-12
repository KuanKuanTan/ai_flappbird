from __future__ import annotations

from dataclasses import dataclass

from ..ai.processor import FrameProcessor
from ..game import AppConfig, GameAssets, GameWorld


@dataclass
class StepResult:
    state: object
    reward: float
    done: bool
    info: object


class FlappyEnv:
    def __init__(
        self,
        config: AppConfig,
        assets: GameAssets,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.world = GameWorld(config, assets, seed=seed)
        self.processor = FrameProcessor(config.preprocessing)

    def reset(self, seed: int | None = None):
        self.world.reset(seed=seed)
        self.world.render()
        frame = self.world.get_frame_rgb()
        self.processor.reset(frame)
        return self.get_state()

    def get_state(self):
        return self.processor.state

    def step(self, action: int) -> tuple[object, float, bool, object]:
        info = self.world.step(action)
        self.world.render()
        frame = self.world.get_frame_rgb()
        self.processor.push(frame)
        reward = float(info.score)
        return self.get_state(), reward, info.done, info
