from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
import random
from typing import Iterable

import numpy as np
import pygame

from ..utils.utils import clamp
from .assets import GameAssets
from .config import AppConfig


@dataclass
class BirdState:
    x: float
    y: float
    width: int
    height: int
    velocity_y: float = -9.0
    rotation: float = 80.0
    frame_index: int = 0

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class PipeState:
    x: float
    gap_y: float
    passed: bool = False


@dataclass
class GameWorldInfo:
    score: int
    done: bool
    collision: str | None
    passed_pipe: bool
    survival_time: float
    distance_traveled: float


class GameWorld:
    def __init__(
        self,
        config: AppConfig,
        assets: GameAssets,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.assets = assets
        self.surface = pygame.Surface((config.game.width, config.game.height))
        self.font = pygame.font.SysFont("Consolas", 16, bold=True)
        self.rng = random.Random(seed if seed is not None else config.game.random_seed)
        self.frame_cycle = cycle([0, 1, 2, 1])
        self.base_x = 0.0
        self.frame_counter = 0
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        elif self.config.game.deterministic_reset:
            self.rng = random.Random(self.config.game.random_seed)

        bird_image = self.assets.bird_frames[0]
        self.bird = BirdState(
            x=int(self.config.game.width * self.config.game.bird_x_ratio),
            y=int((self.config.game.height - bird_image.get_height()) / 2),
            width=bird_image.get_width(),
            height=bird_image.get_height(),
            velocity_y=self.config.game.flap_velocity,
            rotation=80.0,
            frame_index=0,
        )
        self.frame_cycle = cycle([0, 1, 2, 1])
        self.frame_counter = 0
        self.base_x = 0.0
        self.score = 0
        self.done = False
        self.collision_type: str | None = None
        self.distance_traveled = 0.0
        self.survival_time = 0.0
        self.pipes: list[PipeState] = []
        self._spawn_initial_pipes()
        self.render()

    @property
    def floor_y(self) -> float:
        return self.config.game.floor_y

    @property
    def min_y(self) -> float:
        return -2 * self.bird.height

    @property
    def max_y(self) -> float:
        return self.floor_y - self.bird.height * 0.75

    @property
    def current_bird_frame(self) -> pygame.Surface:
        return self.assets.bird_frames[self.bird.frame_index]

    @property
    def current_bird_mask(self) -> pygame.mask.Mask:
        return pygame.mask.from_surface(self.current_bird_frame)

    def _spawn_initial_pipes(self) -> None:
        first_x = self.config.game.width + self.config.game.initial_pipe_offset
        second_x = first_x + self.config.game.pipe_distance
        self.pipes = [
            PipeState(x=first_x, gap_y=self._random_gap_y()),
            PipeState(x=second_x, gap_y=self._random_gap_y()),
        ]

    def _random_gap_y(self) -> float:
        viewport = self.config.game.viewport_height
        gap_y = self.rng.randrange(
            0, max(1, int(viewport * 0.6 - self.config.game.pipe_gap))
        )
        gap_y += int(viewport * 0.2)
        return float(gap_y)

    def _spawn_pipe_if_needed(self) -> None:
        if not self.pipes:
            self.pipes.append(
                PipeState(
                    x=self.config.game.width + 10,
                    gap_y=self._random_gap_y(),
                )
            )
            return

        last_pipe = self.pipes[-1]
        if last_pipe.x <= self.config.game.width - self.config.game.pipe_distance:
            self.pipes.append(
                PipeState(
                    x=self.config.game.width + 10,
                    gap_y=self._random_gap_y(),
                )
            )

    def _remove_old_pipes(self) -> None:
        pipe_width = self.assets.pipe_lower.get_width()
        self.pipes = [pipe for pipe in self.pipes if pipe.x > -pipe_width]

    def flap(self) -> None:
        if self.done:
            return
        if self.bird.y > self.min_y:
            self.bird.velocity_y = self.config.game.flap_velocity
            self.bird.rotation = 80.0
            self.assets.wing_sound.play()

    def _advance_bird(self) -> None:
        self.bird.velocity_y = min(
            self.bird.velocity_y + self.config.game.gravity,
            self.config.game.max_fall_speed,
        )
        self.bird.y = clamp(
            self.bird.y + self.bird.velocity_y,
            self.min_y,
            self.max_y,
        )
        self.bird.rotation = clamp(self.bird.rotation - 3, -90, 20)
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:
            self.bird.frame_index = next(self.frame_cycle)

    def _advance_pipes(self) -> None:
        for pipe in self.pipes:
            pipe.x -= self.config.game.pipe_speed

    def _advance_floor(self) -> None:
        width = self.assets.base.get_width() - self.config.game.width
        self.base_x = -((-self.base_x + 4) % width)

    def get_next_pipe(self) -> PipeState | None:
        for pipe in self.pipes:
            if pipe.x + self.assets.pipe_lower.get_width() >= self.bird.x:
                return pipe
        return self.pipes[0] if self.pipes else None

    def get_expert_action(self) -> int:
        pipe = self.get_next_pipe()
        if pipe is None:
            return 0

        gap_center = pipe.gap_y + self.config.game.pipe_gap / 2
        tolerance = self.config.expert.center_tolerance
        if self.bird.y >= self.floor_y - self.bird.height - self.config.expert.bottom_safe_margin:
            return 1
        if (
            self.bird.center_y > gap_center + tolerance
            or (
                self.bird.center_y > gap_center
                and self.bird.velocity_y > self.config.expert.emergency_descent_velocity
            )
        ):
            return 1
        if self.bird.y <= self.config.expert.top_safe_margin and self.bird.velocity_y < 0:
            return 0
        return 0

    def _check_pipe_pass(self) -> bool:
        passed_pipe = False
        bird_center_x = self.bird.center_x
        pipe_width = self.assets.pipe_lower.get_width()
        for pipe in self.pipes:
            pipe_center_x = pipe.x + pipe_width / 2
            if not pipe.passed and bird_center_x >= pipe_center_x:
                pipe.passed = True
                self.score += 1
                self.assets.point_sound.play()
                passed_pipe = True
        return passed_pipe

    def _pipe_rects(self, pipe: PipeState) -> tuple[pygame.Rect, pygame.Rect]:
        pipe_height = self.assets.pipe_lower.get_height()
        pipe_width = self.assets.pipe_lower.get_width()
        upper = pygame.Rect(int(pipe.x), int(pipe.gap_y - pipe_height), pipe_width, pipe_height)
        lower = pygame.Rect(int(pipe.x), int(pipe.gap_y + self.config.game.pipe_gap), pipe_width, pipe_height)
        return upper, lower

    def _collides_with_pipe(self) -> bool:
        bird_rect = self.bird.rect
        bird_mask = self.current_bird_mask
        upper_mask = pygame.mask.from_surface(self.assets.pipe_upper)
        lower_mask = pygame.mask.from_surface(self.assets.pipe_lower)

        for pipe in self.pipes:
            upper_rect, lower_rect = self._pipe_rects(pipe)
            if upper_rect.colliderect(bird_rect):
                if bird_mask.overlap(upper_mask, (upper_rect.x - bird_rect.x, upper_rect.y - bird_rect.y)):
                    self.collision_type = "pipe"
                    return True
            if lower_rect.colliderect(bird_rect):
                if bird_mask.overlap(lower_mask, (lower_rect.x - bird_rect.x, lower_rect.y - bird_rect.y)):
                    self.collision_type = "pipe"
                    return True
        return False

    def _collides_with_floor(self) -> bool:
        if self.bird.y + self.bird.height >= self.floor_y:
            self.collision_type = "floor"
            return True
        return False

    def step(self, action: int) -> GameWorldInfo:
        if self.done:
            return self.info(False)

        if action == 1:
            self.flap()

        self._advance_bird()
        self._advance_pipes()
        self._advance_floor()
        self._spawn_pipe_if_needed()
        self._remove_old_pipes()
        passed_pipe = self._check_pipe_pass()

        if self._collides_with_floor() or self._collides_with_pipe():
            self.done = True
            self.assets.hit_sound.play()
            self.assets.die_sound.play()

        self.distance_traveled += self.config.game.pipe_speed
        self.survival_time += 1.0 / self.config.game.fps
        return self.info(passed_pipe)

    def info(self, passed_pipe: bool) -> GameWorldInfo:
        return GameWorldInfo(
            score=self.score,
            done=self.done,
            collision=self.collision_type,
            passed_pipe=passed_pipe,
            survival_time=self.survival_time,
            distance_traveled=self.distance_traveled,
        )

    def render(self, overlay_lines: Iterable[str] | None = None) -> pygame.Surface:
        self.surface.blit(
            pygame.transform.scale(
                self.assets.background,
                (self.config.game.width, self.config.game.height),
            ),
            (0, 0),
        )

        for pipe in self.pipes:
            upper_rect, lower_rect = self._pipe_rects(pipe)
            self.surface.blit(self.assets.pipe_upper, upper_rect)
            self.surface.blit(self.assets.pipe_lower, lower_rect)

        rotated = pygame.transform.rotate(self.current_bird_frame, self.bird.rotation)
        rotated_rect = rotated.get_rect(center=self.bird.rect.center)
        self.surface.blit(rotated, rotated_rect)

        self.surface.blit(self.assets.base, (self.base_x, self.floor_y))
        self._draw_score()
        if overlay_lines:
            self._draw_overlay(list(overlay_lines))
        return self.surface

    def _draw_score(self) -> None:
        digits = [int(value) for value in str(self.score)]
        images = [self.assets.numbers[digit] for digit in digits]
        total_width = sum(image.get_width() for image in images)
        x = (self.config.game.width - total_width) / 2
        y = self.config.game.height * 0.1
        for image in images:
            self.surface.blit(image, (x, y))
            x += image.get_width()

    def _draw_overlay(self, lines: list[str]) -> None:
        padding = 8
        for index, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            shadow = self.font.render(line, True, (0, 0, 0))
            y = padding + index * 18
            self.surface.blit(shadow, (padding + 1, y + 1))
            self.surface.blit(text, (padding, y))

    def get_frame_rgb(self) -> np.ndarray:
        frame = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame, (1, 0, 2)).copy()
