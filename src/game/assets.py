from __future__ import annotations

from dataclasses import dataclass

import pygame

from ..utils.constants import BACKGROUNDS, PIPES, PLAYERS
from .config import ROOT_DIR, AppConfig


class SilentSound:
    def play(self) -> None:
        return None


@dataclass
class GameAssets:
    numbers: list[pygame.Surface]
    background: pygame.Surface
    bird_frames: tuple[pygame.Surface, pygame.Surface, pygame.Surface]
    pipe_upper: pygame.Surface
    pipe_lower: pygame.Surface
    base: pygame.Surface
    wing_sound: object
    point_sound: object
    hit_sound: object
    die_sound: object

    @classmethod
    def load(cls, config: AppConfig, enable_audio: bool = True) -> "GameAssets":
        assets_dir = ROOT_DIR / "assets"
        numbers = [
            pygame.image.load(str(assets_dir / "sprites" / f"{index}.png")).convert_alpha()
            for index in range(10)
        ]
        background = pygame.image.load(
            str(ROOT_DIR / BACKGROUNDS[config.game.background_index])
        ).convert()
        bird_frames = tuple(
            pygame.image.load(str(ROOT_DIR / path)).convert_alpha()
            for path in PLAYERS[config.game.player_index]
        )
        pipe_source = pygame.image.load(
            str(ROOT_DIR / PIPES[config.game.pipe_index])
        ).convert_alpha()
        pipe_upper = pygame.transform.flip(pipe_source, False, True)
        pipe_lower = pipe_source
        base = pygame.image.load(str(assets_dir / "sprites" / "base.png")).convert_alpha()

        if enable_audio and pygame.mixer.get_init():
            wing_sound = pygame.mixer.Sound(str(assets_dir / "audio" / "wing.ogg"))
            point_sound = pygame.mixer.Sound(str(assets_dir / "audio" / "point.ogg"))
            hit_sound = pygame.mixer.Sound(str(assets_dir / "audio" / "hit.ogg"))
            die_sound = pygame.mixer.Sound(str(assets_dir / "audio" / "die.ogg"))
        else:
            wing_sound = point_sound = hit_sound = die_sound = SilentSound()

        return cls(
            numbers=numbers,
            background=background,
            bird_frames=bird_frames,
            pipe_upper=pipe_upper,
            pipe_lower=pipe_lower,
            base=base,
            wing_sound=wing_sound,
            point_sound=point_sound,
            hit_sound=hit_sound,
            die_sound=die_sound,
        )
