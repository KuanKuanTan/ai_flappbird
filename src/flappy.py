from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pygame
import torch

from .ai.dataset import (
    collect_dataset,
    get_next_session_index,
    inspect_episode_record,
    save_recording_session,
    trim_episode_tail,
)
from .ai.evaluate import evaluate_model
from .ai.evaluate import load_model
from .ai.processor import FrameProcessor
from .ai.trainer import train_model
from .env import FlappyEnv
from .game import AppConfig, GameAssets, GameWorld, load_config


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def bootstrap_pygame(headless: bool, size: tuple[int, int]) -> pygame.Surface:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    try:
        pygame.mixer.init()
    except pygame.error:
        LOGGER.warning("Audio mixer unavailable; continuing without sound.")
    pygame.display.set_caption("Flappy Bird CNN")
    return pygame.display.set_mode(size)


def load_runtime(config_path: str | Path, headless: bool = False):
    config = load_config(config_path)
    display_size = (
        config.game.width * config.game.render_scale,
        config.game.height * config.game.render_scale,
    )
    screen = bootstrap_pygame(headless=headless, size=(1, 1) if headless else display_size)
    assets = GameAssets.load(config, enable_audio=not headless)
    return config, screen, assets


def blit_scaled(screen: pygame.Surface, world: GameWorld, config: AppConfig) -> None:
    frame = pygame.transform.scale(
        world.surface,
        (
            config.game.width * config.game.render_scale,
            config.game.height * config.game.render_scale,
        ),
    )
    screen.blit(frame, (0, 0))
    pygame.display.flip()


def poll_events() -> tuple[bool, bool]:
    flap_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False, False
            if event.key in (pygame.K_SPACE, pygame.K_UP):
                flap_pressed = True
    return True, flap_pressed


def run_human_mode(config: AppConfig, screen: pygame.Surface, assets: GameAssets) -> int:
    world = GameWorld(config, assets, seed=config.game.random_seed)
    clock = pygame.time.Clock()

    while True:
        running, flap_pressed = poll_events()
        if not running:
            return 0
        action = 1 if flap_pressed else 0
        info = world.step(action)
        overlay = [
            "Mode: Human",
            f"Score: {info.score}",
            f"Survival: {info.survival_time:.1f}s",
        ]
        if info.done:
            overlay.append("Press R to restart or ESC to quit")
        world.render(overlay)
        blit_scaled(screen, world, config)

        if info.done:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return 0
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return 0
                        if event.key == pygame.K_r:
                            world.reset(seed=config.game.random_seed)
                            waiting = False
                world.render(overlay)
                blit_scaled(screen, world, config)
                clock.tick(config.game.fps)
        else:
            clock.tick(config.game.fps)


def run_collect_mode(
    config: AppConfig,
    assets: GameAssets,
    episodes: int,
    controller: str,
) -> int:
    summary = collect_dataset(config, assets, episodes=episodes, controller=controller)
    LOGGER.info(
        "Collection finished: episodes=%s frames=%s avg_score=%.2f",
        summary.episodes,
        summary.frames,
        summary.avg_score,
    )
    return 0


def run_record_mode(
    config: AppConfig,
    screen: pygame.Surface,
    assets: GameAssets,
    episodes: int,
) -> int:
    env = FlappyEnv(config, assets, seed=config.game.random_seed)
    clock = pygame.time.Clock()
    session_index = get_next_session_index(config.dataset_dir)
    episode_index = 0
    total_frames = 0
    session_states: list[np.ndarray] = []
    session_actions: list[int] = []
    session_frames: list[np.ndarray] = []
    segment_lengths: list[int] = []
    segment_scores: list[int] = []

    def save_current_session() -> Path | None:
        if not session_states:
            return None
        return save_recording_session(
            config.dataset_dir,
            session_index,
            session_states,
            session_actions,
            session_frames,
            segment_lengths,
            segment_scores,
        )

    while episode_index < episodes:
        state = env.reset(seed=config.game.random_seed + session_index * 1000 + episode_index)
        states: list[np.ndarray] = []
        actions: list[int] = []
        frames: list[np.ndarray] = []
        done = False
        final_info = None
        started = False

        while not started:
            running, flap_pressed = poll_events()
            if not running:
                saved = save_current_session()
                if saved is not None:
                    LOGGER.info("Saved partial recording session to %s", saved)
                return 0

            env.world.render(
                [
                    "Mode: Record",
                    f"Session File: {session_index:05d}",
                    f"Session Progress: {episode_index + 1}/{episodes}",
                    "Press SPACE/UP to start",
                    "ESC: save and quit",
                ]
            )
            blit_scaled(screen, env.world, config)
            clock.tick(config.game.fps)

            if flap_pressed:
                started = True

        while not done:
            running, flap_pressed = poll_events()
            if not running:
                saved = save_current_session()
                if saved is not None:
                    LOGGER.info("Saved partial recording session to %s", saved)
                return 0

            action = 1 if flap_pressed else 0

            states.append(state.copy())
            frames.append(env.world.get_frame_rgb().copy())
            actions.append(action)

            state, _, done, info = env.step(action)
            final_info = info
            overlay = [
                "Mode: Record",
                f"Session File: {session_index:05d}",
                f"Session Progress: {episode_index + 1}/{episodes}",
                f"Score: {info.score}",
                f"Frames: {len(states)}",
                f"Action: {'FLAP' if action == 1 else 'WAIT'}",
                "SPACE/UP: flap",
            ]
            if done:
                overlay.append("R: next episode, ESC: quit")
            env.world.render(overlay)
            blit_scaled(screen, env.world, config)
            clock.tick(config.game.fps)

        trimmed = 0
        if final_info is not None and final_info.collision is not None:
            states, actions, frames, trimmed = trim_episode_tail(
                states,
                actions,
                frames,
                config.training.trim_death_frames,
                config.training.min_segment_frames,
            )

        if len(states) >= config.training.min_segment_frames:
            session_states.extend(states)
            session_actions.extend(actions)
            session_frames.extend(frames)
            segment_lengths.append(len(states))
            segment_scores.append(env.world.score)
            total_frames += len(states)

        LOGGER.info(
            "Recorded segment %s/%s kept=%s trimmed=%s score=%s collision=%s",
            episode_index + 1,
            episodes,
            len(states),
            trimmed,
            env.world.score,
            final_info.collision if final_info is not None else None,
        )

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    saved = save_current_session()
                    if saved is not None:
                        LOGGER.info("Saved partial recording session to %s", saved)
                    return 0
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        saved = save_current_session()
                        if saved is not None:
                            LOGGER.info("Saved partial recording session to %s", saved)
                        return 0
                    if event.key == pygame.K_r:
                        waiting = False

            env.world.render(
                [
                    "Mode: Record",
                    f"Session: {session_index:05d}",
                    f"Segment saved: {episode_index + 1}/{episodes}",
                    f"Score: {env.world.score}",
                    f"Kept Frames: {len(states)}",
                    f"Trimmed Tail: {trimmed}",
                    "Press R for next episode",
                ]
            )
            blit_scaled(screen, env.world, config)
            clock.tick(config.game.fps)

        episode_index += 1

    file_path = save_current_session()
    LOGGER.info(
        "Recording complete: session=%s segments=%s total_frames=%s output=%s",
        session_index,
        len(segment_lengths),
        total_frames,
        file_path,
    )
    return 0


def run_inspect_mode(
    config: AppConfig,
    episode: int | None,
    frame: int,
    no_show: bool,
) -> int:
    info = inspect_episode_record(
        config.dataset_dir,
        episode_index=episode,
        frame_index=frame,
        show=not no_show,
    )
    LOGGER.info(
        "Inspect: path=%s num_frames=%s score=%s frame=%s action=%s state_shape=%s has_frames=%s",
        info["path"],
        info["num_frames"],
        info["score"],
        info["frame_index"],
        info["action"],
        info["state_shape"],
        info["has_frames"],
    )
    return 0


def run_train_mode(config: AppConfig, resume: bool) -> int:
    summary = train_model(config, resume=resume)
    LOGGER.info(
        "Training finished: model=%s best_val_accuracy=%.4f epochs=%s time=%.1fs",
        summary.model_path,
        summary.best_val_accuracy,
        summary.epochs,
        summary.seconds,
    )
    return 0


def _predict_action(model, device, state):
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    action = int(probs.argmax())
    return action, probs


def run_ai_play_mode(
    config: AppConfig,
    screen: pygame.Surface,
    assets: GameAssets,
    model_path: str | Path,
) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, model_path).to(device)
    env = FlappyEnv(config, assets, seed=config.game.random_seed)
    state = env.reset()
    clock = pygame.time.Clock()

    while True:
        running, _ = poll_events()
        if not running:
            return 0
        action, probs = _predict_action(model, device, state)
        state, _, done, info = env.step(action)
        overlay = [
            "Mode: AI",
            f"Score: {info.score}",
            f"Action: {'FLAP' if action == 1 else 'WAIT'}",
            f"P(wait): {probs[0]:.3f}",
            f"P(flap): {probs[1]:.3f}",
            f"FPS: {clock.get_fps():.1f}",
        ]
        env.world.render(overlay)
        blit_scaled(screen, env.world, config)
        clock.tick(config.game.fps)

        if done:
            pygame.time.delay(1200)
            state = env.reset(seed=config.game.random_seed)


def run_battle_mode(
    config: AppConfig,
    assets: GameAssets,
    model_path: str | Path,
) -> int:
    width = config.game.width * config.game.render_scale
    height = config.game.height * config.game.render_scale
    screen = bootstrap_pygame(False, (width * 2, height))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, model_path).to(device)
    human = GameWorld(config, assets, seed=config.game.random_seed)
    ai_world = GameWorld(config, assets, seed=config.game.random_seed)
    ai_processor = FrameProcessor(config.preprocessing)
    ai_world.render()
    ai_processor.reset(ai_world.get_frame_rgb())
    clock = pygame.time.Clock()

    while True:
        running, flap_pressed = poll_events()
        if not running:
            return 0
        human_action = 1 if flap_pressed else 0
        ai_state = ai_processor.state
        ai_action, probs = _predict_action(model, device, ai_state)

        human_info = human.step(human_action) if not human.done else human.info(False)
        ai_info = ai_world.step(ai_action) if not ai_world.done else ai_world.info(False)
        ai_world.render()
        ai_processor.push(ai_world.get_frame_rgb())

        human_overlay = [
            "Human",
            f"Score: {human_info.score}",
            f"Time: {human_info.survival_time:.1f}s",
            f"Dist: {human_info.distance_traveled:.0f}",
        ]
        ai_overlay = [
            "AI",
            f"Score: {ai_info.score}",
            f"Time: {ai_info.survival_time:.1f}s",
            f"Dist: {ai_info.distance_traveled:.0f}",
            f"A: {'FLAP' if ai_action == 1 else 'WAIT'}",
            f"P(flap): {probs[1]:.3f}",
        ]

        human_frame = pygame.transform.scale(human.render(human_overlay), (width, height))
        ai_frame = pygame.transform.scale(ai_world.render(ai_overlay), (width, height))
        screen.blit(human_frame, (0, 0))
        screen.blit(ai_frame, (width, 0))

        if human.done and ai_world.done:
            if human_info.score > ai_info.score:
                winner = "Winner: Human"
            elif ai_info.score > human_info.score:
                winner = "Winner: AI"
            elif human_info.survival_time > ai_info.survival_time:
                winner = "Winner: Human (time)"
            elif ai_info.survival_time > human_info.survival_time:
                winner = "Winner: AI (time)"
            else:
                winner = "Winner: Draw"
            font = pygame.font.SysFont("Consolas", 22, bold=True)
            text = font.render(winner, True, (255, 255, 0))
            screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(config.game.fps)


def run_evaluate_mode(
    config: AppConfig,
    assets: GameAssets,
    model_path: str | Path,
    episodes: int,
) -> int:
    summary = evaluate_model(config, assets, model_path=model_path, episodes=episodes)
    LOGGER.info(
        "Evaluation: episodes=%s avg_score=%.2f best_score=%s avg_survival=%.2fs avg_distance=%.2f",
        summary.episodes,
        summary.average_score,
        summary.best_score,
        summary.average_survival_time,
        summary.average_distance,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flappy Bird CNN platform")
    parser.add_argument(
        "--config",
        default="config/flappy_ai.toml",
        help="Path to the application TOML config.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("play", help="Play Flappy Bird manually.")
    record_parser = subparsers.add_parser(
        "record", help="Play manually and record images/actions for CNN training."
    )
    record_parser.add_argument("--episodes", type=int, default=1)
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a recorded episode file."
    )
    inspect_parser.add_argument("--episode", type=int, default=None)
    inspect_parser.add_argument("--frame", type=int, default=0)
    inspect_parser.add_argument("--no-show", action="store_true")

    collect_parser = subparsers.add_parser("collect", help="Collect CNN training data.")
    collect_parser.add_argument("--episodes", type=int, default=None)
    collect_parser.add_argument(
        "--controller",
        choices=["expert", "human"],
        default="expert",
    )

    train_parser = subparsers.add_parser("train", help="Train the CNN model.")
    train_parser.add_argument("--resume", action="store_true")

    ai_parser = subparsers.add_parser("ai-play", help="Run the trained AI player.")
    ai_parser.add_argument("--model", default="models/flappy_cnn.pth")

    battle_parser = subparsers.add_parser("battle", help="Human vs AI battle.")
    battle_parser.add_argument("--model", default="models/flappy_cnn.pth")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the CNN model.")
    evaluate_parser.add_argument("--model", default="models/flappy_cnn.pth")
    evaluate_parser.add_argument("--episodes", type=int, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    headless = args.mode in {"train", "evaluate"} or (
        args.mode == "collect" and getattr(args, "controller", "expert") == "expert"
    )
    config, screen, assets = load_runtime(args.config, headless=headless)

    try:
        if args.mode == "play":
            return run_human_mode(config, screen, assets)
        if args.mode == "record":
            return run_record_mode(config, screen, assets, args.episodes)
        if args.mode == "inspect":
            return run_inspect_mode(config, args.episode, args.frame, args.no_show)
        if args.mode == "collect":
            episodes = args.episodes or config.training.collect_episodes
            return run_collect_mode(config, assets, episodes, args.controller)
        if args.mode == "train":
            return run_train_mode(config, resume=args.resume)
        if args.mode == "ai-play":
            return run_ai_play_mode(config, screen, assets, args.model)
        if args.mode == "battle":
            return run_battle_mode(config, assets, args.model)
        if args.mode == "evaluate":
            episodes = args.episodes or config.training.evaluation_episodes
            return run_evaluate_mode(config, assets, args.model, episodes)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    finally:
        pygame.quit()

    return 0
