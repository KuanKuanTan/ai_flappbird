from __future__ import annotations

from collections import deque

import numpy as np

from ..game.config import PreprocessSettings


class FrameProcessor:
    def __init__(self, settings: PreprocessSettings) -> None:
        self.settings = settings
        self.frames: deque[np.ndarray] = deque(maxlen=settings.frame_stack)
        self._cv2 = None

    @property
    def cv2(self):
        if self._cv2 is None:
            try:
                import cv2
            except Exception as exc:
                raise RuntimeError(
                    "OpenCV is unavailable. Verify that opencv-python and numpy "
                    "are installed with compatible versions before running data "
                    "collection, training, AI play, or evaluation."
                ) from exc

            self._cv2 = cv2
        return self._cv2

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2GRAY)
        resized = self.cv2.resize(
            gray,
            (self.settings.frame_width, self.settings.frame_height),
            interpolation=self.cv2.INTER_AREA,
        )
        return resized.astype(np.float32) / 255.0

    def reset(self, frame: np.ndarray) -> np.ndarray:
        processed = self.preprocess(frame)
        self.frames.clear()
        for _ in range(self.settings.frame_stack):
            self.frames.append(processed)
        return self.state

    def push(self, frame: np.ndarray) -> np.ndarray:
        processed = self.preprocess(frame)
        if not self.frames:
            return self.reset(frame)
        self.frames.append(processed)
        return self.state

    @property
    def state(self) -> np.ndarray:
        if not self.frames:
            raise RuntimeError("Frame stack is empty. Call reset() first.")
        return np.stack(self.frames, axis=0).astype(np.float32)
