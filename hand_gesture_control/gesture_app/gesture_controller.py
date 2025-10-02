from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyautogui

# Configure PyAutoGUI failsafes (move mouse to a corner to abort)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # we control rate ourselves


@dataclass
class PointerState:
    screen_width: int
    screen_height: int
    last_position: Optional[Tuple[int, int]] = None
    velocity: np.ndarray = np.zeros(2)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x
        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


class GestureController:
    def __init__(self, pointer_sensitivity: float = 1.0) -> None:
        w, h = pyautogui.size()
        self.state = PointerState(screen_width=w, screen_height=h)
        self.filter = OneEuroFilter(min_cutoff=1.7, beta=0.3, d_cutoff=1.0)
        self.pointer_sensitivity = pointer_sensitivity
        self._last_click_time = 0.0

    def normalized_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = int(np.clip(x, 0.0, 1.0) * self.state.screen_width)
        sy = int(np.clip(y, 0.0, 1.0) * self.state.screen_height)
        return sx, sy

    def move_pointer(self, x_norm: float, y_norm: float) -> None:
        t = time.perf_counter()
        target = np.array(self.normalized_to_screen(x_norm, y_norm), dtype=float)
        filtered = self.filter(target, t)
        if self.state.last_position is None:
            self.state.last_position = (int(filtered[0]), int(filtered[1]))
        else:
            current = np.array(self.state.last_position, dtype=float)
            delta = (filtered - current) * self.pointer_sensitivity
            new_pos = current + delta
            self.state.last_position = (int(new_pos[0]), int(new_pos[1]))
        pyautogui.moveTo(self.state.last_position[0], self.state.last_position[1])

    def left_click(self) -> None:
        now = time.perf_counter()
        if now - self._last_click_time > 0.25:
            pyautogui.click(button="left")
            self._last_click_time = now

    def right_click(self) -> None:
        now = time.perf_counter()
        if now - self._last_click_time > 0.25:
            pyautogui.click(button="right")
            self._last_click_time = now

    def scroll(self, dy: int) -> None:
        pyautogui.scroll(dy)
