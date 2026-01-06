from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import mediapipe as mp
except Exception as exc:
    raise RuntimeError(
        "Failed to import mediapipe. Please install dependencies from requirements.txt"
    ) from exc


@dataclass
class HandLandmarks:
    handedness: str
    landmarks: List[Tuple[float, float, float]]  # x, y, z normalized
    bbox_xywh: Tuple[int, int, int, int]


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[HandLandmarks]]:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self._hands.process(image_rgb)
        image_rgb.flags.writeable = True

        output_bgr = frame_bgr.copy()
        detected: List[HandLandmarks] = []

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                h, w = output_bgr.shape[:2]
                coords = []
                xs, ys = [], []
                for lm in hand_landmarks.landmark:
                    x, y, z = lm.x, lm.y, lm.z
                    coords.append((x, y, z))
                    xs.append(int(x * w))
                    ys.append(int(y * h))

                x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
                y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                hl = HandLandmarks(
                    handedness=handedness.classification[0].label,
                    landmarks=coords,
                    bbox_xywh=bbox,
                )
                detected.append(hl)

                self._mp_draw.draw_landmarks(
                    output_bgr,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )

        return output_bgr, detected

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
