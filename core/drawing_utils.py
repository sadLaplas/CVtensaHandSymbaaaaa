"""Утилиты отрисовки landmarks и служебной информации."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from core.logger import get_logger

try:
    import cv2
except Exception as exc:  # pragma: no cover - зависит от окружения исполнения
    cv2 = None  # type: ignore[assignment]
    _OPENCV_IMPORT_ERROR = exc
else:
    _OPENCV_IMPORT_ERROR = None


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

_LOGGER = get_logger("drawing")


def draw_landmarks(frame: np.ndarray, landmarks: list[list[list[float]]]) -> np.ndarray:
    """Рисует точки landmarks на кадре."""

    _ensure_opencv()

    for hand_landmarks in landmarks:
        for point in hand_landmarks:
            x_coord, y_coord = _landmark_to_pixel(point, frame.shape)
            cv2.circle(frame, (x_coord, y_coord), 4, (0, 215, 255), -1)
            cv2.circle(frame, (x_coord, y_coord), 7, (0, 80, 160), 1)

    return frame


def draw_connections(frame: np.ndarray, landmarks: list[list[list[float]]]) -> np.ndarray:
    """Рисует соединения между landmarks одной руки."""

    _ensure_opencv()

    for hand_landmarks in landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx >= len(hand_landmarks) or end_idx >= len(hand_landmarks):
                continue

            start_point = _landmark_to_pixel(hand_landmarks[start_idx], frame.shape)
            end_point = _landmark_to_pixel(hand_landmarks[end_idx], frame.shape)
            cv2.line(frame, start_point, end_point, (60, 180, 75), 2, cv2.LINE_AA)

    return frame


def draw_status_text(frame: np.ndarray, status: Mapping[str, Any]) -> np.ndarray:
    """Рисует поверх кадра компактную панель статуса."""

    _ensure_opencv()

    if not status:
        return frame

    overlay = frame.copy()
    line_height = 24
    panel_width = 320
    panel_height = 16 + line_height * len(status)

    cv2.rectangle(
        overlay, (10, 10), (10 + panel_width, 10 + panel_height), (20, 20, 20), -1
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for index, (key, value) in enumerate(status.items(), start=1):
        text = f"{key}: {value}"
        cv2.putText(
            frame,
            text,
            (20, 10 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def _landmark_to_pixel(point: list[float], frame_shape: tuple[int, ...]) -> tuple[int, int]:
    height, width = frame_shape[:2]
    x_coord = int(min(max(point[0] * width, 0), width - 1))
    y_coord = int(min(max(point[1] * height, 0), height - 1))
    return x_coord, y_coord


def _ensure_opencv() -> None:
    if _OPENCV_IMPORT_ERROR is not None or cv2 is None:
        _LOGGER.error("OpenCV недоступен для отрисовки.")
        raise RuntimeError(
            "OpenCV недоступен. Установите зависимости проекта из requirements.txt."
        ) from _OPENCV_IMPORT_ERROR
