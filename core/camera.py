"""Сервис для безопасной работы с веб-камерой."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import CameraConfig
from core.logger import get_logger

try:
    import cv2
except Exception as exc:  # pragma: no cover - зависит от окружения исполнения
    cv2 = None  # type: ignore[assignment]
    _OPENCV_IMPORT_ERROR = exc
else:
    _OPENCV_IMPORT_ERROR = None


class CameraError(RuntimeError):
    """Ошибка работы с камерой."""


class CameraDependencyError(CameraError):
    """Ошибка подключения зависимости OpenCV."""


class CameraService:
    """Обертка над OpenCV VideoCapture."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: Any | None = None
        self._logger = get_logger("camera")

    @property
    def is_opened(self) -> bool:
        """Показывает, открыта ли камера."""

        return bool(self._capture is not None and self._capture.isOpened())

    def initialize_camera(self) -> None:
        """Открывает камеру и применяет параметры из конфига."""

        if _OPENCV_IMPORT_ERROR is not None or cv2 is None:
            raise CameraDependencyError(
                "OpenCV недоступен. Установите зависимости проекта из requirements.txt."
            ) from _OPENCV_IMPORT_ERROR

        if self.is_opened:
            return

        self._logger.info(
            "Инициализация камеры: device_id=%s, width=%s, height=%s, fps=%s",
            self._config.device_id,
            self._config.width,
            self._config.height,
            self._config.fps,
        )

        self._capture = cv2.VideoCapture(self._config.device_id, cv2.CAP_ANY)

        if not self._capture.isOpened():
            self.release_camera()
            raise CameraError(
                f"Не удалось открыть камеру с device_id={self._config.device_id}."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        self._capture.set(cv2.CAP_PROP_FPS, float(self._config.fps))

    def read_frame(self) -> np.ndarray | None:
        """Считывает один кадр с камеры."""

        if not self.is_opened:
            raise CameraError("Камера не инициализирована.")

        success, frame = self._capture.read()
        if not success or frame is None:
            self._logger.warning("Не удалось получить кадр с камеры.")
            return None

        return frame

    def release_camera(self) -> None:
        """Освобождает ресурсы камеры."""

        if self._capture is None:
            return

        try:
            self._capture.release()
        finally:
            self._capture = None
            self._logger.info("Ресурсы камеры освобождены.")

    def __del__(self) -> None:
        self.release_camera()
