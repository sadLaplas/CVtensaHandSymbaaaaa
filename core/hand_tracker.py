"""Обертка над MediaPipe Tasks Hand Landmarker."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.config import HandTrackerConfig
from core.logger import get_logger

try:
    import cv2
except Exception as exc:  # pragma: no cover - зависит от окружения исполнения
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except Exception as exc:  # pragma: no cover - зависит от окружения исполнения
    mp = None  # type: ignore[assignment]
    mp_python = None  # type: ignore[assignment]
    vision = None  # type: ignore[assignment]
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None


class HandTrackerError(RuntimeError):
    """Базовая ошибка трекера рук."""


class HandTrackerDependencyError(HandTrackerError):
    """Ошибка отсутствующей зависимости."""


class ModelNotFoundError(HandTrackerError):
    """Ошибка отсутствующей модели."""


@dataclass(slots=True)
class HandTrackingResult:
    """Нормализованный результат детекции рук."""

    hands_detected: int = 0
    handedness: list[str] = field(default_factory=list)
    landmarks: list[list[list[float]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Возвращает результат в словарном виде."""

        return asdict(self)


class HandTracker:
    """Обертка над MediaPipe Tasks Hand Landmarker для видеопотока."""

    def __init__(self, config: HandTrackerConfig) -> None:
        self._config = config
        self._logger = get_logger("hand_tracker")
        self._landmarker: Any | None = None
        self._last_timestamp_ms = 0

    @property
    def model_path(self) -> Path:
        """Возвращает путь к файлу модели."""

        return self._config.model_path

    @property
    def is_initialized(self) -> bool:
        """Показывает, инициализирован ли landmarker."""

        return self._landmarker is not None

    def initialize(self) -> None:
        """Создает экземпляр HandLandmarker."""

        if self.is_initialized:
            return

        if _CV2_IMPORT_ERROR is not None or cv2 is None:
            raise HandTrackerDependencyError(
                "OpenCV недоступен. Установите зависимости проекта из requirements.txt."
            ) from _CV2_IMPORT_ERROR

        if (
            _MEDIAPIPE_IMPORT_ERROR is not None
            or mp is None
            or mp_python is None
            or vision is None
        ):
            raise HandTrackerDependencyError(
                "MediaPipe Tasks недоступен. Установите зависимости проекта из requirements.txt."
            ) from _MEDIAPIPE_IMPORT_ERROR

        if not self._config.model_path.exists():
            raise ModelNotFoundError(
                f"Файл модели Hand Landmarker не найден: {self._config.model_path}"
            )

        self._logger.info("Загрузка модели Hand Landmarker: %s", self._config.model_path)

        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=str(self._config.model_path)
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self._config.num_hands,
                min_hand_detection_confidence=self._config.min_hand_detection_confidence,
                min_hand_presence_confidence=self._config.min_hand_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - зависит от рантайма MediaPipe
            raise HandTrackerError(
                "Не удалось инициализировать MediaPipe Hand Landmarker."
            ) from exc

    def process_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Обрабатывает BGR-кадр из OpenCV и возвращает структурированный результат."""

        if frame_bgr is None or frame_bgr.size == 0:
            self._logger.warning("Получен пустой кадр для обработки.")
            return HandTrackingResult().to_dict()

        if not self.is_initialized:
            self.initialize()

        if cv2 is None or mp is None:
            raise HandTrackerDependencyError(
                "Зависимости трекера недоступны. Проверьте установку проекта."
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        except Exception as exc:  # pragma: no cover - зависит от рантайма MediaPipe
            raise HandTrackerError("Ошибка обработки видеокадра в MediaPipe.") from exc

        if not result.hand_landmarks:
            return HandTrackingResult().to_dict()

        handedness = [self._extract_handedness_label(item) for item in result.handedness]
        landmarks = [
            [[point.x, point.y, point.z] for point in hand_landmarks]
            for hand_landmarks in result.hand_landmarks
        ]

        return HandTrackingResult(
            hands_detected=len(landmarks),
            handedness=handedness,
            landmarks=landmarks,
        ).to_dict()

    def close(self) -> None:
        """Освобождает ресурсы landmarker."""

        if self._landmarker is None:
            return

        close_method = getattr(self._landmarker, "close", None)
        if callable(close_method):
            close_method()

        self._landmarker = None
        self._logger.info("Ресурсы Hand Landmarker освобождены.")

    def __del__(self) -> None:
        self.close()

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = time.monotonic_ns() // 1_000_000
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    @staticmethod
    def _extract_handedness_label(categories: list[Any]) -> str:
        if not categories:
            return "Unknown"

        category = categories[0]
        label = getattr(category, "category_name", None) or getattr(
            category, "display_name", None
        )
        return str(label or "Unknown")
