"""Утилиты подготовки hand landmarks для датасета и будущего инференса."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


EXPECTED_LANDMARKS_PER_HAND = 21
EXPECTED_COORDINATES_PER_POINT = 3
_EPSILON = 1e-8


@dataclass(slots=True, frozen=True)
class LandmarkSample:
    """Подготовленный набор landmarks одной руки."""

    handedness: str
    original_landmarks: list[list[float]]
    normalized_landmarks: list[list[float]]
    feature_vector: list[float]


def validate_hand_landmarks(hand_landmarks: list[list[float]]) -> None:
    """Проверяет, что landmarks имеют ожидаемую форму 21 x 3."""

    if len(hand_landmarks) != EXPECTED_LANDMARKS_PER_HAND:
        raise ValueError(
            "Ожидается 21 landmark на руку, "
            f"получено: {len(hand_landmarks)}."
        )

    for point in hand_landmarks:
        if len(point) != EXPECTED_COORDINATES_PER_POINT:
            raise ValueError(
                "Каждый landmark должен содержать 3 координаты: x, y, z."
            )


def normalize_landmarks(hand_landmarks: list[list[float]]) -> list[list[float]]:
    """Нормализует landmarks относительно запястья и масштаба руки."""

    validate_hand_landmarks(hand_landmarks)

    wrist_x, wrist_y, wrist_z = hand_landmarks[0]

    # Make the representation translation invariant by moving wrist to the origin.
    # Делаем представление инвариантным к сдвигу, перенося запястье в начало координат.
    translated_landmarks = [
        [point[0] - wrist_x, point[1] - wrist_y, point[2] - wrist_z]
        for point in hand_landmarks
    ]

    # Use the furthest landmark as a scale reference to reduce sensitivity to distance.
    # Используем самую удаленную точку как масштаб, чтобы снизить чувствительность к дистанции.
    max_distance = max(
        math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        for point in translated_landmarks
    )

    if max_distance < _EPSILON:
        return [[0.0, 0.0, 0.0] for _ in translated_landmarks]

    return [
        [
            point[0] / max_distance,
            point[1] / max_distance,
            point[2] / max_distance,
        ]
        for point in translated_landmarks
    ]


def flatten_landmarks(hand_landmarks: list[list[float]]) -> list[float]:
    """Преобразует landmarks формата 21 x 3 в вектор длиной 63."""

    validate_hand_landmarks(hand_landmarks)
    return [coordinate for point in hand_landmarks for coordinate in point]


def build_landmark_sample(
    hand_landmarks: list[list[float]], handedness: str
) -> LandmarkSample:
    """Готовит один sample: исходные точки, нормализованные точки и feature vector."""

    normalized_landmarks = normalize_landmarks(hand_landmarks)
    feature_vector = flatten_landmarks(normalized_landmarks)

    return LandmarkSample(
        handedness=handedness,
        original_landmarks=hand_landmarks,
        normalized_landmarks=normalized_landmarks,
        feature_vector=feature_vector,
    )


def extract_primary_hand_sample(
    tracking_result: dict[str, Any], require_single_hand: bool = True
) -> LandmarkSample | None:
    """Извлекает и подготавливает sample первой найденной руки."""

    hands_detected = int(tracking_result.get("hands_detected", 0))
    landmarks = tracking_result.get("landmarks", [])
    handedness = tracking_result.get("handedness", [])

    if hands_detected <= 0 or not landmarks:
        return None

    if require_single_hand and hands_detected != 1:
        return None

    primary_landmarks = landmarks[0]
    primary_handedness = handedness[0] if handedness else "Unknown"
    return build_landmark_sample(primary_landmarks, str(primary_handedness))
