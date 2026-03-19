"""Тесты для подготовки hand landmarks."""

from __future__ import annotations

import unittest

from core.landmark_utils import (
    EXPECTED_LANDMARKS_PER_HAND,
    build_landmark_sample,
    extract_primary_hand_sample,
    flatten_landmarks,
    normalize_landmarks,
)


def _make_hand(offset_x: float = 0.0, offset_y: float = 0.0) -> list[list[float]]:
    return [
        [offset_x + index * 0.01, offset_y + index * 0.015, index * 0.005]
        for index in range(EXPECTED_LANDMARKS_PER_HAND)
    ]


class LandmarkUtilsTests(unittest.TestCase):
    """Проверяет нормализацию и подготовку признаков."""

    def test_normalized_landmarks_start_from_zero_wrist(self) -> None:
        hand = _make_hand()

        normalized_landmarks = normalize_landmarks(hand)
        feature_vector = flatten_landmarks(normalized_landmarks)

        self.assertEqual(normalized_landmarks[0], [0.0, 0.0, 0.0])
        self.assertEqual(len(feature_vector), EXPECTED_LANDMARKS_PER_HAND * 3)

    def test_normalization_is_translation_invariant(self) -> None:
        hand_a = _make_hand()
        hand_b = _make_hand(offset_x=0.25, offset_y=0.75)

        normalized_a = normalize_landmarks(hand_a)
        normalized_b = normalize_landmarks(hand_b)

        for point_a, point_b in zip(normalized_a, normalized_b, strict=True):
            for coordinate_a, coordinate_b in zip(point_a, point_b, strict=True):
                self.assertAlmostEqual(coordinate_a, coordinate_b, places=7)

    def test_extract_primary_hand_sample_requires_single_hand(self) -> None:
        result_payload = {
            "hands_detected": 2,
            "handedness": ["Left", "Right"],
            "landmarks": [_make_hand(), _make_hand(offset_x=0.2)],
        }

        sample = extract_primary_hand_sample(result_payload, require_single_hand=True)

        self.assertIsNone(sample)

    def test_build_landmark_sample_preserves_handedness(self) -> None:
        sample = build_landmark_sample(_make_hand(), "Right")

        self.assertEqual(sample.handedness, "Right")
        self.assertEqual(len(sample.feature_vector), EXPECTED_LANDMARKS_PER_HAND * 3)


if __name__ == "__main__":
    unittest.main()
