"""Тесты для сервиса записи датасета."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.config import DatasetConfig
from core.dataset_collector import DatasetCollector, DatasetCollectorError
from core.landmark_utils import EXPECTED_LANDMARKS_PER_HAND, build_landmark_sample


def _make_hand() -> list[list[float]]:
    return [
        [index * 0.01, index * 0.015, index * 0.005]
        for index in range(EXPECTED_LANDMARKS_PER_HAND)
    ]


class DatasetCollectorTests(unittest.TestCase):
    """Проверяет сохранение raw samples и индексного файла."""

    def test_save_sample_creates_json_and_updates_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = DatasetConfig(
                raw_dir=root / "data" / "raw",
                index_file=root / "data" / "raw" / "dataset_index.csv",
                gesture_labels=("open_palm", "fist"),
                capture_interval_ms=250,
                max_samples_per_label=250,
                require_single_hand=True,
            )
            collector = DatasetCollector(config)
            sample = build_landmark_sample(_make_hand(), "Right")

            saved_sample = collector.save_sample("open_palm", sample)

            self.assertTrue(saved_sample.file_path.exists())
            self.assertEqual(collector.get_label_count("open_palm"), 1)
            self.assertEqual(collector.get_total_samples(), 1)
            self.assertTrue(config.index_file.exists())

            payload = json.loads(saved_sample.file_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["gesture_label"], "open_palm")
            self.assertEqual(payload["feature_vector_length"], 63)

    def test_unknown_label_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = DatasetConfig(
                raw_dir=root / "data" / "raw",
                index_file=root / "data" / "raw" / "dataset_index.csv",
                gesture_labels=("open_palm",),
                capture_interval_ms=250,
                max_samples_per_label=250,
                require_single_hand=True,
            )
            collector = DatasetCollector(config)
            sample = build_landmark_sample(_make_hand(), "Left")

            with self.assertRaises(DatasetCollectorError):
                collector.save_sample("unknown_label", sample)


if __name__ == "__main__":
    unittest.main()
