"""Тесты для модуля конфигурации."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from core.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    """Проверяет загрузку и валидацию YAML-конфига."""

    def test_load_config_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    project:
                      name: CVtensaHandSymbaaaaa

                    camera:
                      device_id: 0
                      width: 640
                      height: 480
                      fps: 30

                    mediapipe:
                      model_path: models/hand_landmarker.task
                      num_hands: 2
                      min_hand_detection_confidence: 0.7
                      min_hand_presence_confidence: 0.5
                      min_tracking_confidence: 0.5

                    ui:
                      window_title: CVtensaHandSymbaaaaa
                      show_debug: true
                    """
                ).strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.project.name, "CVtensaHandSymbaaaaa")
            self.assertEqual(config.camera.width, 640)
            self.assertEqual(
                config.mediapipe.model_path,
                (root / "models" / "hand_landmarker.task").resolve(),
            )

    def test_missing_config_raises_error(self) -> None:
        missing_path = Path("/tmp/non_existent_config.yaml")

        with self.assertRaises(ConfigError):
            load_config(missing_path)

    def test_invalid_probability_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    project:
                      name: Test

                    camera:
                      device_id: 0
                      width: 640
                      height: 480
                      fps: 30

                    mediapipe:
                      model_path: models/hand_landmarker.task
                      num_hands: 2
                      min_hand_detection_confidence: 1.2
                      min_hand_presence_confidence: 0.5
                      min_tracking_confidence: 0.5

                    ui:
                      window_title: Test
                      show_debug: true
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
