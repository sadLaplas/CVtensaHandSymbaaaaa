"""Сервис записи landmarks в сырой датасет."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from core.config import DatasetConfig
from core.landmark_utils import LandmarkSample
from core.logger import get_logger


_INDEX_FIELDS = (
    "sample_id",
    "gesture_label",
    "handedness",
    "captured_at",
    "sample_path",
)


class DatasetCollectorError(RuntimeError):
    """Ошибка подготовки или записи датасета."""


@dataclass(slots=True, frozen=True)
class SavedSampleInfo:
    """Метаданные сохраненного sample."""

    sample_id: str
    gesture_label: str
    handedness: str
    file_path: Path
    captured_at: str
    total_samples_for_label: int


class DatasetCollector:
    """Управляет сохранением сырых samples landmarks и индексного CSV."""

    def __init__(self, config: DatasetConfig) -> None:
        self._config = config
        self._logger = get_logger("dataset_collector")
        self._lock = Lock()
        self._allowed_labels = {
            self.sanitize_label(label) for label in self._config.gesture_labels
        }
        self._label_counts = self._scan_existing_counts()

    @property
    def raw_dir(self) -> Path:
        """Возвращает путь к каталогу сырых данных."""

        return self._config.raw_dir

    @property
    def index_file(self) -> Path:
        """Возвращает путь к индексному CSV."""

        return self._config.index_file

    def ensure_storage(self) -> None:
        """Создает каталоги и индексный файл датасета при необходимости."""

        try:
            self._config.raw_dir.mkdir(parents=True, exist_ok=True)
            self._config.index_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatasetCollectorError(
                "Не удалось подготовить каталоги для датасета."
            ) from exc

        if self._config.index_file.exists():
            return

        # Create the CSV header once to keep incremental appends simple and stable.
        # Создаем заголовок CSV один раз, чтобы последующие добавления были простыми и стабильными.
        try:
            with self._config.index_file.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(_INDEX_FIELDS),
                )
                writer.writeheader()
        except OSError as exc:
            raise DatasetCollectorError(
                f"Не удалось создать индексный файл датасета: {self._config.index_file}"
            ) from exc

    def save_sample(self, gesture_label: str, sample: LandmarkSample) -> SavedSampleInfo:
        """Сохраняет sample в JSON и добавляет запись в индексный CSV."""

        normalized_label = self._validate_label(gesture_label)
        self.ensure_storage()

        captured_at = datetime.now(timezone.utc).isoformat()
        sample_id = self._build_sample_id(normalized_label)
        label_dir = self._config.raw_dir / normalized_label
        file_path = label_dir / f"{sample_id}.json"

        payload = {
            "sample_id": sample_id,
            "gesture_label": normalized_label,
            "captured_at": captured_at,
            "handedness": sample.handedness,
            "feature_vector_length": len(sample.feature_vector),
            "sample": asdict(sample),
        }

        try:
            with self._lock:
                label_dir.mkdir(parents=True, exist_ok=True)
                file_path.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                self._append_to_index(
                    sample_id=sample_id,
                    gesture_label=normalized_label,
                    handedness=sample.handedness,
                    captured_at=captured_at,
                    sample_path=file_path,
                )
                self._label_counts[normalized_label] = (
                    self._label_counts.get(normalized_label, 0) + 1
                )
        except OSError as exc:
            raise DatasetCollectorError(
                f"Не удалось сохранить sample в датасет: {file_path}"
            ) from exc

        self._logger.info(
            "Sample сохранен: label=%s, sample_id=%s, path=%s",
            normalized_label,
            sample_id,
            file_path,
        )

        return SavedSampleInfo(
            sample_id=sample_id,
            gesture_label=normalized_label,
            handedness=sample.handedness,
            file_path=file_path,
            captured_at=captured_at,
            total_samples_for_label=self._label_counts[normalized_label],
        )

    def get_label_count(self, gesture_label: str) -> int:
        """Возвращает количество сохраненных samples для заданного label."""

        normalized_label = self._validate_label(gesture_label)
        return self._label_counts.get(normalized_label, 0)

    def get_total_samples(self) -> int:
        """Возвращает общее количество samples во всех классах."""

        return sum(self._label_counts.values())

    def _append_to_index(
        self,
        sample_id: str,
        gesture_label: str,
        handedness: str,
        captured_at: str,
        sample_path: Path,
    ) -> None:
        with self._config.index_file.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(_INDEX_FIELDS),
            )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "gesture_label": gesture_label,
                    "handedness": handedness,
                    "captured_at": captured_at,
                    "sample_path": sample_path.as_posix(),
                }
            )

    def _scan_existing_counts(self) -> dict[str, int]:
        counts = {self.sanitize_label(label): 0 for label in self._config.gesture_labels}

        if not self._config.raw_dir.exists():
            return counts

        for label_directory in self._config.raw_dir.iterdir():
            if not label_directory.is_dir():
                continue
            counts[label_directory.name] = len(list(label_directory.glob("*.json")))

        return counts

    def _validate_label(self, gesture_label: str) -> str:
        normalized_label = self.sanitize_label(gesture_label)
        if not normalized_label:
            raise DatasetCollectorError("Label жеста не может быть пустым.")

        if normalized_label not in self._allowed_labels:
            raise DatasetCollectorError(
                "Неизвестный label жеста. Добавьте его в configs/config.yaml."
            )

        return normalized_label

    @staticmethod
    def sanitize_label(gesture_label: str) -> str:
        """Нормализует label для безопасного имени каталога."""

        cleaned_label = re.sub(r"[^a-zA-Z0-9_-]+", "_", gesture_label.strip().lower())
        return cleaned_label.strip("_")

    @staticmethod
    def _build_sample_id(gesture_label: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        return f"{gesture_label}_{timestamp}"
