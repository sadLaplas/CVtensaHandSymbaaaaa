"""Загрузка и валидация конфигурации проекта."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Ошибка загрузки или валидации конфигурации."""


@dataclass(slots=True, frozen=True)
class ProjectConfig:
    """Базовая информация о проекте."""

    name: str


@dataclass(slots=True, frozen=True)
class CameraConfig:
    """Параметры подключения и чтения с камеры."""

    device_id: int
    width: int
    height: int
    fps: int


@dataclass(slots=True, frozen=True)
class HandTrackerConfig:
    """Параметры трекинга рук через MediaPipe Tasks."""

    model_path: Path
    num_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float


@dataclass(slots=True, frozen=True)
class UIConfig:
    """Параметры пользовательского интерфейса."""

    window_title: str
    show_debug: bool


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """Параметры записи датасета landmarks."""

    raw_dir: Path
    index_file: Path
    gesture_labels: tuple[str, ...]
    capture_interval_ms: int
    max_samples_per_label: int
    require_single_hand: bool


@dataclass(slots=True, frozen=True)
class AppConfig:
    """Полная конфигурация приложения."""

    project_root: Path
    project: ProjectConfig
    camera: CameraConfig
    mediapipe: HandTrackerConfig
    ui: UIConfig
    dataset: DatasetConfig


def get_default_config_path() -> Path:
    """Возвращает путь к конфигу по умолчанию."""

    return Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Загружает YAML-конфиг и валидирует обязательные поля."""

    path = Path(config_path) if config_path is not None else get_default_config_path()
    path = path.resolve()

    if not path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Не удалось разобрать YAML-конфиг: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Не удалось прочитать файл конфигурации: {path}") from exc

    if not isinstance(raw_config, dict):
        raise ConfigError(
            f"Некорректный формат конфигурации: ожидается YAML-объект в файле {path}"
        )

    project_root = path.parents[1]

    project_section = _require_mapping(raw_config, "project")
    camera_section = _require_mapping(raw_config, "camera")
    mediapipe_section = _require_mapping(raw_config, "mediapipe")
    ui_section = _require_mapping(raw_config, "ui")
    dataset_section = _optional_mapping(raw_config, "dataset")

    project_config = ProjectConfig(name=_require_non_empty_string(project_section, "name"))
    camera_config = CameraConfig(
        device_id=_require_int(camera_section, "device_id"),
        width=_require_positive_int(camera_section, "width"),
        height=_require_positive_int(camera_section, "height"),
        fps=_require_positive_int(camera_section, "fps"),
    )
    hand_tracker_config = HandTrackerConfig(
        model_path=_resolve_model_path(
            project_root, _require_non_empty_string(mediapipe_section, "model_path")
        ),
        num_hands=_require_positive_int(mediapipe_section, "num_hands"),
        min_hand_detection_confidence=_require_probability(
            mediapipe_section, "min_hand_detection_confidence"
        ),
        min_hand_presence_confidence=_require_probability(
            mediapipe_section, "min_hand_presence_confidence"
        ),
        min_tracking_confidence=_require_probability(
            mediapipe_section, "min_tracking_confidence"
        ),
    )
    ui_config = UIConfig(
        window_title=_require_non_empty_string(ui_section, "window_title"),
        show_debug=_require_bool(ui_section, "show_debug"),
    )
    dataset_config = _build_dataset_config(project_root, dataset_section)

    return AppConfig(
        project_root=project_root,
        project=project_config,
        camera=camera_config,
        mediapipe=hand_tracker_config,
        ui=ui_config,
        dataset=dataset_config,
    )


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Раздел '{key}' отсутствует или поврежден.")
    return value


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"Раздел '{key}' должен быть YAML-объектом.")
    return value


def _require_non_empty_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Поле '{key}' обязательно и должно быть непустой строкой.")
    return value.strip()


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Поле '{key}' должно быть целым числом.")
    return value


def _require_positive_int(data: dict[str, Any], key: str) -> int:
    value = _require_int(data, key)
    if value <= 0:
        raise ConfigError(f"Поле '{key}' должно быть больше нуля.")
    return value


def _require_probability(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Поле '{key}' должно быть числом в диапазоне [0, 1].")

    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ConfigError(f"Поле '{key}' должно быть в диапазоне [0, 1].")

    return probability


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Поле '{key}' должно иметь значение true или false.")
    return value


def _require_string_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"Поле '{key}' должно быть непустым списком строк.")

    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"Поле '{key}' должно содержать только непустые строки."
            )
        normalized_values.append(item.strip())

    return tuple(normalized_values)


def _resolve_model_path(project_root: Path, model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _build_dataset_config(
    project_root: Path, dataset_section: dict[str, Any] | None
) -> DatasetConfig:
    # Keep stage 01 configs valid by providing dataset defaults.
    # Сохраняем совместимость с конфигами этапа 01 через значения по умолчанию.
    default_labels = ("open_palm", "fist", "peace", "thumbs_up")

    if dataset_section is None:
        return DatasetConfig(
            raw_dir=(project_root / "data" / "raw").resolve(),
            index_file=(project_root / "data" / "raw" / "dataset_index.csv").resolve(),
            gesture_labels=default_labels,
            capture_interval_ms=250,
            max_samples_per_label=250,
            require_single_hand=True,
        )

    return DatasetConfig(
        raw_dir=_resolve_path(
            project_root, _require_non_empty_string(dataset_section, "raw_dir")
        ),
        index_file=_resolve_path(
            project_root, _require_non_empty_string(dataset_section, "index_file")
        ),
        gesture_labels=_require_string_list(dataset_section, "gesture_labels"),
        capture_interval_ms=_require_positive_int(dataset_section, "capture_interval_ms"),
        max_samples_per_label=_require_positive_int(
            dataset_section, "max_samples_per_label"
        ),
        require_single_hand=_require_bool(dataset_section, "require_single_hand"),
    )
