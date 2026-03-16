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
class AppConfig:
    """Полная конфигурация приложения."""

    project_root: Path
    project: ProjectConfig
    camera: CameraConfig
    mediapipe: HandTrackerConfig
    ui: UIConfig


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

    return AppConfig(
        project_root=project_root,
        project=project_config,
        camera=camera_config,
        mediapipe=hand_tracker_config,
        ui=ui_config,
    )


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Раздел '{key}' отсутствует или поврежден.")
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


def _resolve_model_path(project_root: Path, model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
