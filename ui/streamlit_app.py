"""Streamlit-интерфейс для этапов 01-02."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.camera import CameraError, CameraService
from core.config import AppConfig, ConfigError, load_config
from core.dataset_collector import DatasetCollector, DatasetCollectorError
from core.drawing_utils import draw_connections, draw_landmarks, draw_status_text
from core.hand_tracker import HandTracker, HandTrackerError, ModelNotFoundError
from core.landmark_utils import extract_primary_hand_sample
from core.logger import get_logger, setup_logging

try:
    import cv2
except Exception as exc:  # pragma: no cover - зависит от окружения исполнения
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
LOGGER = get_logger("ui")
DEFAULT_STATS: dict[str, Any] = {
    "hands_detected": 0,
    "fps": 0.0,
    "handedness": [],
    "landmark_points": 0,
    "model_status": "Не проверена",
    "camera_status": "Остановлена",
}
DEFAULT_CAPTURE_STATE: dict[str, Any] = {
    "enabled": False,
    "session_saved": 0,
    "label_saved_total": 0,
    "last_saved_path": "",
    "last_capture_status": "Запись выключена",
    "last_capture_timestamp_ms": 0,
}


def main() -> None:
    """Запускает Streamlit-приложение."""

    setup_logging()
    st.set_page_config(page_title="CVtensaHandSymbaaaaa", layout="wide")

    config = _load_app_config()
    if config is None:
        return

    _initialize_session_state(config)
    _render_page(config)


def _load_app_config() -> AppConfig | None:
    try:
        return load_config(CONFIG_PATH)
    except ConfigError as exc:
        LOGGER.error("Ошибка конфигурации: %s", exc)
        st.title("CVtensaHandSymbaaaaa")
        st.error(f"Не удалось загрузить конфигурацию проекта: {exc}")
        return None


def _initialize_session_state(config: AppConfig) -> None:
    defaults = {
        "run_camera": False,
        "camera_service": None,
        "hand_tracker": None,
        "dataset_collector": None,
        "stats": {
            **DEFAULT_STATS,
            "model_status": _get_model_status(config),
        },
        "capture_state": {
            **DEFAULT_CAPTURE_STATE,
            "label_saved_total": 0,
        },
        "selected_gesture_label": config.dataset.gesture_labels[0],
        "last_error": None,
        "last_frame_time": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state["selected_gesture_label"] not in config.dataset.gesture_labels:
        st.session_state["selected_gesture_label"] = config.dataset.gesture_labels[0]


def _render_page(config: AppConfig) -> None:
    st.title(config.project.name)
    st.caption(
        "Этап 02: захват видео, детекция рук и запись датасета landmarks для будущего обучения."
    )

    _render_warnings(config)
    _render_camera_controls(config)
    _render_dataset_controls(config)

    error_placeholder = st.empty()
    video_placeholder = st.empty()

    if st.session_state["run_camera"]:
        _process_single_frame(config, video_placeholder)
        time.sleep(1.0 / max(config.camera.fps, 1))
    else:
        st.session_state["last_frame_time"] = None
        video_placeholder.info("Видеопоток появится здесь после запуска обработки камеры.")

    if st.session_state.get("last_error"):
        error_placeholder.error(st.session_state["last_error"])
    else:
        error_placeholder.empty()

    _render_status_panel()
    _render_capture_panel(config)
    _render_debug_panel(config)

    if st.session_state["run_camera"]:
        st.rerun()


def _render_warnings(config: AppConfig) -> None:
    if _CV2_IMPORT_ERROR is not None or cv2 is None:
        st.error(
            "OpenCV недоступен. Установите зависимости проекта через requirements.txt."
        )

    if not config.mediapipe.model_path.exists():
        st.warning(
            "Файл модели MediaPipe Tasks не найден. "
            f"Ожидаемый путь: {config.mediapipe.model_path}"
        )


def _render_camera_controls(config: AppConfig) -> None:
    st.subheader("Поток камеры")
    start_col, stop_col = st.columns(2)
    can_start = (
        not st.session_state["run_camera"]
        and config.mediapipe.model_path.exists()
        and _CV2_IMPORT_ERROR is None
    )

    if start_col.button(
        "Запустить обработку",
        use_container_width=True,
        disabled=not can_start,
    ):
        st.session_state["run_camera"] = True
        st.session_state["last_error"] = None
        st.session_state["last_frame_time"] = None

    if stop_col.button(
        "Остановить",
        use_container_width=True,
        disabled=not st.session_state["run_camera"],
    ):
        _stop_processing()


def _render_dataset_controls(config: AppConfig) -> None:
    st.subheader("Сбор датасета")
    st.caption(
        "Выберите жест из конфига и включите запись. "
        "Для качества датасета рекомендуется держать в кадре только одну руку."
    )

    label_col, interval_col, target_col = st.columns(3)
    label_col.selectbox(
        "Gesture label",
        options=list(config.dataset.gesture_labels),
        key="selected_gesture_label",
    )
    interval_col.metric(
        "Интервал записи",
        f"{config.dataset.capture_interval_ms} мс",
    )
    target_col.metric(
        "Лимит на класс",
        config.dataset.max_samples_per_label,
    )

    start_col, stop_col = st.columns(2)
    can_start_capture = st.session_state["run_camera"] and not st.session_state[
        "capture_state"
    ]["enabled"]
    can_stop_capture = st.session_state["capture_state"]["enabled"]

    if start_col.button(
        "Включить запись",
        use_container_width=True,
        disabled=not can_start_capture,
    ):
        _start_capture_session(config)

    if stop_col.button(
        "Остановить запись",
        use_container_width=True,
        disabled=not can_stop_capture,
    ):
        _stop_capture_session("Запись датасета остановлена пользователем.")


def _render_status_panel() -> None:
    stats = st.session_state["stats"]

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Hands detected", stats["hands_detected"])
    metric_col_2.metric("FPS", f"{stats['fps']:.1f}")
    metric_col_3.metric("Количество landmark-точек", stats["landmark_points"])

    info_col_1, info_col_2, info_col_3 = st.columns(3)
    handedness = ", ".join(stats["handedness"]) if stats["handedness"] else "Не определено"
    info_col_1.write(f"**Handedness:** {handedness}")
    info_col_2.write(f"**Статус модели:** {stats['model_status']}")
    info_col_3.write(f"**Статус камеры:** {stats['camera_status']}")


def _render_capture_panel(config: AppConfig) -> None:
    capture_state = st.session_state["capture_state"]
    selected_label = st.session_state["selected_gesture_label"]

    total_samples = 0
    try:
        collector = _get_dataset_collector(config)
        total_samples = collector.get_total_samples()
        capture_state["label_saved_total"] = collector.get_label_count(selected_label)
    except DatasetCollectorError as exc:
        LOGGER.error("Ошибка подготовки датасета: %s", exc)
        st.session_state["last_error"] = str(exc)

    st.subheader("Статус записи")
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Активный label", selected_label)
    metric_col_2.metric("Сохранено за сессию", capture_state["session_saved"])
    metric_col_3.metric("Сохранено для label", capture_state["label_saved_total"])
    metric_col_4.metric("Всего samples", total_samples)

    info_col_1, info_col_2 = st.columns(2)
    recording_status = "Включена" if capture_state["enabled"] else "Выключена"
    info_col_1.write(f"**Запись:** {recording_status}")
    info_col_1.write(f"**Последний статус:** {capture_state['last_capture_status']}")
    info_col_2.write(f"**Каталог данных:** {config.dataset.raw_dir}")
    info_col_2.write(f"**Индексный файл:** {config.dataset.index_file}")

    if capture_state["last_saved_path"]:
        st.caption(f"Последний сохраненный sample: {capture_state['last_saved_path']}")


def _render_debug_panel(config: AppConfig) -> None:
    if not config.ui.show_debug:
        return

    with st.expander("Отладочная информация", expanded=False):
        st.json(
            {
                "config_path": str(CONFIG_PATH),
                "project_root": str(config.project_root),
                "model_path": str(config.mediapipe.model_path),
                "dataset_raw_dir": str(config.dataset.raw_dir),
                "dataset_index_file": str(config.dataset.index_file),
                "run_camera": st.session_state["run_camera"],
                "stats": st.session_state["stats"],
                "capture_state": st.session_state["capture_state"],
            }
        )


def _process_single_frame(config: AppConfig, video_placeholder: Any) -> None:
    try:
        camera_service = _get_camera_service(config)
        hand_tracker = _get_hand_tracker(config)

        frame = camera_service.read_frame()
        if frame is None:
            st.session_state["last_error"] = "Не удалось получить кадр с камеры. Попытка будет повторена."
            st.session_state["stats"]["camera_status"] = "Камера подключена, кадр не получен"
            video_placeholder.warning("Кадр с камеры не был считан. Приложение продолжает работу.")
            return

        result_payload = hand_tracker.process_frame(frame)
        capture_overlay_status = _maybe_capture_sample(config, result_payload)

        annotated_frame = frame.copy()
        draw_connections(annotated_frame, result_payload["landmarks"])
        draw_landmarks(annotated_frame, result_payload["landmarks"])

        fps_value = _update_fps()
        handedness = result_payload["handedness"]
        landmark_points = sum(len(hand) for hand in result_payload["landmarks"])
        status = {
            "Hands": result_payload["hands_detected"],
            "FPS": f"{fps_value:.1f}",
            "Handedness": ", ".join(handedness) if handedness else "-",
            "Points": landmark_points,
            "Dataset": capture_overlay_status,
        }
        draw_status_text(annotated_frame, status)

        rgb_frame = _convert_bgr_to_rgb(annotated_frame)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        st.session_state["stats"] = {
            "hands_detected": result_payload["hands_detected"],
            "fps": fps_value,
            "handedness": handedness,
            "landmark_points": landmark_points,
            "model_status": _get_model_status(config),
            "camera_status": "Активна",
        }
        st.session_state["last_error"] = None
    except ModelNotFoundError as exc:
        LOGGER.error("Модель не найдена: %s", exc)
        st.session_state["last_error"] = str(exc)
        st.session_state["stats"]["model_status"] = "Модель не найдена"
        _stop_processing()
    except CameraError as exc:
        LOGGER.error("Ошибка камеры: %s", exc)
        st.session_state["last_error"] = str(exc)
        st.session_state["stats"]["camera_status"] = "Ошибка камеры"
        _stop_processing()
    except (HandTrackerError, DatasetCollectorError, ValueError) as exc:
        LOGGER.error("Ошибка обработки кадра: %s", exc)
        st.session_state["last_error"] = str(exc)
        _stop_capture_session("Запись датасета остановлена из-за ошибки.")
        _stop_processing()
    except Exception as exc:  # pragma: no cover - защитная ветка
        LOGGER.exception("Непредвиденная ошибка в UI")
        st.session_state["last_error"] = (
            "Произошла непредвиденная ошибка при обработке кадра. "
            f"Подробности в логах: {exc}"
        )
        _stop_capture_session("Запись датасета остановлена из-за непредвиденной ошибки.")
        _stop_processing()


def _maybe_capture_sample(config: AppConfig, result_payload: dict[str, Any]) -> str:
    capture_state = st.session_state["capture_state"]
    if not capture_state["enabled"]:
        capture_state["last_capture_status"] = "Запись выключена"
        return "IDLE"

    selected_label = st.session_state["selected_gesture_label"]
    collector = _get_dataset_collector(config)
    capture_state["label_saved_total"] = collector.get_label_count(selected_label)

    sample = extract_primary_hand_sample(
        result_payload, require_single_hand=config.dataset.require_single_hand
    )
    if sample is None:
        if result_payload.get("hands_detected", 0) == 0:
            capture_state["last_capture_status"] = "Ожидание руки в кадре."
        else:
            capture_state["last_capture_status"] = "Ожидание одной уверенно детектированной руки."
        return "WAIT"

    now_ms = time.monotonic_ns() // 1_000_000
    elapsed_ms = now_ms - int(capture_state["last_capture_timestamp_ms"])
    if elapsed_ms < config.dataset.capture_interval_ms:
        remaining_ms = config.dataset.capture_interval_ms - elapsed_ms
        capture_state["last_capture_status"] = f"Пауза между samples: {remaining_ms} мс."
        return "SYNC"

    current_label_count = collector.get_label_count(selected_label)
    if current_label_count >= config.dataset.max_samples_per_label:
        _stop_capture_session("Достигнут лимит samples для текущего label.")
        return "LIMIT"

    # Save only preprocessed landmarks here to avoid repeating normalization later.
    # Сохраняем уже подготовленные landmarks, чтобы не повторять нормализацию на следующих этапах.
    saved_sample = collector.save_sample(selected_label, sample)
    capture_state["session_saved"] += 1
    capture_state["label_saved_total"] = saved_sample.total_samples_for_label
    capture_state["last_saved_path"] = str(saved_sample.file_path)
    capture_state["last_capture_timestamp_ms"] = now_ms
    capture_state["last_capture_status"] = f"Сохранен sample: {saved_sample.sample_id}"
    return "REC"


def _start_capture_session(config: AppConfig) -> None:
    try:
        collector = _get_dataset_collector(config)
        selected_label = st.session_state["selected_gesture_label"]
        st.session_state["capture_state"] = {
            **DEFAULT_CAPTURE_STATE,
            "enabled": True,
            "label_saved_total": collector.get_label_count(selected_label),
            "last_capture_status": "Запись включена. Ожидание подходящего кадра.",
        }
    except DatasetCollectorError as exc:
        LOGGER.error("Не удалось включить запись датасета: %s", exc)
        st.session_state["last_error"] = str(exc)


def _stop_capture_session(message: str) -> None:
    capture_state = st.session_state["capture_state"]
    capture_state["enabled"] = False
    capture_state["last_capture_status"] = message


def _get_camera_service(config: AppConfig) -> CameraService:
    camera_service = st.session_state.get("camera_service")
    if camera_service is None:
        camera_service = CameraService(config.camera)
        camera_service.initialize_camera()
        st.session_state["camera_service"] = camera_service
    return camera_service


def _get_hand_tracker(config: AppConfig) -> HandTracker:
    hand_tracker = st.session_state.get("hand_tracker")
    if hand_tracker is None:
        hand_tracker = HandTracker(config.mediapipe)
        hand_tracker.initialize()
        st.session_state["hand_tracker"] = hand_tracker
    return hand_tracker


def _get_dataset_collector(config: AppConfig) -> DatasetCollector:
    dataset_collector = st.session_state.get("dataset_collector")
    if dataset_collector is None:
        dataset_collector = DatasetCollector(config.dataset)
        dataset_collector.ensure_storage()
        st.session_state["dataset_collector"] = dataset_collector
    return dataset_collector


def _stop_processing() -> None:
    st.session_state["run_camera"] = False
    st.session_state["last_frame_time"] = None
    _stop_capture_session("Запись датасета остановлена.")

    camera_service = st.session_state.get("camera_service")
    if camera_service is not None:
        camera_service.release_camera()
    st.session_state["camera_service"] = None

    hand_tracker = st.session_state.get("hand_tracker")
    if hand_tracker is not None:
        hand_tracker.close()
    st.session_state["hand_tracker"] = None

    st.session_state["stats"]["camera_status"] = "Остановлена"


def _update_fps() -> float:
    current_time = time.perf_counter()
    previous_time = st.session_state.get("last_frame_time")
    st.session_state["last_frame_time"] = current_time

    if previous_time is None:
        return 0.0

    delta = current_time - previous_time
    if delta <= 0:
        return 0.0

    return 1.0 / delta


def _convert_bgr_to_rgb(frame: Any) -> Any:
    if _CV2_IMPORT_ERROR is not None or cv2 is None:
        raise RuntimeError(
            "OpenCV недоступен. Установите зависимости проекта из requirements.txt."
        ) from _CV2_IMPORT_ERROR
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _get_model_status(config: AppConfig) -> str:
    if config.mediapipe.model_path.exists():
        return "Модель доступна"
    return f"Файл отсутствует: {config.mediapipe.model_path}"


if __name__ == "__main__":
    main()
