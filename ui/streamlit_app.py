"""Streamlit-интерфейс для этапа 01."""

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
from core.drawing_utils import draw_connections, draw_landmarks, draw_status_text
from core.hand_tracker import HandTracker, HandTrackerError, ModelNotFoundError
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
        "stats": {
            **DEFAULT_STATS,
            "model_status": _get_model_status(config),
        },
        "last_error": None,
        "last_frame_time": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_page(config: AppConfig) -> None:
    st.title(config.project.name)
    st.caption(
        "Этап 01: захват видео с камеры, детекция рук через MediaPipe Tasks и отрисовка landmarks."
    )

    _render_warnings(config)
    _render_controls(config)

    video_placeholder = st.empty()
    error_placeholder = st.empty()

    if st.session_state.get("last_error"):
        error_placeholder.error(st.session_state["last_error"])
    else:
        error_placeholder.empty()

    if st.session_state["run_camera"]:
        _process_single_frame(config, video_placeholder)
        time.sleep(1.0 / max(config.camera.fps, 1))
        _render_status_panel()
        _render_debug_panel(config)
        st.rerun()
    else:
        st.session_state["last_frame_time"] = None
        video_placeholder.info("Видеопоток появится здесь после запуска обработки камеры.")
        _render_status_panel()
        _render_debug_panel(config)


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


def _render_controls(config: AppConfig) -> None:
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


def _render_debug_panel(config: AppConfig) -> None:
    if not config.ui.show_debug:
        return

    with st.expander("Отладочная информация", expanded=False):
        st.json(
            {
                "config_path": str(CONFIG_PATH),
                "project_root": str(config.project_root),
                "model_path": str(config.mediapipe.model_path),
                "run_camera": st.session_state["run_camera"],
                "stats": st.session_state["stats"],
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
    except HandTrackerError as exc:
        LOGGER.error("Ошибка трекера рук: %s", exc)
        st.session_state["last_error"] = str(exc)
        st.session_state["stats"]["model_status"] = "Ошибка трекера"
        _stop_processing()
    except Exception as exc:  # pragma: no cover - защитная ветка
        LOGGER.exception("Непредвиденная ошибка в UI")
        st.session_state["last_error"] = (
            "Произошла непредвиденная ошибка при обработке кадра. "
            f"Подробности в логах: {exc}"
        )
        _stop_processing()


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


def _stop_processing() -> None:
    st.session_state["run_camera"] = False
    st.session_state["last_frame_time"] = None

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
