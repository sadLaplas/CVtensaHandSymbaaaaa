"""Минимальная точка входа для запуска Streamlit-приложения."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Запускает Streamlit UI через текущий Python-интерпретатор."""

    project_root = Path(__file__).resolve().parents[1]
    streamlit_app = project_root / "ui" / "streamlit_app.py"

    completed_process = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(streamlit_app)],
        cwd=project_root,
        check=False,
    )
    return completed_process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
