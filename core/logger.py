"""Единая настройка логирования проекта."""

from __future__ import annotations

import logging
from functools import lru_cache


LOGGER_NAME = "CVtensaHandSymbaaaaa"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@lru_cache(maxsize=1)
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Создает и настраивает корневой логгер проекта."""

    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Возвращает корневой логгер проекта или дочерний логгер модуля."""

    root_logger = setup_logging()
    if not name:
        return root_logger
    return root_logger.getChild(name)
