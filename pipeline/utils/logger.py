"""Loguru setup for the Phase 1 pipeline."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from loguru import logger

from utils.redact import collect_env_secrets, redact_secrets

_CONFIGURED = False
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}"
_LOG_ROTATION_BYTES = 20 * 1024 * 1024
_extra_secrets: tuple[str, ...] = ()


def _refresh_secret_values() -> None:
    global _extra_secrets
    _extra_secrets = collect_env_secrets()
    try:
        from config.settings import get_settings

        token = get_settings().HF_TOKEN.strip()
        if len(token) >= 8:
            _extra_secrets = tuple(dict.fromkeys((*_extra_secrets, token)))
    except Exception:
        pass


def _redact_log_line(message: str) -> str:
    return redact_secrets(str(message), extra_secrets=_extra_secrets)


def _logging_patcher(record: dict) -> None:
    record["message"] = redact_secrets(record["message"], extra_secrets=_extra_secrets)


def _stderr_sink(message: str) -> None:
    sys.stderr.write(_redact_log_line(message))


def _rotating_file_sink(message: str) -> None:
    path = PROJECT_ROOT / "logs" / "phase1.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= _LOG_ROTATION_BYTES:
        rotated = path.with_name(f"phase1.{int(time.time())}.log")
        path.replace(rotated)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_redact_log_line(message))


def _file_logging_enabled() -> bool:
    flag = os.environ.get("PIPELINE_NO_FILE_LOG", "").strip().lower()
    return flag not in {"1", "true", "yes", "on"}


def configure_logger(level: str = "INFO", *, force: bool = False) -> None:
    """Send logs to stderr and optionally to a rotating file under pipeline/logs."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    _CONFIGURED = False
    _refresh_secret_values()
    logger.remove()
    logger.configure(patcher=_logging_patcher)
    logger.add(
        _stderr_sink,
        level=level,
        format=_LOG_FORMAT,
        enqueue=False,
        backtrace=False,
        diagnose=False,
        colorize=False,
    )
    if _file_logging_enabled():
        logger.add(
            _rotating_file_sink,
            level="DEBUG",
            format=_LOG_FORMAT,
            enqueue=False,
            backtrace=False,
            diagnose=False,
            colorize=False,
        )
    _CONFIGURED = True


configure_logger()
