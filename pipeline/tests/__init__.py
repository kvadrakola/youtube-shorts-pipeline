"""Disable pipeline file logging during unit tests."""

from __future__ import annotations

import os

os.environ.setdefault("PIPELINE_NO_FILE_LOG", "1")

from utils import logger as _logger_module

_logger_module.configure_logger(force=True)
