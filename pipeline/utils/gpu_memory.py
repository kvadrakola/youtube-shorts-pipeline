"""VRAM cleanup for sequential model stages on a 24 GB GPU."""

from __future__ import annotations

import gc
from typing import Any

from utils.logger import logger

_DEVICE_LOGGED = False


def clear_gpu_cache(*, log: bool = True) -> None:
    """Collect garbage and release unused CUDA caching allocator blocks.

    Logs allocated and reserved VRAM before and after the call when a GPU is present.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        if log:
            logger.info("GPU cache cleanup skipped: torch is not installed")
        return

    if not torch.cuda.is_available():
        if log:
            logger.info("GPU cache cleanup skipped: CUDA is not available")
        return

    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    if log:
        logger.info(
            "VRAM before cleanup: allocated={:.2f} GiB reserved={:.2f} GiB",
            before_allocated / 1024**3,
            before_reserved / 1024**3,
        )
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()
    if log:
        logger.info(
            "VRAM after cleanup: allocated={:.2f} GiB reserved={:.2f} GiB",
            after_allocated / 1024**3,
            after_reserved / 1024**3,
        )


def resolve_device(requested: str | None = None) -> str:
    """Return ``cuda`` or ``cpu``. ``cuda`` requires a visible GPU."""
    global _DEVICE_LOGGED
    from config.settings import get_settings

    device = (requested or get_settings().DEVICE).strip().lower()
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("DEVICE must be 'cuda' or 'cpu'")
    if device == "cpu":
        if not _DEVICE_LOGGED:
            logger.info("Running on CPU")
            _DEVICE_LOGGED = True
        return device

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE is cuda but torch.cuda.is_available() is False. "
            "Install the CUDA build of PyTorch. See README.md."
        )
    if not _DEVICE_LOGGED:
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "Using CUDA device {} with {:.1f} GiB VRAM",
            props.name,
            props.total_memory / 1024**3,
        )
        _DEVICE_LOGGED = True
    return device


def release_model(model: Any) -> None:
    """Drop a model reference and clear the CUDA cache."""
    del model
    clear_gpu_cache()
