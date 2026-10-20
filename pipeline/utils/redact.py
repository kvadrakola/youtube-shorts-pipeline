"""Redact secrets from log messages and tracebacks."""

from __future__ import annotations

import os
import re

_REDACTED = "***REDACTED***"
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")
_BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+", re.IGNORECASE)
_BASIC_AUTH_PATTERN = re.compile(r"Basic\s+[A-Za-z0-9+/=]{8,}", re.IGNORECASE)
_AUTH_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(Authorization|Proxy-Authorization)\s*[=:]\s*['\"]?[^\s'\"\]]+"
)

_SECRET_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "DNS_AUTH_KEY",
    "COMET_API_KEY",
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
)


def collect_env_secrets() -> tuple[str, ...]:
    """Return non-empty secret values from well-known environment variables."""
    values: list[str] = []
    for key in _SECRET_ENV_KEYS:
        raw = os.environ.get(key, "").strip()
        if len(raw) >= 8:
            values.append(raw)
    return tuple(dict.fromkeys(values))


def _mask_hf_token(match: re.Match[str]) -> str:
    token = match.group(0)
    if len(token) <= 11:
        return _REDACTED
    return f"hf_...{token[-4:]}"


def redact_secrets(text: str, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Strip or mask tokens, API keys, and auth headers from arbitrary text."""
    if not text:
        return text
    secrets = extra_secrets or collect_env_secrets()
    for secret in secrets:
        if secret:
            text = text.replace(secret, _REDACTED)
    text = _HF_TOKEN_PATTERN.sub(_mask_hf_token, text)
    text = _BEARER_PATTERN.sub("Bearer ***REDACTED***", text)
    text = _BASIC_AUTH_PATTERN.sub("Basic ***REDACTED***", text)
    text = _AUTH_ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}={_REDACTED}", text)
    return text
