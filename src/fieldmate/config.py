"""Runtime configuration, read from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All knobs in one place. Override with FIELDMATE_* / OLLAMA_HOST env vars."""

    ollama_host: str = field(
        default_factory=lambda: (
            os.environ.get("OLLAMA_HOST_URL")
            or _normalise_host(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        )
    )
    model: str = field(default_factory=lambda: os.environ.get("FIELDMATE_MODEL", "llama3.2:3b"))
    temperature: float = field(default_factory=lambda: _env_float("FIELDMATE_TEMPERATURE", 0.0))
    request_timeout_s: float = field(
        default_factory=lambda: _env_float("FIELDMATE_TIMEOUT_S", 300.0)
    )
    num_ctx: int = field(default_factory=lambda: int(os.environ.get("FIELDMATE_NUM_CTX", "2048")))
    # Low-RAM / shared-host mode: memory-map GGUF weights (page cache, reclaimable) instead of
    # copying them into process memory. See docs/REPORT.md "What went wrong".
    use_mmap: bool = field(
        default_factory=lambda: os.environ.get("FIELDMATE_USE_MMAP", "0").lower() in ("1", "true")
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("FIELDMATE_MAX_RETRIES", "1"))
    )


def _normalise_host(host: str) -> str:
    """OLLAMA_HOST may be '0.0.0.0:11434' or 'http://ollama:11434'; return a URL."""
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.replace("0.0.0.0", "127.0.0.1").rstrip("/")


def get_settings() -> Settings:
    return Settings()
