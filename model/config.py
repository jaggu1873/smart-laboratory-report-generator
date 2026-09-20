"""Model-related configuration.

Configuration is read from environment variables so that no secrets or
API keys are ever hardcoded in the source tree. API keys, when a real
backend is added later, must be read from the environment at call time
via :meth:`ModelConfig.get_api_key` and never stored in defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class ModelConfig:
    """Configuration for the report-generation model layer.

    Attributes:
        backend: Name of the inference backend to build (e.g. ``"mock"``).
        model_id: Identifier of the underlying model (backend-specific).
        temperature: Sampling temperature for generation.
        max_tokens: Maximum number of tokens the backend may produce.
        timeout_seconds: Per-request timeout for a real backend.
        api_key_env: Name of the environment variable that would hold the
            API key for a real backend. The key value itself is never stored
            on the config object.
    """

    backend: str = field(default_factory=lambda: _env_str("LAB_MODEL_BACKEND", "mock"))
    model_id: str = field(default_factory=lambda: _env_str("LAB_MODEL_ID", "mock-lab-v1"))
    temperature: float = field(default_factory=lambda: _env_float("LAB_MODEL_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _env_int("LAB_MODEL_MAX_TOKENS", 1024))
    timeout_seconds: int = field(default_factory=lambda: _env_int("LAB_MODEL_TIMEOUT", 30))
    api_key_env: str = field(default_factory=lambda: _env_str("LAB_MODEL_API_KEY_ENV", "LAB_MODEL_API_KEY"))

    def get_api_key(self) -> str | None:
        """Return the API key from the environment, or ``None`` if unset.

        The key is read fresh from the environment on every call and is never
        cached on the instance, so secrets do not live in memory longer than
        needed nor get serialized with the config.
        """
        return os.environ.get(self.api_key_env)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Build a config populated entirely from environment variables."""
        return cls()
