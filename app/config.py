"""Application configuration.

Module-level constants (consumed by the domain layer) and a Flask ``Config``
class, all read from environment variables with safe defaults. A local ``.env``
is loaded at import time, before any variable is read.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load a local .env into os.environ if present; no-op when absent.
load_dotenv()


def _get_str(name: str, default: str) -> str:
    """Read a string env var, treating empty/whitespace as unset."""

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip()


def _get_int(name: str, default: int) -> int:
    """Read an integer env var.

    Raises:
        RuntimeError: if the value is set but not a valid integer.
    """

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as conversion_error:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got {raw_value!r}."
        ) from conversion_error


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Truthy: 1/true/yes/on (case-insensitive)."""

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


# --- Model / Anthropic ----------------------------------------------------- #

# Model identifier used by every Claude-backed component. ANTHROPIC_API_KEY is
# read by the Anthropic SDK directly from the environment, not here.
MODEL_IDENTIFIER: str = _get_str("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")


# --- Pipeline tunables ----------------------------------------------------- #

# Per-stage maximum output tokens.
ROUTER_MAX_TOKENS: int = _get_int("ROUTER_MAX_TOKENS", 512)
SPECIALIST_MAX_TOKENS: int = _get_int("SPECIALIST_MAX_TOKENS", 1024)
SYNTHESIZER_MAX_TOKENS: int = _get_int("SYNTHESIZER_MAX_TOKENS", 2048)

# Default number of records a specialist's search tool returns (default top_k).
DOCUMENTS_PER_QUERY: int = _get_int("DOCUMENTS_PER_QUERY", 3)

# Maximum accepted length (characters) of a sanitized query.
MAX_QUERY_LENGTH: int = _get_int("MAX_QUERY_LENGTH", 10000)

# Upper bound on concurrent specialist threads.
MAX_WORKER_THREADS: int = _get_int("MAX_WORKER_THREADS", 4)

# Maximum model<->tool round-trips per specialist before forcing a final answer.
MAX_TOOL_ITERATIONS: int = _get_int("MAX_TOOL_ITERATIONS", 4)

# Maximum stored conversation messages per session (user + assistant turns).
MAX_HISTORY_MESSAGES: int = _get_int("MAX_HISTORY_MESSAGES", 6)


# --- HTTP server (consumed by wsgi.py) ------------------------------------- #

SERVER_HOST: str = _get_str("FLASK_RUN_HOST", "127.0.0.1")
SERVER_PORT: int = _get_int("FLASK_RUN_PORT", 8000)
DEBUG: bool = _get_bool("FLASK_DEBUG", False)


class Config:
    """Base Flask configuration."""


class TestingConfig(Config):
    TESTING = True
