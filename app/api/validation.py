"""Request input validation and sanitization for the API layer."""

from __future__ import annotations

import unicodedata
import uuid
from typing import Any


class QueryValidationError(ValueError):
    """Raised when the query field fails validation."""


def sanitize_query(raw_query: Any, max_length: int) -> str:
    """Validate and normalize an incoming query.

    Applies Unicode NFC normalization, removes control and format characters
    (keeping newlines and tabs), and trims surrounding whitespace. This is input
    hygiene and bounding — it does not defend against prompt injection.

    Args:
        raw_query: the raw value taken from the request body.
        max_length: maximum allowed length of the cleaned query.

    Returns:
        The cleaned query string.

    Raises:
        QueryValidationError: if the value is not a string, is empty after
            cleaning, or exceeds ``max_length``.
    """

    if not isinstance(raw_query, str):
        raise QueryValidationError("Field 'query' is required and must be a string.")

    normalized = unicodedata.normalize("NFC", raw_query)
    cleaned = "".join(
        character
        for character in normalized
        if character in ("\n", "\t")
        or not unicodedata.category(character).startswith("C")
    ).strip()

    if not cleaned:
        raise QueryValidationError("Field 'query' must not be empty.")
    if len(cleaned) > max_length:
        raise QueryValidationError(
            f"Field 'query' exceeds the maximum length of {max_length} characters."
        )
    return cleaned


def resolve_session_id(raw_session_id: Any) -> str:
    """Return a usable session id, generating one when none is supplied.

    A non-empty string is honored (trimmed); anything else — missing, null,
    empty, whitespace-only, or a non-string — yields a fresh UUID4.

    Args:
        raw_session_id: the raw value taken from the request body.

    Returns:
        The provided session id, or a newly generated one.
    """

    if isinstance(raw_session_id, str) and raw_session_id.strip():
        return raw_session_id.strip()
    return str(uuid.uuid4())
