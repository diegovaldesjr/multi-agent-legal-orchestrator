"""In-memory, multi-turn conversation history keyed by session id.

Stores only the user message and the final synthesized assistant response per
turn — never tool traces, retrieval payloads, or internal reasoning. State lives
in process memory only; there is no persistence layer.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types import MessageParam


class ConversationStore:
    """Thread-safe store of recent conversation turns per session."""

    def __init__(self, max_messages: int) -> None:
        self._max_messages = max_messages
        self._history: dict[str, list[MessageParam]] = {}
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[MessageParam]:
        """Return a copy of the stored messages for ``session_id`` (oldest first)."""

        with self._lock:
            return list(self._history.get(session_id, []))

    def record_turn(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append one user/assistant turn, trimming to the most recent messages."""

        with self._lock:
            messages = self._history.setdefault(session_id, [])
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})
            if len(messages) > self._max_messages:
                self._history[session_id] = messages[-self._max_messages :]
