"""Response synthesizer: merges specialist analyses into one answer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic
from anthropic.types import TextBlock

from app.config import MODEL_IDENTIFIER, SYNTHESIZER_MAX_TOKENS
from app.orchestrator.models import SpecialistResponse
from app.prompts import SYNTHESIZER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import Message, MessageParam, TextBlockParam


_SYNTHESIZER_SYSTEM: list[TextBlockParam] = [
    {
        "type": "text",
        "text": SYNTHESIZER_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


class SynthesisError(RuntimeError):
    """Raised when synthesis cannot be produced."""


class Synthesizer:
    """Merges specialist analyses into a single user-facing answer."""

    def synthesize(
        self,
        query: str,
        history: list[MessageParam],
        specialist_responses: list[SpecialistResponse],
        client: Anthropic,
    ) -> str:
        """Integrate the successful specialist analyses into one answer.

        Args:
            query: the original user query.
            history: prior conversation turns, oldest first.
            specialist_responses: per-specialist results; only successful ones
                contribute to the synthesis.
            client: an Anthropic client.

        Returns:
            The unified answer text.

        Raises:
            SynthesisError: if no specialist succeeded or the model call fails.
        """

        successful_responses = [
            response for response in specialist_responses if response.succeeded
        ]
        if not successful_responses:
            raise SynthesisError(
                "No specialist produced a successful analysis to synthesize."
            )

        messages: list[MessageParam] = [
            *history,
            {"role": "user", "content": self._build_prompt(query, successful_responses)},
        ]

        try:
            message = client.messages.create(
                model=MODEL_IDENTIFIER,
                max_tokens=SYNTHESIZER_MAX_TOKENS,
                system=_SYNTHESIZER_SYSTEM,
                messages=messages,
            )
        except anthropic.APIError as api_error:
            raise SynthesisError(
                f"Synthesizer model call failed: {api_error}"
            ) from api_error

        return self._extract_text(message)

    @staticmethod
    def _build_prompt(
        query: str, successful_responses: list[SpecialistResponse]
    ) -> str:
        """Build the user-turn prompt listing the specialist analyses."""

        analysis_blocks = "\n\n".join(
            f"### Análisis del especialista {response.specialty.value}\n"
            f"{response.analysis}"
            for response in successful_responses
        )
        return (
            "Consulta del usuario:\n"
            f"{query}\n\n"
            "Análisis de los especialistas:\n\n"
            f"{analysis_blocks}\n\n"
            "Integra estos análisis en una respuesta final unificada y coherente "
            "para el usuario."
        )

    @staticmethod
    def _extract_text(message: Message) -> str:
        """Return the concatenated text blocks of a Claude response."""

        return "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()
