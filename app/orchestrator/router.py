"""Query router: selects specialists via Claude structured tool calling.

A single tool (``route_to_specialists``) is declared with a strict
``input_schema``, and ``tool_choice`` forces the model to emit exactly that
tool; the decision is read from the resulting typed ``tool_use`` block. The
router is a single deterministic call — it does not run a conversational loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anthropic
from anthropic.types import ToolUseBlock

from app.config import MODEL_IDENTIFIER, ROUTER_MAX_TOKENS
from app.orchestrator.models import AgentSpecialty, RoutingDecision
from app.prompts import ROUTER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import (
        Message,
        MessageParam,
        TextBlockParam,
        ToolChoiceToolParam,
        ToolParam,
    )


_ROUTING_TOOL_NAME: str = "route_to_specialists"

_ROUTING_TOOL: ToolParam = {
    "name": _ROUTING_TOOL_NAME,
    "description": (
        "Selecciona qué agentes especialistas deben analizar la consulta del "
        "usuario. Elige todos los pertinentes: una consulta puede requerir varios "
        "(por ejemplo, jurisprudencia y normativa a la vez). Selecciona al menos "
        "uno."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "specialties": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [specialty.value for specialty in AgentSpecialty],
                },
                "minItems": 1,
                "description": (
                    "Especialidades a invocar. "
                    "litigante = jurisprudencia y litigios; "
                    "normativo = normativa tributaria y legal."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Justificación breve de la elección.",
            },
        },
        "required": ["specialties", "reasoning"],
    },
}

_ROUTING_TOOL_CHOICE: ToolChoiceToolParam = {
    "type": "tool",
    "name": _ROUTING_TOOL_NAME,
}

_ROUTER_SYSTEM: list[TextBlockParam] = [
    {
        "type": "text",
        "text": ROUTER_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


class RoutingError(RuntimeError):
    """Raised when the router cannot obtain a valid routing decision."""


class Router:
    """Routes a query to one or more specialists via structured tool calling."""

    def route(
        self,
        query: str,
        history: list[MessageParam],
        client: Anthropic,
    ) -> RoutingDecision:
        """Classify ``query`` (in context) and return the specialists to invoke.

        Args:
            query: the user's legal question.
            history: prior conversation turns, oldest first.
            client: an Anthropic client.

        Returns:
            A :class:`RoutingDecision` with the selected specialties and the
            model's reasoning.

        Raises:
            RoutingError: if the model call fails or returns no tool use.
        """

        messages: list[MessageParam] = [*history, {"role": "user", "content": query}]
        try:
            message = client.messages.create(
                model=MODEL_IDENTIFIER,
                max_tokens=ROUTER_MAX_TOKENS,
                system=_ROUTER_SYSTEM,
                tools=[_ROUTING_TOOL],
                tool_choice=_ROUTING_TOOL_CHOICE,
                messages=messages,
            )
        except anthropic.APIError as api_error:
            raise RoutingError(f"Router model call failed: {api_error}") from api_error

        return self._parse_routing_decision(message)

    def _parse_routing_decision(self, message: Message) -> RoutingDecision:
        """Extract the routing decision from the response's tool_use block.

        Raises:
            RoutingError: if the response carries no usable tool_use block.
        """

        tool_use_block = next(
            (block for block in message.content if isinstance(block, ToolUseBlock)),
            None,
        )
        if tool_use_block is None:
            raise RoutingError(
                "Router response contained no tool_use block; cannot route."
            )

        tool_input = tool_use_block.input
        if not isinstance(tool_input, dict):
            raise RoutingError("Router tool input was not an object.")

        selected_specialties = self._normalize_specialties(
            tool_input.get("specialties", [])
        )
        reasoning = str(tool_input.get("reasoning", "")).strip()

        if not selected_specialties:
            selected_specialties = list(AgentSpecialty)
            reasoning = (
                reasoning
                or "Router returned no valid specialties; using all specialties."
            )

        return RoutingDecision(
            selected_specialties=selected_specialties, reasoning=reasoning
        )

    @staticmethod
    def _normalize_specialties(raw_specialties: Any) -> list[AgentSpecialty]:
        """Convert raw values to specialties, dropping unknowns and duplicates."""

        if not isinstance(raw_specialties, list):
            return []

        normalized: list[AgentSpecialty] = []
        seen: set[AgentSpecialty] = set()
        for raw_value in raw_specialties:
            try:
                specialty = AgentSpecialty.from_value(str(raw_value))
            except ValueError:
                continue
            if specialty not in seen:
                seen.add(specialty)
                normalized.append(specialty)
        return normalized
