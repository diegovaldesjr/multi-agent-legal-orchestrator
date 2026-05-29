"""Specialist agents that ground their analysis through native tool_use loops.

Each agent exposes one search tool to Claude and runs the Anthropic Messages API
tool_use loop: the model decides to call the tool, the backend executes the
search and returns a ``tool_result``, and the model continues until it produces a
final grounded answer. The first call is forced via ``tool_choice`` so a search
always happens; the loop is bounded by ``MAX_TOOL_ITERATIONS``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import anthropic
from anthropic.types import TextBlock, ToolUseBlock

from app.config import (
    DOCUMENTS_PER_QUERY,
    MAX_TOOL_ITERATIONS,
    MODEL_IDENTIFIER,
    SPECIALIST_MAX_TOKENS,
)
from app.orchestrator.models import (
    AgentSpecialty,
    Citation,
    SpecialistResponse,
)
from app.prompts import LITIGATION_SYSTEM_PROMPT, REGULATORY_SYSTEM_PROMPT

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import (
        ContentBlock,
        ContentBlockParam,
        Message,
        MessageParam,
        TextBlockParam,
        ToolChoiceParam,
        ToolParam,
    )

    from app.knowledge_base.retrieval import KnowledgeBaseRetriever


SEARCH_JURISPRUDENCIA_TOOL: ToolParam = {
    "name": "search_jurisprudencia",
    "description": (
        "Busca jurisprudencia chilena relevante en la base de conocimiento y "
        "devuelve los fallos coincidentes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Términos de búsqueda derivados de la consulta.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "Número máximo de fallos a devolver.",
            },
        },
        "required": ["query", "top_k"],
    },
}

SEARCH_NORMATIVA_TOOL: ToolParam = {
    "name": "search_normativa",
    "description": (
        "Busca normativa tributaria y legal chilena relevante en la base de "
        "conocimiento y devuelve las normas coincidentes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Términos de búsqueda derivados de la consulta.",
            },
        },
        "required": ["query"],
    },
}


def _query_from(tool_input: object) -> str:
    if isinstance(tool_input, dict):
        value = tool_input.get("query")
        if isinstance(value, str):
            return value
    return ""


def _top_k_from(tool_input: object, default: int) -> int:
    if isinstance(tool_input, dict):
        value = tool_input.get("top_k")
        if isinstance(value, int) and value > 0:
            return value
    return default


class SpecialistAgent(ABC):
    """Base class running the bounded native tool_use loop for one specialty."""

    def __init__(
        self,
        specialty: AgentSpecialty,
        tool: ToolParam,
        system_prompt: str,
        retriever: KnowledgeBaseRetriever,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        self.specialty: AgentSpecialty = specialty
        self._tool: ToolParam = tool
        self._tool_name: str = tool["name"]
        self._retriever: KnowledgeBaseRetriever = retriever
        self._max_iterations: int = max_iterations
        self._system_blocks: list[TextBlockParam] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def analyze(
        self,
        query: str,
        history: list[MessageParam],
        client: Anthropic,
    ) -> SpecialistResponse:
        """Run the tool_use loop and return a grounded analysis.

        Args:
            query: the user's legal question.
            history: prior conversation turns (user/assistant text), oldest first.
            client: an Anthropic client.

        Returns:
            A :class:`SpecialistResponse`. Citations are the records the search
            tool returned. On model failure the response carries
            ``error_message`` instead of raising, so one failing specialist never
            aborts the orchestration.
        """

        messages: list[MessageParam] = [*history, {"role": "user", "content": query}]
        citations: list[Citation] = []

        try:
            for iteration in range(self._max_iterations):
                message = client.messages.create(
                    model=MODEL_IDENTIFIER,
                    max_tokens=SPECIALIST_MAX_TOKENS,
                    system=self._system_blocks,
                    tools=[self._tool],
                    tool_choice=self._tool_choice_for(iteration),
                    messages=messages,
                )
                tool_uses = [
                    block for block in message.content if isinstance(block, ToolUseBlock)
                ]
                if not tool_uses:
                    return SpecialistResponse(
                        specialty=self.specialty,
                        analysis=self._extract_text(message),
                        citations=citations,
                        error_message=None,
                    )

                messages.append(self._assistant_echo(message.content))
                tool_results: list[ContentBlockParam] = []
                for tool_use in tool_uses:
                    payload, records = self._run_search(tool_use.input)
                    citations.extend(records)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": payload,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
        except anthropic.APIError as api_error:
            return SpecialistResponse(
                specialty=self.specialty,
                analysis="",
                citations=citations,
                error_message=f"Model call failed: {api_error}",
            )

        return SpecialistResponse(
            specialty=self.specialty,
            analysis="",
            citations=citations,
            error_message="Specialist exceeded its tool-iteration budget.",
        )

    def _tool_choice_for(self, iteration: int) -> ToolChoiceParam:
        """Force the search on the first call; forbid tools on the last one."""

        if iteration == 0:
            return {"type": "tool", "name": self._tool_name}
        if iteration == self._max_iterations - 1:
            return {"type": "none"}
        return {"type": "auto"}

    @abstractmethod
    def _run_search(self, tool_input: object) -> tuple[str, list[Citation]]:
        """Execute the search tool and return (tool_result JSON, citations)."""

    @staticmethod
    def _assistant_echo(content: list[ContentBlock]) -> MessageParam:
        """Rebuild the assistant turn (text + tool_use) as request params."""

        blocks: list[ContentBlockParam] = []
        for block in content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return {"role": "assistant", "content": blocks}

    @staticmethod
    def _extract_text(message: Message) -> str:
        """Return the concatenated text blocks of a Claude response."""

        return "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        ).strip()


class LitigationAgent(SpecialistAgent):
    """Litigante: grounds analysis in jurisprudence via search_jurisprudencia."""

    def __init__(
        self,
        retriever: KnowledgeBaseRetriever,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        super().__init__(
            AgentSpecialty.LITIGATION,
            SEARCH_JURISPRUDENCIA_TOOL,
            LITIGATION_SYSTEM_PROMPT,
            retriever,
            max_iterations,
        )

    def _run_search(self, tool_input: object) -> tuple[str, list[Citation]]:
        records = self._retriever.search_jurisprudencia(
            _query_from(tool_input), _top_k_from(tool_input, DOCUMENTS_PER_QUERY)
        )
        payload = json.dumps(
            {"resultados": [record.to_payload() for record in records]},
            ensure_ascii=False,
        )
        return payload, [record.to_citation() for record in records]


class RegulatoryAgent(SpecialistAgent):
    """Normativo: grounds analysis in regulations via search_normativa."""

    def __init__(
        self,
        retriever: KnowledgeBaseRetriever,
        max_iterations: int = MAX_TOOL_ITERATIONS,
    ) -> None:
        super().__init__(
            AgentSpecialty.REGULATORY,
            SEARCH_NORMATIVA_TOOL,
            REGULATORY_SYSTEM_PROMPT,
            retriever,
            max_iterations,
        )

    def _run_search(self, tool_input: object) -> tuple[str, list[Citation]]:
        records = self._retriever.search_normativa(_query_from(tool_input))
        payload = json.dumps(
            {"resultados": [record.to_payload() for record in records]},
            ensure_ascii=False,
        )
        return payload, [record.to_citation() for record in records]
