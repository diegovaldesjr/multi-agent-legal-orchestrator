"""Shared, fully-typed data models passed between the pipeline's components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentSpecialty(StrEnum):
    """The two specialist agents the orchestrator can route to."""

    LITIGATION = "litigante"
    REGULATORY = "normativo"

    @classmethod
    def from_value(cls, raw_value: str) -> AgentSpecialty:
        """Resolve a raw string to an :class:`AgentSpecialty`.

        Raises:
            ValueError: if ``raw_value`` does not match a known specialty.
        """

        try:
            return cls(raw_value)
        except ValueError as conversion_error:
            valid_values = ", ".join(specialty.value for specialty in cls)
            raise ValueError(
                f"Unknown specialty '{raw_value}'. Expected one of: {valid_values}."
            ) from conversion_error


@dataclass(frozen=True)
class Citation:
    """A knowledge-base record cited by a specialist, in uniform shape."""

    reference: str
    source: str
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        return {
            "reference": self.reference,
            "source": self.source,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class JurisprudenceRecord:
    """A jurisprudence ruling from the knowledge base."""

    ruling_id: str
    court: str
    excerpt: str

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> JurisprudenceRecord:
        return cls(
            ruling_id=str(raw.get("id", "")),
            court=str(raw.get("tribunal", "")),
            excerpt=str(raw.get("extracto", "")),
        )

    def to_payload(self) -> dict[str, str]:
        """Serialize back to the Spanish field names the model reasons over."""

        return {"id": self.ruling_id, "tribunal": self.court, "extracto": self.excerpt}

    def to_citation(self) -> Citation:
        return Citation(reference=self.ruling_id, source=self.court, excerpt=self.excerpt)


@dataclass(frozen=True)
class RegulationRecord:
    """A regulatory/tax rule from the knowledge base."""

    code: str
    description: str

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> RegulationRecord:
        return cls(
            code=str(raw.get("codigo", "")),
            description=str(raw.get("descripcion", "")),
        )

    def to_payload(self) -> dict[str, str]:
        """Serialize back to the Spanish field names the model reasons over."""

        return {"codigo": self.code, "descripcion": self.description}

    def to_citation(self) -> Citation:
        return Citation(reference=self.code, source="normativa", excerpt=self.description)


@dataclass(frozen=True)
class RoutingDecision:
    """The router's decision about which specialists should handle a query."""

    selected_specialties: list[AgentSpecialty]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_specialties": [
                specialty.value for specialty in self.selected_specialties
            ],
            "reasoning": self.reasoning,
        }


@dataclass
class SpecialistResponse:
    """The analysis produced by one specialist agent.

    ``error_message`` is populated only when the specialist failed (model or
    tool execution). A failed specialist never aborts the pipeline.
    """

    specialty: AgentSpecialty
    analysis: str
    citations: list[Citation] = field(default_factory=list)
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_message is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialty": self.specialty.value,
            "analysis": self.analysis,
            "citations": [citation.to_dict() for citation in self.citations],
            "succeeded": self.succeeded,
            "error_message": self.error_message,
        }


@dataclass
class OrchestratedResponse:
    """The full result returned by the orchestrator for a single query."""

    query: str
    session_id: str
    routing_decision: RoutingDecision
    specialist_responses: list[SpecialistResponse]
    synthesis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "session_id": self.session_id,
            "routing": self.routing_decision.to_dict(),
            "specialists": [
                response.to_dict() for response in self.specialist_responses
            ],
            "synthesis": self.synthesis,
        }
