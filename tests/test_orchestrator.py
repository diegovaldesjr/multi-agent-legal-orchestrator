"""Tests for the multi-agent legal orchestrator.

Every test uses a mocked Anthropic client — no network, no API key. The fake
drives the real ``tool_use`` loop: on a specialist call it returns a ToolUseBlock
until it sees a tool_result come back, then returns a final text block. Response
blocks are real SDK types so the production ``isinstance`` checks behave exactly
as they do against live responses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from app import create_app
from app.agents.specialist import LitigationAgent, RegulatoryAgent
from app.api.validation import (
    QueryValidationError,
    resolve_session_id,
    sanitize_query,
)
from app.config import MAX_HISTORY_MESSAGES, TestingConfig
from app.conversation import ConversationStore
from app.knowledge_base.retrieval import KnowledgeBaseRetriever
from app.orchestrator.models import AgentSpecialty
from app.orchestrator.pipeline import Orchestrator
from app.orchestrator.router import Router

_HTTP_OK = 200
_HTTP_BAD_REQUEST = 400

# Mandatory challenge test-case queries (Spanish domain).
TURN_1 = (
    "Tengo un contrato de arriendo con el arrendatario Juan Pérez. Lleva 4 meses "
    "sin pagar la renta. ¿Qué acciones legales puedo tomar y cuáles son las "
    "implicancias tributarias de las rentas no percibidas?"
)
TURN_2 = "¿Y si el contrato tiene cláusula de arbitraje, cambia algo?"
TURN_3 = (
    "¿Hay sentencias de la Corte Suprema sobre nulidad de finiquito por error en "
    "el cálculo proporcional de vacaciones?"
)


# --------------------------------------------------------------------------- #
# Test doubles                                                                 #
# --------------------------------------------------------------------------- #


def _message(content_blocks: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(content=content_blocks)


def _last_is_tool_result(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    content = messages[-1].get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    content = messages[-1].get("content") if messages else ""
    return content if isinstance(content, str) else ""


class _FakeMessages:
    def __init__(self, owner: FakeAnthropicClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self._owner.record_call(kwargs)
        tools = kwargs.get("tools") or []
        tool_names = {tool["name"] for tool in tools}

        if "route_to_specialists" in tool_names:
            return _message(
                [
                    ToolUseBlock(
                        type="tool_use",
                        id="toolu_route",
                        name="route_to_specialists",
                        input={
                            "specialties": [
                                specialty.value
                                for specialty in self._owner.routing_specialties
                            ],
                            "reasoning": "Routing decided by the test double.",
                        },
                    )
                ]
            )

        if tools:  # specialist call
            messages = kwargs["messages"]
            if _last_is_tool_result(messages):
                return _message([TextBlock(type="text", text=self._owner.analysis_text)])
            tool_name = tools[0]["name"]
            query = _last_user_text(messages)
            tool_input: dict[str, Any] = {"query": query}
            if tool_name == "search_jurisprudencia":
                tool_input["top_k"] = 3
            return _message(
                [ToolUseBlock(type="tool_use", id="toolu_search", name=tool_name, input=tool_input)]
            )

        return _message([TextBlock(type="text", text=self._owner.synthesis_text)])


class FakeAnthropicClient:
    """Stand-in for ``anthropic.Anthropic`` that drives the tool_use loop."""

    def __init__(
        self,
        routing_specialties: list[AgentSpecialty],
        synthesis_text: str = "Respuesta final unificada simulada.",
        analysis_text: str = "Análisis simulado del especialista.",
    ) -> None:
        self.routing_specialties = routing_specialties
        self.synthesis_text = synthesis_text
        self.analysis_text = analysis_text
        self.calls: list[dict[str, Any]] = []
        self.messages = _FakeMessages(self)

    def record_call(self, kwargs: dict[str, Any]) -> None:
        self.calls.append(kwargs)

    def specialist_calls(self) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if call.get("tools")
            and call["tools"][0]["name"] != "route_to_specialists"
        ]


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def retriever() -> KnowledgeBaseRetriever:
    return KnowledgeBaseRetriever()


# --------------------------------------------------------------------------- #
# Native tool_use loop                                                         #
# --------------------------------------------------------------------------- #


def test_litigation_agent_runs_native_tool_loop(retriever) -> None:
    """The agent forces a search, feeds the tool_result back, then answers."""

    client = FakeAnthropicClient(routing_specialties=[AgentSpecialty.LITIGATION])
    response = LitigationAgent(retriever).analyze(TURN_1, [], client)

    assert response.succeeded
    assert response.analysis == client.analysis_text
    assert response.citations  # jurisprudence matched the arriendo query

    calls = client.specialist_calls()
    assert len(calls) >= 2  # at least one tool round-trip + final answer
    assert calls[0]["tool_choice"] == {
        "type": "tool",
        "name": "search_jurisprudencia",
    }
    assert any(_last_is_tool_result(call["messages"]) for call in calls)


def test_regulatory_agent_runs_native_tool_loop(retriever) -> None:
    """The regulatory agent uses search_normativa in a real tool_use loop."""

    client = FakeAnthropicClient(routing_specialties=[AgentSpecialty.REGULATORY])
    response = RegulatoryAgent(retriever).analyze(
        "implicancias tributarias de las rentas de arrendamiento no percibidas",
        [],
        client,
    )

    assert response.succeeded
    assert response.citations
    assert client.specialist_calls()[0]["tool_choice"]["name"] == "search_normativa"


def test_specialist_empty_retrieval_yields_no_citations(retriever) -> None:
    """A query with no matching records produces zero citations (no fabrication)."""

    client = FakeAnthropicClient(routing_specialties=[AgentSpecialty.LITIGATION])
    response = LitigationAgent(retriever).analyze(TURN_3, [], client)

    assert response.succeeded
    assert response.citations == []  # nothing to ground on; nothing invented


# --------------------------------------------------------------------------- #
# Routing                                                                       #
# --------------------------------------------------------------------------- #


def test_router_selects_both_specialties() -> None:
    """The router returns the structured decision from its tool_use block."""

    expected = [AgentSpecialty.LITIGATION, AgentSpecialty.REGULATORY]
    client = FakeAnthropicClient(routing_specialties=expected)

    decision = Router().route(TURN_1, [], client)

    assert decision.selected_specialties == expected
    assert decision.reasoning
    assert client.calls[0]["tool_choice"]["name"] == "route_to_specialists"


# --------------------------------------------------------------------------- #
# End-to-end pipeline                                                           #
# --------------------------------------------------------------------------- #


def test_end_to_end_runs_selected_specialists(retriever) -> None:
    """Both specialists run (in parallel) and the synthesis is returned."""

    expected = [AgentSpecialty.LITIGATION, AgentSpecialty.REGULATORY]
    client = FakeAnthropicClient(routing_specialties=expected)
    result = Orchestrator(client=client, retriever=retriever).process_query(
        TURN_1, "case-001"
    )

    assert [r.specialty for r in result.specialist_responses] == expected
    assert all(r.succeeded for r in result.specialist_responses)
    assert result.synthesis == client.synthesis_text


def test_conversation_history_persists_across_turns(retriever) -> None:
    """Turn 2 receives Turn 1 as context, and history is stored per session."""

    store = ConversationStore(MAX_HISTORY_MESSAGES)
    client = FakeAnthropicClient(routing_specialties=[AgentSpecialty.LITIGATION])
    orchestrator = Orchestrator(
        client=client, retriever=retriever, conversation_store=store
    )

    orchestrator.process_query(TURN_1, "s1")
    assert len(store.history("s1")) == 2  # user + assistant

    client.calls.clear()
    orchestrator.process_query(TURN_2, "s1")
    assert len(store.history("s1")) == 4  # two turns retained

    # Turn 2's router call carried the prior turn as context.
    router_messages = client.calls[0]["messages"]
    assert len(router_messages) > 1
    assert any(
        message.get("role") == "user" and "arriendo" in str(message.get("content", ""))
        for message in router_messages
    )


# --------------------------------------------------------------------------- #
# Retrieval                                                                     #
# --------------------------------------------------------------------------- #


def test_search_jurisprudencia_matches_and_misses(retriever) -> None:
    matches = retriever.search_jurisprudencia("arrendatario renta arriendo", top_k=3)
    assert matches and matches[0].ruling_id.startswith("ROL")

    assert retriever.search_jurisprudencia("zzz qqq nada relevante", top_k=3) == []


def test_search_normativa_matches_and_misses(retriever) -> None:
    matches = retriever.search_normativa("rentas de arrendamiento no percibidas")
    assert matches and matches[0].code

    assert retriever.search_normativa("zzz qqq nada relevante") == []


# --------------------------------------------------------------------------- #
# HTTP endpoint + input validation                                             #
# --------------------------------------------------------------------------- #


def _endpoint_client(retriever: KnowledgeBaseRetriever):
    client = FakeAnthropicClient(routing_specialties=[AgentSpecialty.LITIGATION])
    orchestrator = Orchestrator(client=client, retriever=retriever)
    flask_app = create_app(orchestrator=orchestrator, config_object=TestingConfig)
    return client, flask_app.test_client()


def test_query_endpoint_returns_structured_json(retriever) -> None:
    client, test_client = _endpoint_client(retriever)
    http_response = test_client.post(
        "/query", json={"query": TURN_1, "session_id": "case-001"}
    )

    assert http_response.status_code == _HTTP_OK
    body = http_response.get_json()
    assert body["session_id"] == "case-001"
    assert body["routing"]["selected_specialties"] == ["litigante"]
    assert body["synthesis"] == client.synthesis_text
    assert isinstance(body["specialists"], list) and body["specialists"]


def test_query_endpoint_autogenerates_session_id(retriever) -> None:
    _, test_client = _endpoint_client(retriever)
    http_response = test_client.post("/query", json={"query": TURN_1})

    assert http_response.status_code == _HTTP_OK
    session_id = http_response.get_json()["session_id"]
    assert isinstance(session_id, str) and session_id


def test_query_endpoint_rejects_empty_query(retriever) -> None:
    _, test_client = _endpoint_client(retriever)
    http_response = test_client.post(
        "/query", json={"query": "   ", "session_id": "s1"}
    )

    assert http_response.status_code == _HTTP_BAD_REQUEST
    assert "query" in http_response.get_json()["error"]


def test_query_endpoint_rejects_oversized_query(retriever) -> None:
    _, test_client = _endpoint_client(retriever)
    http_response = test_client.post(
        "/query", json={"query": "a" * 20000, "session_id": "s1"}
    )

    assert http_response.status_code == _HTTP_BAD_REQUEST
    assert "maximum length" in http_response.get_json()["error"]


def test_sanitize_query_strips_control_chars_and_normalizes() -> None:
    assert sanitize_query("  hola\x00\x07 mundo\r\n  ", max_length=100) == "hola mundo"


def test_sanitize_query_rejects_non_string_and_empty() -> None:
    for bad_value in (None, 123, "", "   ", "\x00\x00"):
        with pytest.raises(QueryValidationError):
            sanitize_query(bad_value, max_length=100)


def test_resolve_session_id_honors_value_or_generates() -> None:
    assert resolve_session_id("  given-id  ") == "given-id"
    for missing in (None, "", "   ", 123):
        generated = resolve_session_id(missing)
        assert isinstance(generated, str) and generated
    assert resolve_session_id(None) != resolve_session_id(None)
