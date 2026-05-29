"""Orchestration pipeline: Query -> Router -> Parallel Specialists -> Synthesizer.

Selected specialists run concurrently in a thread pool; each runs its own native
tool_use loop. Conversation history is threaded through every stage and persisted
per session in an in-memory store.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import anthropic

from app.agents.specialist import LitigationAgent, RegulatoryAgent, SpecialistAgent
from app.config import MAX_HISTORY_MESSAGES, MAX_WORKER_THREADS
from app.conversation import ConversationStore
from app.knowledge_base.retrieval import KnowledgeBaseRetriever
from app.orchestrator.models import (
    AgentSpecialty,
    OrchestratedResponse,
    SpecialistResponse,
)
from app.orchestrator.router import Router, RoutingError
from app.orchestrator.synthesizer import SynthesisError, Synthesizer

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam


class OrchestrationError(RuntimeError):
    """Raised when the pipeline cannot produce a response."""


def build_default_client() -> Anthropic:
    """Construct the default Anthropic client from the environment.

    Returns:
        An Anthropic client.

    Raises:
        anthropic.AnthropicError: if no credentials (``ANTHROPIC_API_KEY`` or
            ``ANTHROPIC_AUTH_TOKEN``) are configured.
    """

    client = anthropic.Anthropic()
    if not client.api_key and not client.auth_token:
        raise anthropic.AnthropicError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY (see "
            ".env.example)."
        )
    return client


class Orchestrator:
    """Coordinates routing, parallel specialist analysis, synthesis, and memory."""

    def __init__(
        self,
        client: Anthropic | None = None,
        retriever: KnowledgeBaseRetriever | None = None,
        router: Router | None = None,
        synthesizer: Synthesizer | None = None,
        conversation_store: ConversationStore | None = None,
        max_worker_threads: int = MAX_WORKER_THREADS,
    ) -> None:
        self._injected_client: Anthropic | None = client
        self._default_client: Anthropic | None = None
        self._client_lock = threading.Lock()
        self._retriever: KnowledgeBaseRetriever = (
            retriever or KnowledgeBaseRetriever()
        )
        self._router: Router = router or Router()
        self._synthesizer: Synthesizer = synthesizer or Synthesizer()
        self._store: ConversationStore = conversation_store or ConversationStore(
            MAX_HISTORY_MESSAGES
        )
        self._max_worker_threads: int = max_worker_threads
        self._agents: dict[AgentSpecialty, SpecialistAgent] = {
            AgentSpecialty.LITIGATION: LitigationAgent(self._retriever),
            AgentSpecialty.REGULATORY: RegulatoryAgent(self._retriever),
        }

    def _resolve_client(self) -> Anthropic:
        """Return the injected client, or build and cache a default one once.

        Raises:
            anthropic.AnthropicError: if a default client cannot be built.
        """

        if self._injected_client is not None:
            return self._injected_client
        if self._default_client is None:
            with self._client_lock:
                if self._default_client is None:
                    self._default_client = build_default_client()
        return self._default_client

    def process_query(self, query: str, session_id: str) -> OrchestratedResponse:
        """Run the full pipeline for a single query within its conversation.

        Loads prior history for ``session_id``, routes, runs the selected
        specialists in parallel, synthesizes, and records the new turn.

        Args:
            query: the user's legal question.
            session_id: conversation key; its prior turns are sent as context.

        Returns:
            An :class:`OrchestratedResponse`.

        Raises:
            ValueError: if ``query`` is empty.
            OrchestrationError: if the client cannot be initialized, or routing
                or synthesis fails.
        """

        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        try:
            client = self._resolve_client()
        except anthropic.AnthropicError as client_error:
            raise OrchestrationError(
                "Could not initialize the Anthropic client "
                f"(is ANTHROPIC_API_KEY set?): {client_error}"
            ) from client_error

        history = self._store.history(session_id)

        try:
            routing_decision = self._router.route(query, history, client)
        except RoutingError as routing_error:
            raise OrchestrationError(str(routing_error)) from routing_error

        specialist_responses = self._run_specialists(
            routing_decision.selected_specialties, query, history, client
        )

        try:
            synthesis = self._synthesizer.synthesize(
                query, history, specialist_responses, client
            )
        except SynthesisError as synthesis_error:
            raise OrchestrationError(str(synthesis_error)) from synthesis_error

        self._store.record_turn(session_id, query, synthesis)

        return OrchestratedResponse(
            query=query,
            session_id=session_id,
            routing_decision=routing_decision,
            specialist_responses=specialist_responses,
            synthesis=synthesis,
        )

    def _run_specialists(
        self,
        specialties: list[AgentSpecialty],
        query: str,
        history: list[MessageParam],
        client: Anthropic,
    ) -> list[SpecialistResponse]:
        """Run the given specialists concurrently, in the order they were given.

        A thread-level failure is captured as an errored :class:`SpecialistResponse`
        rather than propagated.
        """

        worker_count = min(self._max_worker_threads, max(len(specialties), 1))
        responses_by_specialty: dict[AgentSpecialty, SpecialistResponse] = {}

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_specialty = {
                executor.submit(
                    self._agents[specialty].analyze, query, history, client
                ): specialty
                for specialty in specialties
            }
            for future, specialty in future_to_specialty.items():
                try:
                    responses_by_specialty[specialty] = future.result()
                except Exception as worker_error:  # noqa: BLE001 - boundary guard
                    responses_by_specialty[specialty] = SpecialistResponse(
                        specialty=specialty,
                        analysis="",
                        citations=[],
                        error_message=f"Specialist worker crashed: {worker_error}",
                    )

        return [responses_by_specialty[specialty] for specialty in specialties]
