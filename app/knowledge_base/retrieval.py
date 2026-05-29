"""Mocked knowledge base retrieval.

Retrieval is mocked against ``mock_data.json`` (package data beside this module)
using a stopword-filtered lexical overlap score, standing in for a vector kNN
search. The method signatures match what a production retriever would expose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.orchestrator.models import JurisprudenceRecord, RegulationRecord

# Mock corpus path, resolved relative to this module (working-directory safe).
_DEFAULT_MOCK_DATA_PATH: Path = Path(__file__).resolve().parent / "mock_data.json"

_TOKEN_PATTERN = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)

# Common Spanish function words, excluded so relevance reflects content words.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "al", "ante", "con", "como", "de", "del", "desde", "e", "el",
        "ella", "ello", "en", "entre", "es", "esa", "ese", "eso", "esta",
        "estas", "este", "estos", "hasta", "hay", "la", "las", "le", "les",
        "lo", "los", "mas", "más", "mi", "no", "o", "para", "pero", "por",
        "que", "se", "segun", "según", "si", "sin", "sobre", "son", "su",
        "sus", "tu", "un", "una", "unas", "unos", "y", "u",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lower-case content tokens (stopwords removed) for the relevance score."""

    return {
        match.group(0).lower()
        for match in _TOKEN_PATTERN.finditer(text)
        if match.group(0).lower() not in _STOPWORDS
    }


def _document_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Join the given fields of a raw record into one searchable string."""

    return " ".join(str(raw.get(key, "")) for key in keys)


def _overlap_score(query_tokens: set[str], document_text: str) -> int:
    """Number of content tokens shared between the query and the document."""

    return len(query_tokens & _tokenize(document_text))


class KnowledgeBaseRetrievalError(RuntimeError):
    """Raised when the mock corpus cannot be loaded or is malformed."""


class KnowledgeBaseRetriever:
    """Loads the mock corpus once and serves per-category lexical retrieval."""

    def __init__(self, mock_data_path: Path | None = None) -> None:
        self._mock_data_path: Path = mock_data_path or _DEFAULT_MOCK_DATA_PATH
        corpus = self._load_corpus()
        self._jurisprudence: list[dict[str, Any]] = corpus.get("jurisprudencia", [])
        self._regulations: list[dict[str, Any]] = corpus.get("normativa", [])

    def _load_corpus(self) -> dict[str, list[dict[str, Any]]]:
        try:
            raw_text = self._mock_data_path.read_text(encoding="utf-8")
        except OSError as read_error:
            raise KnowledgeBaseRetrievalError(
                f"Could not read mock corpus at {self._mock_data_path}: {read_error}"
            ) from read_error

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as parse_error:
            raise KnowledgeBaseRetrievalError(
                f"Mock corpus at {self._mock_data_path} is not valid JSON: {parse_error}"
            ) from parse_error

        if not isinstance(parsed, dict):
            raise KnowledgeBaseRetrievalError("Mock corpus must be a JSON object.")
        return parsed

    def search_jurisprudencia(
        self, query: str, top_k: int
    ) -> list[JurisprudenceRecord]:
        """Return the ``top_k`` most relevant jurisprudence rulings for ``query``.

        Mocked stand-in for a vector kNN search: the signature and return type
        match a production retriever, so only the scoring (lexical overlap)
        differs. An empty list is a valid result and must be surfaced as "no
        jurisprudence found", never invented around.

        Raises:
            ValueError: if ``top_k`` is not positive.
        """

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_tokens = _tokenize(query)
        scored = [
            (_overlap_score(query_tokens, _document_text(raw, ("extracto",))), raw)
            for raw in self._jurisprudence
        ]
        relevant = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: item[0],
            reverse=True,
        )
        return [JurisprudenceRecord.from_raw(raw) for _, raw in relevant[:top_k]]

    def search_normativa(self, query: str) -> list[RegulationRecord]:
        """Return the regulations relevant to ``query``.

        Mocked stand-in for a vector kNN search. An empty list is a valid result.
        """

        query_tokens = _tokenize(query)
        scored = [
            (_overlap_score(query_tokens, _document_text(raw, ("descripcion",))), raw)
            for raw in self._regulations
        ]
        relevant = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: item[0],
            reverse=True,
        )
        return [RegulationRecord.from_raw(raw) for _, raw in relevant]
