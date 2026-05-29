"""API routes (Flask Blueprint): POST /query and GET /health."""

from __future__ import annotations

from typing import Any, cast

from flask import Blueprint, current_app, jsonify, request

from app.api.validation import (
    QueryValidationError,
    resolve_session_id,
    sanitize_query,
)
from app.config import MAX_QUERY_LENGTH
from app.orchestrator.pipeline import OrchestrationError, Orchestrator

api_blueprint = Blueprint("api", __name__)


def _orchestrator() -> Orchestrator:
    """Return the shared Orchestrator stored on the application."""

    return cast(Orchestrator, current_app.extensions["orchestrator"])


@api_blueprint.post("/query")
def query() -> Any:
    """Run the orchestration pipeline for a JSON ``{query, session_id?}`` body.

    ``query`` is required and sanitized; ``session_id`` is optional and
    auto-generated when absent (the value used is echoed in the response).
    Returns the orchestrated result as JSON (200), a validation error (400), or
    a structured upstream/credentials error (502).
    """

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        user_query = sanitize_query(payload.get("query"), MAX_QUERY_LENGTH)
    except QueryValidationError as validation_error:
        return jsonify({"error": str(validation_error)}), 400

    session_id = resolve_session_id(payload.get("session_id"))

    try:
        result = _orchestrator().process_query(user_query, session_id)
    except ValueError as validation_error:
        return jsonify({"error": str(validation_error)}), 400
    except OrchestrationError as orchestration_error:
        return jsonify({"error": str(orchestration_error)}), 502

    return jsonify(result.to_dict()), 200
