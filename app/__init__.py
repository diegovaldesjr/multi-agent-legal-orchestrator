"""Application package and Flask application factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Flask
from flask.json.provider import DefaultJSONProvider

from app.config import Config

if TYPE_CHECKING:
    from app.orchestrator.pipeline import Orchestrator


def create_app(
    orchestrator: Orchestrator | None = None,
    config_object: type[Config] = Config,
) -> Flask:
    """Build and configure the Flask application.

    Args:
        orchestrator: a pre-built orchestrator (e.g. with a mocked Anthropic
            client) for tests. When omitted, a default one is created.
        config_object: the configuration class to load.

    Returns:
        The configured Flask application.
    """

    # Imported inside the factory to avoid an import cycle.
    from app.api import api_blueprint
    from app.orchestrator.pipeline import Orchestrator as DefaultOrchestrator

    application = Flask(__name__)
    application.config.from_object(config_object)

    # Preserve non-ASCII characters in JSON responses.
    if isinstance(application.json, DefaultJSONProvider):
        application.json.ensure_ascii = False

    application.extensions["orchestrator"] = orchestrator or DefaultOrchestrator()
    application.register_blueprint(api_blueprint)
    return application
