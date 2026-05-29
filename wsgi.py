"""WSGI entry point.

Exposes the module-level ``app`` (``gunicorn wsgi:app``, ``flask --app wsgi
run``) and runs the development server directly (``python wsgi.py``).
"""

from __future__ import annotations

from app import create_app
from app.config import DEBUG, SERVER_HOST, SERVER_PORT

app = create_app()


if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG)
