# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-agent orchestrator for Chilean legal analysis, built **only** on the Anthropic Python SDK with native `tool_use` (no LangChain/LangGraph or any agent framework). Flow:

```
Query → Router → Parallel Specialists (native tool_use) → Synthesizer → Response
                          │
                  Conversation history (per session)
```

Two specialists (see `AgentSpecialty`): `litigante` → `LitigationAgent` (tool `search_jurisprudencia`), `normativo` → `RegulatoryAgent` (tool `search_normativa`).

**Language split:** code (classes, modules, typing, comments) is in **English**; the legal domain surface (prompts in `app/prompts.py`, tool names, `mock_data.json`, model-facing text) is in **Spanish** — the system answers Chilean legal questions in Spanish.

## Commands

Setup and the full quality gate (all run fully offline — no API key needed):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # app package + pytest, ruff, mypy

python -m pytest                 # tests (mocked Anthropic client)
python -m pytest tests/test_orchestrator.py::test_litigation_agent_runs_native_tool_loop  # single test
ruff check .                     # lint  (config in pyproject.toml)
mypy                             # strict type-check of app/ + wsgi.py

python wsgi.py                   # run the API (needs ANTHROPIC_API_KEY to serve queries)
# also: flask --app wsgi run  |  gunicorn wsgi:app
```

All three checks (`pytest`, `ruff check .`, `mypy`) are expected to pass with **zero findings** — keep it that way when editing.

## Architecture

Single application package `app/`, standard Flask layout (application factory + blueprint). The web layer is thin; the domain layer is framework-agnostic.

- **`app/__init__.py`** — `create_app()` factory. Builds the `Orchestrator` once and stores it on `app.extensions["orchestrator"]`; routes read it from there via `current_app`. Blueprint and pipeline imports happen **inside** the factory to avoid an import cycle.
- **`app/orchestrator/pipeline.py`** — `Orchestrator.process_query()` is the entry point: loads conversation history for the `session_id`, routes, runs the selected specialists concurrently in a `ThreadPoolExecutor` (deliberately not asyncio — the WSGI app is sync), synthesizes, and records the new turn. Results are returned in router-selection order.
- **`app/agents/specialist.py`** — base `SpecialistAgent` runs the **native `tool_use` loop**; `LitigationAgent` / `RegulatoryAgent` supply the tool + search. The loop forces the search on the first call (`tool_choice` = the tool), lets the model continue (`auto`), and forces a final answer on the last iteration (`tool_choice` = `none`), bounded by `MAX_TOOL_ITERATIONS`. The backend executes the search and returns a real `tool_result` block. `analyze()` **never raises**: model/tool failures are recorded on `SpecialistResponse.error_message`. The assistant turn (text + `tool_use`) is rebuilt as request params via `_assistant_echo` to stay `mypy --strict`-clean.
- **`app/orchestrator/router.py`** — routing is a single **structured tool-calling** call: one tool (`route_to_specialists`) with a strict `input_schema`, forced via `tool_choice`. The decision is read from the typed `tool_use` block. Not a conversational loop; never parse JSON out of free text.
- **`app/orchestrator/synthesizer.py`** — merges only the *successful* specialist analyses into one answer.
- **`app/conversation.py`** — `ConversationStore`: thread-safe in-memory `dict[session_id, list[MessageParam]]`. Stores only user + final synthesized assistant turns (never tool traces), trimmed to `MAX_HISTORY_MESSAGES`. Threaded into router, specialists, and synthesizer so follow-up turns have context.
- **`app/knowledge_base/retrieval.py`** — mocked retrieval over `mock_data.json` (stopword-filtered lexical overlap), standing in for a vector kNN search. Two methods: `search_jurisprudencia(query, top_k)` and `search_normativa(query)`, returning typed records. Preserve the signatures/return types if swapping in a real backend. An empty list is a valid result (drives the no-hallucination path).
- **`app/prompts.py`** — all system prompts (Spanish), one per component. The specialist prompts mandate using the search tool and forbid inventing rulings/rules when a search returns nothing. Tune prompts here, not inline.
- **`app/config.py`** — every tunable is read from the environment with safe defaults (`.env` auto-loaded via `python-dotenv`). `MODEL_IDENTIFIER` is the single source of truth for the model string.
- **`app/orchestrator/models.py`** — shared typed dataclasses: `AgentSpecialty` (StrEnum), `JurisprudenceRecord` / `RegulationRecord` (mirror the two mock categories; English fields, `to_payload()` re-emits Spanish keys for the model, `to_citation()` unifies), `Citation`, `RoutingDecision`, `SpecialistResponse`, `OrchestratedResponse`. `to_dict()` defines the JSON API shape.

## Conventions that span files (read before editing)

- **Conversation memory is in-process, per `session_id`.** `session_id` is optional (a UUID4 is generated when omitted) and echoed back; reusing it continues a conversation. History is in-memory only (no DB/persistence) and stores only user + synthesized-assistant turns.
- **Native `tool_use`, no faking.** Specialists must reach the knowledge base through the `tool_use` loop (real `tool_use`/`tool_result` blocks), never by retrieving directly and pre-stuffing context. Hallucination prevention is structural: empty retrieval → empty `tool_result` + empty citations + prompt instruction to say "no encontré" and invent nothing.
- **Input handling at the API boundary** lives in `app/api/validation.py`: `sanitize_query` (Unicode NFC, control-char stripping, `MAX_QUERY_LENGTH` bound — input hygiene, **not** prompt-injection defense) and `resolve_session_id`. The orchestrator keeps its own minimal non-empty-query guard.
- **Error funneling.** Router/synthesizer failures and missing/invalid credentials all become a single `OrchestrationError`, which the `/query` route maps to a **502 with a structured JSON body** — never a 500/traceback. Credentials are checked up front in `build_default_client` because the SDK constructs without a key and only fails deep inside `messages.create`.
- **Client lifecycle.** Tests inject a mocked client; production builds one lazily and caches it (double-checked locking). Never construct `anthropic.Anthropic()` at import time (the app must boot with no key).
- **Typed SDK usage.** Tool/system params use the SDK's `TypedDict`s (`ToolParam`, `ToolChoiceToolParam`, `TextBlockParam`); response blocks are narrowed with `isinstance(block, TextBlock | ToolUseBlock)`. This keeps `mypy --strict` clean.
- **Tests drive the real loop with real SDK block types.** The fake client returns a `ToolUseBlock` until it sees a `tool_result` come back, then a final `TextBlock` — exercising the actual `tool_use` control flow. Blocks are genuine `TextBlock`/`ToolUseBlock` (not `SimpleNamespace`) so production `isinstance` narrowing behaves as it does live.
- **Package `__init__.py` files are intentionally lightweight** (no re-exports) to keep the import graph acyclic. Import concrete classes from their modules.
- **Comments declare behavior** (what / args / returns / raises) — do not add change-narrative or decision-justification prose.
- **Model string:** `claude-sonnet-4-5-20250929` (full dated id; the bare `claude-sonnet-4-5` is invalid). Sonnet 4.5 does not support `effort`/adaptive-thinking params, so calls are kept plain.
