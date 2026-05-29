# multi-agent-legal-orchestrator

Multi-agent orchestration system for Chilean legal analysis, built directly on
the Anthropic Python SDK with native `tool_use` — no agent frameworks. A router
selects specialists, the specialists ground their answers through real tool-use
loops over a mocked knowledge base, and a synthesizer merges the results into one
coherent response. Conversation history is preserved across turns.

```
Query → Router → Parallel Specialists (native tool_use) → Synthesizer → Response
                          │
                  Conversation history (per session)
```

Two specialists, matching the legal domain:

| Agent (domain) | Class             | Tool exposed to Claude            |
| -------------- | ----------------- | --------------------------------- |
| Litigante      | `LitigationAgent` | `search_jurisprudencia(query, top_k)` |
| Normativo      | `RegulatoryAgent` | `search_normativa(query)`         |

The code (classes, modules, typing, comments) is in English; the legal domain
surface (prompts, tool names, mock data, model-facing text) is in Spanish,
because the system reasons about and answers Chilean legal questions.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # installs the app package + dev tools (pytest, ruff, mypy)

# Configuration: copy the template and fill in your key.
cp .env.example .env             # then edit ANTHROPIC_API_KEY (.env is gitignored)

# Quality gate — all three pass with zero findings, fully offline (no key needed).
python -m pytest                 # 15 tests, mocked Anthropic client
ruff check .                     # lint
mypy                             # strict type-check of app/ + wsgi.py

# Run the API (a real key is only needed to serve live queries).
python wsgi.py                   # or: flask --app wsgi run   |   gunicorn wsgi:app
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Tengo un contrato de arriendo y el arrendatario lleva 4 meses sin pagar la renta. ¿Qué acciones legales puedo tomar y cuáles son las implicancias tributarias de las rentas no percibidas?", "session_id": "case-001"}'
```

Response shape:

```json
{
  "query": "...",
  "session_id": "case-001",
  "routing": {
    "selected_specialties": ["litigante", "normativo"],
    "reasoning": "..."
  },
  "specialists": [
    {
      "specialty": "litigante",
      "analysis": "...",
      "citations": [
        {"reference": "ROL-1234-2024", "source": "Corte de Apelaciones de Santiago", "excerpt": "Se condena al arrendatario..."}
      ],
      "succeeded": true,
      "error_message": null
    },
    {
      "specialty": "normativo",
      "analysis": "...",
      "citations": [
        {"reference": "Circular SII N°44-2022", "source": "normativa", "excerpt": "Las rentas de arrendamiento no percibidas..."}
      ],
      "succeeded": true,
      "error_message": null
    }
  ],
  "synthesis": "..."
}
```

`session_id` is optional: send the same value across requests to continue a
conversation; omit it and a UUID4 is generated and echoed back.

### Configuration

All settings are read from the environment; a local `.env` (gitignored) is
loaded automatically at startup via `python-dotenv`, and every variable has a
safe default except the API key. See [`.env.example`](.env.example) for the full
template.

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required to serve queries) | Read directly by the Anthropic SDK |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5-20250929` | Model identifier used by every stage |
| `ROUTER_MAX_TOKENS` | `512` | Max output tokens for the router |
| `SPECIALIST_MAX_TOKENS` | `1024` | Max output tokens per specialist loop call |
| `SYNTHESIZER_MAX_TOKENS` | `2048` | Max output tokens for synthesis |
| `DOCUMENTS_PER_QUERY` | `3` | Default `top_k` for a specialist's search tool |
| `MAX_WORKER_THREADS` | `4` | Upper bound on parallel specialist threads |
| `MAX_TOOL_ITERATIONS` | `4` | Model↔tool round-trips per specialist before a final answer is forced |
| `MAX_HISTORY_MESSAGES` | `6` | Conversation messages retained per session |
| `MAX_QUERY_LENGTH` | `10000` | Max characters of a sanitized query (over this → 400) |
| `FLASK_RUN_HOST` / `FLASK_RUN_PORT` | `127.0.0.1` / `8000` | Bind address for `python wsgi.py` / `flask run` |
| `FLASK_DEBUG` | `0` | Flask debug mode |

`.env` is loaded once at import time in `app/config.py`, before any value is
read, so every entry point (`wsgi.py`, the Flask CLI, pytest, a direct import)
sees the same configuration. Secrets are never hard-coded — `ANTHROPIC_API_KEY`
lives only in the environment, and `.env` is gitignored.

---

## Project layout

Standard Flask layout (official docs: *Application Factories* + *Large
Applications as Packages*): a single application package (`app/`) with the
factory in `__init__.py`, routes in a Blueprint, configuration in `config.py`,
and a `wsgi.py` entry point at the root. The domain layer (orchestrator, agents,
knowledge base) lives as subpackages of `app/`.

```
multi-agent-legal-orchestrator/
├── pyproject.toml              # Build metadata + tool config (ruff, mypy, pytest)
├── wsgi.py                     # Entry point: `app = create_app()` (gunicorn / flask / python)
├── app/                        # The application package
│   ├── __init__.py             # create_app() application factory; registers the blueprint
│   ├── config.py               # Config class + env-driven constants
│   ├── prompts.py              # System prompts (Spanish), one per component
│   ├── conversation.py         # In-memory, per-session conversation history
│   ├── api/
│   │   ├── routes.py           # Blueprint: POST /query, GET /health
│   │   └── validation.py       # Query sanitization + session-id resolution
│   ├── orchestrator/
│   │   ├── models.py           # Typed dataclasses (records, citations, responses)
│   │   ├── router.py           # Routing via structured tool calling
│   │   ├── synthesizer.py      # Merges specialist analyses into one answer
│   │   └── pipeline.py         # Orchestrator: router → parallel specialists → synthesizer
│   ├── agents/
│   │   └── specialist.py       # LitigationAgent + RegulatoryAgent (native tool_use loops)
│   └── knowledge_base/
│       ├── retrieval.py        # Mocked kNN retrieval (production-shaped interface)
│       └── mock_data.json      # Mock corpus: jurisprudencia + normativa (package data)
└── tests/
    └── test_orchestrator.py    # 15 tests, mocked Anthropic client (real SDK block types)
```

---

## Architecture decisions

**Native `tool_use`, no frameworks.** Plain Python objects — router, two
specialists, synthesizer — sequenced by a pipeline. Every Claude interaction is a
direct `client.messages.create(...)`; the control flow is explicit, not hidden
behind a framework state machine.

The orchestration layer is deterministic application code, not an agentic LLM
loop. The model is invoked at three well-defined points — the router (a single
structured routing call), each specialist's bounded native `tool_use` loop, and
the synthesizer (a single call) — while the top-level flow that ties them
together (routing, parallel dispatch, synthesis sequencing, and conversation
persistence) is explicit Python control flow.

**Retrieval is a capability the model uses while reasoning, not a fixed
preprocessing step.** Each specialist exposes one search tool
(`search_jurisprudencia(query, top_k)`, `search_normativa(query)`) and runs the
real Messages API loop: the model emits a `tool_use` block, the backend executes
the search and returns a genuine `tool_result`, and the model continues until it
answers. Because the search is the model's to wield, it derives the query itself
(the user's phrasing is rarely the best query), chooses `top_k`, refines when hits
are weak, and — decisive for a legal tool — observes an empty result and says so
instead of inventing rulings. Nothing is faked; no JSON is parsed out of prose.

**Grounding is mandatory; the loop is bounded.** The first call forces the search
via `tool_choice`, so a specialist never answers from priors — grounding is
required for legal responses. Later iterations are the model's choice (`auto`),
capped by `MAX_TOOL_ITERATIONS`, and the final iteration forces a written answer
(`tool_choice: none`). Agentic flexibility, bounded latency and cost, no "forgot
to search" path.

**Hallucination prevention is structural.** Empty retrieval → empty `tool_result`,
empty citations, and a prompt that requires the specialist to state that nothing
was found. The synthesizer integrates only what was actually grounded;
fabrication is the harder path, not merely a discouraged one.

**Routing is one deterministic call.** The router forces a single
`route_to_specialists` tool whose `specialties` field is an `enum`; the decision
is read from the typed `tool_use` block (no JSON-from-prose). Unknown values are
rejected by the schema; an empty result falls back to all specialists. It
classifies — it does not converse.

**Specialists run in parallel.** Selected specialists execute concurrently in a
`ThreadPoolExecutor` — the right tool for blocking SDK I/O under a synchronous
Flask app — with results collected in routing order.

**Conversation memory: per session, in process.** `ConversationStore` maps
`session_id → messages` under a lock, storing only the user turn and the final
synthesized answer (never tool traces or reasoning), trimmed to
`MAX_HISTORY_MESSAGES`. Prior turns are passed to the router, specialists, and
synthesizer, so a follow-up like *"¿Y si el contrato tiene cláusula de
arbitraje?"* is understood in context. No persistence layer — in-memory is the
right scope here.

**Mocked retrieval, real interface.** `retrieval.py` scores `mock_data.json` by
stopword-filtered lexical overlap behind the exact signatures and return types a
production retriever would expose. Swapping in a real vector kNN search (e.g.
OpenSearch) is a body change, not an interface change.

Supporting engineering decisions:

- **Contained failures.** A specialist that fails records the error on its
  `SpecialistResponse` and returns it — one failure never aborts the run. Router
  and synthesizer failures, plus missing/invalid credentials, funnel into a single
  `OrchestrationError` → a clean **502 with a JSON body**, never a 500/traceback
  (credentials are validated up front, since the SDK only fails deep inside
  `messages.create`).
- **Client built once.** The real client is built lazily and cached behind a lock;
  tests inject a mock. The app boots with no key.
- **Typed against the SDK.** Tool params, message turns, and response blocks use
  the SDK's own types (narrowed with `isinstance`); the package passes
  `mypy --strict`.
- **Boundary hygiene.** Queries are sanitized at the API edge (Unicode NFC,
  control-char stripping, `MAX_QUERY_LENGTH` bound). One `MODEL_IDENTIFIER` in
  `config.py` is shared by every stage. Static system prompts are cached
  (`cache_control: ephemeral`). Standard Flask layout: app factory + blueprint,
  `wsgi.py` entry point.

---

## Tests

`pytest` runs fully offline against a mocked Anthropic client. The fake **drives
the real `tool_use` loop**: on a specialist call it returns a `ToolUseBlock`, and
once it sees the resulting `tool_result` come back it returns a final text block —
exercising the exact control flow the live model would. Response blocks are
constructed from the **real SDK types** (`TextBlock`, `ToolUseBlock`), so the
production `isinstance` narrowing behaves identically to a live response.

The suite covers:

- **Native `tool_use` loops** — both specialists force the search, feed the
  `tool_result` back, and produce a grounded answer; the first call is asserted to
  force the correct tool, and a `tool_result` round-trip is asserted to occur.
- **Empty retrieval / hallucination prevention** — a query with no matching
  records yields zero citations and nothing fabricated.
- **Routing** — the structured decision is read from the `tool_use` block.
- **Parallel multi-agent execution** — both specialists run and the synthesis is
  returned, in routing order.
- **Conversation history** — a second turn receives the first as context, and
  history is retained per session.
- **Retrieval** — relevant queries match; unrelated queries return nothing.
- **HTTP + input validation** — structured JSON response, `session_id`
  auto-generation, query sanitization, and length bounds.

```bash
python -m pytest        # 15 passed
ruff check .            # clean
mypy                    # clean (strict)
```

---

## Scope and boundaries

- **Retrieval is mocked.** The mock preserves the production interface (same
  method signatures and return types); moving to a real vector kNN search is a
  body change, not an interface change.
- **Conversation history is in-process memory by design** — no external store.
  History survives within a running process for the configured window; it does not
  persist across restarts, which matches the scope of this service.
- **No live end-to-end run is bundled here** (no API key in the development
  environment). The full pipeline is exercised against the mocked client in the
  test suite; with a valid key, `python wsgi.py` boots and `POST /query` returns
  the full structured response — the live path is code-complete, not stubbed.
