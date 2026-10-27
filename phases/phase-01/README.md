# Phase 1 — Same Agent, Two Frameworks

A calculator agent (`add`, `subtract`, `multiply`, `divide`) implemented in LangGraph and Pydantic AI with identical tools, instructions, scenarios and scoring.

## Prerequisites
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- A model provider with tool calling and an OpenAI-compatible endpoint (Gemini, OpenRouter, Groq, Ollama Cloud, OpenAI)
- Optional: a free Langfuse Cloud (Hobby) project for traces

## Environment variables
Pick an environment with `APP_ENV` (`local` default, `dev`, `prod`). Copy `.env.<env>.example` to `.env.<env>` (git-ignored; optional shared `.env`) and fill in. Real process env vars always win over files. `prod` refuses to start tracing without Langfuse keys. Example: `APP_ENV=dev uv run python phase-01/run_eval.py`.


| Var | Meaning |
|---|---|
| `MODEL_PROVIDER` | `gemini`, `openrouter`, `groq`, `ollama-cloud`, `openai` (sets base URL) |
| `MODEL_NAME` | model id for that provider |
| `API_KEY` | provider key |
| `BASE_URL` | optional override |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_OTEL_HOST` | Langfuse Cloud tracing (EU default; US host is `https://us.cloud.langfuse.com`) |

Provider base URLs in `lab_common/config.py` were not verified against live endpoints (no keys were available when written) — check them on first run.

## Install
```
uv sync
```
Pinned versions are in `uv.lock`; key ones: langgraph 1.2.18, pydantic-ai-slim 2.62.0, langchain-openai 1.7.3, openinference-instrumentation-langchain 0.1.80, openinference-instrumentation-pydantic-ai 0.1.30, opentelemetry-sdk 1.45.1.

## Run
```
uv run python phase-01/langgraph-agent/lg_agent.py "Calculate 120 + 45 and then multiply the result by 3."
uv run python phase-01/pydantic-ai-agent/pa_agent.py "Calculate 120 + 45 and then multiply the result by 3."
```
(Run from the repo root; each script also enables tracing.)

## Tests
```
uv run pytest tests            # offline mechanics + scorer; live tests auto-skip without .env
uv run pytest tests -m live    # real model, all 6 scenarios x 2 frameworks
```

## Tracing
- With Langfuse keys set, spans are exported via OTLP/HTTP to Langfuse Cloud. Inspect: Langfuse UI → Tracing.
- Without keys: `TRACE_CONSOLE=1` prints raw spans.
- Offline structure demo (no key needed): `uv run python phase-01/trace_demo.py langgraph|pydantic-ai seq|zero|badargs`.

## Evaluation
```
uv run python phase-01/run_eval.py --only=langgraph --trace
uv run python phase-01/run_eval.py --only=pydantic-ai --trace
```
Prints a table and saves trajectories to `phase-01/results/live_results.json` (git-ignored).

## Known limitations
See [phase-1-decision.md](phase-1-decision.md). Main one: live-model runs and the Langfuse export were not executed in the session that built this (no credentials available).
