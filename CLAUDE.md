# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A learning lab for AI agents, not a framework. Governing rule (see `docs/reuse-policy.md`): **REUSE → ADAPT → COMPOSE → BUILD**. Before adding any capability or dependency, check the Phase 0 audit (`audits/phase-00/final-decision.md`) and do a focused license/maintenance check for anything not approved there. A BUILD needs a written decision record (see `phases/phase-01/decision-record.md` for the format). Do not build tracing, eval frameworks, agent frameworks, memory/vector stores, MCP/A2A implementations or tool registries. Phases are sequential; do not start the next phase unprompted (`ROADMAP.md`).

Docs state evidence levels (`[V]` verified, `[S]` secondary, `[K]` from memory). Keep that honesty: mark unobserved results as pending rather than inferring them, and prefer observed behavior over reading framework source.

## Commands

Python 3.12, managed with `uv`. Run from repo root.

```
uv sync
uv run pytest tests -q                                   # offline; live tests auto-skip without a model key
uv run pytest tests/phase_01/test_pydantic_ai_agent.py::test_unknown_tool_name_fed_back_not_crash   # single test
uv run pytest tests -m live                              # real model, 6 scenarios x 2 frameworks
uv run python phase-01/run_eval.py --only=langgraph --trace   # or --only=pydantic-ai; writes phase-01/results/live_results.json
uv run python phase-01/trace_demo.py langgraph|pydantic-ai seq|zero|badargs   # span trees, no key needed
uv run python phase-01/langgraph-agent/lg_agent.py "question"
```

No linter or build step is configured. Set `PYDANTIC_AI_NO_BANNER=1` to silence Pydantic AI's startup banner.

## Architecture

**Same agent, two frameworks, one comparison harness.** Everything that must be identical lives in `phase-01/common/lab_common/` (tools, system instructions, `MAX_STEPS`, scenarios, scorer, config, tracing). `phase-01/langgraph-agent/lg_agent.py` and `phase-01/pydantic-ai-agent/pa_agent.py` only wrap those plain functions with each framework's native mechanism and both expose `run(question, model=None, ...) -> RunResult`. Never add framework-specific tools, prompts or scenarios; that contaminates the comparison.

`RunResult`/`ToolCallRecord` (`dataset.py`) is the framework-neutral trajectory. Each agent has a `_fill` function that extracts it from native message types, so framework message-format changes break there (e.g. Pydantic AI v2 reports failed/retried tools as `ToolReturnPart.outcome != "success"`, not `RetryPromptPart`). Escaped exceptions are caught in `run` and stored in `RunResult.error`.

**Imports:** `phase-01/*` directories are hyphenated and not packages. Imports work via `pythonpath` in `pyproject.toml` (pytest) and a `sys.path` bootstrap at the top of the agent scripts and `run_eval.py`. New modules go in those directories and are imported as top-level names (`lg_agent`, `pa_agent`, `lab_common.*`).

**Testing has two tiers.** Offline tests use scripted models (`tests/phase_01/conftest.py`: `ScriptedChatModel` for LangChain, `FunctionModel` for Pydantic AI) driven by a list of `("call", [(tool, args)])` / `("text", str)` steps; they verify framework mechanics only. `@live` tests (skip without configured provider) verify real model behavior. Do not claim model-behavior results from offline tests.

**Config.** `lab_common/envfile.py` loads `.env.<APP_ENV>` (`local` default, `dev`, `prod`) then optional `.env`; real process env wins. Model comes from `MODEL_PROVIDER` / `MODEL_NAME` / `API_KEY` (+ optional `BASE_URL`) through OpenAI-compatible endpoints for both frameworks (`config.py` holds the base-URL table). `tracing.py` sends OpenTelemetry + OpenInference spans to Langfuse Cloud via OTLP/HTTP; `APP_ENV=prod` refuses to start tracing without Langfuse keys. Only `*.example` env files are tracked; real `.env*` files are git-ignored.

Docs in `phases/phase-01/` (comparison, failure-analysis, observations, evaluation, phase-1-decision) record findings; update them when behavior is newly observed, especially the `[live pending]` items.

## State

Local git repo on branch `phase-01`, remote `origin` = github.com/anjinapp-aryan/ai-agent-lab (was empty), nothing committed yet. Live eval ran on Groq (openai/gpt-oss-120b; Gemini 3.6 tool calls via LangGraph returned 400); check `phases/phase-01/phase-1-decision.md` for current limitations.
