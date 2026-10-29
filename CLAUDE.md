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
uv run python phase-01/experiments.py parallel-offline|parallel-live 3|timeout|timeout-trace langgraph   # instrumented-tool parallelism/timeout experiments -> results/experiments.json
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

Phase 1 is closed as GO WITH LIMITATIONS on branch `phase-01` (pushed to `origin`; `main` does not exist on the remote). Live runs use Groq `openai/gpt-oss-120b`; Gemini 3.6 tool calls through LangGraph return HTTP 400. Verified: parallel vs sequential tool execution, timeouts, live eval (see `phases/phase-01/phase-1-decision.md`). Still blocked: Langfuse Cloud export (no credentials). `experiments.py` takes optional `tools=`/`step_timeout=`/`run_timeout=` on the agent `run()` functions; defaults leave the agents unchanged. Live tests run whenever the active `.env.<APP_ENV>` has a provider, so `pytest` then uses real API quota.

Phase 2 (tool calling): code in `phase-02/` (`exp_batch*.py`, `common/lab2`), tests in `tests/phase_02/`, docs in `phases/phase-02/`. `uv run python phase-02/exp_batch1.py [exp...]` runs scripted/wire-level experiments (no API quota). `lab2.wire` runs the real provider clients against a mock HTTP transport to capture exact request payloads. Hypotheses are written in `phases/phase-02/experiments.md` before runs; keep Expected vs Actual honest.
Phase 2 Batch 3 (authorization + tool-output security): `phase-02/exp_batch3.py` (scripted tier + `live`), fixtures in `lab2/security.py` (tiny dicts, not an authz framework), docs `phases/phase-02/{authorization,tool-output-security,security-boundaries}.md`. The Phase 1 agents gained optional hook/context arguments (`wrap_tool_call`, `context`, `deps`, `capabilities`); defaults unchanged. On Windows pipe live output with `PYTHONIOENCODING=utf-8` (model replies contain non-cp1252 characters).
Phase 2 Batch 4 (concurrency): `phase-02/exp_batch4.py` (scripted + `live`), timeline fixtures in `lab2/concurrency.py`; use `time.perf_counter()` for ordering (Windows `time.monotonic()` ticks ~15.6 ms). A shared Pydantic AI `Agent(max_concurrency=N)` deadlocked when two threads each called `run_sync` (`phase-02/repro_pa_max_concurrency.py`); use it within one event loop. Run long experiment groups with `python -u` and a `timeout`, never piped through `grep` in the foreground.
Phase 2 Batch 5 (tool selection): `phase-02/exp_batch5.py` (offline groups, `live [exp...]`, `ablation`), fixtures/probe/classifier in `lab2/selection.py`, docs `phases/phase-02/{tool-selection,architecture}.md`. Groq has a 200K tokens/day limit per model that is not shown in response headers; the 120b run hit it, so Batch 5 live used `MODEL_NAME=openai/gpt-oss-20b` (process env overrides `.env.local`). The frameworks send different request envelopes (Pydantic AI adds `additionalProperties:false`/`strict`/`tool_choice`; the Phase 1 Pydantic AI model sends no temperature): compare captured requests before attributing a behavior difference to a framework. `RunResult.ToolCallRecord` has an optional `call_id`.
Phase 2 is closed. Phase 3 (agent loop / ReAct execution): 3.0 reuse audit only, docs in `phases/phase-03/` (no code). Decision: reuse LangGraph + Pydantic AI, reference the other frameworks by source. LangGraph 1.2.18's default `recursion_limit` is 10007 (`langgraph/_internal/_config.py`), so always set it explicitly. Phase 3.1 runs batch by batch on explicit request (`phase-03/exp3_batch*.py`, recorder `phase-03/common/lab3/trajectory.py`, tests `tests/phase_03/`); never start the next batch unprompted. LangGraph succeeds only if `recursion_limit` > super-steps (2r+1 for r tool rounds): at `limit == steps` it raises `GraphRecursionError` after the final answer was produced. Pydantic AI `request_limit` counts model requests, including output re-prompts and `ModelRetry` retries; provider-SDK HTTP retries are invisible to both.
