# Phase 1 GO / NO-GO (final)

Verdict: **GO WITH LIMITATIONS**

Provider/model for every live result: Groq `openai/gpt-oss-120b` (same for both frameworks).

| Criterion | Status | Evidence |
|---|---|---|
| Same agent implemented in both frameworks | PASS | `lg_agent.py`, `pa_agent.py` |
| Same tools | PASS | single `lab_common/tools.py`, wrapped natively |
| Same model | PASS | Groq gpt-oss-120b for all live runs |
| Real model execution | PASS | live eval 11/12 plus live pytest; S4 on Pydantic AI asserts the documented escape |
| Sequential tool calls | PASS | live S2 add->multiply on both |
| Independent tool behavior understood | PASS | measured: frameworks parallelize batched calls (1.5s overlap); live model never batched (6/6 sequential); `Tool(sequential=True)` serializes |
| Tool error behavior understood | PASS | live S4, failure-analysis.md F-01 |
| Invalid argument behavior understood | PASS with caveat | offline mechanics verified; live model never produced invalid args (S5 refused in text) |
| Timeout behavior understood | PASS | experiments.py timeout, F-06 (scripted model; live-model reaction unmeasured) |
| State understood | PASS | observations.md, architecture.md |
| Agent loop understood | PASS | framework-free-mental-model.md, comparison.md |
| Langfuse export verified | **BLOCKED** | no `LANGFUSE_*` credentials anywhere; nothing was sent to Langfuse |
| Offline tracing verified | PASS | in-memory OTel+OpenInference span trees, both frameworks, including timeout/error spans |
| Evaluation implemented | PASS | pytest scorer, `run_eval.py` |
| Failure injection completed | PASS | F-01..F-07 incl. timeout; live: exception, unknown capability |
| Reuse rule followed | PASS | decision-record.md; no new dependencies in closure |
| Documentation complete | PASS | all required files updated |
| Tests pass | PASS | 43 passed, 0 failed, 0 skipped (live tests included) |

## Why not GO
Langfuse Cloud delivery and trace rendering (input, model, tool args/results, latency in the UI) remain unverified because credentials are unavailable. Nothing was fabricated; offline span structure is the only trace evidence.

## Remaining limitations
1. Langfuse export unverified (needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env.dev` or `.env.prod`, then `APP_ENV=dev uv run python phase-01/run_eval.py --only=<fw> --trace`). OTLP endpoint/auth format is from memory.
2. One live model (gpt-oss-120b), one run per eval cell, 3 runs per parallelism cell. Other models may batch tool calls or call tools with bad arguments.
3. Timeout reaction of a real model not measured; thread-based timeout leaves the worker running.
4. Gemini 3.6 tool calls through LangGraph returned HTTP 400 (cause unconfirmed).
5. Transitive dependency licenses unaudited; Langfuse Hobby limits unchecked.

## Final report answers
1. **Learned:** loop, state, error and timeout policy are the architecture; frameworks differ in defaults and what they hide.
2. **Agent:** goal + tools + observations fed back + model-driven decisions + state + termination.
3. **Minimum architecture:** framework-free-mental-model.md.
4. **LangGraph:** explicit `StateGraph(MessagesState)`, `ToolNode`, errors fed back, `recursion_limit`, `step_timeout`.
5. **Pydantic AI:** `Agent(tools, retries)`, hidden loop, exceptions escape, `Tool(timeout/sequential)`, `UsageLimits`.
6. **Abstractions:** comparison.md.
7. **Tool failures:** failure-analysis.md.
8. **State:** message list; no persistence across runs.
9. **Trajectory:** reconstructable from spans offline; cloud unverified.
10. **Evaluation:** evaluation.md results tables.
11. **Choice:** LangGraph for explicit control/state/HITL; Pydantic AI for typed, validated agents with built-in retry/timeout knobs.
12. **Framework-independent:** structured action request -> execute -> observe -> repeat; budget; error/timeout policy; model-vs-scheduler split on parallelism.
13. **Not built:** provider layer, tracer, eval framework, tool registry, retry/timeout framework.
14. **Phase 2:** Deep Tool Calling (schemas, structured outputs, retries, parallel-call control, permissions, idempotency, cancellation, injection via tool output). Recommended only; not started.
