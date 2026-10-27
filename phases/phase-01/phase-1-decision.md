# Phase 1 GO / NO-GO

Verdict: **GO WITH LIMITATIONS**

| Criterion | Status | Evidence |
|---|---|---|
| Same agent implemented in both frameworks | Done | `phase-01/langgraph-agent/lg_agent.py`, `phase-01/pydantic-ai-agent/pa_agent.py` |
| Same tools | Done | single `lab_common/tools.py`, wrapped natively by each |
| Same tasks | Done | `lab_common/dataset.py` (6 scenarios), shared scorer |
| Tool calling works | Done | live S1-S3 pass on both frameworks (Groq gpt-oss-120b) |
| Sequential tool calls work | Done | live S2: add→multiply on both |
| Failure scenarios tested | Partial | F-01,02,03,05,07 observed offline; F-04 weather refusal observed live (no fabrication); **F-06 timeout not tested** |
| State understood | Done | observations.md §State, architecture.md |
| Agent loop understood | Done | framework-free-mental-model.md, comparison.md abstraction table |
| Tracing works | Partial | OTel+OpenInference span trees verified in-memory for both frameworks; **Langfuse Cloud export not run (no keys)** |
| Evaluation works | Done | `run_eval.py` live table, 11/12 (see evaluation.md) |
| Framework comparison complete | Done, with `[unobserved]` cells | comparison.md |
| Reuse rule followed | Done | decision-record.md; one tiny BUILD (D4) documented |
| Documentation complete | Done | all 8 required files present |
| Tests pass | Pass | `24 passed, 12 skipped` (skips = live tests) |

## Limitations (not hidden)
1. Live run done once (Groq `openai/gpt-oss-120b`, 11/12 pass, single run per cell, no repeats). Other models unobserved; S5 never exercised live argument validation; parallel-call behavior not inspected.
2. Langfuse Cloud delivery and UI not verified; OTLP endpoint/auth format from memory.
3. Provider base URLs unverified live.
4. F-06 (timeout) untested; Hobby-plan limits unchecked; transitive licenses unaudited.
5. Not a git commit: work is on local branch `phase-01`, nothing committed or pushed.

## To close the limitations
Fill `.env`, then `uv run pytest tests -m live`, `uv run python phase-01/run_eval.py --only=langgraph --trace` and `--only=pydantic-ai --trace`, and inspect both traces in Langfuse; update evaluation.md and observations.md `[live pending]` items.

## Final report
1. **Learned**: the loop, state and error policy are the architecture; frameworks differ mainly in defaults (LangGraph resilient on tool exceptions, Pydantic AI strict) and visibility (explicit graph vs hidden loop).
2. **Agent**: goal + tools + observations fed back + model-driven decisions + state + termination.
3. **Minimum architecture**: see framework-free-mental-model.md.
4. **LangGraph**: `StateGraph(MessagesState)`, `agent` node, `ToolNode`, `tools_condition`, `recursion_limit`.
5. **Pydantic AI**: `Agent(model, instructions, tools, retries)`, internal loop, `UsageLimits`.
6. **Abstractions**: comparison.md table.
7. **Tool failures**: failure-analysis.md.
8. **State**: messages list; explicit vs implicit; nothing persists across runs.
9. **Trajectory**: 14 (LangGraph) vs 6 (Pydantic AI) spans for S2, one trace each.
10. **Evaluation**: pytest + scorer; live results pending.
11. **Choice**: LangGraph for control-flow/stateful/HITL; Pydantic AI for typed, validated, test-friendly agents.
12. **Framework-independent**: model-requests-action → execute → observe → repeat; budget; error policy.
13. **Not built**: provider layer (reused OpenAI-compatible clients), tracing, eval framework, tool registry, retry logic.
14. **Phase 2**: Deep Tool Calling — schemas, structured outputs, retries, parallel calls, permissions, idempotency, timeouts, injection via tool output; recommended only, not started.
