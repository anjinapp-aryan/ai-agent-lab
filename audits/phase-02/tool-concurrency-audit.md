# Concurrency, Limits, Cancellation (capability I)

## Already proven in Phase 1 (do not repeat as discovery)
- Framework schedules parallel execution when the model batches calls: both frameworks overlapped 1.5s/1.5s (scripted).
- `gpt-oss-120b` issued one call per turn in 6/6 live runs, so live execution was sequential across turns.
- `Tool(sequential=True)` (Pydantic AI) serializes.
- Timeouts: no default; Pydantic AI `Tool(timeout)` (model-visible retry) and `run_sync(timeout)` (abort); LangGraph `step_timeout` is late on sync, on time with `ainvoke`.

## Existing mechanisms to reuse
| Mechanism | Project | License | Evidence | Use |
|---|---|---|---|---|
| `UsageLimits(tool_calls_limit, request_limit, ...)` | Pydantic AI | MIT | fields present `[V]` | Exp 14 |
| `Agent(tool_timeout=...)`, `Tool(timeout=)` | Pydantic AI | MIT | params present `[V]` | Exp 7 |
| `recursion_limit`, `step_timeout`, `Send` (fan-out) | LangGraph | MIT | `Send` imports `[V]`; step_timeout Phase 1 `[V]` | Exp 7, 13, 14 |
| `ToolCallLimitMiddleware` / `ModelCallLimitMiddleware` | langchain `agents/middleware/tool_call_limit.py`, `model_call_limit.py` | MIT | files exist `[F]` | COMPOSE-candidate for Exp 14 (adds `langchain`) |
| AnyIO task groups, cancel scopes | agronholm/anyio | MIT, 4.15.1, pushed 10-26 | `[V]` health; installed as a transitive dependency (4.15.1) `[V]` | REFERENCE for real cancellation (async tools) |
| Durable cancellation/retry | Temporal, DBOS | MIT | `[V]` | REFERENCE (Phase 6) |

## Gaps to investigate (not assumed)
- **Dependency-aware execution:** neither framework appears to plan a DAG of tool calls; ordering is whatever the model emits per turn. Exp 13 tests whether the model sequences A->B across turns and whether the frameworks do anything when A's output is B's input in the same batch.
- **Concurrency limits** (max N tools at once): no per-framework knob was found in signatures; verify in Exp 13/14. If absent, state it as a finding; do not build a scheduler.
- **Cancellation:** Python threads cannot be killed (Phase 1). Real cancellation needs async tools (AnyIO/asyncio) — Exp 7 should include one async tool as a controlled comparison if cheap; otherwise record as limitation.

Build: none. Test harness = Phase 1 `experiments.py` pattern (monotonic timestamps) adapted for three tools.
