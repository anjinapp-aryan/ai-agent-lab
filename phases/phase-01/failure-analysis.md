# Failure Analysis

Evidence level: **scripted-model runs** (the "model" emits a fixed tool call, so framework behavior is real and deterministic; model reactions are NOT observed). Reproduce: `uv run pytest tests -q` and `phase-01/trace_demo.py`. Live-model reactions are pending (see phase-1-decision.md).

| # | Failure | LangGraph 1.2.18 | Pydantic AI 2.62.0 |
|---|---|---|---|
| F-01 | Tool raises (`ZeroDivisionError`) | Caught by `ToolNode`; `ToolMessage(status="error", content="Error: ZeroDivisionError('float division by zero')\n Please fix your mistakes.")` is fed to the model; run continues. With `handle_tool_errors=False` the exception escapes `invoke()` and the run ends. | **Exception escapes `run_sync`; run ends**, no further model call. Must opt in by raising `ModelRetry` from the tool or catching outside. |
| F-02 | Invalid args (`b="hello"`) | Pydantic validation inside the tool wrapper → `ToolInvocationError(... b: Input should be a valid number ...)` returned to the model as an error ToolMessage. No retry counter: the model may retry until the step cap. | Validation error returned as `ToolReturnPart(outcome="retried")` with structured error list; **per-tool retry budget** (`retries`, default 1). Exhausted → `UnexpectedModelBehavior`, run ends. |
| F-03 | Division by zero | Same as F-01 | Same as F-01 |
| F-04 | Unavailable capability (weather) | Model-dependent; live (gpt-oss-120b S6): refusal text, 0 tools, no fabrication. | Same |
| F-05 | Model calls non-existent tool (`sqrt`) | `Error: sqrt is not a valid tool, try one of [add, subtract, multiply, divide].` fed back; run continues. | `retried: Unknown tool name: 'sqrt'. Available tools: ...` fed back; counts against retries. |
| F-06 | Tool timeout | Tested (see below). No default timeout; `ToolNode` has none; native `step_timeout` is enforced late on sync `invoke` and on time with `ainvoke`. | Tested (see below). No default timeout; native `Tool(timeout=)` and `run_sync(timeout=)`. |
| F-07 | Model never stops calling tools | `GraphRecursionError` at `recursion_limit` (2*8+1) — caught in our wrapper | `UsageLimitExceeded` at `request_limit=8` — caught in our wrapper |

## Failure path summary
Failure → framework converts to text/part → model sees it on next call (or run ends) → model recovers, retries or explains → final answer or terminated run.

## Trace evidence (span status), from `trace_demo.py`
- LangGraph F-01/F-02: `divide [TOOL] ERROR` span, but root `LangGraph` and later `agent` spans are `OK`; final answer produced. Run completes.
- Pydantic AI F-01: `execute_tool divide ERROR` **and** root `invoke_agent` span `ERROR`; no further `chat` span.
- Pydantic AI F-02: `execute_tool divide ERROR` then another `chat` span, root `OK`.

## Key takeaway
Same user-visible tool failure has opposite defaults: LangGraph = *resilient by default* (errors become model input); Pydantic AI = *strict by default* for tool exceptions (but retry-aware for validation). Whether the model "recovers" is a model property, not a framework property (live S4 on gpt-oss-120b confirmed the difference end to end).

## F-06 Tool timeout (experiment, scripted model, tool sleeps 4s, limit 1s)
Reproduce: `uv run python phase-01/experiments.py timeout` and `... timeout-trace langgraph|pydantic-ai`.

| Case | Wall time | Outcome |
|---|---|---|
| Control, no timeout, LangGraph | 4.01s | waits, completes |
| Control, no timeout, Pydantic AI | 4.03s | waits, completes |
| Pydantic AI `Tool(timeout=1)` | 1.02s | timeout becomes `ToolReturnPart(outcome="retried", "Timed out after 1.0 seconds.")`; model gets another turn; run completes |
| Pydantic AI `run_sync(timeout=1)` | 1.03s | whole run aborted, `RunTimedOut`; no model recovery |
| LangGraph `step_timeout=1`, sync `invoke` | **4.0s** | `TimeoutError` raised only after the tool finished: not enforced in time |
| LangGraph `step_timeout=1`, async `ainvoke` | 1.02s | `TimeoutError` raised, run aborted |
| Tool-level thread wrapper raising `TimeoutError`, LangGraph | 1.02s | error fed back to model, run completes |
| Same wrapper, Pydantic AI | 1.03s | exception escapes, run ends |

Findings
- Neither framework times out a tool by default; a hung tool hangs the run.
- Pydantic AI has a per-tool timeout that is model-visible and recoverable, plus a whole-run timeout. LangGraph has only a graph-step timeout; for the same recovery you write the timeout inside the tool.
- Python cannot kill a running thread: the wrapper abandons the worker, which keeps running (side effects continue). Real cancellation needs async tools or process isolation (Phase 2/6).
- Trace: the timed-out tool span is `ERROR` with an `exception` event, duration ~1.0s in both. Root stays OK when the error is fed back.
- Only scripted models were used for timeouts; live-model reaction to a timeout message was not measured.
