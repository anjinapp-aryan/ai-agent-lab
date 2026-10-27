# Failure Analysis

Evidence level: **scripted-model runs** (the "model" emits a fixed tool call, so framework behavior is real and deterministic; model reactions are NOT observed). Reproduce: `uv run pytest tests -q` and `phase-01/trace_demo.py`. Live-model reactions are pending (see phase-1-decision.md).

| # | Failure | LangGraph 1.2.18 | Pydantic AI 2.62.0 |
|---|---|---|---|
| F-01 | Tool raises (`ZeroDivisionError`) | Caught by `ToolNode`; `ToolMessage(status="error", content="Error: ZeroDivisionError('float division by zero')\n Please fix your mistakes.")` is fed to the model; run continues. With `handle_tool_errors=False` the exception escapes `invoke()` and the run ends. | **Exception escapes `run_sync`; run ends**, no further model call. Must opt in by raising `ModelRetry` from the tool or catching outside. |
| F-02 | Invalid args (`b="hello"`) | Pydantic validation inside the tool wrapper → `ToolInvocationError(... b: Input should be a valid number ...)` returned to the model as an error ToolMessage. No retry counter: the model may retry until the step cap. | Validation error returned as `ToolReturnPart(outcome="retried")` with structured error list; **per-tool retry budget** (`retries`, default 1). Exhausted → `UnexpectedModelBehavior`, run ends. |
| F-03 | Division by zero | Same as F-01 | Same as F-01 |
| F-04 | Unavailable capability (weather) | Model-dependent; with scripted text reply: 1 LLM call, 0 tools, END. Live behavior: not yet observed. | Same |
| F-05 | Model calls non-existent tool (`sqrt`) | `Error: sqrt is not a valid tool, try one of [add, subtract, multiply, divide].` fed back; run continues. | `retried: Unknown tool name: 'sqrt'. Available tools: ...` fed back; counts against retries. |
| F-06 | Tool timeout | **Not tested.** Neither framework imposes a tool timeout by default (not verified in source; no test written). Phase 2/6. | Same |
| F-07 | Model never stops calling tools | `GraphRecursionError` at `recursion_limit` (2*8+1) — caught in our wrapper | `UsageLimitExceeded` at `request_limit=8` — caught in our wrapper |

## Failure path summary
Failure → framework converts to text/part → model sees it on next call (or run ends) → model recovers, retries or explains → final answer or terminated run.

## Trace evidence (span status), from `trace_demo.py`
- LangGraph F-01/F-02: `divide [TOOL] ERROR` span, but root `LangGraph` and later `agent` spans are `OK`; final answer produced. Run completes.
- Pydantic AI F-01: `execute_tool divide ERROR` **and** root `invoke_agent` span `ERROR`; no further `chat` span.
- Pydantic AI F-02: `execute_tool divide ERROR` then another `chat` span, root `OK`.

## Key takeaway
Same user-visible tool failure has opposite defaults: LangGraph = *resilient by default* (errors become model input); Pydantic AI = *strict by default* for tool exceptions (but retry-aware for validation). Whether the model "recovers" is a model property, not a framework property — to be measured live.
