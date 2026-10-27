# Retry, Error Ownership and Limits (Batch 1 portion: Experiments 2 and 14)

Experiment 6 (tool exceptions) and 7 (timeouts) are Batch 2; Phase 1 already established the base behavior.

## What gets retried (validation errors)
- A failed validation is **not** a retry of the tool: the tool never ran. The framework turns the error into a message and asks the model to try again.
- **LangGraph**: no retry budget. A model that keeps sending a bad call is bounded only by `recursion_limit`: 9 model calls and `GraphRecursionError` at the default 17; 0 tool executions.
- **Pydantic AI**: a per-tool budget (`retries`, default 1): 2 model calls then `UnexpectedModelBehavior`; with `retries=3`, 4 model calls. 0 tool executions.
- **Who owns the error?** Framework wrapper for validation; the model decides what to do next; the framework (budget in PA, step cap in LG) decides when to stop.

## Limits on runaway tool use (Experiment 14)
| Limit | Owner | Unit | Observed |
|---|---|---|---|
| `UsageLimits(tool_calls_limit=3)` | Pydantic AI | tool calls | 3 executed, 4th refused before running; `UsageLimitExceeded` |
| `UsageLimits(tool_calls_limit=2)` with a batch of 3 | Pydantic AI | tool calls | **0 executed** (whole batch refused) |
| `UsageLimits(request_limit=4)` | Pydantic AI | model requests | 4 executions, then error |
| Phase 1 default `request_limit=8` | Pydantic AI | model requests | 8 executions, error |
| `recursion_limit` 5 / 9 / 17 | LangGraph | graph supersteps | 2 / 4 / 8 executions, `GraphRecursionError` |
| tool-call count limit | LangGraph | — | **does not exist** (batch of 3 ran all 3) |
| time | `run_sync(timeout)` (PA), `step_timeout` (LG) | seconds | Phase 1 |

- **Who stops runaway tool execution?** The framework, with different units; Pydantic AI has the finer control (per tool-call count), LangGraph only a step count.
- **State after the stop:** Pydantic AI messages remain obtainable with `capture_run_messages` (9 messages at limit 3). LangGraph's error carries no state; a checkpointer might preserve it (not tested).
- A `GraphRecursionError`/`UsageLimitExceeded` is an exception to the caller, not a message to the model: the model gets no chance to wrap up. Our `run()` wrappers convert it to `RunResult.error`.

---

# Batch 2 additions: exceptions, three retry mechanisms, ownership

## Exceptions
| Config | Behavior |
|---|---|
| LangGraph default | every exception class (`ValueError`, `RuntimeError`, `TimeoutError`, pydantic `ValidationError`, `ToolException`, even `ModelRetry`) becomes an error ToolMessage; run continues |
| LangGraph `handle_tool_errors=False` | every class escapes and ends the run |
| LangGraph `handle_tool_errors=(ValueError,)` | only `ValueError`; pydantic `ValidationError` (wrapped as `ToolInvocationError`) is not matched |
| Pydantic AI default | `ValueError`, `RuntimeError`, `TimeoutError`, `ToolException` escape; pydantic `ValidationError` and `ModelRetry` are model-visible |

Neither framework classifies errors as recoverable vs terminal automatically; the decision is configuration (LangGraph) or opt-in exception type (Pydantic AI).

## Who retries? (keep these apart)
- (M) model-requested: the model sees the error and calls again. LangGraph: no budget (step cap). Pydantic AI: `retries`/`ModelRetry` budget.
- (F) framework-automatic: LangGraph `RetryPolicy` re-runs the *node* without a model call, but only if the error handler does not swallow the exception first, and only for `retry_on` classes (default skips `RuntimeError`/`ValueError`). Pydantic AI has no equivalent for plain exceptions.
- (T) tool-internal: invisible to the framework (3 runs, 2 model calls).
- Caller-level: re-running the whole run (needed for Pydantic AI plain exceptions).

Who owns the error: the framework turns it into a message or an exception; the model or the caller decides what happens next. Who owns the *effect state*: nobody except the backend.
