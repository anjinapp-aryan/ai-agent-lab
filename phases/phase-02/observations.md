# Phase 2 Observations (cumulative; Batch 1)

## Architect questions answerable so far
| Question | Answer from experiments |
|---|---|
| Who creates the schema? | The framework, from signature and docstring. Pydantic AI also sets `additionalProperties:false` and `strict:true`; LangChain does neither. |
| Who validates arguments? | The framework wrapper (Pydantic) before the function runs. |
| What happens to `"10"` for an int? | Coerced to `10` in both (also `10.0`, `True`→`1`); `10.5`, `"hello"`, `None` rejected. |
| Extra arguments? | LangGraph: dropped, tool runs. Pydantic AI: rejected. `**kwargs` flips them. |
| What exactly is retried? | Validation failures are retried by asking the model again (tool does not run). PA has a budget; LG only a step cap. |
| Who owns the error? | The wrapper creates it, the model sees it, the framework decides when to stop. |
| Who validates tool output? | Nobody; serialization is the only gate and differs between frameworks. |
| Who stops runaway tool execution? | Framework limits: PA by calls or requests; LG by supersteps only. |
| Timeout, authorization, idempotency, injection, concurrency, dependency ordering, selection | Not yet (Batches 2-5); Phase 1 covers timeout/concurrency baselines. |
| Can we reconstruct the trajectory? | Phase 1 offline spans yes; Batch 1 used execution logs and wire payloads instead of spans. |

## Surprises (Expected ≠ Actual)
1. `strict: true` goes on the wire with Pydantic AI but not LangChain.
2. `**kwargs` turns Pydantic AI's rejection of extras into pass-through.
3. `True` is accepted as `1` for an `int` parameter.
4. A non-serializable tool result ends a Pydantic AI run before the second model request, but LangGraph sends `repr()`.
5. A Pydantic AI tool-call limit rejects an over-limit batch entirely (0 executions).
6. Neither framework truncates or validates tool results; a 1M-character result goes straight into the model context.

## Method lessons
- A scripted model is not enough to know what the provider receives; a mock transport under the real client is. It exposed the `strict` flag and the serialization difference with zero API cost.
- Hypotheses written first made the surprises countable.

---

# Batch 2 observations

## Architect questions answered by Batch 2
| Question | Answer from experiments |
|---|---|
| Retry: what exactly gets retried? | Depends on who retries: the model re-issues the call (M), LangGraph `RetryPolicy` re-runs the whole node (F), or the tool retries itself (T). |
| Error handling: who owns the error? | The framework wraps it; recoverability is configuration/opt-in; neither knows if the side effect happened. |
| Timeout: what layer owns it? | The framework owns the wait only; sync work continues in its thread, async work is cancelled. |
| Idempotency: who prevents duplicate side effects? | Only the backend/tool, with a stable key committed atomically. Neither framework. |
| Dependency of effect on retry owner | model, framework and caller retries can all duplicate. |

## Surprises (Expected ≠ Actual)
1. A pydantic `ValidationError` raised inside a tool body is a model-visible retry in Pydantic AI (predicted: escape).
2. LangGraph's `RetryPolicy` is inert under the default error handler; it re-runs the whole node, repeating already-successful calls in the batch.
3. A sync tool keeps running after both frameworks report a timeout.
4. `Tool(sequential=True)` accidentally hides a non-atomic guard's race.
5. The model reused the idempotency key 8/8 times but also reused the *same invented key text* across unrelated runs.
6. The wording of an error (`... Please fix your mistakes.` vs a bare `ModelRetry` message that states the order was created) changed whether the model retried.

## Method lessons
- Separating (M)/(F)/(T) retry mechanisms explained what looked like contradictory "the framework retries" claims.
- Setting the side effect *before* the failure in the fixture is what exposes the duplicate; a failure before the effect hides it.
