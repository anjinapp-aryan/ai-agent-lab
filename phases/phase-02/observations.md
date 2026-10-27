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
