# Idempotency and Duplicate Calls (Experiments 8 and 9)

```
Agent/framework -> tool call -> idempotency key -> backend/tool boundary -> side effect
   (retry owner)                 (must be stable)     (checks + commits atomically)
```

## Framework retry != business idempotency
- A retry mechanism answers "should I attempt this call again?" (model, framework, tool, or the caller).
- Idempotency answers "if I attempt it again, does the effect happen only once?".
- Measured: neither LangGraph nor Pydantic AI deduplicates by arguments or call id; every retry owner duplicated a side effect that had already happened ([experiments.md](experiments.md), B2.3).

## What the experiments show
| Situation | Orders created |
|---|---|
| same tool/args twice (sequential or one batch), no key | 2 |
| lost response, model re-calls, no key | 2 (both frameworks) |
| lost response, LangGraph `RetryPolicy` (no model call), no key | 2 |
| lost response, Pydantic AI plain exception | 1, run dead; caller re-run → 2 |
| exact duplicate with key | 1 |
| lost response + same key | 1 |
| lost response + key changed by the model | 2 |
| same key, different args | 1 + `ConflictError` |
| different key, same args | 2 |
| two identical keyed calls in one batch, non-atomic guard | 2 (race) |
| same, `Tool(sequential=True)` | 1 |

## Design rules supported by the evidence
1. The key is created outside the model (request id, business id). Models can reuse a key (8/8 live) but can also produce the same text for unrelated orders.
2. The backend commits the effect and the key atomically; the guard here is deliberately not (it lost the race).
3. A reused key with different arguments must be an explicit conflict, not a silent replay.
4. A timeout or a lost response means "unknown outcome": retry only against an idempotent operation.
5. Framework retry policies (`RetryPolicy`, `ModelRetry`, caller retries) do not know the effect state; they are safe only on idempotent tools.

The guard is the approved ~10-line fixture inside the simulated `create_order`; no idempotency service or lock manager was built. Production alternatives (durable execution, DB unique constraints) remain Phase 6 material.
