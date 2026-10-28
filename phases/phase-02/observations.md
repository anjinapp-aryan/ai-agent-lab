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

---

# Batch 3 observations

## Architect questions answered by Batch 3
| Question | Answer from experiments |
|---|---|
| Where does authorization live? | Application layer, enforced at a boundary every tool call crosses (framework hook or tool body), with the data layer as last line. |
| Can the LLM be trusted for authorization? | No: prompt rules are not a boundary; live injection obeyed 20/20. |
| Can tool arguments carry identity? | No: argument-supplied identity is a bypass. |
| Tenant/resource authorization? | Needs a resource check beyond the tool permission. |
| What does framework retry do to a denial? | `RetryPolicy` retried it 3×; model retry loops re-issue it; Pydantic AI plain exception ends the run. |
| Is tool output trusted? | Frameworks deliver it unchanged as context; the model may obey it. |
| Does either framework defend against injection? | No. |
| Can authorization stop an injected second call? | Yes for unauthorized actions (0 effects), not for authorized ones. |

## Surprises (Expected ≠ Actual)
1. The model obeyed injected prose despite an explicit "only do what the user asked" instruction, 20/20.
2. The same intent as a JSON `next_action` field (and as a JSON string) was followed 0/20.
3. In Pydantic AI a correct denial ended the run: injection became denial-of-service for a harmless request.
4. "Do not tell the user" was obeyed: a denied attempted refund was not disclosed in LangGraph's answers.
5. My fixture's denial message leaked the other tenant's name.
6. LangGraph's safety against forged identity args rests on the "drop extras" default, which differs from Pydantic AI's explicit rejection.

## Method lessons
- Counting `authz_checks == tool calls` turned "every call is authorized" into a testable claim.
- Separating scripted (does the boundary hold?) from live (does the model attempt the action?) kept framework and model findings apart.

---

# Batch 4 observations

## Architect questions answered by Batch 4
| Question | Answer from experiments |
|---|---|
| When do tools run concurrently? | Only when the model batches them; then by default in both frameworks (threads for sync, one event loop for async). |
| Who controls concurrency? | Model (batching), framework (whether/limits), runtime (threads vs loop). |
| Is result order guaranteed? | Model sees call order; completion order differs. |
| Can shared state race / does async help? | Yes / no (only between suspension points). |
| Who prevents duplicate effects under concurrency? | An atomic business/data guard; sequential execution only hides the race. |
| What happens to siblings on failure / cancel? | Sync siblings always finish and commit; async siblings are cancelled when the failure escapes. |
| Who enforces dependency order? | The model's turn-taking or the application; never the framework. |
| Can identity leak across concurrent requests? | Yes with global/shared identity; no with per-request or framework per-call context. |

## Concurrency guarantees by layer (question 17)
| Layer | Guarantees observed | Does not guarantee |
|---|---|---|
| LLM | whether calls are batched; turn-by-turn dependency handling (10/10 live) | batching independent calls (0/10 live) |
| Framework | runs a batch together; result order = call order; limits (LangGraph sync `max_concurrency`, Pydantic AI `sequential`) | dependency awareness, shared-state safety, dedup, rollback, identity isolation |
| Runtime | threads (parallel, uncancellable) or event loop (concurrent, cancellable) | atomicity |
| Tool | whatever it implements (locks, async) | anything about other tools |
| Application | request-scoped identity, locks, atomic idempotency | — |
| Database | (not tested here; the atomic boundary in production) | — |

## Surprises (Expected ≠ Actual)
1. Windows `time.monotonic()` (~15.6 ms ticks) made start order look like completion order until switched to `perf_counter()`.
2. Pydantic AI `run_sync` returned in ~9 ms after a sibling failure while the other sync tools committed effects later.
3. Escaping failures cancel async siblings in both frameworks.
4. A shared `Agent(max_concurrency=1)` deadlocked across threads calling `run_sync`.
5. LangGraph `invoke` turns async-only tools into silent error messages.
6. The live model never batched independent calls, so framework parallelism was unused in practice.
7. A placeholder fixture result ("get_time done") made one model loop to the step cap and the other invent times; those runs were discarded.

## Method lessons
- Use a high-resolution clock for ordering; verify suspicious orderings before believing them.
- A no-framework control (raw threads, plain `asyncio.gather`) separated "framework causes it" from "the runtime does it".
- Fixtures must return realistic values in live trials, or the model's behavior measures the fixture.
