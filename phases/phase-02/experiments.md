# Phase 2 Experiments

Model for any live run: Groq `openai/gpt-oss-120b` (same for both frameworks). Scripted tier first. Hypotheses below were written **before** the experiments were run; results are recorded beside them and any mismatch is called out, never edited away.

## Tier classification (all 15)
| # | Experiment | Tier | Plan |
|---|---|---|---|
| 1 | Simple typed tool | A | scripted |
| 2 | Missing argument | A (C follow-up: does a real model recover from the feedback text?) | scripted first |
| 3 | Wrong type | A (C follow-up as 2) | scripted first |
| 4 | Extra argument | A | scripted |
| 5 | Invalid enum | A (C follow-up as 2) | scripted first |
| 6 | Tool exception | C | scripted, then live if interesting |
| 7 | Timeout | A | scripted (Phase 1 baseline reused) |
| 8 | Duplicate call | A (+B: does a real model duplicate?) | scripted first |
| 9 | Side-effecting create_order | A | scripted |
| 10 | Authorization | A (+B: model reaction to denial) | scripted first |
| 11 | Result validation | A | scripted |
| 12 | Tool-output injection | B | live (simulated) |
| 13 | Parallel and dependent calls | C | scripted, then live |
| 14 | Tool-call limits | A | scripted |
| 15 | Tool selection | B | live, 5 trials per framework |

Note on the "C follow-up" for 2/3/5: a real model almost never emits an invalid call on demand (Phase 1 S5), so the live question is only "given the framework's error text, does the model self-correct?" That needs a replay-then-live hybrid model. It is deferred and will be run only if the scripted results make it informative.

## Batch 1 — pre-registered hypotheses
Tools: `calculate(a: int, b: int) -> int`, `convert(amount: float, currency: Literal["USD","EUR","INR"])`.

**E1 Simple typed tool.** H: both build JSON Schema from type hints and use the docstring as description; Pydantic AI marks the schema `additionalProperties: false`, LangChain does not; args arrive as ints; result handed back to the model: LangGraph stringifies (`"5"`), Pydantic AI keeps a typed value and serializes later.

**E2 Missing argument.** H: both detect before calling the tool (tool never runs) and feed an error to the model. One bad call then a good call: both recover. Persistent bad calls: Pydantic AI stops after its retry budget (`UnexpectedModelBehavior`); LangGraph has no budget and stops only at `recursion_limit` (`GraphRecursionError`).

**E3 Wrong type.** H: Pydantic lax coercion in both: `"10"`→10 and `10.0`→10 accepted; `"hello"`, `10.5`, `None`, `[1]` rejected; `True` accepted as 1. Identical in both frameworks (same validator).

**E4 Extra argument.** H (from audit PoC): LangGraph executes and drops the extra; Pydantic AI rejects. LangChain `args_schema` with `extra="forbid"` makes LangGraph reject. A function with `**kwargs`: extras may be passed through (unknown, to be observed).

**E5 Invalid enum.** H: `"XYZ"` and lowercase `"usd"` rejected by both; the error lists allowed values; a Python `Enum` class behaves like `Literal`.

**E11 Result validation.** H: neither framework validates a return value against its annotation (a function annotated `-> int` that returns `"abc"` is accepted); `None` and dicts are serialized; non-JSON-serializable objects are stringified or raise; a 1,000,000-character result is passed to the model whole with no truncation.

**E14 Tool-call limits.** H: Pydantic AI `UsageLimits.tool_calls_limit` and `request_limit` raise `UsageLimitExceeded`; LangGraph has no tool-call-count limit, only `recursion_limit` (`GraphRecursionError`). With a model that batches 3 calls and a limit of 2, Pydantic AI refuses before executing any or executes only 2 (unknown which). Partial results are lost on error unless messages are captured separately (Pydantic AI `capture_run_messages`).

## Batch 1 results (2026-10-27)
Common setup: LangGraph 1.2.18 and Pydantic AI 2.62.0 via the Phase 1 agents (`lg_agent.run`, `pa_agent.run`); model = Phase 1 **scripted model** (no live model, so no quota used); the wire-level rows use the real `ChatOpenAI` / `OpenAIChatModel` clients against a mock HTTP transport (`lab2/wire.py`) and record the exact request JSON. Tracing: not captured in Batch 1 (the evidence is tool execution logs, framework error text and wire payloads). Timing is negligible for all but E14 (milliseconds). Reproduce: `uv run python phase-02/exp_batch1.py [exp...]`; pinned as `tests/phase_02/test_batch1_tool_mechanics.py` (57 tests).

**Where Expected ≠ Actual** (hypothesis was wrong or incomplete):
1. E1: I expected Pydantic AI to keep a typed tool result while LangGraph stringifies. On the wire both send the string `"5"`; any difference is internal.
2. E1: not predicted — Pydantic AI sends `"strict": true` on the tool definition (the audit had marked strict mode as unknown); LangChain sends no `strict`.
3. E4: for a `**kwargs` function, LangGraph drops the extra and the function receives `{}`; Pydantic AI **passes it through** (`{'secret': 'unexpected'}`). This was listed as unknown.
4. E11: `None` and `NaN` serialize differently (`"null"`/`"NaN"` vs `""`/`"null"`), and an unserializable object **aborts** the Pydantic AI run before the second model request. I had only predicted "stringified or raises".
5. E14: unpredicted exactly how Pydantic AI's limit behaves: the limit is checked before executing, and a batch that would exceed it executes **none** of its calls.

### E1 Simple typed tool — hypothesis mostly confirmed
- Input: `calculate(a:int,b:int)`, call `{a:2,b:3}`. Expected: schema from hints, docstring as description. Actual: both send `{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}`; Pydantic AI adds `"additionalProperties": false` and `"strict": true`. Args arrive as `int`; result `5` reaches the model as `"5"` in both.
- Interpretation: the schema is created by the framework, not the model or the provider; the two frameworks make different choices about closing it and about asking the provider for strict decoding.

### E2 Missing argument — confirmed
- Expected: detected before execution, fed back. Actual: tool body ran 0 times on the bad call; one bad call then a good call recovers in both (3 model calls). Persistent bad calls: LangGraph made 9 model calls and stopped on `GraphRecursionError` (no retry budget); Pydantic AI stopped after 2 model calls (`retries=1`) and 4 (`retries=3`) with `UnexpectedModelBehavior`. Executions: 0.
- Interpretation: detection is the framework wrapper's job in both. Retry ownership differs: Pydantic AI owns a per-tool budget; LangGraph owns only a global step limit.

### E3 Wrong type — confirmed (+1)
- Accepted (coerced to `int`): `"10"`, `10.0`, `True` (→ `1`). Rejected: `"hello"`, `10.5`, `None`, `[1]`, and also `"1e2"` (added). Identical in both frameworks.
- Interpretation: the validator is Pydantic lax mode in both; "wrong type" is only wrong if it cannot be coerced. `True` becoming `1` is a notable surprise to design around.

### E4 Extra argument — confirmed, plus new finding
- LangGraph default: **executed, extra silently dropped**. Pydantic AI default: **rejected** (`extra_forbidden`, error fed back). LangGraph with a Pydantic `args_schema` using `extra="forbid"`: rejected. `**kwargs` function: LangGraph drops, Pydantic AI passes through.
- Interpretation: "extras are ignored" is a LangGraph default, not a Pydantic AI one. A tool that takes `**kwargs` hands the model a channel to inject arbitrary parameters under Pydantic AI.

### E5 Invalid enum — confirmed (+1)
- `"XYZ"`, `"usd"` and `"EUR "` (trailing space) are rejected for both `Literal` and `Enum`; `"USD"` accepted. No case or whitespace normalization. Error lists the allowed values.

### E11 Result validation — confirmed with differences
- A function annotated `-> int` returning `"abc"` is accepted by both (no return validation). Huge result: 1,000,000 characters sent whole by both (no truncation). `None`: LangGraph `"null"`, Pydantic AI `""`. `NaN`: `"NaN"` vs `"null"`. dict: both JSON (spacing differs). Non-serializable object: LangGraph sends `repr()`; Pydantic AI raises `PydanticSerializationError` and ends the run (1 model request instead of 2).
- Interpretation: nobody validates tool output; serialization is the only gate, and the two frameworks have different failure semantics. Truncation and size limits are the tool author's responsibility.

### E14 Tool-call limits — partly predicted
- Pydantic AI `tool_calls_limit=3` on a never-ending model: 3 executions, then `UsageLimitExceeded` ("next tool call(s) would exceed") — the check is **before** executing. `request_limit=4`: 4 executions. Batch of 3 calls with limit 2: **0 executions**. 9 messages remained available via `capture_run_messages`.
- LangGraph: only `recursion_limit` (error `GraphRecursionError`): limit 5→2 executions, 9→4, 17→8; no tool-call-count limit exists (batch of 3 ran all 3); the run result carries no partial state. A checkpointer was not tested.
- Interpretation: stopping runaway tool use is the framework's job but with different units: calls vs requests (Pydantic AI) or graph supersteps (LangGraph). Time limits remain as in Phase 1.

### Not done in Batch 1 (honest scope)
Live-model follow-ups for E2/E3/E5 (does a real model self-correct from the error text?) were deliberately not run: they need a replay-then-live hybrid model and the scripted results were fully deterministic. 0 live trials were used for Batch 1.

---

# Batch 2 — pre-registered hypotheses (written before any Batch 2 run)

Three retry mechanisms are kept apart throughout: **(M) model-requested** (error goes to the model, the model re-issues the call), **(F) framework-automatic** (the framework re-runs without asking the model), **(T) tool-internal** (the tool retries itself; the framework never knows).

**B2.1 Tool exception.**
- H1: LangGraph (installed default) feeds every exception class (`ValueError`, `RuntimeError`, `TimeoutError`, pydantic `ValidationError`, `ToolException`) back to the model; Pydantic AI lets all of them escape the run except `ModelRetry`, which is model-visible and budgeted. Neither distinguishes recoverable from terminal by exception class; Pydantic AI has an explicit opt-in (`ModelRetry`), LangGraph has a config switch.
- H2: LangGraph `handle_tool_errors=(ValueError,)` handles only `ValueError`; other classes escape.
- H3: Pydantic AI `ModelRetry` raised twice then success: `retries=1` fails with `UnexpectedModelBehavior`, `retries=3` succeeds; model calls = attempts + 1. (M)
- H4: LangGraph `RetryPolicy` on the tools node never fires while the default handler swallows exceptions; with `handle_tool_errors=False` it re-runs the node with **no model call between attempts** (F); LangGraph's default `retry_on` skips `ValueError`/`RuntimeError`, so `retry_on=Exception` is needed.
- H5: tool-internal retry (T) leaves model calls and tool-call count unchanged.

**B2.2 Timeout.**
- H6: Phase 1 results hold (LangGraph sync `step_timeout` fires late, `ainvoke` on time; Pydantic AI `Tool(timeout)` is model-visible, `run_sync(timeout)` aborts).
- H7: for **sync** tools both frameworks stop *waiting* but the worker thread keeps running and finishes later; for **async** tools the coroutine is cancelled at the threshold.

**B2.3 Duplicate calls.**
- H8: neither framework deduplicates by arguments or by call id: same tool/same args twice executes twice, in sequence and within one batch.
- H9: failure **after** the side effect + a model-requested retry gives 2 ledger entries in both frameworks; Pydantic AI default (exception escapes) gives 1 entry and a dead run.
- H10: LangGraph `RetryPolicy` re-runs the whole tools node, so a batch `[ok side-effect call, failing call]` repeats the *successful* side effect too.

**B2.4 Idempotency.**
- H11: a backend idempotency key (committed atomically with the side effect) yields 1 order for exact duplicates and for retry-after-lost-response in all frameworks.
- H12: same key + different args is detected by the backend (`ConflictError`), surfaced as a tool exception: fed back in LangGraph, escapes in Pydantic AI. Different key + same args creates a second order.
- H13: a non-atomic check-then-commit guard is defeated by two identical calls executed in parallel in one batch (2 orders despite the key).
- H14 (live, ≤5 trials per framework): a real model that must supply the key itself may not reuse the same key on retry, which would defeat the guard; a key derived outside the model avoids this.

## Batch 2 results (2026-10-27)
Common setup: LangGraph 1.2.18, Pydantic AI 2.62.0, Phase 1 agents and scripted models, simulated tools only (no real side effects, no network). Reproduce: `uv run python phase-02/exp_batch2.py [exceptions retry timeout duplicates idempotency]`, `trace <fw>`, `live`; pinned by `tests/phase_02/test_batch2_failure_and_idempotency.py` (40 tests). Raw data: `phase-02/results/batch2.json` (git-ignored). Live: only H14, Groq `openai/gpt-oss-120b`, 5 trials per framework (10 runs total).

**Where Expected ≠ Actual**
1. **H1 (wrong):** I predicted a pydantic `ValidationError` raised *inside a tool body* would escape Pydantic AI. It is treated as a model-visible retry (structured error list, run continues). LangGraph wraps the same exception as `ToolInvocationError` (fed back by default; escapes with `handle_tool_errors=False` or a `(ValueError,)` filter even though pydantic's `ValidationError` subclasses `ValueError`).
2. **H14 (not observed):** I expected a real model might change the idempotency key on retry. In all 8 retries it reused the same key.
3. **Unpredicted:** 2 of 5 Pydantic AI live trials did not retry. My simulated error text said "response lost after the order was created", so the model concluded the order existed and reported success. LangGraph's wrapper text (`Error: ... Please fix your mistakes.`) led to a retry in 5/5. The wording of the error is a behavioral lever and a confound here (N=5, one model).
4. **Unpredicted:** the model invented the same key (`order-20231027-001`) in 8 of 10 trials, independent of the order. A model-chosen key can collide across *different* orders.

### Result table
| Experiment | LangGraph | Pydantic AI | Evidence |
|---|---|---|---|
| Tool exception | all 6 classes fed back to the model by default; `handle_tool_errors=False` → all escape; `(ValueError,)` → only `ValueError` handled | `ValueError`/`RuntimeError`/`TimeoutError`/`ToolException` escape; pydantic `ValidationError` and `ModelRetry` are model-visible | `b21_exceptions` (24 rows) |
| Exception retry | (M) model re-issues call, no budget: 3 tool runs, 4 model calls. (F) `RetryPolicy` inert under the default handler; with `handle_tool_errors=False` and `retry_on=Exception`: 3 tool runs, **2** model calls; default `retry_on` skips `RuntimeError` | (M) `ModelRetry` budget: `retries=1` fails (`UnexpectedModelBehavior`, 2 tool runs); `retries=3` succeeds (3 runs, 4 model calls). No automatic retry of plain exceptions | `b21_retry` |
| Timeout | see matrix below | see matrix below | `b22_timeout` |
| Sync timeout | `step_timeout` on `invoke` fires only after the tool returns (3.0 s for a 3 s tool, limit 1 s) | `Tool(timeout)` / `run_sync(timeout)` return at ~1.0 s but the worker thread finishes at ~3.0 s | `b22_timeout` |
| Async timeout | `ainvoke` returns at 1.0 s; async tool cancelled | `Tool(timeout)` and `run(timeout)` return at ~1.0 s; async tool cancelled | `b22_timeout` |
| Duplicate call | no dedup by args or call id: 2 executions (sequential and same batch), 2 distinct call ids | same | `b23_dup` |
| Failure after side effect | (M) 2 orders; (F) `RetryPolicy` 2 orders with 2 model calls; batch retry also re-ran a *succeeded* call (2 executions) | (M) `ModelRetry` 2 orders; plain exception: 1 order and a dead run, caller re-run → 2 orders | `b23_after_effect` |
| Idempotency key | duplicate → 1 order (1 replay); lost response + same key → 1 order; model changes key → 2 orders; key from context → 1; `RetryPolicy` + key → 1 | same | `b24_idempotency` |
| Same key / different args | backend raises `ConflictError`; fed back to the model; 1 order | `ConflictError` escapes the run; 1 order | `b24_idempotency` |
| Non-atomic guard race | 2 identical keyed calls in one batch → **2 orders** | **2 orders**; with `Tool(sequential=True)` → 1 order, 1 replay | `b24_idempotency` |

### B2.1 Tool exception (6 classes × 4 configs)
- **Hypothesis:** H1, H2 (see above). **Tool:** `failing_tool(kind)` raising `ValueError`, `RuntimeError`, `TimeoutError`, pydantic `ValidationError` (from inside the body), `ToolException`, `ModelRetry`. **Input:** one call, then a scripted `"handled"` text. **Model:** scripted.
- **Actual:** LangGraph default recovered in 6/6 (model calls 2); `handle=False`: 0/6 (run error, 1 model call); `handle=(ValueError,)`: 1/6 (only `ValueError`). Pydantic AI: recovered only for `validation` and `model_retry`.
- **Does the framework separate recoverable from terminal errors?** Not by class. LangGraph: by *configuration* (a filter you set). Pydantic AI: by *opt-in exception type* (`ModelRetry`) plus validation errors. LangGraph attaches no special meaning to `ModelRetry` (it is just another exception whose text is shown).
- **Architectural conclusion:** the recoverable/terminal decision is yours to design in both. Defaults point in opposite directions (LangGraph resilient, Pydantic AI strict), which is the Phase 1 finding now shown across exception classes.

### B2.1 Retry mechanisms — three different things
| Tag | Who retries | What happens | Observed |
|---|---|---|---|
| (M) model-requested | the model, after seeing the error | new model call, new tool call | LangGraph: unbounded until the step cap; Pydantic AI: bounded by `retries` |
| (F) framework-automatic | the framework, no model call | node re-run | LangGraph `RetryPolicy` only; Pydantic AI has none for plain exceptions |
| (T) tool-internal | the tool itself | invisible to the framework | 3 runs inside one tool call; model calls stay 2 |
`RetryPolicy` interacts with the error handler: the default handler swallows the exception first, so `RetryPolicy` never sees it.

### B2.2 Timeout matrix (tool sleeps 3 s, limit 1 s, scripted model; times in seconds from run start)
| Framework | Mechanism | Sync/Async | Timeout fires / run returns | Tool stops? | Run stops? | Model can recover? |
|---|---|---|---|---|---|---|
| LangGraph | `step_timeout`, `invoke` | sync + sync tool | 3.0 (late) | no (it finished at 3.0) | yes, `TimeoutError` | no |
| LangGraph | `step_timeout`, `ainvoke` | async + sync tool | 1.02 | **no — kept running, finished at 3.02** | yes, `TimeoutError` | no |
| LangGraph | `step_timeout`, `ainvoke` | async + async tool | 1.03 | **yes, cancelled at 1.03** | yes | no |
| Pydantic AI | `Tool(timeout=1)` | sync + sync tool | 1.02 | **no — finished at 3.02** | no (run continues) | **yes** (retry message) |
| Pydantic AI | `Tool(timeout=1)` | async + async tool | 1.03 | yes, cancelled | no | yes |
| Pydantic AI | `run_sync(timeout=1)` | sync + sync tool | 1.03 | **no — finished at 3.02** | yes, `RunTimedOut` | no |
| Pydantic AI | `run(timeout=1)` | async + async tool | 1.01 | yes, cancelled | yes, `RunTimedOut` | no |
| Pydantic AI | `Agent(tool_timeout=1)` | sync + sync tool | 1.02 | no — finished at 3.0 | no | yes |
**Ownership:** the framework owns the *wait*, not the *work*. A sync tool is never stopped; only an async tool is cancelled. Hypotheses H6 and H7 confirmed.

### B2.3 Duplicate execution
- Same tool, same args, two turns or one batch → 2 executions in both frameworks; call ids are distinct (the framework cannot tell it is a duplicate). No deduplication exists.
- **Failure after side effect** (response lost after the order was created):
  - (M) both frameworks: model re-calls → 2 orders (LangGraph 3 model calls; Pydantic AI via `ModelRetry` 3 model calls).
  - (F) LangGraph `RetryPolicy` (`handle_tool_errors=False`): 2 orders with only 2 model calls — the duplicate happens with no model involvement.
  - Pydantic AI default exception: 1 order and the run is dead; the *caller* re-running the whole run produces order 2.
  - H10 confirmed: `RetryPolicy` re-runs the whole tools node. In a batch `[record_action ok, flaky fails once]`, `record_action` executed **2×** (Pydantic AI, no retry: 1×, run dead).
- **Conclusion:** every retry owner (model, framework, caller) can duplicate a side effect. None of them knows whether the effect already happened.

### B2.4 Idempotency (simulated `create_order`, tiny backend guard)
- Key committed together with the side effect: exact duplicate → 1 order; lost response + same key → 1 order; `RetryPolicy` + key → 1 order. Different key, same args → 2 orders (legitimately a different operation to the backend).
- Same key, different args → `ConflictError`, a *tool exception*, so its handling follows the framework's exception policy (fed back vs escapes).
- **The key must not depend on the model remembering it:** a changed key → 2 orders. Live: the model reused the key 8/8 but invented the *same* key text across unrelated runs, so a model-chosen key is also a collision risk.
- **Guard limits (measured, not fixed):** a check-then-commit guard without atomicity lost a race (2 orders from 2 identical keyed calls in one batch, in both frameworks). Serializing the calls (`Tool(sequential=True)`, Pydantic AI only) hid the race. Real systems need an atomic reserve/unique constraint in the backend; building that was out of scope.
- **Neither LangGraph nor Pydantic AI provides business-level idempotency.** Retry answers "should I call again?"; idempotency answers "if I call again, does the effect happen once?".

### Live (H14): model supplies the key; lost response after the effect; 5 trials per framework
| Framework | Trials | Retried | Same key on retry | Orders created (per trial) | No retry |
|---|---|---|---|---|---|
| LangGraph (RuntimeError text) | 5 | 5 | 5/5 | 1,1,1,1,1 | 0 |
| Pydantic AI (`ModelRetry`) | 5 | 3 | 3/3 | 1,1,1,1,1 | 2 (model inferred the order existed from the message) |
Limitations: one model, N=5, confounded by exception text; the simulated message states the order was created. Counts only; no percentages implied.

### Tracing evidence (offline spans, OTel + OpenInference)
Failure-after-side-effect with a model-requested retry reconstructs as: `LLM → create_order [TOOL, ERROR, exception event] → LLM → create_order [TOOL, OK] → LLM` in both frameworks (LangGraph adds `agent`/`tools` node spans); the root span stays `OK` because the error was handled. 2 orders were created. `RetryPolicy` re-runs (F) were not span-captured.

### Not done / limits
Live timeout recovery and live exception recovery were not run (mechanics deterministic; Phase 1 covered live exception recovery). Live H14 used one model. A distributed or atomic idempotency store, LangGraph checkpoint-based recovery, and async-tool-with-LangGraph `invoke` were not tested.

