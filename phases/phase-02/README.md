# Phase 2 — Deep Tool Calling (implementation)

Audit and design: [audits/phase-02/](../../audits/phase-02/). Hypotheses and results: [experiments.md](experiments.md).

## Status
| Batch | Experiments | Status |
|---|---|---|
| 1 | 1 schema, 2 missing arg, 3 wrong type, 4 extra arg, 5 enum, 11 result validation, 14 limits | **done** (scripted + wire-level, 0 live trials) |
| 2 | 6 exception, 7 timeout, 8 duplicate, 9 create_order | **done** (scripted; live H14 only: 10 runs on Groq `openai/gpt-oss-120b`) |
| 3 | 10 authorization, 12 output injection | **done** (scripted + 54 live runs; 43 + 2 tests) |
| 4 | 13 parallel/dependent (+ races, limits, partial failure, cancellation, concurrent authz) | **done** (scripted + 40 live runs, 10 discarded; 43 tests) |
| 5 | 15 tool selection (5.1–5.18: metadata, count, order, ambiguity, missing capability, tool_choice, description/output injection, authz vs selection) | **done** (scripted + wire + 436 live runs on `gpt-oss-20b`/`120b` + 55 ablation calls; 53 + 2 tests) |

## How it is built (reuse)
- Phase 1 agents (`lg_agent.run`, `pa_agent.run`) are reused unchanged in behavior; optional arguments were added (`recursion_limit`, `usage_limits`, LangGraph `retry_policy`, `wrap_tool_call`, `context`, `context_schema`, `configurable`; Pydantic AI `deps`, `capabilities`) so limit, retry and authorization experiments can set them; defaults are unchanged.
- Phase 1 scripted models are reused by path ([phase-02/common/lab2/scripted.py](../../phase-02/common/lab2/scripted.py)) and wrapped to record what the model receives.
- `lab2/wire.py` uses `httpx.MockTransport` with the real provider clients to capture exact request payloads. No new dependency.
- `lab2/fixtures.py` holds simulated tools (test data). There is no tool framework.

## Run
```
uv run python phase-02/exp_batch1.py            # all Batch 1 experiments, prints observations
uv run python phase-02/exp_batch1.py 4 11       # selected experiments
uv run python phase-02/exp_batch2.py [exceptions retry timeout duplicates idempotency]   # Batch 2, scripted
uv run python phase-02/exp_batch2.py trace langgraph|pydantic-ai   # offline span tree
uv run python phase-02/exp_batch2.py live        # H14, 5 trials per framework, Groq quota
uv run python phase-02/exp_batch3.py [basic designs resource failures bypass delivery flows]   # Batch 3, scripted
uv run python phase-02/exp_batch3.py live       # injection trials, <=5 per framework per case, Groq quota
uv run python phase-02/exp_batch4.py [independent order counter asyncrace duplicate limits partial dependent authz output cancel combined]   # Batch 4
uv run python phase-02/exp_batch4.py live       # batching/dependency/concurrent-output trials, Groq quota
uv run python phase-02/exp_batch5.py [mechanics wire choice security]   # Batch 5, offline
uv run python phase-02/exp_batch5.py live [5.3 5.16 ...] | ablation      # Batch 5 live (Groq quota; MODEL_NAME=openai/gpt-oss-20b was used)
uv run pytest tests/phase_02                    # 236 offline regression tests (+4 live)
uv run pytest                                   # everything (Phase 1 live tests use Groq quota)
```
Results JSON: `phase-02/results/batch1.json` (git-ignored).

## Documents
[experiments.md](experiments.md) (hypotheses and results) · [tool-schema.md](tool-schema.md) · [validation.md](validation.md) · [retry-and-error-handling.md](retry-and-error-handling.md) · [timeout.md](timeout.md) · [idempotency.md](idempotency.md) · [observations.md](observations.md) · [comparison.md](comparison.md)

## Batch 3 summary (authorization + tool-output security)
Documents: [authorization.md](authorization.md) · [tool-output-security.md](tool-output-security.md) · [security-boundaries.md](security-boundaries.md).
- The LLM requests; the application boundary decides. Prompt rules are not a boundary; argument-supplied identity is a bypass; permission-only authorization leaks across tenants.
- Native hooks (`wrap_tool_call`, `Hooks.before_tool_execute`) stop the tool body from starting; both frameworks hand the application a trusted context.
- Frameworks deliver tool output verbatim and add no injection protection. Live `gpt-oss-120b` followed injected prose 20/20 (0/20 for a JSON field); with authorization enforced there were 0 side effects.
- Denial semantics matter: Pydantic AI's plain-exception denial ended the run; LangGraph's `RetryPolicy` retried a denial 3×.
- Limits: one model, N=5, no HITL approval flow tested, no mitigations tried. Next: Batch 4 (concurrency) only after review.

## Batch 4 summary (concurrency)
Documents: [concurrency.md](concurrency.md) · [race-conditions.md](race-conditions.md) · [partial-failure.md](partial-failure.md).
- Concurrency is an execution property: both frameworks overlap batched tools (threads for sync, one loop for async), deliver results in call order, and do nothing for shared state, duplicates, rollback, dependencies or identity isolation.
- Races were reproduced without any framework (3/100), under both frameworks (3–5/100), and under asyncio with an `await` in the critical section (1/100); stdlib locks fix them; sequential execution only hides them.
- Failure/cancel: async siblings are cancelled, sync siblings always finish and commit, sometimes after the run has returned.
- The live model never batched independent calls and always respected dependencies by taking turns.
- Unexpected: a shared Pydantic AI `Agent(max_concurrency=1)` deadlocked across threads calling `run_sync` (`phase-02/repro_pa_max_concurrency.py`).
- Limits: local simulation only (no DB/multi-process), one model, N=5; timestamps before Batch 4 have ~16 ms resolution. Next: Batch 5 (tool selection), only after review.

## Batch 5 summary (tool selection)
Documents: [tool-selection.md](tool-selection.md) · [architecture.md](architecture.md) · audit [tool-selection-audit.md](../../audits/phase-02/tool-selection-audit.md) (REUSE NOT REQUIRED).
- The model selects; both frameworks execute exactly the selected valid tool and reject invented names and invalid args.
- Benign metadata changes (name, description, schema, count up to 20, order) did not change selection in this sample; among synonyms the name echoing the user's verb won.
- Instructions inside tool descriptions were obeyed 20/20 (including an unrequested `cancel_order`); injected tool output redirected the next selection 10/10; application authorization held (0 effects).
- The frameworks send different requests (Pydantic AI: `additionalProperties:false`, `strict`, `tool_choice:auto`; LangGraph: none, and Phase 1 PA sent no temperature). `additionalProperties:false` alone flipped `gpt-oss-20b` from honest refusal to pretended capability (5/5 vs 5/5).
- Groq enforces `tool_choice=required` by rejecting text answers (HTTP 400).
- Limits: one model family (20b primary, 120b partial after a daily-token quota stop), N ≤ 5, toy domain. Next: Phase 2 review; nothing further started.
