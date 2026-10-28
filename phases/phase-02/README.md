# Phase 2 — Deep Tool Calling (implementation)

Audit and design: [audits/phase-02/](../../audits/phase-02/). Hypotheses and results: [experiments.md](experiments.md).

## Status
| Batch | Experiments | Status |
|---|---|---|
| 1 | 1 schema, 2 missing arg, 3 wrong type, 4 extra arg, 5 enum, 11 result validation, 14 limits | **done** (scripted + wire-level, 0 live trials) |
| 2 | 6 exception, 7 timeout, 8 duplicate, 9 create_order | **done** (scripted; live H14 only: 10 runs on Groq `openai/gpt-oss-120b`) |
| 3 | 10 authorization, 12 output injection | **done** (scripted + 54 live runs; 43 + 2 tests) |
| 4 | 13 parallel/dependent | not started |
| 5 | 15 tool selection (live, 5 trials per framework) | not started |

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
uv run pytest tests/phase_02                    # 140 offline regression tests (+2 live)
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
