# Phase 2 — Deep Tool Calling (implementation)

Audit and design: [audits/phase-02/](../../audits/phase-02/). Hypotheses and results: [experiments.md](experiments.md).

## Status
| Batch | Experiments | Status |
|---|---|---|
| 1 | 1 schema, 2 missing arg, 3 wrong type, 4 extra arg, 5 enum, 11 result validation, 14 limits | **done** (scripted + wire-level, 0 live trials) |
| 2 | 6 exception, 7 timeout, 8 duplicate, 9 create_order | not started |
| 3 | 10 authorization, 12 output injection | not started |
| 4 | 13 parallel/dependent | not started |
| 5 | 15 tool selection (live, 5 trials per framework) | not started |

## How it is built (reuse)
- Phase 1 agents (`lg_agent.run`, `pa_agent.run`) are reused unchanged in behavior; two optional arguments were added (`recursion_limit`, `usage_limits`) so limit experiments can set them.
- Phase 1 scripted models are reused by path ([phase-02/common/lab2/scripted.py](../../phase-02/common/lab2/scripted.py)) and wrapped to record what the model receives.
- `lab2/wire.py` uses `httpx.MockTransport` with the real provider clients to capture exact request payloads. No new dependency.
- `lab2/fixtures.py` holds simulated tools (test data). There is no tool framework.

## Run
```
uv run python phase-02/exp_batch1.py            # all Batch 1 experiments, prints observations
uv run python phase-02/exp_batch1.py 4 11       # selected experiments
uv run pytest tests/phase_02                    # 57 offline regression tests
uv run pytest                                   # everything (Phase 1 live tests use Groq quota)
```
Results JSON: `phase-02/results/batch1.json` (git-ignored).
