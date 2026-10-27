# Phase 2 Learning Design — 15 Experiments

Rules: same model for every live comparison (Groq `openai/gpt-oss-120b`), same prompt/tools/schemas/tasks/scorer for both frameworks; Phase 1 baselines are reused, not re-run. Each experiment has a **hypothesis about ownership** that it can falsify.

Two tiers per experiment (Phase 1 lesson: separate framework from model):
- **A — mechanics:** scripted model, deterministic, offline, free. Establishes what the *framework* does.
- **B — live:** real model, N=5 trials per framework (report counts), only where model behavior matters. Rate limits are recorded, not hidden.

## Matrix
| # | Experiment | Hypothesis (owner) | Tier | Reused mechanism | Metrics | New code |
|---|---|---|---|---|---|---|
| 1 | Simple typed `calculate(a:int,b:int)` | schema = framework; args = model | A+B | both frameworks | exact payload sent to model, args, result | fixture |
| 2 | Missing argument | validation = framework wrapper; retry decision = model (LG) / framework budget (PA) | A+B | native validation | who detects, attempts, feedback text, final outcome | none |
| 3 | Wrong type (`"hello"`) | same as 2; coercion boundary | A+B | native | attempts, coercion cases (`"10"`) | none |
| 4 | Extra argument | PoC predicts LG drops, PA rejects; test `extra="forbid"` args_schema in LG | A | native + Pydantic config | rejected/ignored/coerced/passed-through; side effect on secret field | none |
| 5 | Invalid enum (`currency="XYZ"`) | validation = framework; model can self-correct from error | A+B | native | attempts to recovery | fixture |
| 6 | Tool exception: model-visible vs terminating | policy = framework default; configurable | A+B | LG `handle_tool_errors`, `RetryPolicy`; PA `ModelRetry`, `retries` | retry count, latency, tokens/LLM calls overhead, outcome | none |
| 7 | Timeout: per-tool vs run vs step | Phase 1 mapped; add async-tool control | A | `Tool(timeout)`, `run_sync(timeout)`, `step_timeout`, `tool_timeout` | wall time, owner, recovery, side effect continues? | none (extends Phase 1 `experiments.py` pattern) |
| 8 | Duplicate tool call (same tool, same args) | no framework dedup | A+B | native | executions per run, model behavior live | fixture |
| 9 | Side-effecting `create_order` (simulated) | idempotency belongs to the tool/backend | A | idempotency-key guard | duplicate executions with/without guard; retry-after-timeout duplicate | tiny guard (BUILD record) |
| 10 | Authorization: safe / dangerous / needs approval | policy layer is optional and framework-specific | A+B | PA `requires_approval`, `FilteredToolset`; LG `interrupt`, `wrap_tool_call` | allowed/denied/paused; resume behavior; does model learn of denial? | none |
| 11 | Result validation: valid / malformed / unexpected / huge | frameworks do not validate or truncate results by default | A | typed returns, Pydantic | what reaches the model, size, truncation | fixture |
| 12 | Tool-output injection (simulated) | model treats tool output as semi-trusted; mitigation = prompt + tool-set restriction, not framework | B | prompt framing, restricted toolset, AgentDojo patterns (reference) | followed injected instruction? (counts over N=5), per mitigation | fixture (simulated dangerous tool, no real effect) |
| 13 | Parallel A,B,C and dependent A→B | batching = model; scheduling = framework; dependency = model across turns (not framework) | A+B | Phase 1 timestamp harness | serial vs parallel duration, overlap, turn count, wrong-order incidents | extends harness |
| 14 | Limits: max tool calls / iterations / time | each limit has a different owner | A | `tool_calls_limit`, `request_limit`, `recursion_limit`, timeouts | where raised, state at stop, partial result retained? | none |
| 15 | Tool selection among similar tools (weather trio) | selection quality = model + descriptions | B | pytest scorer (+ optional Inspect AI) | correct / wrong / unnecessary / none (counts, N=5 per prompt, 6+ prompts) | fixture |

Approx. live call budget: Exp 1,2,3,5,6,8,10,12,13,15 ≈ 10 experiments × 2 frameworks × N=5 × ≤3 LLM calls ≈ 300 calls. Groq free-tier limits unverified; run experiments one at a time, record any 429, and shrink N rather than adding retry infrastructure.

## Refined architecture hypothesis (to be corrected by experiments)
```
model --(tool_call: name,args)--> [schema contract: advisory, framework closes/doesn't close extras]
  -> validation (framework wrapper; coercion boundary)  --fail--> error text -> model (LG: unbounded / PA: budgeted)
  -> policy (optional; approval/allowlist; framework-specific, absent by default)
  -> runner (timeout: none by default; concurrency: framework; retry: infra vs model-visible)
  -> result (stringified; no schema/truncation by default?)
  -> model (treats result as context: injection surface)
```
Known from evidence: validation owner, extra-arg difference, error-feedback semantics, timeout owners, parallelism split. Everything else is a hypothesis.

## Execution order
Group 1 (offline, free): 4, 11, 14, 7, 9. Group 2 (offline+live validation): 1, 2, 3, 5, 6, 8. Group 3 (live-heavy): 13, 15, 10, 12 last (largest quota use, hardest to interpret).

## Deliverables at implementation time
`phase-02/` code (fixtures + experiment drivers), `tests/phase_02/` deterministic regression for Tier A, `phases/phase-02/` docs mirroring Phase 1 (observations, failure-analysis, comparison, evaluation, decision). Langfuse stays optional (blocked).

## Stop conditions
Stop and re-audit if any experiment needs a new dependency beyond the optional two (`langchain`, `inspect-ai`), a server/database, real network or file access, or a real credential.
