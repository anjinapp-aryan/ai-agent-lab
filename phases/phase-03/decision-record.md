# Phase 3 Decision Records

## DR-3.1-1: offline trajectory labels (`phase-03/common/lab3/patterns.py`, Batch 4)

**Decision:** BUILD a 30-line pure function `classify(executions) -> REPEATED_CALL | OSCILLATION | NO_PROGRESS | NONE`. Measurement only.

**Capability needed:** label recorded experiment trajectories, to find out how much a syntactic check can and cannot decide. It is needed to show false alarms and misses as evidence, not to protect anything.

**Existing OSS considered** (audit: [reuse-audit.md](reuse-audit.md#batch-4-focused-audit-loop-repetition-and-no-progress-detection-2026-10-31)):
- `agent-loop-guard` (MIT): the closest match, and the one that nearly replaces this. Not adopted, because adding it is a new dependency for a 1-star, v0.1.1 project whose commits since 2026-04 are documentation, and it is a runtime gate (`ALLOW/WARN/STOP`) rather than an offline label. Its design is the reference.
- Framework limits (`recursion_limit`, `UsageLimits`) and LangChain `ToolCallLimitMiddleware` count events; they do not recognize patterns.
- `agentevals` compares against a reference trajectory. These runaway shapes have no reference.

**What it is:** input is already-recorded `(tool, args, outcome)` tuples. Thresholds: 3+ consecutive identical calls, 2+ exact A,B cycles. Standard library only (`json`).

**Excluded:** it never runs during an agent run, and it cannot retry, stop, block or execute. It is not middleware, not a hook, not a policy. A test asserts it imports nothing but `json` and exposes one function. Fuzzy or semantic matching is deliberately excluded.

**Known weaknesses (measured, kept visible):**
- False alarms on legitimate polling.
- Misses reworded no-progress.
- Cannot decide changing-argument oscillation.
- Counts a framework retry as repetition.

**Revisit if:** a later phase needs runtime protection. In that case, evaluate OSS and framework hooks first; do not promote this function.
