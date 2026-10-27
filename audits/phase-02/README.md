# Phase 2 Audit — Deep Tool Calling (design only)

Date: 2026-10-27. Phase 1 is closed (commit 4f8e25d) and untouched.
Scope: reuse audit + experiment design. **No Phase 2 experiments were implemented.**

## Method
- GitHub API (`gh api`) for license, latest release, `pushed_at`, `archived`, stars on **46 repositories**; LICENSE text read for every `NOASSERTION`; directory listing of `langchain-ai/langchain` middleware.
- Installed package introspection (Pydantic AI 2.62.0, LangGraph 1.2.18, langchain-core 1.7.0) for API surface.
- One tiny offline proof-of-concept, [poc_arg_validation.py](poc_arg_validation.py), to verify schema generation and missing/extra/wrong-type/enum/coercion handling in both current frameworks (scripted model, no API calls). It is evidence for the audit, not a Phase 2 experiment.
- Not measured: contributor counts, test coverage, issue responsiveness, transitive licenses.

## Evidence tiers
- `[V]` verified this session (GitHub API, LICENSE text, installed-package introspection, or the PoC).
- `[F]` file/API exists (verified) but its runtime semantics were not run.
- `[S]` secondary source.
- `[K]` prior knowledge, not re-verified. Phase 2 experiments must confirm before any claim is relied on.

## Files
| File | Content |
|---|---|
| [tool-calling-ecosystem.md](tool-calling-ecosystem.md) | Layers, candidate frameworks/SDKs, health |
| [tool-schema-audit.md](tool-schema-audit.md) | Schemas (capability A) + PoC schema result |
| [tool-validation-audit.md](tool-validation-audit.md) | Validation (C, F) + PoC results |
| [tool-retry-audit.md](tool-retry-audit.md) | Retries (D) and idempotency (H) |
| [tool-security-audit.md](tool-security-audit.md) | Authorization (E) and safety/injection (G) |
| [tool-concurrency-audit.md](tool-concurrency-audit.md) | Parallelism, limits, cancellation (I) |
| [tool-observability-audit.md](tool-observability-audit.md) | Tracing (J) |
| [tool-evaluation-audit.md](tool-evaluation-audit.md) | Tool-call evaluation (K) |
| [reuse-decisions.md](reuse-decisions.md) | REUSE / ADAPT / COMPOSE / REFERENCE / REJECT / BUILD per capability |
| [phase-2-learning-design.md](phase-2-learning-design.md) | 15-experiment matrix |
| [phase-2-decision.md](phase-2-decision.md) | Design gate |
