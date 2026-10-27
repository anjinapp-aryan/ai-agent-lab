# Retries (D) and Idempotency / Duplicate Calls (H)

## Two different retries — do not conflate
1. **Model-visible retry** (the model is told what was wrong and tries again): validation errors, `ModelRetry`, error ToolMessages. Phase 1: Pydantic AI has a budget (`retries`, per-tool `max_retries`); LangGraph has none (bounded only by `recursion_limit`).
2. **Infrastructure retry** (code re-runs a failing call without the model): backoff on transient errors (429, timeouts, network).

## Candidates
| Project | Repo | License | Latest | Pushed | Role | Decision |
|---|---|---|---|---|---|---|
| Pydantic AI retries | pydantic/pydantic-ai | MIT | v2.62.0 | 10-27 | model-visible: `retries`, `Tool(max_retries)`, `ModelRetry` `[V]` | REUSE |
| LangGraph `RetryPolicy` | langchain-ai/langgraph | MIT | 1.2.18 | 10-27 | node-level infra retry: `initial_interval, backoff_factor, max_interval, max_attempts, jitter, retry_on` `[V fields]` | REUSE (Exp 6) |
| LangChain `tool_retry` / `model_retry` middleware | langchain-ai/langchain (`langchain` pkg 1.4.7, MIT) | MIT | 1.4.7 | 10-27 | tool-level retry with backoff `[F]` file exists; semantics `[K]` | COMPOSE if Exp 6 needs it (adds `langchain` dependency) |
| tenacity | jd/tenacity | Apache-2.0 | 9.2.1 (10-07) | 10-12 | general retry/backoff decorator; already a transitive dependency `[V]` (in uv.lock via LangChain stack) | REUSE only if a tool needs transient-error retry (Exp 6) |
| stamina | hynek/stamina | MIT | 26.1.0 (04-13) | 10-19 | opinionated retries on tenacity | REFERENCE |
| backoff | litl/backoff | MIT | v2.2.1 (2022) | 2024-05-02 | retry decorator | **REJECT — repository archived** `[V]` |

## Idempotency / duplicate calls
- Neither framework deduplicates identical tool calls by default `[K]`; Experiment 8 verifies (do not assume).
- Idempotency is a property of the *side-effecting tool/backend* (idempotency key), not of the LLM loop. Durable-execution engines give exactly-once-ish semantics at workflow level:
  | Project | License | Latest | Pushed | Decision |
  |---|---|---|---|---|
  | DBOS Transact (Python) | MIT | 3.3.0 (10-15) | 10-27 | REFERENCE → possible COMPOSE (Phase 6); too heavy for Exp 9 |
  | Temporal Python SDK | MIT | 1.34.0 (09-30) | 10-27 | REFERENCE (needs Temporal server) |
  | Restate | BSL 1.1 (Phase 0) | — | — | REFERENCE ONLY, LEGAL REVIEW REQUIRED |
- For Exp 9 the only needed mechanism is an idempotency key check inside a *simulated* `create_order` (~10 lines: dict keyed by key -> stored result). That is a **tiny, concept-learning BUILD**; production-grade stores are out of scope (documented in reuse-decisions.md).

## Open questions for experiments
Does a timeout-induced retry re-execute a side effect? Does Pydantic AI count timeouts against `max_retries`? (Phase 1: `Tool(timeout)` produced `outcome="retried"`.) Does LangGraph `RetryPolicy` re-run non-idempotent tools after partial success?
