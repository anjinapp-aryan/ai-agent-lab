# M. Reliability / Durable Execution

| Project | License | Latest | Pushed | Decision |
|---|---|---|---|---|
| LangGraph checkpoints, step limits (`recursion_limit`) `[K]` | MIT | 1.2.18 | 10-27 | REUSE — baseline |
| DBOS Transact (Python) | MIT | — | 10-26 | REUSE (Phase 6) — durable workflows as library on Postgres/SQLite `[K]`; lightest durable option |
| Temporal | MIT | v1.32.1 (10-08) | 10-27 | REFERENCE → optional Phase 6 run; heavy server; mature |
| Restate | **BSL 1.1** (Additional Use Grant restricts use) `[V]` | v1.8.1 (10-21) | 10-27 | REFERENCE ONLY — LEGAL REVIEW REQUIRED before use |
| Inngest | license reported NOASSERTION, not inspected | — | 10-27 | REFERENCE ONLY until license read |
| Pydantic AI durable-execution integrations (Temporal/DBOS) `[K]` | MIT | | | COMPOSE |
| tenacity / httpx timeouts for plain retries | not evaluated | | | REUSE when needed |

Build: none. Circuit breakers / infinite-loop guards: use framework limits first; custom budget guard may be a few lines of glue (not a framework).
Failure-injection experiments in Phase 6: kill process mid-tool-call, duplicate tool call (idempotency), model returns garbage.
