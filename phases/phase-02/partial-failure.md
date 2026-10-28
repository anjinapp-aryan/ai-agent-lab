# Partial Failure and Cancellation (Experiments 4.12, 4.13, 4.17)

## Answers
12. **What happens when one concurrent tool fails?** It depends on how the failure surfaces. Model-visible (LangGraph default handler, Pydantic AI `ModelRetry`): all three results reach the model and it can recover. Escaping exception: the run ends and the model sees nothing; LangGraph `invoke` waits ~620 ms for siblings before raising, Pydantic AI `run_sync` returns after ~9 ms while siblings keep running.
13. **Are other tools cancelled?** Async siblings: yes, when the exception escapes (both frameworks). Sync siblings: never; they finish and commit their side effects, even after the run has already returned (Pydantic AI `run_sync`).
14. **Can successful side effects be rolled back?** Not by either framework; nothing was compensated in any case. Partial success, partial failure and rollback are three separate concerns; only the first two are ever reported, and rollback/compensation is application work.

## Cancellation (4.17)
Cancelling the run at ~1 s: async tools cancelled immediately (0 side effects, `finally` ran); sync tools reported nothing, ran to 5 s and committed their side effects ~4 s after the run reported `cancelled`. Batch 2 said TIMEOUT ≠ CANCELLATION; Batch 4 adds CANCELLED RUN ≠ CANCELLED WORK for sync tools.

## Design consequences
- Treat a failed or cancelled run as "outcome unknown" for every sync side-effecting tool that was in flight.
- Make side-effecting tools idempotent (Batch 2) and, where cancellation must stop work, async.
- Decide explicitly whether a tool failure should be model-visible (recover with partial results) or terminal (and then reconcile the siblings' effects).
