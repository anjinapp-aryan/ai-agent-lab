# Race Conditions (Experiments 4.6–4.9, 4.15, 4.18)

## Answers
6. **Can shared mutable state race?** Yes. 100 concurrent read–sleep–write increments ended at 3–5 instead of 100, with raw threads (no framework), LangGraph, and Pydantic AI alike. The frameworks neither cause nor prevent it.
7. **Does async eliminate races?** No. With an `await` between read and write, asyncio lost 99 of 100 updates in all three settings. Without a suspension point between read and write the update was exact; that is the only guarantee asyncio gives.
8. **Which primitives?** Standard library: `threading.Lock` for thread-run tools (100/100, at the cost of serializing the critical section: ~0.6 s for 100 × 5 ms), `asyncio.Lock` for async tools (100/100). For side effects: an atomic check-and-commit at the business/data boundary (here a lock around check + effect + commit; in production a unique constraint or transactional insert).
9. **Can concurrent calls duplicate side effects?** Yes: two identical concurrent `create_order` calls created 2 orders without a key and 2 with a non-atomic key guard, in both frameworks; 2 concurrent identical authorized *requests* did the same (5/5 trials).
10. **Where must idempotency live?** At the business/data boundary, atomically with the effect. The atomic guard gave 1 order + 1 replay in every case.
11. **Does sequential execution solve business correctness?** It hides the race: `sequential` mode gave 100/100 with an unsafe counter, and a run-level limit (`Agent(max_concurrency=1)`) gave 1 order with a non-atomic guard. The tool and guard are still wrong and break as soon as anything else runs concurrently (another worker, process, or replica).

## Identity under concurrency (4.15)
15. **Can request-scoped identity leak?** Yes when it is not request-scoped: a global `current_user` and a tool set built once and shared both leaked in 10 of 20 concurrent requests, turning correct denials into unauthorized grants. Tools built per request and the frameworks' per-call context (`runtime.context`, `ctx.deps`) leaked 0 of 20.
16. **Is global mutable security context safe?** No. Identity must travel with the call (framework context or per-request construction), never through shared mutable state.
