# Concurrency (Batch 4)

Terms as measured: **concurrent** = overlapping in time; **parallel** = overlapping on separate threads; **async** = overlapping on one thread by cooperative scheduling; **batched** = several tool calls in one model response; **dependent** = B needs A's output.

## Who decides what (measured)
```
MODEL          decides whether calls are batched (gpt-oss-120b: never, 0/10 independent; dependency respected 10/10)
FRAMEWORK      decides whether a batch is run together (both: yes by default) and what limits exist
RUNTIME        thread pool (sync tools: parallel) or event loop (async tools: concurrent on one thread)
TOOL           sync or async; only async can be cancelled
APPLICATION    locks, atomic guards, request-scoped identity
DATA           atomicity / uniqueness (not exercised: local simulation only)
```

## Answers
1. **When does LangGraph run tools concurrently?** When the model batches calls: `invoke` runs sync tools on a thread pool (2 threads, 1000 ms overlap for two 1 s tools); `ainvoke` overlaps sync tools on threads and async tools on one thread. `invoke` cannot run async-only tools at all (they become error messages).
2. **When does Pydantic AI?** When the model batches calls, by default, for sync (threads) and async (tasks) tools, under both `run_sync` and `run`; `sequential` mode or `Tool(sequential=True)` serializes.
3. **Sync vs async:** sync tools give real parallelism on threads but cannot be cancelled; async tools overlap on one thread and can be cancelled. Async does not remove races (1/100 with an `await` between read and write).
4. **Is tool-call order guaranteed?** START order followed CALL order in every run; COMPLETION order followed duration (B,C,A).
5. **Is result order guaranteed?** Yes in the observed runs: the model received results in CALL order regardless of completion order, in both frameworks.
- **Limits:** LangGraph `config["max_concurrency"]` caps sync `invoke` exactly (1/2/4/8/10) and is ignored by `ainvoke`; Pydantic AI has no per-run tool cap (`Agent(max_concurrency)` limits runs), only `parallel`/`sequential`/`parallel_ordered_events` modes.
- **Dependencies:** no framework detects them; a batched dependent call ran with a guessed argument. Ordering comes from the model taking turns, or from the application.

## Framework difference worth remembering
A shared Pydantic AI `Agent(max_concurrency=1)` deadlocked when used from two threads that each call `run_sync`; in one event loop it serialized runs correctly. Use it within a single loop (or verify) before relying on it.
