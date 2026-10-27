# Timeout (Experiment 7)

Evidence: scripted model, tool sleeps 3 s, limit 1 s, monotonic timestamps; full matrix in [experiments.md](experiments.md#b22-timeout-matrix-tool-sleeps-3-s-limit-1-s-scripted-model-times-in-seconds-from-run-start). Pinned in `tests/phase_02/test_batch2_failure_and_idempotency.py` (short durations).

## Answers
- **What layer owns the timeout?** The framework owns the *wait*; the tool body owns the *work*. No layer stops a running sync tool.
- **Sync vs async:** sync tools run in a worker thread that cannot be killed: both frameworks return at ~1.0 s while the worker finishes at ~3.0 s (side effects continue after the "timeout"). Async tools are cancelled at ~1.0 s (a `CancelledError` reaches the tool).
- **LangGraph `step_timeout`:** only works usefully with `ainvoke`; on sync `invoke` it fires after the tool returns (3.0 s).
- **Pydantic AI:** `Tool(timeout)` and `Agent(tool_timeout)` turn the timeout into a model-visible retry (the model can recover); `run_sync(timeout)`/`run(timeout)` abort the whole run.
- **Recovery:** only Pydantic AI's per-tool timeout lets the model continue. In LangGraph you must raise the timeout from inside the tool to get the same effect (Phase 1: wrapper) and rely on the default error handler.
- **Design consequence:** a timeout is not cancellation. If the tool has a side effect, a timeout means "unknown outcome", which leads directly to idempotency (see [idempotency.md](idempotency.md)).
