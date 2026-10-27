# LangGraph vs Pydantic AI — Tool Calling (Batch 1; same scripted inputs, same tools)

| Aspect | LangGraph (1.2.18) | Pydantic AI (2.62.0) |
|---|---|---|
| Schema closed to extras | no | yes (`additionalProperties:false`) |
| Provider `strict` flag | not sent | `strict:true` sent |
| Extra arg (plain function) | dropped, tool runs | rejected |
| Extra arg (`**kwargs`) | dropped | passed through |
| Coercion | Pydantic lax | Pydantic lax (identical) |
| Retry on bad args | no budget; step cap | per-tool budget (`retries`) |
| Return validation | none | none |
| Result serialization | `repr`/JSON-ish string | strict JSON; unserializable aborts run |
| Result size limit | none | none |
| Tool-call count limit | none | `UsageLimits(tool_calls_limit)` (checked before executing, whole batch) |
| Step/request limit | `recursion_limit` | `request_limit` |
| State after limit error | not in error | retrievable via `capture_run_messages` |

No overall winner. The notable design point is that defaults diverge exactly at the security-relevant edges (extra args, serialization, limits): treat framework defaults as unverified until tested.

## Batch 2 additions
| Aspect | LangGraph (1.2.18) | Pydantic AI (2.62.0) |
|---|---|---|
| Tool exception default | fed back to model (all classes) | escapes (except `ValidationError`, `ModelRetry`) |
| Recoverable-vs-terminal switch | `handle_tool_errors` (bool / type / callable) | raise `ModelRetry` (opt-in) |
| Automatic retry (F) | `RetryPolicy` on the tools node (needs handler off; re-runs the whole node) | none for plain exceptions |
| Model-visible retry budget (M) | none (step cap) | `retries` / `max_retries` |
| Tool timeout | none; `step_timeout` (works with `ainvoke`) | `Tool(timeout)`, `Agent(tool_timeout)`, run timeout |
| Timeout is recoverable by model | only if raised inside the tool | yes (`Tool(timeout)`) |
| Sync tool after timeout | keeps running | keeps running |
| Async tool after timeout | cancelled | cancelled |
| Dedup of duplicate calls | none | none |
| Serialize parallel tool calls | no knob found | `Tool(sequential=True)` |
| Error text shown to the model | `Error: ... Please fix your mistakes.` | bare message / structured error list |
