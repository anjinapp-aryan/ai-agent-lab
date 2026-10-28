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

## Batch 3 additions
| Aspect | LangGraph (1.2.18) | Pydantic AI (2.62.0) |
|---|---|---|
| Native pre-execution interception | `ToolNode(wrap_tool_call)` | `Hooks(before_tool_execute)`, also `wrap_tool_execute` |
| Trusted run context | `runtime.context` (`context_schema` + `invoke(context=)`) / `configurable` | `ctx.deps` (`deps_type` + `run(deps=)`) |
| Hide tools per identity | bind only permitted tools | `prepare_tools` hook |
| Native approval gate | `interrupt` (not tested) | `requires_approval` / deferred tool calls (default output type errors) |
| Denial as plain exception | fed to the model; run continues | run ends |
| Denial interacts with retry | `RetryPolicy` retries it; model may re-issue | `ModelRetry` denial invites retry; plain exception terminal |
| Forged identity args (tool without those params) | dropped; trusted identity applies | rejected before tool body |
| Tool output delivery | verbatim string / JSON text | verbatim string / compact JSON text |
| Injection protection | none | none |
