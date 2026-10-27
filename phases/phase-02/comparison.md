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
