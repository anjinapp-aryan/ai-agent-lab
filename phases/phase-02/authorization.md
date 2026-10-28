# Authorization (Experiments 3.1 – 3.7)

Evidence: [experiments.md](experiments.md#batch-3-results-2026-10-28). Authorization here is **application** logic (a dict of roles and permissions plus a tenant check); the frameworks only supply places to run it.

## Answers
1. **Where should authorization live?** At a boundary that every tool call crosses and that the model cannot influence: a framework interception hook (LangGraph `ToolNode(wrap_tool_call)`, Pydantic AI `Hooks(before_tool_execute)`) or inside each tool body, with the business/data layer as the last line. The hook runs before the tool body starts (0 bodies started in the experiments) and covers every tool in that agent; a check inside the body works but depends on every tool author remembering it.
2. **Can the LLM itself be trusted?** No. A prompt rule did not prevent execution (an unguarded tool ran when the model requested it), and live models followed injected instructions 20/20 despite an "only do what the user asked" instruction. The LLM decides what it *wants*; it is not the authority on whether it is allowed.
3. **Can tool arguments contain trusted identity?** No. A tool that read `role`/`tenant`/`authorized` from arguments was bypassed in both frameworks. Identity must come from trusted context: a closure bound by the application, LangGraph `request.runtime.context` (`context_schema` + `invoke(context=...)`), or Pydantic AI `ctx.deps`.
4. **How should tenant/resource authorization work?** As a second check on the *resource*, not only on the *tool*: permission-only authorization let tenant-A cancel and read tenant-B's order; adding the tenant check blocked both. Tool authorization ≠ resource authorization. (The business/data layer should enforce tenant scoping too; the check here is a stand-in.)
5. **Does authorization run before every tool execution?** Only when it is placed where every call passes. In the injected flows 2 of 2 tool calls (the model's first and its injected second call) were checked in each protected variant. Hiding a tool (visibility) is not enforcement.
6. **What happens when it fails?** Framework-specific: LangGraph default feeds the denial to the model and continues; Pydantic AI ends the run on a plain exception; returning a structured denial result keeps both runs alive.
7. **Does framework retry interact badly?** Yes. LangGraph `RetryPolicy` retried a deterministic denial 3×; a `ModelRetry`-style denial invited model retries until the budget ran out; LangGraph's default lets the model re-issue the same denied call. Effects stayed 0 and every attempt was re-checked, but retry is meant for transient failures and a denial is not one.

## Native mechanisms found
| Mechanism | Framework | Before tool body | Trusted context | How a deny surfaces |
|---|---|---|---|---|
| `ToolNode(wrap_tool_call)` | LangGraph | yes | `request.runtime.context` | raise (fed to model by default) or return an error `ToolMessage` |
| `Hooks(before_tool_execute)` | Pydantic AI | yes | `ctx.deps` | raise (ends run) or `ModelRetry` (model-visible, budgeted) |
| `prepare_tools` / bind only permitted tools | Pydantic AI / LangGraph | n/a (visibility) | `ctx.deps` / application | model's call becomes "unknown tool" |
| `Tool(requires_approval=True)` | Pydantic AI | yes (not executed) | n/a | run error unless the output type allows deferred calls; approval resume not tested |

## Design guidance supported by the evidence
- Centralize enforcement in a hook; keep identity out of the model's arguments and out of the tool schema.
- Return a non-retryable, structured denial to the model; keep denials out of `RetryPolicy`/`ModelRetry`.
- Denial messages must not reveal other tenants' details (the fixture's message did).
- Log every decision (the audit trail made "2 of 2 calls checked" provable).
