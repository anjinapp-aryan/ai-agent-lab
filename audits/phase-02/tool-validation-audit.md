# Tool Argument and Result Validation (capabilities C, F)

## Verified behavior (PoC, scripted model; `convert(amount: float, currency: Literal[USD,EUR,INR])`)
| Case | LangGraph 1.2.18 | Pydantic AI 2.62.0 |
|---|---|---|
| Valid | executes | executes |
| Missing required arg | rejected: `ToolInvocationError ... currency: Field required...` fed to model | rejected: `retried` with `[{'type':'missing','loc':['currency']}]` |
| Wrong type (`"hello"` for float) | rejected, error fed to model | rejected, `float_parsing` error, `retried` |
| **Extra arg (`secret="unexpected"`)** | **EXECUTED; extra silently dropped** | **rejected**: `extra_forbidden`, `retried` |
| Invalid enum (`"XYZ"`) | rejected, error fed to model | rejected: `literal_error: Input should be 'USD','EUR' or 'INR'` |
| Coercible string (`"10"` for float) | executes (coerced to 10.0) | executes (coerced to 10.0) |

`[V]` all rows. Consequences: (1) LangGraph's default is *permissive on extras* — a model-supplied unexpected parameter vanishes without any signal; (2) both coerce lax types, so "wrong type" is only rejected when coercion fails.

## Pydantic AI hooks `[V]` (installed signature)
`Tool(args_validator=..., strict=..., max_retries=..., prepare=..., require_parameter_descriptions=...)`; `Agent(validation_context=..., output_type=...)`; `ModelRetry` for tool-raised retry requests. Custom `args_validator` allows semantic checks after schema validation.

## LangChain/LangGraph hooks
`StructuredTool(args_schema=...)` with a Pydantic model can set `extra="forbid"` `[K]` to close the extras gap (to be verified in Experiment 4); `ToolException`/`handle_tool_error` `[V names]`; `InjectedToolArg` hides arguments from the model schema `[V name]`; `ToolNode(wrap_tool_call=...)` interception point `[V param]`; langchain middleware `tool_error.py` `[F]`.

## Result validation (capability F)
- Pydantic AI: return type annotations; `include_return_schema` exposes return schema to the model `[V param]`; results are serialized into `ToolReturnPart` `[V]`.
- LangGraph: tool return is stringified into `ToolMessage.content` (Phase 1: `165.0`) — no result schema validation by default `[K]`.
- Truncation of huge results: not found as a built-in in either from signatures; verify in Experiment 11.

## Libraries
| Project | License | Latest | Pushed | Decision |
|---|---|---|---|---|
| Pydantic | MIT | v2.14.1 | 10-27 | REUSE — the validator in both frameworks |
| Guardrails AI | Apache-2.0 | v0.11.0 (08-14) | 10-27 | REFERENCE — output re-ask validators; overlaps framework hooks |
| Instructor | MIT | v1.17.0 | 10-26 | REFERENCE |
| NeMo Guardrails | LICENSE.md header says Apache-2.0 (GitHub: NOASSERTION) | v0.24.1 (09-16) | 10-27 | REFERENCE — dialogue rails, heavier than needed; verify full LICENSE.md before any use |

Build: none. Result validation for Experiment 11 uses Pydantic models / framework hooks only.
