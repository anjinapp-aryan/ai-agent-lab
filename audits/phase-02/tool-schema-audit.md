# Tool Schemas (capability A)

## Candidates
| Project | Repo | License | Latest | Pushed | Relevant capability | Decision |
|---|---|---|---|---|---|---|
| Pydantic | pydantic/pydantic | MIT | v2.14.1 (10-11) | 10-27 | Model/type -> JSON Schema, validation; underlies both frameworks `[V]` | REUSE (already) |
| jsonschema | python-jsonschema/jsonschema | MIT | v4.26.0 (2026-01-07) | 10-26 | Validate arbitrary JSON against schema | REFERENCE: only needed to validate *received* schemas independently; not required |
| Instructor | 567-labs/instructor | MIT | v1.17.0 (09-09) | 10-26 | Structured output via Pydantic with retry-on-validation | REFERENCE: overlaps Pydantic AI output validation |
| Outlines | dottxt-ai/outlines | Apache-2.0 | 1.3.3 (08-06) | 09-21 | Constrained decoding (schema-guaranteed output) for local models | REFERENCE: not applicable to hosted tool-calling APIs |
| Guardrails AI | guardrails-ai/guardrails | Apache-2.0 | v0.11.0 (08-14) | 10-27 | Output validators/re-ask | REFERENCE (see validation audit) |
| LangChain tools | langchain-ai/langchain | MIT | core 1.7.0 | 10-27 | `convert_to_openai_tool`, `StructuredTool`, `args_schema` | REUSE (already) |
| Pydantic AI tools | pydantic/pydantic-ai | MIT | v2.62.0 | 10-27 | schema from type hints/docstrings; `strict`, `function_schema`, `schema_generator`, `include_return_schema` `[V]` | REUSE (already) |

## Verified: schema sent to the model for the same typed tool (PoC, `convert(amount: float, currency: Literal["USD","EUR","INR"])`)
- LangChain: `{"properties": {"amount": {"type":"number"}, "currency": {"enum":[...],"type":"string"}}, "required": [...], "type":"object"}` — **no `additionalProperties`**.
- Pydantic AI: same properties plus **`"additionalProperties": false`**.
Docstring becomes the tool description in both (Phase 1).

## Findings
- Both generate JSON Schema from Python type hints via Pydantic; enums come from `Literal`.
- The schema is only a *request* to the model; enforcement happens in the validation layer (next audit). The two frameworks differ on closing the schema (`additionalProperties`), which changes extra-argument behavior.
- Provider-side `strict` mode: Pydantic AI exposes `Tool(strict=...)`; whether Groq/gpt-oss honors strict schemas is `[K]`/unverified. Experiment 1 should record the exact payload sent.
- Do not build schema tooling. Nothing else here needs adding.
