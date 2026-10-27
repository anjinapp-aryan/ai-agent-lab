# Tool Schema — who creates it, and what is sent (Experiment 1)

Evidence: wire-level capture of the real `ChatOpenAI` and `OpenAIChatModel` clients (mock transport), `calculate(a: int, b: int) -> int`.

| | LangGraph / LangChain | Pydantic AI |
|---|---|---|
| Creator | LangChain tool layer from type hints + docstring | Pydantic AI from type hints + docstring |
| `parameters` | `properties`, `required`, `type:object` | same + **`additionalProperties: false`** |
| `strict` on the tool definition | not sent | **`"strict": true`** |
| Description | docstring | docstring |
| Tool result on wire | `{"role":"tool","content":"5"}` | same |

Architectural answers
- **Who creates the schema?** The framework, deterministically, from the Python signature. The model and the provider only consume it.
- **Is the schema a contract?** It is advisory to the model. Enforcement is the framework's validation step; whether extras are rejected is decided by the framework (see validation.md), and `strict: true` additionally asks the provider to constrain decoding (provider support for gpt-oss on Groq not tested in Batch 1; Phase 1 live runs with Pydantic AI worked).
- Consequence: the same Python function advertises a different contract through each framework. Anything that must be strict should say so explicitly rather than rely on a framework default.
