# C. Tool Calling

Tool abstraction is built into every framework in agent-frameworks.md. Do not build a registry.

| Candidate | License | Decision |
|---|---|---|
| Pydantic AI tools (schema from type hints, validation, retry on validation error) `[K]` | MIT `[V]` | REUSE — best study subject for validation + error feedback |
| LangGraph/LangChain tools + ToolNode `[K]` | MIT `[V]` | REUSE |
| OpenAI Agents SDK function tools + guardrails `[K]` | MIT `[V]` | REUSE for comparison |
| MCP (external tools) | see mcp.md | REUSE |
| LiteLLM (provider normalization) | MIT outside `enterprise/` `[V]` | REUSE optional, only if multi-provider needed |

Build: none.
Experiments: malformed tool args, tool exception, tool timeout, prompt injection via tool output.
