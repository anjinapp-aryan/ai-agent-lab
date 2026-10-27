# Tool-Calling Ecosystem (capability B)

## Layers (who can own what)
```
Model provider API (tool schema in, tool_call out)   <- OpenAI-compatible, Anthropic, Gemini SDKs
Schema layer        (function/Pydantic -> JSON Schema)
Validation layer    (JSON args -> typed args)
Policy layer        (allowlist, approval, limits)
Runner layer        (execute, timeout, concurrency, retry)
Result layer        (serialize back into the model's context)
Observability layer (spans)
```
Frameworks bundle several layers; the experiments in Phase 2 identify which layer owns each behavior.

## Frameworks and SDKs (all `[V]` from GitHub API, 2026-10-27)
| Project | Repo | License | Stars | Latest release | Pushed | Tool-calling role |
|---|---|---|---|---|---|---|
| LangGraph | langchain-ai/langgraph | MIT | 44.0k | 1.2.18 (10-26) | 10-27 | `ToolNode`, `RetryPolicy`, `interrupt`, `Send` |
| LangChain | langchain-ai/langchain | MIT | 148.6k | langchain-core 1.7.0 (10-22); `langchain` pkg 1.4.7 on PyPI, MIT | 10-27 | tool abstractions; `agents/middleware` (tool_retry, tool_call_limit, human_in_the_loop, tool_error, tool_selection, tool_emulator, model_retry, pii, shell_tool) `[F]` |
| Pydantic AI | pydantic/pydantic-ai | MIT | 20.9k | v2.62.0 (10-27) | 10-27 | `Tool(max_retries, args_validator, strict, sequential, requires_approval, timeout, prepare)`, toolsets, `UsageLimits.tool_calls_limit` `[V]` |
| OpenAI Agents SDK | openai/openai-agents-python | MIT | 30.3k | v0.24.0 (10-15) | 10-27 | function tools, guardrails `[K]`; pre-1.0 |
| Google ADK | google/adk-python | Apache-2.0 | 22.0k | v2.13.0 (10-23) | 10-27 | tool confirmation, parallel tool execution `[K]` |
| Microsoft Agent Framework | microsoft/agent-framework | MIT | 14.3k | dotnet-1.26.0 (10-23) | 10-27 | approval modes `[K]`; .NET-first release tags |
| smolagents | huggingface/smolagents | Apache-2.0 | 30.1k | v1.26.0 (05-29) | 10-22 | code-as-action tools `[K]` |
| LlamaIndex | run-llama/llama_index | MIT | 52.7k | v0.14.25 (09-21) | 10-26 | function tools `[K]`; RAG-centric, REFERENCE |
| OpenAI Python SDK | openai/openai-python | Apache-2.0 | 32.0k | v3.40.0 (10-27) | 10-27 | tool-call wire format |
| Anthropic Python SDK | anthropics/anthropic-sdk-python | Apache-2.0 | 4.0k | v2.3.0 (10-23) | 10-23 | tool-use wire format; tool runner `[K]` |
| MCP Python SDK | modelcontextprotocol/python-sdk | MIT | 24.7k | v2.3.0 (10-02) | 10-27 | out of Phase 2 scope (no MCP) |

Adoption signal here = stars; stars were not used for decisions. Provider wire formats are accessed through the two frameworks, not directly.

## Conclusion
Phase 1's two frameworks already cover schema, validation, retry, approval, limits and concurrency hooks. No additional framework is needed; adding `langchain` (the package, for middleware) is the only framework-adjacent addition worth evaluating (see reuse-decisions.md).
