# A. Agent Frameworks (+ B. Runtime, C. Tool-calling basics)

Health `[V]` 2026-10-27. All pushed within 5 days except smolagents (pushed 10-22; last release 2026-05-29, ~5 months, slowing).

| Project | License | Latest release | Notes | Decision |
|---|---|---|---|---|
| LangGraph | MIT | 1.2.18 (10-26) | Graph/state machine, checkpointers, interrupts `[K]` | REUSE (primary: state, HITL, multi-agent) |
| Pydantic AI | MIT | v2.62.0 (10-27) | Typed agents, validation, MCP/A2A/durable-exec integrations `[K]` | REUSE (primary: tool calling, typed outputs) |
| OpenAI Agents SDK | MIT | v0.24.0 (10-15) | Minimal: agents, handoffs, guardrails, tracing `[K]`; pre-1.0 | REUSE (minimal-loop reading, comparison) |
| Google ADK | Apache-2.0 | v2.13.0 (10-23) | Multi-agent, workflow agents, A2A `[K]` | REUSE for comparison (Phase 5) |
| Microsoft Agent Framework | MIT | dotnet-1.26.0 (10-23) | AutoGen + Semantic Kernel successor `[K]`; .NET + Python | REFERENCE ONLY (Python-first lab) |
| CrewAI | MIT | 1.15.30 (10-24) | Role/task crews, flows | REFERENCE ONLY; run once in Phase 5 for contrast |
| smolagents | Apache-2.0 | v1.26.0 (05-29) | Code-as-action agents, small readable codebase `[K]` | REUSE for reading/learning |
| Agno | Apache-2.0 | v3.1.10 (10-23) | Batteries-included agents + runtime | REFERENCE ONLY (overlap, sprawl) |
| Haystack | Apache-2.0 | v3.4.1 (10-21) | Pipelines, RAG-first | REFERENCE ONLY |
| LangChain | MIT | — | Integration layer under LangGraph | COMPOSE (transitive, only as needed) |
| AutoGen | repo license reported as CC-BY-4.0 (unusual for code) | — | Superseded by Agent Framework `[K]` | REJECT for new work; LEGAL REVIEW REQUIRED if ever used |
| OpenAI Swarm | MIT | — | Educational; pushed 2026-04-15, replaced by Agents SDK `[K]` | REFERENCE ONLY |
| Claude Agent SDK (Python) | MIT | v0.2.174 (10-24) | Claude Code loop as library; Claude-oriented | REFERENCE ONLY (coding-agent runtime study) |

Architecture contrast to teach: graph state machine (LangGraph) vs typed function-style agents (Pydantic AI) vs minimal handoff loop (OpenAI SDK) vs code-as-action (smolagents).

Lock-in: OpenAI SDK and Claude SDK lean toward their vendors; LangGraph, Pydantic AI, ADK are model-agnostic `[K]`.
