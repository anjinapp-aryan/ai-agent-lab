# L. Human-in-the-Loop

| Candidate | License | Decision |
|---|---|---|
| LangGraph `interrupt` + checkpointer resume, edit state, approve/reject `[K]` | MIT `[V]` | REUSE — primary |
| OpenAI Agents SDK tool approval / guardrails `[K]` | MIT `[V]` | REFERENCE |
| Pydantic AI deferred tools / approval `[K]` | MIT `[V]` | REUSE for contrast |
| Google ADK tool confirmation `[K]` | Apache-2.0 `[V]` | REFERENCE |
| OpenHands / Codex approval modes | MIT / Apache-2.0 `[V]` | REFERENCE — real-world UX |

Build: none (CLI prompt UI is trivial glue).
Tests: approval after process restart (needs durable checkpoint), rejection with feedback, timeout on no human.
