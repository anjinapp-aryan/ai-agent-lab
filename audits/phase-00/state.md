# D. Agent State / Checkpointing

| Candidate | License | Decision |
|---|---|---|
| LangGraph state + checkpointers (SQLite/Postgres), time-travel `[K]` | MIT `[V]` | REUSE — primary |
| Pydantic AI graph / message history `[K]` | MIT `[V]` | REUSE for contrast |
| Google ADK sessions/state `[K]` | Apache-2.0 `[V]` | REFERENCE |
| OpenAI Agents SDK sessions `[K]` | MIT `[V]` | REFERENCE |

Build: none. Lesson: separate conversation history, graph state, and durable workflow state.
