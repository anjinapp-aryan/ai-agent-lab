# H. Routing

| Option | Decision |
|---|---|
| Framework-native: LangGraph conditional edges, OpenAI SDK handoffs, ADK sub-agent transfer, Pydantic AI delegation `[K]` | REUSE |
| LiteLLM router for model routing/fallback (MIT outside `enterprise/`, `[V]`) | REUSE optional |
| Dedicated semantic-router libraries | Not evaluated; audit in Phase 5 if needed |

Build: none.
