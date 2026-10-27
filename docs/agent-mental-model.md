# Agent Mental Model

An agent = model + tools + loop + state. Loop: observe → decide (LLM) → act (tool) → update state → stop or repeat.

Layers (each audited separately in audits/phase-00):
- **Framework**: defines loop, tools, state, handoffs (LangGraph, Pydantic AI, ADK, ...).
- **Runtime**: durability, retries, resume (checkpointers, Temporal, DBOS).
- **Tool protocol** (MCP): agent ↔ tools/data.
- **Agent protocol** (A2A): agent ↔ agent across vendors/processes.
- **Memory**: persistence beyond the context window.
- **Evals + observability**: trajectories, traces, regression.

MCP is not A2A; neither is a framework.
