# Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Ecosystem + license audit, stack decision | Done (GO WITH LIMITATIONS) |
| 1 | Same agent in LangGraph and Pydantic AI, tracing, eval, closure experiments | GO WITH LIMITATIONS (Langfuse export blocked; see phases/phase-01/phase-1-decision.md) |
| 2 | Deep Tool Calling: audit + Batches 1-5 (experiments 1-15, tool selection included) (phases/phase-02) | Closed |
| 3 | Agent loop / ReAct execution: 3.0 reuse audit done (phases/phase-03); 3.1 experiments not started | 3.0 GO WITH LIMITATIONS |
| — | Memory (Mem0 vs Graphiti vs LangMem comparison); the Phase 3 brief names Phase 4 = State & Context, so later rows are to be re-sequenced at review | Planned |
| 4 | MCP client/server usage; A2A interop | Planned |
| 5 | Multi-agent, planning, reflection, routing | Planned |
| 6 | Reliability: durable execution (DBOS/Temporal), failure injection | Planned |
| 7 | Evaluation + observability at depth | Planned |
| 8 | Browser + coding agents (run existing ones) | Planned |

Every phase begins with a fresh reuse audit for its capability (re-run `gh api` health checks; versions drift fast).
