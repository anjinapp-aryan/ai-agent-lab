# Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Ecosystem + license audit, stack decision | Done (GO WITH LIMITATIONS) |
| 1 | Same agent in LangGraph and Pydantic AI, tracing, eval, closure experiments | GO WITH LIMITATIONS (Langfuse export blocked; see phases/phase-01/phase-1-decision.md) |
| 2 | Deep Tool Calling: audit done; Batches 1-2 done (experiments 1-9, 11, 14); Batches 3-5 pending (phases/phase-02) | In progress |
| 3 | Memory (Mem0 vs Graphiti vs LangMem comparison) | Planned |
| 4 | MCP client/server usage; A2A interop | Planned |
| 5 | Multi-agent, planning, reflection, routing | Planned |
| 6 | Reliability: durable execution (DBOS/Temporal), failure injection | Planned |
| 7 | Evaluation + observability at depth | Planned |
| 8 | Browser + coding agents (run existing ones) | Planned |

Every phase begins with a fresh reuse audit for its capability (re-run `gh api` health checks; versions drift fast).
