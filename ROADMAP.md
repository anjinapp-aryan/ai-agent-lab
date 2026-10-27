# Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Ecosystem + license audit, stack decision | Done (GO WITH LIMITATIONS) |
| 1 | Run + inspect one agent loop with tools, tracing, tiny eval (see final-decision.md §Phase 1) | GO WITH LIMITATIONS: live runs + Langfuse export pending (phases/phase-01/phase-1-decision.md) |
| 2 | State, checkpointing, human-in-the-loop (LangGraph) | Planned |
| 3 | Memory (Mem0 vs Graphiti vs LangMem comparison) | Planned |
| 4 | MCP client/server usage; A2A interop | Planned |
| 5 | Multi-agent, planning, reflection, routing | Planned |
| 6 | Reliability: durable execution (DBOS/Temporal), failure injection | Planned |
| 7 | Evaluation + observability at depth | Planned |
| 8 | Browser + coding agents (run existing ones) | Planned |

Every phase begins with a fresh reuse audit for its capability (re-run `gh api` health checks; versions drift fast).
