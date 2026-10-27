# K. A2A (agent ↔ agent protocol)

Distinction: framework = builds the agent; **MCP** = agent ↔ tools; **A2A** = agent ↔ agent across vendors/processes.

## Status
- `a2aproject/A2A`: Apache-2.0, latest GitHub release **v1.0.1** (2026-05-28), pushed 2026-10-21 `[V]`.
- Linux Foundation governance, 150+ supporting organizations (LF press release, 2026-04-09) `[S]`.
- A claim of "v1.2 with signed Agent Cards" appeared in one secondary source `[S]` and is **unconfirmed**; GitHub shows v1.0.1 as latest release. Treat v1.0.x as the target and recheck.
- `a2aproject/a2a-python`: Apache-2.0, v1.3.0 (2026-10-27), pushed same day `[V]`.
- `a2aproject/a2a-samples`: Apache-2.0, pushed 10-20 `[V]`.
- Concepts: Agent Card (discovery/capabilities), tasks, async/streaming messages `[K]`.

## Decisions
| Item | Decision |
|---|---|
| a2a-python SDK | REUSE (Phase 4) |
| a2a-samples | REUSE / ADAPT as study material |
| Framework A2A adapters (ADK, Pydantic AI, Agent Framework) `[K]` | COMPOSE |
| Own A2A implementation | BUILD forbidden |

Risk: younger than MCP; ecosystem adoption claims are vendor-sourced. Do a real two-process interop test before trusting.
