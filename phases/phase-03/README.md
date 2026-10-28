# Phase 3 — Agent Loop / ReAct Execution

Status: **3.0 reuse audit done — GO WITH LIMITATIONS.** Phase 3.1 in progress: **Batch 1 (trace the loop) done** — see [experiments.md](experiments.md); Batches 2–10 not started.

| Document | Content |
|---|---|
| [reuse-audit.md](reuse-audit.md) | candidates, health, licenses, reuse decision matrix, maintenance risks |
| [source-evidence.md](source-evidence.md) | verified loop locations (file / function / line, commit SHA) for 9 frameworks + references |
| [agent-loop-mental-model.md](agent-loop-mental-model.md) | framework-free loop, ownership, stopping-condition, failure-mode and state matrices, Q1–Q14 |

## Executive summary
- **The agent loop always lives in the framework, but in different places:**
  - a generic graph runtime driving an application-defined cycle (LangGraph);
  - an agent-run node loop (Pydantic AI, OpenAI Agents SDK, ADK, smolagents);
  - an event-driven router flow (CrewAI);
  - the chat client (Microsoft Agent Framework);
  - the model class (Agno);
  - a recursive event loop (Strands).
- **All of them reduce to the same five operations:** call the model; finish if there are no tool calls; execute the calls; append the results; check external stop conditions.
- **The model decides continuation and natural termination.** Everything else that stops a loop is a framework counter or an application decision.
- **Iteration-cap defaults span 10 to 10007 for the same concept.** LangGraph 1.2.18's default `recursion_limit` is 10007 (source), not the "25" remembered from older versions.
- **No framework detects repeated calls, oscillation or lack of progress.** Only count caps exist.
- **Decision:** REUSE LangGraph + Pydantic AI (already installed; no new dependency); REFERENCE the rest by source reading. Build nothing but fixtures and, if justified, a measurement-only trajectory checker.

## Phase 3.1 recommendation (what to study next, not started)
Use the Phase 1 agents and Phase 2 fixtures; scripted models for mechanics, live runs only for model behavior.
1. **Trace one loop end to end** in both frameworks: every model request, tool call and observation, with step counters. Confirm the five-part model against running behavior, not just source.
2. **Stopping conditions:** natural stop; each cap at its default and at a low value; timeout; cancellation; policy denial; `tool_choice=required`. For each, record who stopped it, what state and messages survive, and whether side effects are left dangling.
3. **Failure modes A–G:**
   - Provoke them with scripted models (deterministic) and with tool outputs that invite repetition or oscillation (live, N≤5).
   - Measure cost (requests, tokens) until each framework's guard fires.
4. **Retries inside the loop:** model re-issue vs framework retry vs iteration cap, and the side-effect count per stop.
5. **Human interruption** (left untested in Phase 2): LangGraph `interrupt` and Pydantic AI deferred tools, observed as a loop stop and resume (in-memory checkpointer only).
6. **Manual stepping** (`agent.iter()` / `graph.stream`) as the application's hook into each iteration.
7. **Reference reading:** compare the one-page mini-swe-agent loop with both frameworks to separate the essential parts from framework machinery.

## What we must NOT build
- an agent loop, ReAct engine, `AgentEngine` or `AgentRuntime`
- a tool dispatcher or executor
- an iteration or recursion limiter
- a timeout or cancellation layer
- a retry framework
- a checkpointer or state store
- an interrupt or approval system
- a tracing system
- a text-ReAct parser (CrewAI, smolagents and the ReAct paper exist; native tool calling replaces it)
- a loop-control middleware (LangChain has limit middleware; frameworks have hooks)

## Open questions (for experiments)
1. What survives when each stop condition fires (messages, usage, partial results), in each framework?
2. How many requests and tokens does a looping agent burn before each default cap fires?
3. Does the live model repeat identical calls after an error or an empty result, and how often?
4. Can contradictory tool outputs make the model oscillate?
5. Can no-progress be detected from the trajectory alone, without task knowledge?
6. How do framework retries count against iteration caps (does a retry consume a step)?
7. Does `remaining_steps`-style "last step" signalling (LangGraph prebuilt) change model behavior near the cap?
8. How does stop/resume via interrupt behave when a side-effecting tool was in flight?

## PHASE 3.0 GO / NO-GO
**GO WITH LIMITATIONS**
- **GO because:**
  - Mature, MIT-licensed loops already run our agents.
  - Their source locations and stop mechanics are verified.
  - No new dependency is needed.
  - The open questions are measurable with existing fixtures.
- **Limitations:**
  - Seven frameworks were studied by source only, not run.
  - Line numbers are version-specific.
  - The "no loop detection" finding comes from keyword search of the loop files, not whole repositories.
  - Live experiments remain bound by Groq quotas (200K tokens per day per model).
  - Langfuse export is still blocked.
  - Any trajectory checker needs a decision record before it is written.
