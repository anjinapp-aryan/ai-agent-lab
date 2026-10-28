# Phase 3.0 — Agent Loop / ReAct Execution: Reuse Audit

Date 2026-10-28. Health `[V]` via `gh api repos/...` and PyPI JSON. Loop locations: [source-evidence.md](source-evidence.md). Mental model and ownership: [agent-loop-mental-model.md](agent-loop-mental-model.md).

## Candidates
`deps` = number of `requires_dist` entries on PyPI, including extras: a rough footprint signal, not the installed size.

| Project | License `[V]` | Stars | Last push | Latest release | Open issues | Loop style | Recommendation |
|---|---|---|---|---|---|---|---|
| langchain-ai/langgraph | MIT | 44.0k | 2026-10-28 | 1.2.18 (10-26), deps 6 | 498 | generic Pregel super-step runtime; agent loop = graph cycle chosen by a conditional edge | **REUSE** (already in the stack) |
| pydantic/pydantic-ai | MIT | 20.9k | 2026-10-28 | 2.63.0 (10-28), deps 70 | 1579 | node graph `UserPrompt → ModelRequest → CallTools → … → End`; `agent.iter()` steps it | **REUSE** (already in the stack) |
| openai/openai-agents-python | MIT | 30.4k | 2026-10-27 | 0.24.0 (10-15), deps 56 | 10 | `while True` turn loop; `max_turns` 10; `tool_use_behavior` | **REFERENCE** (pre-1.0; Responses-API centric) |
| google/adk-python | Apache-2.0 | 22.0k | 2026-10-28 | 2.13.0 (10-23), deps 274 | 396 | `while True` over steps until a final-response event; `max_llm_calls` 500 | **REFERENCE** (heavy footprint) |
| microsoft/agent-framework | MIT | 14.4k | 2026-10-28 | core 1.23.0 (10-22), deps 38 | 846 | loop inside the chat-client function-invocation layer; `max_iterations` 40 | **REFERENCE** |
| crewAIInc/crewAI | MIT | 60.2k | 2026-10-28 | 1.15.32 (10-27), deps 57 | 861 | event-driven `Flow` routers; text ReAct parser + native tools; `max_iter` 25 | **REFERENCE** |
| huggingface/smolagents | Apache-2.0 | 30.1k | 2026-10-22 | 1.26.0 (**2026-05-29**), deps 53 | 1018 | `while not final and step <= max_steps`; code-as-action option | **REFERENCE** (release cadence slowed) |
| agno-agi/agno | Apache-2.0 | 42.9k | 2026-10-28 | 3.1.11 (10-27), deps 406 | 1939 | `while True` inside `Model.response`; `tool_call_limit` | **REFERENCE** (very large footprint; issue load) |
| strands-agents/harness-sdk (was sdk-python) | Apache-2.0 | 9.0k | 2026-10-28 | strands-agents 1.62.0 (10-21), deps 123 | 1063 | recursive `event_loop_cycle` on `stop_reason` | **REFERENCE** |
| langchain-ai/langchain (`create_agent`) | MIT | 148.6k | 2026-10-28 | langchain 1.4.7 (10-21), deps 22 | 691 | builds a LangGraph graph; limit middleware | **REFERENCE**; declined dependency (Phase 2) — would need a user decision |
| SWE-agent/mini-swe-agent | MIT | 8.9k | 2026-10-26 | 2.4.6 (07-23), deps 35 | 126 | 190-line `while True: step()`; exit message, step/cost/time limits | **REFERENCE** (best minimal loop to read; executes shell commands, so not run here) |
| ysymyth/ReAct | MIT | 4.2k | 2024-02-06 | none | 5 | paper notebooks, prompt-based ReAct | **REFERENCE** (historical) |
| microsoft/autogen | CC-BY-4.0 (docs) + MIT (`LICENSE-CODE`) | 61.6k | 2026-10-16 | python-v0.7.5 (2025-09-30) | 1124 | — | **REJECT**: README says it is transitioning to archived and advises against production use; successor is Agent Framework |
| run-llama/llama_index | MIT | 52.7k | 2026-10-26 | v0.14.25 (09-21) | 1087 | `ReActAgent` `[K]`, not inspected | **NOT AUDITED**: no capability beyond the above |

## Reuse decision matrix
| Capability | Existing OSS (evidence) | License | Mature? | Decision | Reason |
|---|---|---|---|---|---|
| Agent loop | LangGraph Pregel + graph cycle; Pydantic AI `AbstractAgent.run` | MIT | yes | **REUSE** | both already run our agents; source located |
| ReAct execution | native tool calling in both; text ReAct in CrewAI `parser.py`, smolagents, ReAct paper | MIT / Apache | yes | **REUSE** (native tool calling) + **REFERENCE** (text ReAct) | modern providers return structured tool calls; text parsing is the historical form |
| Tool dispatch | LangGraph `ToolNode`; Pydantic AI `CallToolsNode` | MIT | yes | **REUSE** | measured in Phase 2 |
| Iteration limit | `recursion_limit`; `UsageLimits.request_limit` | MIT | yes | **REUSE** | defaults differ wildly (10007 vs 50): a Phase 3 measurement target |
| Recursion protection | same | MIT | yes | **REUSE** | — |
| Tool-call limit | `UsageLimits.tool_calls_limit`; LangGraph: none native (LangChain middleware, declined) | MIT | partial | **REUSE** (PA) / **REFERENCE** (LangChain) | Phase 2 Batch 1 measured PA's check |
| Timeout | LangGraph `step_timeout`; PA `run_timeout`, `Tool(timeout)` | MIT | yes | **REUSE** | Phase 2 Batch 2 |
| Cancellation | asyncio task cancel; PA `AgentRun.cancel()`; LangGraph drain/interrupt | MIT | yes | **REUSE** | Phase 2 Batch 4 measured sync vs async |
| Stop conditions | `tools_condition` (LG); `End` from `CallToolsNode` (PA) | MIT | yes | **REUSE** | the model's output picks the branch |
| Loop state | LangGraph `MessagesState`/channels; PA message history + `RunUsage` | MIT | yes | **REUSE** | Phase 4 owns state in depth |
| Checkpointing | `langgraph-checkpoint` (installed); PA none in-core `[S]`, durable via DBOS/Temporal integrations `[S]` | MIT | yes | **REUSE** (LangGraph, in-memory saver only) | needed only to observe resume and replay of a loop |
| Failure recovery | LG error `ToolMessage`/`RetryPolicy`; PA `ModelRetry`/`retries` | MIT | yes | **REUSE** | Phase 2 Batch 2 |
| Parallel tool execution | `ToolNode` thread pool; PA parallel mode | MIT | yes | **REUSE** | Phase 2 Batch 4 |
| Human interruption | LangGraph `interrupt` / `interrupt_before`; PA deferred tools / `requires_approval` | MIT | yes | **REUSE** | not yet measured (Phase 2 left HITL untested) |
| Observability | OTel + OpenInference (Phase 1) | Apache-2.0 | yes | **REUSE** | Langfuse still blocked |
| Loop/oscillation/no-progress detection | none found in any loop file (keyword search) | — | **no** | **BUILD (experiment-only detector over recorded trajectories)** if Phase 3.1 needs it | see the decision note below |

### Decision note: the only BUILD candidate
- **Capability:** detect repeated identical calls, A↔B oscillation and no-progress in a recorded trajectory.
- **Existing OSS considered:** per-tool caps (Pydantic AI `tool_calls_limit`, LangChain `ToolCallLimitMiddleware`, Agno `tool_call_limit`, MAF `max_function_calls`) limit counts but do not recognize patterns.
- **What would be built:** a pure function `trajectory → findings` over `RunResult.tool_calls` (≈20 lines), used for *measurement only*: no loop control, no middleware, no framework hook.
- **Excluded:** anything that intervenes in the loop. If intervention is needed, use the frameworks' own hooks (`wrap_tool_call`, `Hooks`, limits).
- **Status:** not built in 3.0. Requires a written decision record in Phase 3.1 if used.

## License review
| Classification | Projects |
|---|---|
| SAFE TO REUSE | LangGraph (MIT), Pydantic AI (MIT), langgraph-checkpoint (MIT, installed), OpenTelemetry/OpenInference (Apache-2.0) |
| SAFE FOR LEARNING (reference reading) | OpenAI Agents SDK (MIT), ADK (Apache-2.0), Agent Framework (MIT), CrewAI (MIT), smolagents (Apache-2.0), Agno (Apache-2.0; per-package LICENSE files under `libs/`), Strands (Apache-2.0), LangChain (MIT), mini-swe-agent (MIT), ReAct (MIT) |
| LEGAL REVIEW REQUIRED (only if ever reused) | **Agno**: Apache-2.0 at repo level, but its platform/hosted features (AgentOS UI, cloud) are separate products `[K]`; the repo has multiple LICENSE files `[V]`. **CrewAI**: MIT core, but the enterprise platform (AMP) is commercial `[K]`. Neither is reused here |
| REJECT | **AutoGen**: transitioning to archived. Its repo license is detected as CC-BY-4.0 for docs, with MIT code in `LICENSE-CODE`; the license is not the reason for rejection, the maintenance status is |
| Note | Several PyPI entries (`google-adk`, `agent-framework-core`, `crewai`, `smolagents`) carry no `license` field in metadata; their repository LICENSE files were used |

## Maintenance risks
- **Fast-moving versions:** Pydantic AI released 2.63.0 on the audit day, and LangGraph 1.2.18 is two days old. Pin versions; re-verify line numbers per version.
- **LangGraph's prebuilt ReAct agent is deprecated** in favor of LangChain `create_agent`. Our explicit `StateGraph` is unaffected; tutorials based on `create_react_agent` will drift.
- **Strands moved repository** (`sdk-python` → `harness-sdk`).
- **smolagents' last release is 5 months old** despite recent pushes.
- **High open-issue counts:** Agno 1939, Pydantic AI 1579. Activity, not quality: they do not by themselves disqualify a project.
- **OpenAI Agents SDK is pre-1.0** (0.24.0).

## Conclusion
**REUSE** LangGraph and Pydantic AI (no new dependency). **REFERENCE** the other seven frameworks, mini-swe-agent and the ReAct paper by source reading only. No framework or loop is to be written. The single possible BUILD is a measurement-only trajectory checker, deferred to Phase 3.1 with a decision record.
