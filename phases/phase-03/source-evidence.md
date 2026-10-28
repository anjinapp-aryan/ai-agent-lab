# Phase 3.0 — Source Evidence: where each framework's agent loop lives

Collected 2026-10-28. Evidence levels:
- `[V-installed]`: read in this repo's `.venv` (the exact versions we run).
- `[V-GitHub]`: read from the default branch at the commit SHA shown, via `gh api .../contents`.
- `[S]`: documentation or secondary source.
- `NOT VERIFIED`: location not confirmed.

Line numbers are for the version or commit given and will drift.

## LangGraph 1.2.18 `[V-installed]`
| What | Where |
|---|---|
| Loop | `langgraph/pregel/main.py`, `Pregel.stream`: `while loop.tick():` (sync L3055, async L3526); each tick runs the ready tasks through `runner.tick(..., timeout=self.step_timeout)` |
| One iteration | `langgraph/pregel/_loop.py`, `PregelLoop.tick` (L650) |
| Stop: out of steps | `self.step > self.stop` → `status = "out_of_steps"` |
| Stop: natural | no tasks left → `"done"` |
| Stop: interrupt | `interrupt_before` → `GraphInterrupt` |
| Stop: drain | `drain_requested` → `"draining"` |
| Step budget | `self.stop = self.step + config["recursion_limit"] + 1` (L1887) |
| Limit error | `GraphRecursionError("Recursion limit of N reached without hitting a stop condition")` (`main.py` L3102) |
| **Default limit** | `DEFAULT_RECURSION_LIMIT = 10007` (env `LANGGRAPH_DEFAULT_RECURSION_LIMIT`, `langgraph/_internal/_config.py` L42) |
| Continue-or-stop decision | `langgraph/prebuilt/tool_node.py`, `tools_condition` (L1616): last message has `tool_calls` → `"tools"`, else `"__end__"` |
| Prebuilt ReAct agent | `langgraph/prebuilt/chat_agent_executor.py`, `create_react_agent` (L278), **deprecated**: "moved to `langchain.agents` … `create_agent`"; uses `remaining_steps` (default 25 in `AgentStatePydantic`) to force a stop message before the limit |

**Observation:** the *runtime* loop (Pregel super-steps) is generic and knows nothing about models or tools. The *agent* loop is a graph cycle `agent → (tools_condition) → tools → agent`, written by the application (our `lg_agent.build_graph`) or a prebuilt. The model's output (presence of `tool_calls`) chooses the edge.

The default limit of 10007 is no practical guard for a looping agent (~5000 model calls); Phase 1 sets `recursion_limit = 2*MAX_STEPS+1 = 17`. Memory said "25" `[K]`; the source says 10007. **Source wins.**

## Pydantic AI 2.62.0 `[V-installed]` (2.63.0 released 2026-10-28)
| What | Where |
|---|---|
| Loop | `pydantic_ai/agent/abstract.py`, `AbstractAgent.run` (L678): `while not isinstance(node, End): node = await agent_run.next(node)` |
| Nodes | `pydantic_ai/_agent_graph.py`: `UserPromptNode` (L696) → `ModelRequestNode` (L1610) → `CallToolsNode` (L2661) → either `ModelRequestNode` (tool results / retry prompt) or `End` via `_handle_final_result` (L3159–3179) |
| Request limit | `UsageLimits.check_before_request` before every model request (`_agent_graph.py` L2157); `UsageLimits.request_limit` default **50**, `tool_calls_limit` default None (`pydantic_ai/usage.py` L528, L537); `check_before_tool_call` (L687) |
| End strategy | `end_strategy` default changed to `'graceful'` in v2 (`_agent_graph.py` L198) |
| Cancellation | checked in `AgentRun.__anext__` (`pydantic_ai/run.py` L343+); `AgentRun.cancel()` (L776) |
| Manual stepping | `agent.iter()` → `AgentRun.next(node)` (`run.py` L579): the application can drive or stop the loop node by node |

**Observation:** the loop is framework code over a small graph library (`pydantic_graph`). The stop decision is the model's (text/output vs tool calls), checked by `CallToolsNode`.

## OpenAI Agents SDK 0.24.0 `[V-GitHub openai/openai-agents-python@cd26bbbc52]`
| What | Where |
|---|---|
| Loop | `src/agents/run.py`, `AgentRunner._run_impl` (L629): `while True:` (L1032); streaming: `src/agents/run_internal/run_loop.py`, `start_streaming` (L987), `while True:` (L1353) |
| Turn limit | `max_turns` default `DEFAULT_MAX_TURNS = 10` (`run_config.py` L45); `if max_turns is not None and current_turn > max_turns` → `MaxTurnsExceeded` unless an error handler is configured (`run.py` L1598) |
| Next step | `src/agents/run_internal/run_steps.py`: `NextStepFinalOutput`, `NextStepRunAgain`, `NextStepHandoff`, `NextStepInterruption` |
| Stop after tools | `tool_use_behavior` = `"run_llm_again"` (default), `"stop_on_first_tool"`, `{"stop_at_tool_names": [...]}` or a callable (`run_internal/turn_resolution.py` L810–832) |

## Google ADK 2.13.0 `[V-GitHub google/adk-python@3fbf2b5c22]`
| What | Where |
|---|---|
| Loop | `src/google/adk/flows/llm_flows/base_llm_flow.py`, `BaseLlmFlow.run_async` (L288): `while True:` → `_run_one_step_async`; `break` if `invocation_context.is_aborted`, no event, `last_event.is_final_response()`, or a partial event |
| Call limit | `RunConfig.max_llm_calls`, default **500** (env `ADK_MAX_LLM_CALLS`; `src/google/adk/agents/run_config.py` L38–48) |

## Microsoft Agent Framework (Python `agent-framework-core` 1.23.0) `[V-GitHub microsoft/agent-framework@4a272d2b8e]`
| What | Where |
|---|---|
| Loop | in the **chat-client layer**, not the agent: `python/packages/core/agent_framework/_tools.py`, `FunctionInvocationLayer` (L5577) `._get_response_with_function_invocation` (L5719): `for attempt_idx in range(attempt_start, max_iterations)` (L5812; streaming L6090) |
| Limits | `FunctionInvocationConfiguration` (L1871): `max_iterations` (default `DEFAULT_MAX_ITERATIONS = 40`, L109), `max_function_calls`, `max_duration_seconds`, `max_consecutive_errors_per_request` (default 3, L110), `terminate_on_unknown_calls` (default False) |

## CrewAI 1.15.32 `[V-GitHub crewAIInc/crewAI@b95aa8bbde]`
| What | Where |
|---|---|
| Loop | `lib/crewai/src/crewai/agents/agent_executor.py`, `AgentExecutor(Flow[...])` (L187): an event-driven `Flow` of `@router` methods, not a `while`. Cycle: `check_max_iterations` (L2416) → `call_llm_and_parse` (L1624) → `route_by_answer_type` (L1815) → `execute_tool_action` (L1829) → back |
| Limits | `max_iter` default **25** (L211); `max_method_calls = max_iter * 10` (L262); `force_final_answer` route → `handle_max_iterations_exceeded` (L1600–1610) |
| Text ReAct | `agents/parser.py`, `parse()` (L62): `Thought / Action / Action Input / Final Answer` → `AgentAction` / `AgentFinish`; a native tool-call path also exists (`native_tool_calls` router, L1916) |

## smolagents 1.26.0 `[V-GitHub huggingface/smolagents@96f33faaf0]`
| What | Where |
|---|---|
| Loop | `src/smolagents/agents.py`, `MultiStepAgent._run_stream` (L540): `while not returned_final_answer and self.step_number <= max_steps:` (L545) |
| Limits | `max_steps` default **20** (L300). On exhaustion, `_handle_max_steps_reached` (L625) makes one more LLM call (`provide_final_answer`) and records `AgentMaxStepsError` |
| Variants | `ToolCallingAgent` (L1215, JSON tool calls); `CodeAgent` (L1505, actions are Python code) |

## Agno 3.1.11 `[V-GitHub agno-agi/agno@3b7042246f]`
| What | Where |
|---|---|
| Loop | in the **model layer**: `libs/agno/agno/models/base.py`, `Model.response` (L651), `while True:` (L705); async `aresponse` (L884/L929) |
| Limits / exits | `tool_call_limit` (L657, counted at L817); breaks on no tool calls, `stop_after_tool_call`, `requires_confirmation`, `external_execution_required`, `requires_user_input` (L839–860) |

## Strands Agents 1.62.0 `[V-GitHub strands-agents/harness-sdk@15ca21f15c]`
`strands-agents/sdk-python` now redirects to `strands-agents/harness-sdk`; the Python SDK is under `strands-py/`.
| What | Where |
|---|---|
| Loop | **recursive**, not a `while`: `strands-py/src/strands/event_loop/event_loop.py`, `event_loop_cycle` (L155): `stop_reason == "tool_use"` (L306) → `_handle_tool_execution` (L855) → `recurse_event_loop` (L420) → next cycle; `"end_turn"` ends |
| Other exits | `max_tokens` handling (L295, L687), `cancelled` (L673), interrupts (`_stop_for_interrupts`, L817) |

## References (not candidates for the comparison)
| Project | Where |
|---|---|
| LangChain `create_agent` 1.4.7 `[V-GitHub langchain-ai/langchain@facf2788ae]` | `libs/langchain_v1/langchain/agents/factory.py`, `create_agent` (L824+) builds a **LangGraph** graph; edges `_make_model_to_tools_edge` (L1991), `_make_tools_to_model_edge` (L2072); middleware `model_call_limit.py`, `tool_call_limit.py` (per-tool, thread/run limits, `exit_behavior`). The `langchain` package was declined in Phase 2 |
| mini-swe-agent 2.4.6 `[V-GitHub SWE-agent/mini-swe-agent@04d809ceab]` | `src/minisweagent/agents/default.py` (190 lines), `DefaultAgent.run` (L88): `while True: self.step()` (L96); `step = execute_actions(query())`. Exits when the last message has role `exit` (environment raises `Submitted` when output starts with `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, `environments/local.py` L48), `LimitsExceeded` (`step_limit`, `cost_limit` 3.0, L132), `TimeExceeded`, `RepeatedFormatError`. The smallest production-used loop found |
| ReAct paper code `[V-GitHub ysymyth/ReAct]` | notebooks only (`hotpotqa.ipynb`, `FEVER.ipynb`, `alfworld.ipynb`, `WebShop.ipynb`); prompt-based Thought/Action/Observation; last push 2024-02-06 |

## Searched, not found
No repeated-call, oscillation or no-progress detector was found in any of the loop files above. Keyword search of the downloaded files for "repeated/identical/duplicate tool call", "loop detect", "oscillat" found nothing.
- The closest mechanisms are per-tool call caps: LangChain `ToolCallLimitMiddleware`, Agno `tool_call_limit`, Pydantic AI `tool_calls_limit`, MAF `max_function_calls`.
- **NOT VERIFIED** for whole repositories (only the loop files were searched).
