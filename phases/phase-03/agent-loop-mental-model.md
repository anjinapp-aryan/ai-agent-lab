# Phase 3.0 — Agent Loop Mental Model (from source + Phase 1/2 evidence)

Evidence tags:
- `[src]`: [source-evidence.md](source-evidence.md).
- `[P1]` / `[P2-Bn]`: measured in Phase 1 / Phase 2 Batch n.
- `[open]`: to be measured in Phase 3.1+.

## The loop without a framework
Every framework audited reduces to the same five operations; only *where* they live differs.

```
state = [system, user]                                   # message list (+ counters, identity, metadata)
while not stop(state):                                   # (5) stop: limits, cancellation, policy  -> framework/app/runtime
    response = model(state, tool_definitions)            # (1) ask the model                      -> model decides
    if no tool calls in response: return response        # (2) natural termination                -> model decides, framework checks
    results = execute(response.tool_calls)               # (3) dispatch + validate + run          -> framework + runtime + tool
    state = state + [response] + results                 # (4) append observations                -> framework
```

| Part | Fundamental to an agent? | Framework-specific? |
|---|---|---|
| (1) model call with tool definitions | yes | request construction details (Phase 2 Batch 5: envelopes differ) |
| (2) "no tool calls → finish" | yes, in every framework audited `[src]` | how it is checked: conditional edge (LangGraph `tools_condition`), node transition (Pydantic AI `CallToolsNode` → `End`), `NextStepFinalOutput` (OpenAI), `is_final_response()` (ADK), `stop_reason == "end_turn"` (Strands), parsed `AgentFinish` (CrewAI text ReAct) |
| (3) execute tool calls | yes | validation, parallelism, error→message mapping, retries `[P2-B1..B4]` |
| (4) append results as observations | yes | message shape, ordering (call order `[P2-B4]`), serialization `[P2-B1]` |
| (5) external stop conditions | **not fundamental** to the idea, **essential** in production | every framework has a different counter, default and failure mode (table below) |

The ReAct paper's "Thought → Action → Observation" is the same loop with the reasoning made visible as text and the action parsed from text (CrewAI `parser.py`, ReAct notebooks `[src]`). With native tool calling, the "Thought" is the model's hidden or visible reasoning, the "Action" is a structured tool call, and the "Observation" is the tool message.

## Where the loop lives (by framework)
| Framework | Loop owner | Mechanism | Default iteration cap `[src]` |
|---|---|---|---|
| LangGraph | framework runtime (Pregel) + application-defined graph cycle | `while loop.tick()`; edge chosen by `tools_condition` | `recursion_limit` **10007** super-steps (Phase 1 sets 17) |
| Pydantic AI | framework | `while not isinstance(node, End)` over agent nodes | `request_limit` **50** model requests |
| OpenAI Agents SDK | framework | `while True` per turn; `NextStep*` | `max_turns` **10** |
| Google ADK | framework | `while True` until a final-response event | `max_llm_calls` **500** |
| Microsoft Agent Framework | framework, **inside the chat client** | `for attempt in range(max_iterations)` | `max_iterations` **40** |
| CrewAI | framework (event-driven Flow) | router cycle | `max_iter` **25** |
| smolagents | framework | `while not final and step <= max_steps` | `max_steps` **20** |
| Agno | framework, **inside the model class** | `while True` in `Model.response` | none found as a default iteration cap; `tool_call_limit` optional |
| Strands | framework | recursion per cycle on `stop_reason` | **NOT VERIFIED** (no iteration cap found in `event_loop.py`) |
| mini-swe-agent | the 190-line agent class | `while True: step()` | `step_limit` 0 (off), `cost_limit` $3 |

**Defaults range from 10 to 10007 for the same concept.** "It stops eventually" is a framework default, not a property of agents.

## Loop ownership matrix
| Capability | Model | Framework | Runtime | Application | Tool/Data |
|---|---|---|---|---|---|
| Decide another iteration is needed (emit tool calls) | **owns** `[P2-B5]` | checks presence | — | can force via `tool_choice` `[P2-B5]` | — |
| Decide it is finished (emit no tool calls / final answer) | **owns** | detects; may also stop on a tool (`stop_on_first_tool`, `return_direct`, `stop_after_tool_call`) `[src]` | — | can end it (policy, `iter()`/`next()` stepping, interrupts) | — |
| Iteration cap | — | **enforces** (counter + exception) | — | **chooses the value** (defaults unsafe or arbitrary) | — |
| Tool dispatch, validation | — | **owns** `[P2-B1, B5]` | threads/event loop `[P2-B4]` | — | executes |
| Timeout | — | per step/run `[P2-B2]` | **enforces** (cancellation reaches async only) `[P2-B4]` | sets budgets | — |
| Cancellation | — | API (`cancel()`, drain) | **owns** delivery | **initiates** | sync work may finish anyway `[P2-B4]` |
| Retries | model-requested `[P2-B2]` | framework-automatic (`RetryPolicy`, `retries`) `[P2-B2]` | — | policy | tool-internal |
| Loop state | context window (what the model sees) | **holds** messages/counters for the run | process memory | identity/deps (trusted) `[P2-B3]` | checkpoints/DB (if configured) |
| Business termination (policy, approval, budget) | — | provides hooks/interrupts | — | **owns** | enforces invariants |
| Side-effect safety across iterations | — | none | — | authorization `[P2-B3]` | **idempotency** `[P2-B2, B4]` |

## Stopping conditions matrix
| Condition | LangGraph | Pydantic AI | Owner |
|---|---|---|---|
| Natural: no tool calls | `tools_condition` → `__end__` `[src][P1]` | `CallToolsNode` → `End` `[src][P1]` | model decides; framework detects |
| Max iterations | `recursion_limit` → `GraphRecursionError`, state lost from `invoke` result `[P1][P2-B1]` | `request_limit` → `UsageLimitExceeded`; messages via `capture_run_messages` `[P2-B1]` | framework enforces; application must set |
| Tool-call limit | none native | `tool_calls_limit` (checked before the whole batch) `[P2-B1]` | framework |
| Token/cost budget | none native `[open]` | `UsageLimits` token limits `[src]` | framework (PA) / application |
| Timeout | `step_timeout` per super-step `[P2-B2]` | `run_timeout`, `Tool(timeout)` `[P2-B2]` | framework + runtime |
| Cancellation | task cancel; drain | `AgentRun.cancel()` `[src]`; task cancel `[P2-B4]` | application initiates, runtime delivers |
| Tool exception | default: fed back, loop continues; `handle_tool_errors=False` raises `[P2-B2]` | raises unless `ModelRetry` `[P2-B2]` | framework default, application chooses |
| Invalid tool / args | fed back, continues `[P2-B5]` | fed back within `retries` budget, then raises `[P2-B1]` | framework |
| Provider/model error | raises `[P2-B5]` (e.g. HTTP 400 on `tool_choice=required`) | raises | framework surfaces; application handles |
| Policy denial | fed back or raised (hook design) `[P2-B3]` | plain exception ends the run `[P2-B3]` | application |
| Human approval | `interrupt` `[src]`, not measured `[open]` | deferred tools / `requires_approval` (errors with default output type `[P2-B3]`) | application |
| Forced continuation | `tool_choice=required` never finishes → `GraphRecursionError` `[P2-B5]` | `UsageLimitExceeded` `[P2-B5]` | application misconfiguration; framework cap saves it |

## Failure modes matrix (what exists vs what must be measured)
| Mode | Native protection | Known from Phase 2 | Phase 3.1 question |
|---|---|---|---|
| A. Infinite loop | only the iteration cap; LangGraph default 10007 | `required` every step hits the cap `[P2-B5]` | cost and time to hit each default cap; what state survives |
| B. Repeated identical call | none found `[src]` | the model re-issued denied calls (LangGraph default) `[P2-B3]` | does the live model repeat after errors or empty results? |
| C. Oscillation A↔B | none found | — | can it be provoked with contradictory tool outputs? |
| D. No progress | none found | — | can progress be observed from the trajectory alone? |
| E. Wrong tool → recovery | errors fed back | unknown-tool and invalid-args recovery `[P2-B5]` | multi-step recovery rates live |
| F. Result triggers another call | normal chaining | dependent calls sequenced 10/10; injection triggered calls 10/10 `[P2-B4, B5]` | trajectory length and order vs plan |
| G. Failure → retry | model, framework, tool retries all exist | framework retry duplicated side effects without idempotency `[P2-B2]` | interaction of retries with the iteration cap |
| H. Several calls per response | LG thread pool; PA parallel/sequential modes | measured `[P2-B4]`; the live model rarely batched | none (Phase 2 covered it) |

## State
- **Within a run, the loop is stateful in the framework:** messages, step/request counters, usage, retry counters. LangGraph keeps channels and `remaining_steps`; Pydantic AI keeps `GraphAgentState` and `RunUsage` `[src]`.
- **The model is stateless:** every iteration re-sends the whole message list (the context window *is* the state the model sees) `[P2-B5 wire captures]`.
- **Across runs, nothing persists** unless the application adds message history or a LangGraph checkpointer. Durable state is a Phase 4 topic; Phase 3 only needs to observe what is lost at a stop (e.g. `GraphRecursionError` returns no state from `invoke`) `[P1]`.

## Answers to the required questions (audit-level; experiments will confirm)
| Q | Answer |
|---|---|
| Q1 Where does the loop live? | In the framework, in every case audited. The *location* varies: graph runtime (LangGraph), agent run (Pydantic AI, OpenAI, ADK, smolagents, CrewAI Flow), chat client (MAF), model class (Agno), recursive event loop (Strands) `[src]` |
| Q2 Who decides another iteration? | The model, by emitting tool calls; the framework only checks for them `[src]`. The application can override (`tool_choice`, stop-on-tool, interrupts) |
| Q3 Who decides it is finished? | Normally the model (no tool calls). The framework or application ends it otherwise: cap, timeout, cancellation, policy, stop-on-tool |
| Q4 Who enforces max iterations? | The framework enforces it; the application must choose the value (defaults 10…10007) |
| Q5 Who owns loop state? | The framework for the run; the model sees a re-sent copy; persistence is application/checkpointer-owned |
| Q6 Who owns cancellation? | The application initiates, the runtime delivers, the framework exposes the API; sync tool work can outlive it `[P2-B4]` |
| Q7 Who owns retries? | Three owners: model (re-issue), framework (automatic policies), tool (internal). The application must pick one per failure type `[P2-B2]` |
| Q8 Who owns business-level termination? | The application (policy, approval, budgets), through framework hooks/interrupts |
| Q9 Can the model be trusted to terminate? | No. Termination is usually model-driven, but a misconfiguration (`required`), an injection, or a model error can prevent it; the cap is the guarantee `[P2-B5]` |
| Q10 Infinite loop? | It runs until the framework cap, a timeout, or the provider/quota stops it. With LangGraph's default, that is ~5000 model calls `[src]`. Pattern detection does not exist natively |
| Q11 Framework retry creates side effects? | They are duplicated unless the data layer is idempotent `[P2-B2, B4]` |
| Q12 Can a framework guarantee progress? | No. It can only bound effort (iterations, tokens, time). Progress is task semantics, so it belongs to application checks or evaluation |
| Q13 Minimum machinery? | Message list, model call with tools, tool-call extraction, executor, append, stop check plus cap: the five parts above (mini-swe-agent does it in ~190 lines, `[src]`) |
| Q14 What must we never implement? | See [README](README.md#what-we-must-not-build) |
