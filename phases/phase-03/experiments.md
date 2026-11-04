# Phase 3.1 — Agent Loop Experiments

Reuse: Phase 1 agents (`lg_agent.build_graph`, `pa_agent.build_agent`), shared calculator tools, prompt and `MAX_STEPS`; the Phase 2 mock HTTP transport (`lab2.wire`); the existing provider clients. New measurement code: `phase-03/common/lab3/trajectory.py`, a recorder with three vantage points:
- **WIRE:** httpx event hooks on the provider client.
- **TOOL:** a `functools.wraps` wrapper; the schema is unchanged, which a test checks.
- **FRAMEWORK:** each step yielded by the framework's own iteration API (LangGraph `graph.stream(stream_mode=["debug","values"])`, Pydantic AI `agent.iter()`).

**Faithfulness of the view:** both APIs are what `invoke` and `run` use internally (`[V]` source: `Pregel.invoke` iterates `self.stream(...)`; `AbstractAgent.run` loops `agent_run.next(node)`). A test also checks that the plain Phase 1 `run()` produces identical wire and tool accounting.

LANGFUSE = NOT AVAILABLE (no credentials); evidence is local structured trajectories.

Protocol note: the Batch 1 expectations below come from the Phase 3.0 mental model committed in `95f44e9` before any Phase 3.1 run. This file itself was written after the first offline run, and that is stated here rather than presented as pre-registration.

---

# Batch 1 — Trace the agent loop (2026-10-28)

## Hypotheses (from the Phase 3.0 five-part model)
- **H1.1 (3.1.1):** one loop is `request(system,user) → response(tool_call) → tool runs → request(system,user,assistant,tool) → response(text) → stop`: 2 model requests, 1 tool execution. The framework adds nothing the model did not ask for.
- **H1.2 (3.1.2):** each tool result appears in the *next* model request and not before. The whole history is re-sent each time (the model is stateless). 3 model requests, 2 tool executions.
- **H1.3 (3.1.3):** if the model returns two tool calls in one response, both execute before the next request and come back as two tool messages in one request (2 requests). If it returns them one at a time, there are 3 requests. Which one a live model does is unknown; Phase 2 Batch 4 saw 0/10 batching.
- **H1.4:** "iteration" is counted differently. Model requests, framework steps and tool executions are three different numbers, and the frameworks' step units differ.

## Setup
- **Offline:** real `ChatOpenAI` / `OpenAIChatModel` over the mock transport replaying scripted completions (mechanics only, no API).
- **Live:** Groq `openai/gpt-oss-120b`, temperature 0 in both frameworks (equalized; the Phase 1 Pydantic AI model sends none), 3 trials per framework per task.
- **Live quota:** 17/18 trials valid. The 18th (3.1.3, Pydantic AI, trial 3) hit Groq's tokens-per-day limit (199,779 / 200,000 used). Live work stopped there; the model was not switched.
- **Reproduce:** `uv run python -u phase-03/exp3_batch1.py` (offline), `... live [3.1.1 3.1.2 3.1.3]`. Data in `phase-03/results/batch1.json` (git-ignored); summary in [results/batch1-summary.json](results/batch1-summary.json). Tests: `tests/phase_03/test_batch1_loop_trace.py` (21 offline).

## Results
### Loop accounting (identical in offline and live runs; live tokens vary slightly)
| Task | Framework | Model requests | Tool calls requested | Tool executions | Framework steps | Final responses | Live tokens (total) | Live model latency (median per request) |
|---|---|---|---|---|---|---|---|---|
| 3.1.1 `25 × 17` | LangGraph | 2 | 1 | 1 | 3 super-steps | 1 | 706 ×3 | 445 ms |
| | Pydantic AI | 2 | 1 | 1 | 6 nodes | 1 | 770 ×3 | 597 ms |
| 3.1.2 `120+45, ×3` | LangGraph | 3 | 2 | 2 | 5 | 1 | 1150 ×3 | 695 ms |
| | Pydantic AI | 3 | 2 | 2 | 8 | 1 | 1274–1311 | 582 ms |
| 3.1.3 `10×5 and 20+7`, live | LangGraph | 3 | 2 | 2 | 5 | 1 | 1158–1178 | 643 ms |
| | Pydantic AI | 3 | 2 | 2 | 8 | 1 | 1299 ×2 | 549 ms |
| 3.1.3 scripted batched | both | 2 | 2 | 2 | 3 / 6 | 1 | — | — |

- **Formulas** (r = model responses that requested tools, so requests = r + 1):
  - LangGraph super-steps = 2r + 1 (one `agent` and one `tools` node per round, plus the final `agent`).
  - Pydantic AI nodes = 2 + 2(r + 1): `UserPromptNode`, then a `ModelRequestNode` + `CallToolsNode` pair per request, then `End`. Pydantic AI's `run_step` = `usage.requests` = number of model requests.
- **Batched calls:** one response with two tool calls is **one** round (one `tools` super-step, one `CallToolsNode`) with two tool executions.
- **Tool execution cost:** under 0.1 ms per tool. Framework time between events is about 0.5–3 ms. Model requests take 445–1800 ms: practically all loop latency is model latency.
- **Token cost:** Pydantic AI requests cost about 30 tokens more each, for the same model and conversation. This matches its `strict`/`additionalProperties:false` tool schemas (Phase 2 Batch 5).

### Traces (live, 3.1.2, trial 1)
LangGraph:
```
   171 ms  WIRE -> request #1: [system, user]
   756 ms  WIRE <- response #1: add({"a":120,"b":45})            finish=tool_calls
   758 ms  FW   super-step 1: node 'agent' finished              state: 2 messages
   759 ms  TOOL add(120, 45) -> 165.0
   759 ms  FW   super-step 2: node 'tools' finished              state: 3 messages
   762 ms  WIRE -> request #2: [system, user, assistant, tool]
  1500 ms  WIRE <- response #2: multiply({"a":165,"b":3})        finish=tool_calls
  1501 ms  FW   super-step 3: node 'agent' finished              state: 4 messages
  1502 ms  TOOL multiply(165, 3) -> 495.0
  1502 ms  FW   super-step 4: node 'tools' finished              state: 5 messages
  1504 ms  WIRE -> request #3: [system, user, assistant, tool, assistant, tool]
  3305 ms  WIRE <- response #3: text 'The final result is 495.'   finish=stop
  3306 ms  FW   super-step 5: node 'agent' finished -> tools_condition -> END
```
Pydantic AI (same task, offline trace; the live one has the same shape):
```
  FW   about to run UserPromptNode       messages=0 run_step=0
  FW   about to run ModelRequestNode     messages=0 run_step=0
  WIRE -> request #1: [system, user]            <- response: add(120,45)
  FW   about to run CallToolsNode        messages=2 run_step=1
  TOOL add -> 165.0
  FW   about to run ModelRequestNode     messages=2 run_step=1   (tool return becomes part of the next ModelRequest)
  WIRE -> request #2: [system, user, assistant, tool]   <- response: multiply(165,3)
  FW   about to run CallToolsNode        messages=4 run_step=2
  TOOL multiply -> 495.0
  FW   about to run ModelRequestNode     messages=4 run_step=2
  WIRE -> request #3: [system, user, assistant, tool, assistant, tool]   <- response: text
  FW   about to run CallToolsNode        messages=6 run_step=3   (a final text response still passes through CallToolsNode)
  FW   about to run End
```

### Expected vs actual
| H | Expected | Actual |
|---|---|---|
| H1.1 | 2 requests, 1 tool execution | **confirmed**: both frameworks, offline and live 6/6 |
| H1.2 | each result in the next request; history re-sent | **confirmed**: every tool result was found in the next request's `tool` message (all runs), never earlier. Request *n* carries the full history: 2 → 4 → 6 messages |
| H1.3 | batched = 2 requests; sequential = 3 | **mechanics confirmed** (scripted). **Live: the model never batched (5/5 trials: `multiply`, then `add`)**, matching Phase 2 Batch 4. So in practice independent calls cost an extra model round trip |
| H1.4 | the counts differ | **confirmed**. For 3.1.2: 3 model requests, 2 tool executions, 5 LangGraph super-steps, 8 Pydantic AI nodes, Pydantic AI `run_step` = 3 |

### Additional observations
- **The system prompt is not part of loop state in LangGraph.** `lg_agent`'s `agent` node prepends it on every call (state holds 2 → 6 messages; the wire carries 1 more). This is an application choice in our graph, not a framework rule. In Pydantic AI the instructions are likewise re-sent on every request (wire); where they are stored was not inspected.
- **Pydantic AI's history unit is a request/response pair:** tool returns are parts of the next `ModelRequest`, not separate messages. LangGraph stores one `ToolMessage` per call. On the wire both become `role: tool` messages.
- **Completion signal:** the provider response with `finish_reason=stop` and no `tool_calls`. LangGraph's `tools_condition` routes to `END` with no extra step. Pydantic AI routes the final response through `CallToolsNode` to `End`. Neither framework made a request after the final answer.
- **Arguments are coerced by the framework:** the model sent integers; the tools received `25.0`, `17.0` (Pydantic validation against `float` annotations).

## Architectural conclusions (Batch 1)
- **The loop is real and observable:** request → response → (tools) → request carrying the observations → … → text response. The **model** chooses each next action and when to stop. The **framework** turns that choice into execution and into the next request. The **runtime** runs tool bodies (threads in both frameworks, from Phase 2). Nothing else intervenes in a normal run.
- **The model is stateless:** the loop state is the message list the framework re-sends. Cost per iteration grows with history (706 → 1150 tokens from one to two rounds).
- **"Iteration" has no single meaning:**
  - The natural unit is a **model request** (one decision).
  - A tool call is not an iteration: a batch of 2 is one decision.
  - LangGraph limits count **super-steps** (≈ 2 per decision); Pydantic AI's `request_limit` counts **model requests**. Batch 3 will measure the limits.

## Limits (Batch 1)
- One model (`gpt-oss-120b`), 3 trials (3.1.3 Pydantic AI: 2).
- Calculator tools only.
- Offline token counts come from the mock (2 per response), so token figures are live-only.
- LangGraph per-step state is taken from the `values` stream (state after each super-step), not from a checkpointer.

---

# Batch 2 — Natural termination (hypotheses written 2026-10-28, before any Batch 2 run)

Batch-local IDs follow the Batch 2 brief (`B2-3.1.4` … `B2-3.1.10`); they are not the master-prompt numbering.
- **H2.1 (B2-3.1.4, no tool call):** both frameworks make exactly 1 model request, 0 tool executions, and return the model's text. LangGraph: 1 super-step (`agent`), then `tools_condition` routes to `END`. Pydantic AI: `UserPromptNode → ModelRequestNode → CallToolsNode → End` (4 nodes). No hidden extra request.
- **H2.2 (B2-3.1.5 / 3.1.8, tool then final):** 2 requests, 1 tool execution, no third request (a sentinel completion is served if one is ever asked for).
- **H2.3 (B2-3.1.6, what is inspected):** the termination signal is *the absence of tool calls in the parsed response*, not `finish_reason`. Expected: `finish_reason` = `length` / `tool_calls` / `content_filter` with text and no calls still terminates; tool calls with `finish_reason=stop` still execute. Unknown: whether Pydantic AI treats `length` or `content_filter` as an error (it has provider-specific handling).
- **H2.4 (B2-3.1.7, empty content):** LangGraph terminates with an empty final answer (its edge only checks `tool_calls`). Pydantic AI may *not* accept an empty response as a `str` output; it may retry (another request) or raise. Expected ≠ certain.
- **H2.5 (B2-3.1.9, state):** LangGraph returns channel state (messages without the system prompt, which our node adds per call); Pydantic AI returns request/response pairs plus usage and `run_step`.
- **H2.6 (B2-3.1.10, ownership):** both return the model's final text unchanged (no trimming, no wrapping); in LangGraph the *application* (`lg_agent._fill`) picks the last message; in Pydantic AI the framework's output processing produces `result.output`.

## Batch 2 results (2026-10-28)
Setup:
- **Offline:** the real provider clients over a mock transport (`raw_transport` in `phase-03/exp3_batch2.py`). Each scripted completion sets `content`, `tool_calls` and `finish_reason` independently. After the script ends, a **sentinel** completion is served, so any extra request shows up.
- **Reuse:** Batch 1 runners (`graph.stream` / `agent.iter` over the Phase 1 graph/agent) and recorder.
- **Tests:** `tests/phase_03/test_batch2_natural_termination.py` (26 offline).
- **Data:** `phase-03/results/batch2.json` (git-ignored).

### Results table (offline; measured)
| Experiment | Framework | Model requests | Tool calls | Tool executions | Steps | Final result | Termination signal |
|---|---|---:|---:|---:|---:|---|---|
| B2-3.1.4 no tool call | LangGraph | 1 | 0 | 0 | 1 super-step | model text, unchanged | `tools_condition`: last `AIMessage.tool_calls` empty → `__end__` |
| B2-3.1.4 | Pydantic AI | 1 | 0 | 0 | 4 nodes | model text, unchanged | `CallToolsNode`: no tool-call parts and non-empty text → `End` |
| B2-3.1.5 tool → final | LangGraph | 2 | 1 | 1 | 3 | model text | same, on response #2 |
| B2-3.1.5 | Pydantic AI | 2 | 1 | 1 | 6 | model text | same, on response #2 |
| B2-3.1.6 text + `finish_reason` ∈ {stop, length, tool_calls, content_filter} | LangGraph | 1 | 0 | 0 | 1 | model text | `finish_reason` **ignored** for routing (kept in `response_metadata`) |
| B2-3.1.6 | Pydantic AI | 1 | 0 | 0 | 4 | model text | `finish_reason` **ignored** when text is present (stored; `tool_calls` normalized to `tool_call`) |
| B2-3.1.6 tool call + `finish_reason=stop` | both | 2 | 1 | 1 | 3 / 6 | model text | the tool **ran**: routing follows the parsed tool calls, not `finish_reason` |
| B2-3.1.7 empty `""` or `null` content, no calls | LangGraph | 1 | 0 | 0 | 1 | `""`, returned as a successful answer | no tool calls → `__end__` |
| B2-3.1.7 | Pydantic AI | **2** | 0 | 0 | 6 | the *second* completion (sentinel) | **not terminal**: the framework re-prompts with a `system` message "Please return text or call a tool." |
| B2-3.1.7 empty + `finish_reason=length` | LangGraph / Pydantic AI | 1 / 1 | 0 | 0 | — | `""` / **`UnexpectedModelBehavior`** "Model token limit … exceeded before any response was generated" | Pydantic AI reads `finish_reason` only for empty responses |
| B2-3.1.7 empty + `finish_reason=content_filter` | LangGraph / Pydantic AI | 1 / 1 | 0 | 0 | — | `""` / **`ContentFilterError`** | same |
| B2-3.1.7 empty twice | LangGraph / Pydantic AI | 1 / 2 | 0 | 0 | — | `""` / **`UnexpectedModelBehavior: Exceeded maximum output retries (1)`** | the re-prompt consumes the output-retry budget (`retries=1` in `pa_agent`) |
| B2-3.1.8 tool → final, sentinel armed | both | 2 | 1 | 1 | 3 / 6 | model text | **no third request** (sentinel never served) |
| B2-3.1.10 `"  **The answer is 425.**\n\n"` | both | 1 | 0 | 0 | 1 / 4 | byte-identical to the wire content; Pydantic AI `output` is a `str`; Phase 1 `RunResult.final_text` identical | — |

### Natural termination traces (offline, B2-3.1.5)
LangGraph:
```
REQUEST #1   [system, user]
RESPONSE #1  tool_calls=[multiply(25,17)]  finish=tool_calls
FRAMEWORK    super-step 1 'agent' -> tools_condition: tool_calls non-empty -> 'tools'
TOOL         multiply -> 425.0                         (super-step 2 'tools')
REQUEST #2   [system, user, assistant, tool]
RESPONSE #2  tool_calls=[]  "No further tools are required. The answer is 425."  finish=stop
FRAMEWORK    super-step 3 'agent' -> tools_condition: tool_calls empty -> '__end__' -> no tasks -> loop.tick() returns False
FINAL        state.messages[-1].content, extracted by the caller (lg_agent._fill / our runner)
```
Pydantic AI:
```
UserPromptNode -> ModelRequestNode
REQUEST #1   [system, user]
RESPONSE #1  tool_calls=[multiply(25,17)]
CallToolsNode  tool-call parts present -> execute -> next ModelRequestNode carrying the tool return
TOOL         multiply -> 425.0
REQUEST #2   [system, user, assistant, tool]
RESPONSE #2  text only
CallToolsNode  no tool-call parts, non-empty text -> output processing (str) -> End(FinalResult)
FINAL        run.result.output == response text
```

### Expected vs actual
| H | Expected | Actual |
|---|---|---|
| H2.1 | 1 request; 1 / 4 steps | **confirmed** |
| H2.2 | no third request | **confirmed** (sentinel never served) |
| H2.3 | absence of tool calls is the signal; `finish_reason` ignored | **confirmed for non-empty responses** in both frameworks. Pydantic AI does consult `finish_reason` for *empty* responses (`length` / `content_filter` → exception) |
| H2.4 | Pydantic AI may retry or raise on empty | **retries**: one framework-written re-prompt ("Please return text or call a tool.", sent as role `system`), then `UnexpectedModelBehavior` once the output-retry budget is spent. LangGraph returns `""` as a successful final answer |
| H2.5 | state shapes differ | **confirmed**. LangGraph: `[Human, AI(tool_calls), Tool, AI]`; the system prompt is not stored (our node adds it per call). Pydantic AI: `[ModelRequest, ModelResponse] × n`, plus `run_step`, usage and `finish_reason` per response |
| H2.6 | final text unchanged | **confirmed**: byte-identical, including whitespace and markdown. In LangGraph the caller extracts the last message; in Pydantic AI the framework's output processing returns it (`str` output) |

### Live validation
**Partly blocked by Groq quota (tokens per day 200,000 / 200,000).** 3 trials ran on `openai/gpt-oss-120b` (LangGraph 2, Pydantic AI 1) before the limit; the model was not switched.
- **No live no-tool case was observed.** For "What is 2 + 2?" the model called `add(2,2)` in 3/3 trials, then answered. The Phase 1 system prompt says "only by calling the provided tools", so the *application's* prompt decided that even trivial arithmetic goes through a tool.
- **The provider client retried an HTTP 429 on its own.** In LangGraph trial 2, request #2 got a 429 and the OpenAI client retried it after ~7.5 s. The framework saw one model call; the wire saw two HTTP requests. That makes the provider SDK a fourth retry owner below the framework, relevant to the Batch 5 retry experiments. The trajectory accounting now counts `http_errors` separately.

### Who decides when to stop? (evidence-based)
| Concern | Model | Framework | Runtime | Application |
|---|---|---|---|---|
| Produces final response | **yes**: its text is returned unchanged (B2-3.1.10) | Pydantic AI may *request another* response on empty output (B2-3.1.7) | — | the system prompt shapes whether tools are used first (live) |
| Indicates no tool calls | **yes**: by emitting none | — | — | can force or forbid via `tool_choice` (Phase 2 Batch 5) |
| Detects no tool calls | — | **yes**: `tools_condition` / `CallToolsNode` parse the response | — | in LangGraph the condition is wired by the application's graph (`add_conditional_edges("agent", tools_condition)`) |
| Ends loop | — | **yes**: no next task (LangGraph `tick()` → done) / `End` node (Pydantic AI) | — | could stop earlier via `iter()`/`stream` (Batch 7) |
| Returns final result | — | Pydantic AI: `result.output`; LangGraph: final state | — | LangGraph: the application picks `messages[-1]` (`lg_agent._fill`) |
| Chooses whether another tool is needed | **yes** | Pydantic AI overrides an empty "nothing" with a re-prompt; LangGraph does not | — | prompt and `tool_choice` influence it |

**"The model decides when the agent stops" is incomplete.**
- The model *proposes* an end by emitting a response without tool calls.
- The framework *decides* whether that response is terminal, and the rules differ:
  - **LangGraph:** any response without `tool_calls` ends the loop, even an empty one.
  - **Pydantic AI:** a response without tool calls ends the loop only if it yields valid output. Empty → framework re-prompt or exception; `finish_reason` is consulted only then.
- The provider's `finish_reason` is **not** the signal in either framework.
- In LangGraph the "framework rule" is itself wired by the application (the conditional edge in our graph).

### Refined mental model
```
                 MODEL
                   | response (content?, tool_calls?, finish_reason)
                   v
        +------------------------------+
        | FRAMEWORK: parse response    |
        | tool calls present?          |-- YES --> validate + EXECUTE (runtime) --> TOOL RESULT --> next MODEL request
        +------------------------------+
                   | NO
                   v
        +------------------------------+
        | acceptable final output?     |  LangGraph: always (even "")  --> END; state returned; app extracts text
        | (framework output policy)    |  Pydantic AI: non-empty text  --> End(FinalResult.output)
        +------------------------------+     empty + finish=length/content_filter --> raise
                                             empty otherwise --> framework re-prompt "Please return text or call a tool."
                                             --> MODEL ... until output retries are exhausted --> raise
```
`finish_reason` sits beside the arrow: it is recorded, not routed on (except in Pydantic AI's empty case).

### Limits (Batch 2)
- Offline mechanics, plus only 3 live trials; no live no-tool case, because the system prompt requires tools.
- Not tested: malformed tool-call JSON, multiple choices (`n>1`), refusals, streaming.
- `pa_agent` uses `retries=1`; other budgets change how many re-prompts happen.

---

# Batch 3 — Iteration limits, retries and loop budget (hypotheses written 2026-10-29, before any Batch 3 run)

Batch-local IDs follow the Batch 3 brief (`B3-3.1.11` … `B3-3.1.22`). The limit knobs:
- **LangGraph:** `config["recursion_limit"]`.
- **Pydantic AI:** `UsageLimits(request_limit=…, tool_calls_limit=…)` plus the agent/tool `retries` budget.

Layers kept apart: application (chooses limits), framework (loop + counters), model adapter, provider client (HTTP retries; the OpenAI SDK default is `max_retries=2`), provider. Retry kinds kept apart: model-visible tool retry, framework tool retry, tool-internal retry, output re-prompt, HTTP retry.

- **H3.1 (B3-3.1.11):** a final-only run needs LangGraph `recursion_limit ≥ 1` (one super-step) and Pydantic AI `request_limit ≥ 1`. The natural final response consumes one unit in both.
- **H3.2 (B3-3.1.12):** one tool round needs LangGraph ≥ 3 (super-steps agent, tools, agent) and Pydantic AI ≥ 2 (model requests). Below that: `GraphRecursionError` / `UsageLimitExceeded`.
- **H3.3 (B3-3.1.13):** two rounds need LangGraph ≥ 5 and Pydantic AI ≥ 3. So **LangGraph's unit is super-steps (2r+1); Pydantic AI's unit is model requests (r+1)**, and equal numbers are not comparable.
- **H3.4 (B3-3.1.14, final-step safety):** LangGraph checks the limit *between* super-steps. When it trips after an `agent` step that requested a tool, the tool is **not** executed. When it trips after a `tools` step, the tool **was** executed and no further request is made. Pydantic AI checks `request_limit` *before each request*, so the last requested tool **is** executed and the run fails before the next request. `tool_calls_limit` is checked *before* executing a tool batch, so the tool is **not** executed.
- **H3.5 (B3-3.1.15):** Pydantic AI's empty-output re-prompt is a real model request: it consumes `request_limit` *and* the output-retry budget. LangGraph has no re-prompt (it returns `""`).
- **H3.6 (B3-3.1.16):** persistent empty output is stopped by the output-retry budget (`retries=N` → N+1 requests, then `UnexpectedModelBehavior`), unless `request_limit` is lower and trips first.
- **H3.7 (B3-3.1.17):**
  - **Pydantic AI:** a tool raising `ModelRetry` triggers a model-visible retry. Each retry is a new model request (consumes `request_limit`) and also counts against the tool's own retry budget; exhausting it raises `UnexpectedModelBehavior`. A plain exception is not retried (it escapes).
  - **LangGraph default:** the error is fed back and the model retries, costing 2 super-steps per retry.
  - **LangGraph `RetryPolicy`** (`handle_tool_errors=False`): the framework re-runs the tools node *inside the same super-step*. No model request, no extra super-step.
- **H3.8 (B3-3.1.18):** a framework tool retry does **not** consume loop budget; a model-visible retry does. Pydantic AI has no native framework-only tool re-execution (all its tool retries go through the model).
- **H3.9 (B3-3.1.19):** a transient HTTP 500 followed by 200 is retried by the provider client: 2 HTTP attempts, 1 framework model call, no extra framework step, no extra model response.
- **H3.10 (B3-3.1.20):** at the minimum successful limit, a run with an HTTP retry still succeeds. HTTP retries do not consume loop budget in either framework.
- **H3.11 (B3-3.1.21):** neither our LangGraph graph nor Pydantic AI warns the model or forces a final answer near the limit; requests near the limit are byte-identical in shape. The deprecated LangGraph prebuilt `create_react_agent` (installed, `langgraph-prebuilt`) uses `remaining_steps` and may return a framework-written "need more steps" message instead of raising. That is a reference observation only.
- **H3.12 (B3-3.1.22):** side effects committed before the limit trips remain (no rollback) in both frameworks.

## Batch 3 results (2026-10-29)
**Setup.** `phase-03/exp3_batch3.py` reuses the Batch 1/2 harness and the Phase 2 failure fixtures (`make_flaky`, `Orders`).
- New fixture `http_flaky`: the first *k* HTTP attempts return 500/429 (`retry-after-ms: 1`).
- Runners keep partial state when a limit exception ends the run.
- Tests: `tests/phase_03/test_batch3_limits_retries.py` (40 offline).
- Data: `phase-03/results/batch3.json` (git-ignored).
- Live: 6 runs on `openai/gpt-oss-120b` after the quota recovered.

### Glossary (as measured)
| Term | Meaning | Where it is counted |
|---|---|---|
| model request | one framework call to the model adapter | Pydantic AI `usage.requests` / `run_step`; LangGraph: count of `AIMessage`s |
| HTTP attempt | one HTTP request on the wire | provider client (OpenAI SDK), invisible to the framework |
| model response | one successful completion | wire (`status 200`) |
| tool call | one call the model asked for | inside a model response |
| tool execution | one run of a tool body | the tool itself (our `traced` log) |
| framework step | LangGraph super-step (one node run) / Pydantic AI node | `graph.stream` / `agent.iter` |
| model-visible tool retry | error returned to the model, which asks again | costs a new model request |
| framework tool retry | framework re-runs the tool, no model | LangGraph `RetryPolicy`; no Pydantic AI equivalent |
| output re-prompt | framework asks the model again after unusable output | Pydantic AI only; costs a model request |
| HTTP retry | provider client re-sends the same request | SDK `max_retries` (default 2) |
| iteration | **no framework defines it.** Natural unit: one model decision (= one model request) | — |

### Iteration accounting (B3-3.1.11 … 3.1.14, offline; B3-live)
| Script | LangGraph minimum `recursion_limit` | super-steps used | Pydantic AI minimum `request_limit` | requests used |
|---|---:|---:|---:|---:|
| final only | **2** | 1 | **1** | 1 |
| one tool round + final | **4** | 3 | **2** | 2 |
| two tool rounds + final | **6** | 5 | **3** | 3 |
- **LangGraph:** a run succeeds only if `recursion_limit` **>** super-steps (= 2r+1).
  - Cause `[V]`: `PregelLoop.tick` checks `step > stop` *before* "no tasks → done", so the tick after the finishing step is already out of budget.
  - Consequence: at `limit == steps` the model has **already produced the final answer**, it is in the streamed state, and the run still raises `GraphRecursionError`. `invoke()` then returns no state, so the answer is lost.
  - Observed offline and **live 2/2**: the real answer "The product of 25 and 17 is 425." was produced and discarded at limit 3; limit 4 succeeded.
- **Pydantic AI:** `request_limit` = model requests, exactly (r+1). It is checked before each request.
- **Comparing equal numbers is meaningless:** LangGraph 4 ≈ Pydantic AI 2 (one round); LangGraph 6 ≈ Pydantic AI 3.
- **Phase 1 cap:** LangGraph `2*MAX_STEPS+1 = 17` allows 8 decisions; Pydantic AI `request_limit=8` allows 8. They were equivalent by construction, now verified.

### Final-step safety (B3-3.1.14, endless tool calls)
| Framework / limit | requests | tool calls requested | tool executions | last event | outcome |
|---|---:|---:|---:|---|---|
| LangGraph 2 / 4 / 6 (even) | 1 / 2 / 3 | 1 / 2 / 3 | 1 / 2 / 3 | tool executed | `GraphRecursionError` |
| LangGraph 3 / 5 (odd) | 2 / 3 | 2 / 3 | **1 / 2** | **model's tool call left pending, not executed** | `GraphRecursionError` |
| Pydantic AI `request_limit` 2…6 | n | n | **n** | the last requested tool **executed**, then the next request refused | `UsageLimitExceeded` |
| Pydantic AI `tool_calls_limit` 0 / 1 / 2 | n+1 | n+1 | **n** | refused **before** executing; the model request was already paid for | `UsageLimitExceeded` |
- LangGraph stops between super-steps, so whether the last *requested* tool runs depends on the parity of the limit.
- Pydantic AI `request_limit` always lets the last tool batch run; `tool_calls_limit` never lets the excess tool run.

### Retry accounting (B3-3.1.15 … 3.1.18)
| Event | Model requests | Framework steps | Tool executions | Consumes the loop limit? |
|---|---:|---:|---:|---|
| Pydantic AI output re-prompt (empty → final) | +1 | +2 | 0 | **yes**: `request_limit=1` fails, 2 succeeds; also uses the output-retry budget |
| Pydantic AI persistent empty, `retries=N` | N+1 | — | 0 | stopped by `UnexpectedModelBehavior: Exceeded maximum output retries (N)`; with `request_limit=2` and N ≥ 2, `UsageLimitExceeded` trips first |
| Pydantic AI `ModelRetry` (tool fails once, `retries=1`) | +1 | +2 | +1 | **yes**: 3 requests total; `request_limit=2` fails. `retries=0` → `Tool 'flaky' exceeded max retries count of 0` at the first failure |
| Pydantic AI plain `RuntimeError` | 0 | — | 0 retries | not retried: the exception escapes (1 request) |
| LangGraph model-visible retry (default error `ToolMessage`) | +1 | **+2** | +1 | **yes**: limit 4 fails, 6 succeeds |
| LangGraph `RetryPolicy` (framework re-runs `tools` node) | **0** | **0** | +1 per failure | **no**: 2 failures → 3 executions inside one super-step; succeeds at the minimum limit 4 |
| LangGraph empty response | — | — | — | not a retry: `""` is returned (Batch 2) |

### Provider HTTP retry accounting (B3-3.1.19 / 3.1.20)
| Wire | HTTP attempts | Framework model calls | Framework steps | At the minimum limit (LangGraph 4 / Pydantic AI 2) |
|---|---:|---:|---:|---|
| 1× 500 then 200 | 3 | 2 | 3 / 6 (unchanged) | succeeds |
| 2× 500 then 200 | 4 | 2 | unchanged | succeeds |
| 1× 429 then 200 | 3 | 2 | unchanged | — |
| 3× 500 | 3 (1 + SDK default 2 retries) | 0 | — | error escapes: `OpenAIAPIError` (LangGraph) / `ModelServerError` (Pydantic AI) |

HTTP retries are below the framework: invisible to every loop counter, and they never consume loop budget. They do cost time and provider quota (each attempt counts against provider rate limits).

### Near the limit (B3-3.1.21)
- **No warning, no forced answer:** our LangGraph graph and Pydantic AI send the same system/user messages on every request up to the limit, then raise.
- **Reference, the deprecated `create_react_agent` (LangGraph prebuilt):**
  - At limit 4 it returned a **framework-written final answer**, "Sorry, need more steps to process this request.", with **no error**, after 2 real model calls.
  - At limits 3 and 5 it raised `GraphRecursionError`.
  - Its `remaining_steps` guard only fires at some limit values; callers then receive a successful run whose text the model never wrote.

### Side effects before the limit (B3-3.1.22)
`create_order` (Phase 2 `Orders.unguarded`) on every turn: the ledger equals the tool executions in every case (LangGraph 4 → 2, 5 → 2 plus 1 requested but not executed; Pydantic AI 2 → 2, 3 → 3). **Nothing is rolled back.** The limit exception leaves committed effects and no final answer.

### Critical matrix
| Event | Model request count | Framework step count | Tool retry count | HTTP retry count | Loop budget? |
|---|---|---|---|---|---|
| normal model response | +1 | LangGraph +1 (`agent`); Pydantic AI +2 (request + call-tools nodes) | — | — | **yes**, both |
| tool call (requested) | 0 (part of a response) | — | — | — | no, by itself |
| tool execution | 0 | LangGraph +1 per batch (`tools`); Pydantic AI 0 extra | — | — | LangGraph: yes (a super-step); Pydantic AI: no |
| model-visible tool retry | +1 | LangGraph +2; Pydantic AI +2 | Pydantic AI per-tool retry +1 | — | **yes**, both |
| framework tool retry (LangGraph `RetryPolicy`) | 0 | 0 | policy attempt +1 | — | **no** |
| Pydantic AI output re-prompt | +1 | +2 | output retry +1 | — | **yes** |
| provider HTTP retry | 0 | 0 | — | +1 | **no** |
| natural termination | the final request counts | LangGraph: the *next* tick is checked against the limit | — | — | LangGraph: needs one unit of headroom; Pydantic AI: no |
| iteration-limit failure | — | — | — | — | LangGraph `GraphRecursionError` (state dropped by `invoke`); Pydantic AI `UsageLimitExceeded` (history on `run.ctx`) |

### Expected vs actual
| H | Expected | Actual |
|---|---|---|
| H3.1 | LangGraph ≥ 1, Pydantic AI ≥ 1 | Pydantic AI ✔. **LangGraph needs 2**: limit = super-steps fails even though the work finished |
| H3.2 | LangGraph ≥ 3, Pydantic AI ≥ 2 | Pydantic AI ✔. **LangGraph needs 4**, and at 3 a finished answer is discarded (live 2/2) |
| H3.3 | LangGraph ≥ 5, Pydantic AI ≥ 3 | Pydantic AI ✔. **LangGraph needs 6**. Units confirmed: super-steps vs model requests |
| H3.4 | LangGraph parity-dependent; Pydantic AI runs the last tool; `tool_calls_limit` refuses before | **confirmed** |
| H3.5 | re-prompt consumes `request_limit` and output retries | **confirmed** |
| H3.6 | `retries=N` → N+1 requests unless `request_limit` trips first | **confirmed** |
| H3.7 | `ModelRetry` costs requests; plain exception escapes; LangGraph default +2 steps; `RetryPolicy` 0 steps | **confirmed** |
| H3.8 | framework retry consumes no budget; Pydantic AI has none | **confirmed** (no Pydantic AI framework-only tool re-execution was found or observed) |
| H3.9 | 2 attempts, 1 framework call | **confirmed** (plus: after 3 failures the SDK gives up and the error escapes) |
| H3.10 | HTTP retry succeeds at the minimum limit | **confirmed** with the measured minimum (LangGraph 4). My first run used 3, from H3.2, and failed for the H3.2 reason, not because of the HTTP retry |
| H3.11 | no warning; the prebuilt may write a message | **confirmed**; the prebuilt writes one only at some limit values |
| H3.12 | no rollback | **confirmed** |

### Ownership (annotated stack)
```
USER
 └ APPLICATION ........ chooses recursion_limit / UsageLimits / retries / RetryPolicy / tool_calls_limit
                         owns rollback/compensation (no layer below does it) and what to do with a limit error
    └ AGENT FRAMEWORK .. loop + counters: LangGraph super-steps (checked before "done"), Pydantic AI requests (before each request)
    |                    model-visible tool retries (both), framework tool retry (LangGraph RetryPolicy), output re-prompt (Pydantic AI)
       └ MODEL ADAPTER . ChatOpenAI / OpenAIChatModel: one framework call -> one logical request
          └ PROVIDER CLIENT  OpenAI SDK: HTTP retries (max_retries=2), invisible above
             └ PROVIDER ....  rate limits count every HTTP attempt
TOOL / DATA ........... side effects commit when the body runs; never rolled back by a limit
```
Cancellation was not measured in this batch (Phase 2 Batch 4: the runtime delivers it; sync tools finish anyway).

### Answers
| Q | Answer |
|---|---|
| Q1 | LangGraph has no "iteration": it counts **super-steps** (one node run each; 2 per tool round + 1), and the run fails unless the limit exceeds them |
| Q2 | Pydantic AI's budget counts **model requests** (including re-prompts and retry requests); its graph nodes are not counted |
| Q3 | Framework tool retry (LangGraph `RetryPolicy`): no. Model-visible tool retry: yes |
| Q4 | Yes: +1 request (Pydantic AI) / +2 super-steps (LangGraph) |
| Q5 | Yes: +1 request in Pydantic AI (LangGraph has no re-prompt) |
| Q6 | No: invisible to both frameworks; costs provider quota and time |
| Q7 | LangGraph: at an odd limit the requested tool is **not** executed (pending in state). Pydantic AI `request_limit`: the tool **is** executed, then the next request is refused. `tool_calls_limit`: not executed |
| Q8 | The effect stays; nothing is rolled back; the run ends in an exception with no final answer |
| Q9 | No (our graph, Pydantic AI). The deprecated prebuilt sometimes writes "Sorry, need more steps…" and returns success |
| Q10 | No. LangGraph counts super-steps with one unit of headroom; Pydantic AI counts model requests. Equivalent: LangGraph `2·k` ≈ Pydantic AI `k` model decisions |

### Live validation
Quota recovered on 2026-10-29; 6 runs on `openai/gpt-oss-120b`, temperature 0:
- LangGraph limit 3: `GraphRecursionError` after the real final answer, 2/2.
- LangGraph limit 4: success, 1/1.
- Pydantic AI limit 1: tool executed, then `UsageLimitExceeded`, 2/2.
- Pydantic AI limit 2: success, 1/1.

The offline mechanics were reproduced exactly.

### Limits (Batch 3)
- Scripted mechanics only; the live check covers only the limit boundary.
- No streaming, no async `ainvoke` path, no Pydantic AI token limits.
- LangGraph `RetryPolicy` was measured on the `tools` node only (not on the `agent`/model node).
- Cancellation was out of scope.

---

# Batch 4 — Repeated calls, oscillation and no progress (hypotheses written 2026-10-31, before any Batch 4 run)

Batch-local IDs follow the Batch 4 brief (`B4-3.1.23` … `B4-3.1.38`). Budgets are stated in model decisions and set per framework using the Batch 3 measurements: for k decisions on an endless script, LangGraph `recursion_limit = 2k`, Pydantic AI `request_limit = k`.

Kept apart: same tool, same arguments, same call ID, different call IDs, same result, different text, same semantic state, same business operation.

- **H4.1 (B4-3.1.23, 3.1.24):** neither framework detects identical repeated calls. With different call IDs, every call executes and gets its own tool message; only the loop limit stops an endless script.
- **H4.1a (B4-3.1.24, same ID, added):** unknown. A reused call ID may be executed again, or collapsed where a framework keys results by ID. Expected: executed, but no strong prior.
- **H4.2 (B4-3.1.26, 3.1.27):** neither framework detects A→B→A, with or without changing arguments. The limit stops it.
- **H4.3 (B4-3.1.28, 3.1.29):** neither framework compares results between iterations, whether the text is identical or only semantically the same. Requests carry no injected warning.
- **H4.4 (B4-3.1.38):** the limit is the only fallback for all three runaway shapes, and it cannot tell them from legitimate work. A budget too small for legitimate polling ends that run the same way.
- **H4.5 (B4-3.1.32–3.1.34):** the frameworks provide no business-level duplicate protection. Same key: 3 executions in both frameworks; ledger 3 without the guard, 1 with it. Different keys: 3 orders. The same business intent under a different key: 2 orders, because the key defines "same", not the framework.
- **H4.6 (B4-3.1.25, 3.1.30, 3.1.31):** repeated calls can be legitimate. Polling and different-argument sequences run and terminate naturally, with no special treatment before or after the repetition.
- **H4.7 (B4-3.1.37):** trajectory history is sufficient for *syntactic* patterns only. Expected labels from a pure function (`REPEATED_CALL` = 3+ consecutive same tool and args; `NO_PROGRESS` = the same with the same result; `OSCILLATION` = 2+ exact A,B cycles):

| Trajectory | Expected label | Honest verdict |
|---|---|---|
| 3.1.23 identical read-only | NO_PROGRESS | correct |
| 3.1.25 different arguments | NONE | correct |
| 3.1.26 A,B,A,B | OSCILLATION | correct |
| 3.1.27 A(1),B(1),A(2),… | NONE | undecidable from syntax; may be pagination or a runaway |
| 3.1.28 PENDING × n | NO_PROGRESS | correct |
| 3.1.29 reworded PENDING | REPEATED_CALL | misses no-progress (same state, different text) |
| 3.1.30 PROCESSING, PROCESSING, SHIPPED | REPEATED_CALL | **false alarm** on legitimate polling |
| 3.1.31 PENDING, PENDING, COMPLETE | REPEATED_CALL | **false alarm** on legitimate polling |
| 3.1.32 same key, guard on | NO_PROGRESS (same result, replay) | correct on syntax, blind to the ledger |
| 3.1.33 different keys | NONE | correct |
| 3.1.36 empty results | NO_PROGRESS | plausible; depends on task |

- **H4.8 (B4-3.1.35):**
  - LangGraph default: each error is fed back and the loop limit stops it.
  - LangGraph `RetryPolicy` with `handle_tool_errors=False`: bounded by the policy, then the exception escapes after 1 model request.
  - LangGraph `RetryPolicy` with the default handler: the handler consumes the error first, so the policy never fires.
  - Pydantic AI `ModelRetry`: stopped by the retry budget, the only native stop here that reacts to repetition, and only to repeated *failure*. Expected (`[K]`, not read): the counter resets on success, so alternating fail/success never trips it.
  - Pydantic AI plain exception: escapes at the first failure.
- **H4.9 (B4-3.1.36):** an empty result has no loop semantics: it is a success on the wire, unlike an error.

## Batch 4 results (2026-10-31)
**Setup.** `phase-03/exp3_batch4.py` reuses the Batch 3 runners and scripted transport, Phase 2 `Orders`/`make_flaky`, and the recorder.
- Budget: 6 model decisions (LangGraph `recursion_limit=12`, Pydantic AI `request_limit=6`) unless stated.
- Tests: `tests/phase_03/test_batch4_repetition.py` (55 offline; 3 runs, 55/55 each).
- Data: `phase-03/results/batch4.json` (git-ignored); counts in [results/batch4-summary.json](results/batch4-summary.json).
- Audit: [reuse-audit.md](reuse-audit.md). Classifier decision: [decision-record.md](decision-record.md).

### Results (offline; LangGraph / Pydantic AI, identical unless noted)
| Experiment | Case | Model requests | Tool executions | Native steps | Stopped by | Offline label |
|---|---|---:|---:|---|---|---|
| B4-3.1.23 | `get_order(123)` forever | 6 | 6 | 12 / 14 | `GraphRecursionError` / `UsageLimitExceeded` | NO_PROGRESS |
| B4-3.1.24 | 3 identical, different IDs, then final | 4 | 3 | 7 / 10 | natural | NO_PROGRESS |
| B4-3.1.24 | 3 identical, **same ID across turns** | 4 | 3 | 7 / 10 | natural | NO_PROGRESS |
| B4-3.1.24 | 2 identical, **same ID in one response** | 2 / **1** | 2 / **0** | 3 / 3 | natural / **`UnexpectedModelBehavior`: "Function tool calls must have unique `tool_call_id` values"** | NONE |
| B4-3.1.24 | 2 identical, distinct IDs in one response | 2 | 2 | 3 / 6 | natural | NONE |
| B4-3.1.25 | control: 123, 124, 125, 126 | 5 | 4 | 9 / 12 | natural | NONE |
| B4-3.1.26 | A,B,A,B forever | 6 | 6 | 12 / 14 | limit | OSCILLATION |
| B4-3.1.27 | A(1),B(1),A(2),… forever | 6 | 6 | 12 / 14 | limit | NONE |
| B4-3.1.28 | PENDING forever | 6 | 6 | 12 / 14 | limit | NO_PROGRESS |
| B4-3.1.29 | reworded PENDING forever | 6 | 6 | 12 / 14 | limit | REPEATED_CALL |
| B4-3.1.30 | PROCESSING, PROCESSING, SHIPPED | 4 | 3 | 7 / 10 | natural | REPEATED_CALL (false alarm) |
| B4-3.1.31 | PENDING, PENDING, COMPLETE | 4 | 3 | 7 / 10 | natural | REPEATED_CALL (false alarm) |
| B4-3.1.31 | same, budget of 2 decisions | 2 | 2 | 4 / 6 | **limit kills legitimate work** | NONE |
| B4-3.1.36 | `search` → `[]` forever | 6 | 6 | 12 / 14 | limit | NO_PROGRESS |

No run contains a framework warning: the system and user messages were unchanged on every request, and no error names repetition. The only exception is the duplicate-ID rejection, a protocol check.

### Side effects (B4-3.1.32 – 3.1.34, Phase 2 `Orders`, both frameworks identical)
| Case | Tool executions | Ledger | Replays |
|---|---:|---:|---:|
| no guard, 3 identical calls | 3 | **3** | 0 |
| guard, same key × 3 | 3 | **1** | 2 |
| guard, keys A, B, C | 3 | 3 | 0 |
| guard, same key A, 2 call IDs | 2 | **1** | 1 |
| guard, same intent, keys A then B | 2 | **2** | 0 |

The frameworks ran every call. Only the business layer changed the ledger, and it understands "same" only through the key it is given.

### Repetition after an error (B4-3.1.35)
| Setup | Model requests | Tool executions | Stopped by |
|---|---:|---:|---|
| LangGraph default handler | 6 | 6 | loop limit; the error text goes back to the model every time |
| LangGraph default, alternating fail/ok | 6 | 6 | loop limit |
| LangGraph `RetryPolicy(3)`, `handle_tool_errors=False` | **1** | 3 | `RuntimeError` escapes after the policy's 3 attempts |
| LangGraph `RetryPolicy(3)`, default handler | 6 | 6 | loop limit: **the handler consumes the error, the policy never fires** |
| Pydantic AI `ModelRetry`, `retries=1` / `3` | 2 / 4 | 2 / 4 | retry budget: `Tool 'flaky' exceeded max retries count of N` |
| Pydantic AI `ModelRetry` alternating, `retries=1` | 6 | 6 | request limit: **the budget resets on success**, so it counts consecutive failures, not repetition |
| Pydantic AI plain exception, `retries=3` | 1 | 1 | `RuntimeError` escapes at once |

### Empty result (B4-3.1.36)
A success with no loop semantics in either framework. On the wire LangGraph sends `content: []` and Pydantic AI sends `content: ""`. Only the mock transport received this; how a provider treats an empty list was not tested.

### What stops a runaway (B4-3.1.38; A,A,A / A,B,A,B / PENDING)
| Stop | Reacts to the pattern? | Result |
|---|---|---|
| LangGraph `recursion_limit` | no, counts super-steps | 6 decisions at limit 12, then error |
| Pydantic AI `request_limit` | no, counts model requests | 6 at limit 6; **50 at the default** (measured) |
| Pydantic AI `tool_calls_limit=3` | no, counts tool calls | 3 executed, the 4th refused |
| Tool retry budget | only consecutive *failures* | not triggered by successful repetition |
| Natural termination | the model chooses | never, on these scripts |
| Application stop | not exercised | — |

LangGraph's default limit (10007) was not run; a runaway under it was not measured.

### Trajectory analysis (B4-3.1.37)
Every label matched the table written before the run (H4.7). The record is enough for syntax and no further:
- **Correct:** exact repeats, exact A,B cycles, identical results.
- **False alarm:** legitimate polling that made progress at the third call (B4-3.1.30, 3.1.31).
- **Miss:** reworded no-progress reads as REPEATED_CALL, not NO_PROGRESS (B4-3.1.29).
- **Undecidable:** A(1),B(1),A(2),… may be pagination or a runaway (B4-3.1.27).
- **Blind:** the label cannot see the ledger. Three calls read NO_PROGRESS with the guard (1 order, safe) and REPEATED_CALL without it (3 orders, harmful); the label only reflects that the result text changed, not which outcome is the problem.
- **Not predicted:** LangGraph `RetryPolicy` is labelled REPEATED_CALL although the model decided once. The recorded `reexecution` flag and the single call ID tell them apart; the label, which reads executions, does not.

### Live (`openai/gpt-oss-120b`, temperature 0, 2 trials per framework per case, budget 6 decisions, 14,116 tokens)
Not evidence of framework detection. It shows model behavior, and it exposed a confound.
| Case | LangGraph | Pydantic AI |
|---|---|---|
| "keep checking until COMPLETE", always PENDING | `check_status` × 6, then `GraphRecursionError` (2/2) | 1 check, then text saying it is PENDING (2/2) |
| `ping_a` / `ping_b` outputs telling the model to call the other | a, b, a, then the model stopped itself (2/2) | a, b, then the model stopped itself (2/2) |
| "keep checking until it has shipped", PROCESSING, PROCESSING, SHIPPED | 3 checks, correct answer (2/2) | 1 check, reports not yet shipped (2/2): **natural termination, task unfinished** |

### The discrepancy, traced: Pydantic AI echoes `reasoning`
Same model and prompt, opposite behavior. Traced in three steps:
1. **Replay of each framework's request built against the mock** (no `reasoning` in it), with `additionalProperties`/`strict` toggled: all four variants polled, 12/12. The schema envelope was **not** the cause; my first suspicion was wrong.
2. **Real second requests captured:** Pydantic AI sends the model's previous `reasoning` back in the assistant message; LangGraph drops it. That text read "we have only one call per turn… we can inform them".
3. **Direct replay, changing only that field (3 trials each):**

| Request | Next decision |
|---|---|
| Pydantic AI real request, with `reasoning` | text, 3/3 |
| Pydantic AI real request, `reasoning` removed | **tool call, 3/3** |
| LangGraph real request, no `reasoning` | tool call, 3/3 |
| LangGraph real request + the same `reasoning` | **text, 3/3** |

An offline test pins the echo itself. Consequences:
- **What the frameworks send after the first request is not the same conversation.** Any multi-turn live comparison between them carries this confound, including Phases 1–2 and Batches 1–3. Phase 2's 5.10 finding is a first-request effect and is unaffected.
- **Correction to Batch 1:** it attributed Pydantic AI's extra tokens per request to its schemas. The echo also adds tokens after the first request; the two were not separated.
- **Not established:** whether echoing helps or harms in general. N=3, one model, one prompt, one reasoning text.
- **Also seen:** one Pydantic AI reply promised "I'll keep monitoring it and let you know as soon as it changes". The run had ended and nothing was monitoring.

### Critical classification
| Dimension | Framework sees it? | Evidence |
|---|---|---|
| same tool | runs it | 3.1.25 |
| same arguments | no comparison | 3.1.23 |
| same call ID, across turns | no comparison; executed again | 3.1.24 |
| same call ID, within a response | Pydantic AI rejects (protocol); LangGraph runs both | 3.1.24 |
| different call IDs | pairs each result with its call | 3.1.24 |
| same result | no comparison | 3.1.28 |
| different text, same semantic state | no comparison | 3.1.29 |
| same business operation | no concept of it | 3.1.34 |

### Expected vs actual
| H | Expected | Actual |
|---|---|---|
| H4.1 | no detection of identical calls | **confirmed** |
| H4.1a | reused ID: unknown | across turns, executed in both. Within a response: **Pydantic AI rejects, LangGraph runs both** |
| H4.2 | no oscillation detection | **confirmed** |
| H4.3 | no result comparison | **confirmed** |
| H4.4 | limits are the fallback and cannot discriminate | **confirmed**: the same counter ends a runaway and legitimate polling |
| H4.5 | no business duplicate protection | **confirmed** |
| H4.6 | legitimate repetition runs normally | **confirmed** |
| H4.7 | syntactic labels only | **confirmed**, plus the unpredicted `RetryPolicy` false alarm |
| H4.8 | stops as listed; budget resets on success | **confirmed**; the reset was a from-memory guess, now observed |
| H4.9 | empty result has no loop semantics | **confirmed**; wire encoding differs |
| not predicted | — | the live polling split between frameworks, caused by the `reasoning` echo |

### Ownership (evidence-based)
| Capability | Model | Framework | Runtime | Application | Business/Data |
|---|---|---|---|---|---|
| Decide next tool | **owns** | — | — | shapes it (prompt, catalog) | — |
| Execute tool | requests | **dispatches** | **runs the body** | authorizes | effect happens here |
| Detect repeated call | may notice, unreliable | **no** | no | **owns policy**: which tools, what "same" means | — |
| Detect oscillation | may notice (stopped itself live) | **no** | no | **owns policy** | — |
| Detect no progress | may notice, unreliable | **no** | no | **owns**, with the tool reporting real state | holds the true state |
| Enforce loop limit | — | **enforces** | — | **chooses the value** | — |
| Prevent duplicate side effect | — | **no** | no | supplies the key | **owns** |
| Idempotency | may supply a key | **no** | no | should supply the key | **owns** the atomic check |
| Authorization | — | hook point only | — | **owns** | enforces scoping |
| Business constraint | — | no | no | may pre-check | **owns** |

### If the framework does not detect repetition, who should?
It splits, by what each layer can actually know:
- **Model:** first line, and sometimes it works (it stopped both live oscillations). Not a guarantee: it polled to the limit under one framework and quit early under the other.
- **Framework:** owns the blunt bound, and it should keep it. It has no way to know that PROCESSING to SHIPPED is progress. A generic detector would false-alarm exactly as ours did; the external detectors make "same" configurable per tool for this reason.
- **Tool:** owns meaning. It knows whether a result is terminal or whether a retry is pointless, and can say so in its result. Not tested here.
- **Application policy:** owns the decision, per tool: expected repetition, budget, and what to do when it trips. Framework hooks are the place for it.
- **Business/data:** owns duplicate effects. No upper layer can substitute for it, because a model can always send the call again.

### Is repeated tool calling itself an error?
**No.** Polling, pagination, retries and status checks all repeat by design, and an idempotent replay repeats safely (ledger 1, replays 2). Repetition only means something relative to a definition of "same", an expected progress, and a side-effect risk. The same trajectory, three calls of `check_status`, was correct when the third result was SHIPPED and a runaway when it never changed.

### Limits (Batch 4)
- Scripted mechanics; thresholds in the labels are arbitrary.
- Live: one model, 2 trials, and N=3 for the `reasoning` result.
- Not tested: tool-reported terminal state, application hooks acting on repetition, LangGraph's default limit, provider handling of empty tool content.
- `agent-loop-guard` was read, not run.


---

# Batch 5 - Retry and failure recovery within the loop (hypotheses written 2026-11-04, before any Batch 5 run)

Batch-local IDs follow the Batch 5 brief (`B5-3.1.39` ... `B5-3.1.47`). Gap analysis: [batch5-gap-analysis.md](batch5-gap-analysis.md).

## Methodological note: reasoning echo (applies to every live comparison in Phase 3.1)
Batch 4 found that Pydantic AI sends the model's earlier `reasoning` text back in later requests and LangGraph drops it, and that changing only that field flipped the model's next decision (3/3, both directions). Consequences, stated without rewriting any earlier result:
- **Offline scripted findings** (Batches 1-4: counters, limits, stop mechanics, state shapes) do not depend on model reasoning and stay valid.
- **Earlier multi-turn live comparisons between the frameworks** (Phases 1-2, Batches 1-3, and Batch 4's live polling) may be confounded by that difference. They are not re-run or re-labelled here; read them as native-behavior observations, not as a controlled comparison.
- **From Batch 5 on, a live cross-framework comparison must name its mode:**
  - **Mode A, native:** each framework as it ships.
  - **Mode B, controlled parity:** no earlier reasoning in either framework's requests. LangGraph never sends it, so LangGraph's native cell is already Mode B. Pydantic AI Mode B uses its documented profile option `openai_chat_send_back_thinking_parts=False`.
- Native and controlled cells are never compared as if equivalent.
- Not built: a condition with reasoning echoed in both frameworks (it needs request rewriting below the framework). Residual request-envelope differences remain (Phase 2 Batch 5).

## Hypotheses
Tools raise `ModelRetry` for model-visible failures. Phase 2 measured that LangGraph's default handler turns it into an error `ToolMessage` and Pydantic AI treats it as a retry prompt, so one tool object serves both frameworks.
- **H5.1 (B5-3.1.39):** the error text reaches the model on the next request, each framework wrapping it in its own template (LangGraph adds a "fix your mistakes" suffix). The failed call and its error stay in the history under the same call ID. The retry is a new call with a new ID and a new model decision; the final answer comes from the model alone.
- **H5.2 (B5-3.1.40):** neither framework tells transient from permanent. If the scripted model gives up after three failures, both frameworks accept its truthful final answer, and the execution record still shows three failed executions. Pydantic AI needs `retries>=3` for the run not to die first. Live model behavior (persistence, honesty) is unknown, and I expect it to differ between models and between native and controlled cells.
- **H5.3 (B5-3.1.41):** an earlier success stays in the next requests together with the later failure and the recovery result. The framework does not re-run A; whether a model does is model behavior.
- **H5.4 (B5-3.1.42):** ledger counts match Phase 2's table (not reclaimed here). New at loop level:
  - After a lost response the model sees only an ambiguous timeout.
  - Without a guard the retry makes a second order and the framework accepts any final answer, including one that claims a single order.
  - With a context-supplied key the retry replays (ledger 1).
  - Same key with different arguments raises `ConflictError`: a plain exception, so LangGraph feeds it back and the model can recover, while Pydantic AI's run ends.
- **H5.5 (B5-3.1.43):** a scripted model cannot show wording effects. What scripted runs can show is that each framework passes the tool's text through verbatim inside its own template and classifies nothing (including "Permission denied"). Wording may change a live model's decision; I expect "Permission denied" to stop retries more often than "Temporary failure". Unknown. The framework templates (LangGraph's suffix, Pydantic AI's JSON wrapper) are part of what the model reads.
- **H5.6 (B5-3.1.44):** with A ok, B and C failing in one response and both retried in the next, the successes are kept and not re-run, retries are independent, and each result keeps its own call ID.
- **H5.7 (B5-3.1.45):** a final answer after a failure is accepted without any check. The failure stays in the history and in the execution record whether or not the answer mentions it. Honesty belongs to the model.
- **H5.8 (B5-3.1.46):** a failed run does not mean lost state, but where it survives depends on the API layer:
  - LangGraph keeps the state only through `stream` (the last chunk). The error path through `invoke` returns nothing.
  - Pydantic AI keeps history on the run object (`iter`).
  - The Phase 1 application-level `run()` skips its `_fill` when an exception escapes (source read, not yet measured), so its `RunResult` keeps no tool calls in either framework.
  - A side effect that already committed is in neither framework's state.
- **H5.9 (B5-3.1.47):** nine separate counters per scenario reconcile without a universal "iteration". Tool-internal retries and business effects are invisible to both frameworks.

## Batch 5 results (2026-11-04)
**Setup.** `phase-03/exp3_batch5.py`. It reuses the Batch 3 runners (`e3.run`), the Batch 4 trajectory view and wire check, the Phase 2 `Orders` and `make_flaky_internal_retry` fixtures, the recorder and the Phase 1 `run()` entry points.
- New measurement code only: scripted tool fixtures, `lab3/history.py` (a neutral view of retained conversation state), a `reasoning` option on the recorder's Pydantic AI model, and `error_body` capture for non-200 responses.
- Tests: `tests/phase_03/test_batch5_recovery.py` (48 offline; 3 runs, 48/48 each).
- Data: `phase-03/results/batch5.json` (git-ignored); live counts in [results/batch5-live-summary.json](results/batch5-live-summary.json).
- Gap analysis: [batch5-gap-analysis.md](batch5-gap-analysis.md).
- Failures are raised as `ModelRetry`, which both frameworks make model-visible, so one tool object serves both frameworks. Each framework puts its own template around the text (below).
- Tools are synchronous; batched calls run on worker threads, and result order follows call order (Phase 2 Batch 4).

### What was not rerun
| Question | Why not |
|---|---|
| retry-budget arithmetic, step counts per retry | ALREADY COVERED by Batch 3 (B3-3.1.17/18). Used only as reference in the accounting table |
| ledger counts for no guard / same key / different key / key changed by the model | ALREADY COVERED by Phase 2 (idempotency table) and Batch 4 (B4-3.1.32-34). Asserted in passing, not claimed anew |
| repeated identical failures to the loop limit | Batch 4 B4-3.1.35 |
| sibling results inside one batch | Phase 2 4.12-4.13 |

### Reasoning-echo handling
Live cells: **LG native (= controlled)**, **PA native** (echoes reasoning) and **PA controlled** (`reasoning="stripped"`, the documented profile option `openai_chat_send_back_thinking_parts=False`). Offline tests check on the wire that LangGraph and PA controlled send no `reasoning` field, native PA sends it, and the controlled mode changes nothing else in the requests.
Not removed, so "controlled" means *only* the echo is equalized:
1. the tool-error template (LangGraph: `Error: ModelRetry('<text>')\n Please fix your mistakes.`; Pydantic AI: `{"error":"<text>"}`),
2. Pydantic AI's `strict`/`additionalProperties:false`/`tool_choice:auto` envelope (Phase 2 Batch 5),
3. message formatting.
Not built: reasoning echoed in both frameworks.

### Scripted results (LangGraph / Pydantic AI, identical unless noted)
| Exp | Case | Result |
|---|---|---|
| B5-3.1.39 | A fails once, model retries, then final | 3 model requests, 2 tool executions, 1 failed; call IDs `call_1_0` (failed) and `call_2_0` (retry). The error is in request 2 under its own template, and **both calls stay in every later request and in the retained history** (`error`/`retried`, then `success`). The retry is a new model decision; the final answer is the scripted model's text |
| B5-3.1.40 | A always fails; model gives up after 3 (`retries=3`) | accepted: natural stop, 3 failures recorded, final text kept. Neither framework tells transient from permanent |
| B5-3.1.40 | same script, Phase 1 default `retries=1` (Pydantic AI) | **the run dies at the 2nd failure** (`exceeded max retries count of 1`); the model's give-up is never requested. The framework's retry budget pre-empts the model's decision |
| B5-3.1.41 | A ok, B fails, C ok, final | A, B, C each executed once; request 4 carries A's result, B's error and C's result under IDs 1_0, 2_0, 3_0; A is not re-run |
| B5-3.1.42 A | no guard, lost response, model retries | ledger **2**, 2 executions; the model sees only "The order service timed out before returning a response." (no word of creation); a final "Your order has been created." is accepted unchecked |
| B5-3.1.42 B | context key, same | ledger 1, 1 replay; the replay result reads like a fresh order (`order #1 created: 1 x widget`) |
| B5-3.1.42 C | same key, different args | ledger 1. **LangGraph:** the `ConflictError` becomes an error message, the model answers, natural stop. **Pydantic AI:** the run ends with `ConflictError` (a plain exception), no final answer |
| B5-3.1.43 | 4 wordings, raised | the text arrives verbatim inside the template; requests, executions and steps are identical across wordings. The framework classifies nothing, "Permission denied" included |
| B5-3.1.43 | "Permission denied" returned as a normal result | arrives verbatim, no template, 0 failed executions |
| B5-3.1.44 | `[A,B,C]` (B, C fail) then `[B,C]` then final | executions A 1, B 2, C 2; IDs `1_0,1_1,1_2` then `2_0,2_1`; request 3 holds all five results under the right IDs; successes not re-run |
| B5-3.1.45 | A fails, honest final / final that does not mention it | both accepted. The failure stays in the retained history and in the execution record either way. The Phase 1 `run()` keeps the error text in `tool_calls` for a completed run |

### Retry exhaustion and retained state (B5-3.1.46)
| Case | Stopped by | Executions (failed) | Model requests | Native retained state | Phase 1 `run()` |
|---|---|---:|---:|---|---|
| E1 LangGraph default, limit 6 | `GraphRecursionError` | 3 (2) | 3 | user, A call, A ok, B call, B error, B call, B error | no tool calls, no text, 0 LLM calls |
| E2 LangGraph `RetryPolicy(3)`, `handle=False` | `ModelRetry` (escaped) | 4 (3) | 2 | user, A call, A ok, **B call (pending)**; the 3 B executions leave no trace | same |
| E3 Pydantic AI `ModelRetry`, `retries=1` | `UnexpectedModelBehavior` | 3 (2) | 3 | user, A call, A ok, B call, B retry prompt, **B call (pending)** | same |
| E4 Pydantic AI plain `RuntimeError` | `RuntimeError` | 2 (1) | 2 | user, A call, A ok, **B call (pending)** | same |
| E5 LangGraph default, `create_order` always loses its response, limit 6 | `GraphRecursionError` | 3 (3) | 3 | 3 calls, 3 errors; ledger **3** | — |
| E6 Pydantic AI `ModelRetry`, `retries=2` | `UnexpectedModelBehavior` | 3 (3) | 3 | 2 retry prompts + **1 pending call**; ledger **3** | — |
- A failed run keeps a usable record, but only in the native view (`graph.stream` last chunk, `agent.iter` run state). The Phase 1 application-level `run()` returns **no tool calls, no text and 0 LLM calls** after any failed run, in both frameworks, because its `_fill` is skipped when an exception escapes.
- **A pending call in retained state does not mean "not executed".** E2 ran B three times, E3 and E6 ran their last call, and the state shows an unanswered call in each. For a side-effecting tool that is "outcome unknown", the same shape as a lost response.
- Retained state never mentions a committed effect: in E5/E6 the ledger has 3 orders, no message says "created", and in E6 the third commit appears in no retained message.

### Recovery accounting (B5-3.1.47; nine separate counters)
| Scenario | HTTP attempts | HTTP errors | Model requests | Model responses | Tool executions | Steps (LG / PA) | Model-visible retries | Framework retries | Tool-internal retries | Business effects |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| model-visible retry (39) | 3 | 0 | 3 | 3 | 2 | 5 / 8 | 1 | 0 | 0 | — |
| LangGraph `RetryPolicy`, 2 failures | 2 | 0 | 2 | 2 | 3 | 3 / — | 0 | 2 | 0 | — |
| tool-internal retry, 2 internal failures | 2 | 0 | 2 | 2 | **1** | 3 / 6 | 0 | 0 | **2** | — |
| HTTP retry, 1x 500 | **3** | 1 | 2 | 2 | 1 | 3 / 6 | 0 | 0 | 0 | — |
| lost response + model retry, no guard | 3 | 0 | 3 | 3 | 2 | 5 / 8 | 1 | 0 | 0 | **2** |
No row has an "iteration" column. Tool-internal retries and business effects appear in neither framework's counters; the framework sees one execution where the tool ran three times.

### Live results (`openai/gpt-oss-120b`, temperature 0; 72 runs, about 70K tokens; no daily-quota stop)
Each wording was run 3 times per cell with a 5-decision budget, a tool that always fails, and Pydantic AI `retries=8` so its retry budget does not pre-empt the model. Outcomes are assigned from what the run recorded.

| Wording | LG native (= controlled) | PA native | PA controlled |
|---|---|---|---|
| Temporary failure | budget exhausted, no final: 3/3 (5 executions each) | gave up and said so: 3/3 (2-3 executions) | budget exhausted, no final: 3/3 (5) |
| Please retry the operation | provider rejected the call: 2/3 + budget 1/3 (re-run: 3/3 rejected) | provider rejected the call: 3/3 (re-run: 3/3) | budget exhausted: 3/3 (re-run: 3/3) |
| Operation unavailable | **empty final accepted: 3/3** (2 executions) | gave up and said so: 3/3 (1) | gave up and said so: 3/3 (1) |
| Permission denied | gave up and said so: 3/3 (1) | 3/3 (1) | 3/3 (1) |

Three findings from this table:
1. **Wording changes the model's decision.** "Permission denied" ended the attempts after one execution in 9/9 runs. "Temporary failure" produced retrying in 6/9 (to the budget). The framework treated all four the same.
2. **The echo changes it too (Mode A vs B).** For "Temporary failure", native Pydantic AI stopped and reported honestly 3/3, while LangGraph and controlled Pydantic AI kept retrying to the budget 6/6. This agrees with Batch 4's polling result.
3. **Mode B does not make the frameworks equivalent.** With the echo removed, LangGraph and Pydantic AI still differed ("Operation unavailable": LangGraph ended on an empty reply after 2 executions, controlled Pydantic AI gave an honest final after 1; "Please retry": LangGraph hit provider rejections 5/6, controlled Pydantic AI 0/6). The remaining candidates are the error template and the tool-schema envelope, neither separable without changing framework behavior.

**Provider rejection (new failure class).** After "Please retry the operation", the model changed `report_id` from the string `"7"` to the number `7`. Groq rejected it with HTTP 400 `tool_use_failed` ("parameters for tool fetch_report did not match schema: `/report_id`: expected string, but got number") **before either framework saw a response**, so neither could feed the error back. Both runs died. The re-run reproduced it exactly for both native cells (6/6 each, counting the original runs for Pydantic AI; LangGraph 5/6). This was classified as provider-side, not as a framework failure. The recapture also showed one Groq requests-per-minute 429 (limit 30), retried transparently by the SDK, which my classifier first mislabeled; the classifier was fixed and a test added.

**Empty final answer in the wild.** LangGraph accepted an empty reply as the final answer 3/3 for "Operation unavailable". Batch 2 predicted this from a scripted empty reply; this is its first live occurrence. The user would have received an empty string reported as success.

**Lost response (B5-3.1.42 live, 9 runs per case, 3 per cell, all cells identical):** in every run the model retried once after the timeout and then said "Your order has been placed successfully."
- No guard: ledger **2** in 9/9; the final answer names "Order #2" and mentions no earlier attempt, so it does not report the possible duplicate.
- Context key: ledger **1** in 9/9, 1 replay each; the answer is accurate.

**Recovery after an earlier success (B5-3.1.41 live, 9 runs):** all cells identical. The model looked up the customer once, tried `send_receipt` once, did not retry it, used `log_note` as instructed, and said so. It never repeated A. The prompt named the fallback, so this tests following a documented recovery path, not inventing one.

### Expected vs actual
| H | Expected | Actual |
|---|---|---|
| H5.1 | error visible under a template; failed call kept; retry new decision | **confirmed**; IDs `call_1_0`/`call_2_0` in every later request |
| H5.2 | no transient/permanent split; give-up accepted; `retries>=3` needed | **confirmed**, plus the unplanned point: the framework's budget can end the run before the model's decision (Pydantic AI default `retries=1`) |
| H5.3 | successes kept, A not re-run | **confirmed** offline; live 9/9 |
| H5.4 | loop-level lost-response and conflict behavior | **confirmed**, including LangGraph recovering from `ConflictError` while Pydantic AI's run ends. Live: duplicate not reported in 9/9 |
| H5.5 | scripted shows pass-through; wording may matter live | **confirmed**: scripted pass-through in templates; live: wording changed decisions, and the echo changed them again |
| H5.6 | independent retries, IDs paired | **confirmed** |
| H5.7 | any final accepted; failure stays in record | **confirmed** (and the live empty final) |
| H5.8 | state survives only in some API layers | **confirmed**; plus the unplanned "pending call may be executed" and "state never shows a committed effect" |
| H5.9 | nine counters reconcile | **confirmed**; no universal "iteration" needed |
| not predicted | — | provider rejection of a type-changed retry; live empty final; Mode B still not equivalent |

### Ownership (evidence-based)
| Concern | Model | Framework | Runtime | Application | Business/Data |
|---|---|---|---|---|---|
| Decide whether to retry | **owns** (wording and echo change it) | pre-empts with budgets (`retries`, loop limit) | — | sets the budgets | — |
| Expose tool error | — | **owns the text and template** | — | chooses raise vs return | tool chooses the message |
| Execute retry | requests it (model-visible) | LangGraph `RetryPolicy` re-runs without the model | runs the body | — | tool-internal retries are invisible above |
| Preserve conversation history | in context only | **holds it** (native view) | — | must capture it: Phase 1 `run()` drops it on failure | — |
| Preserve successful results | — | holds them in history; no retries of successes | — | — | the effect itself |
| Enforce retry limits | — | **enforces** | — | **chooses values** | — |
| Prevent duplicate business effects | — | **no** | no | supplies a stable key | **owns** |
| Maintain idempotency | may carry a key | no | no | should generate it | **owns** the atomic check and the conflict |
| Produce a truthful final response | **owns**; unreliable (duplicate unreported 9/9, empty final 3/3) | accepts any final | — | must verify against the ledger if it matters | holds the ground truth |

### Limits (Batch 5)
- Scripted mechanics are exact for these versions; live is one model, N=3 per cell, and the wording set is four phrases.
- "Controlled" equalizes the echo only; error template and schema envelope still differ.
- A condition with reasoning echoed in both frameworks was not built.
- Honesty labels come from a keyword check plus reading; "did not mention a duplicate" was read, not machine-verified.
- Not tested: checkpointed resume of a failed run, automatic retry of the model call itself, streaming, tools raising non-`ModelRetry` exceptions in live runs.
- Langfuse: NOT AVAILABLE.


---

# Batch 6 - Timeout, cancellation and token limits (hypotheses written 2026-11-05, before any Batch 6 run)

Batch-local IDs follow the Batch 6 brief (`B6-3.1.48` ... `B6-3.1.58`). Gap analysis, mechanism table and reuse audit: [batch6-gap-analysis.md](batch6-gap-analysis.md). Reasoning-echo modes (Batch 5 note) apply to any live comparison; this batch is scripted and local, so they do not arise unless stated.

Terms kept apart: **timeout** (a deadline fired), **cancellation request** (a caller asks), **cancellation acknowledged** (the callee saw it and stopped), **work stopped** (the underlying thread, coroutine or server-side job actually ended).

- **H6.1 (B6-3.1.48, model request timeout):**
  - The timeout is enforced by the provider client (httpx through the OpenAI SDK), not by either framework.
  - The SDK retries a timed-out attempt (default 2 retries), so one framework model call costs up to 3 HTTP attempts and about 3 x timeout of wall time. The framework sees one call.
  - With retries off, the error surfaces after one timeout. It ends the run: no retry prompt, no recovery by the model.
  - The server-side work continues after the client gives up (the handler finishes and its write fails).
  - Pydantic AI's `request_timeout` is a separate clock over all attempts: with default retries it cuts the request at about `request_timeout` and raises `ModelRequestTimeout`, while `timeout` alone yields about 3 x.
- **H6.2 (B6-3.1.49, synchronous tool):** when the caller gets the timeout or cancellation, the worker thread is still alive and the tool finishes later; its effect commits after the caller returned. Pydantic AI's `Tool(timeout)` turns it into a model-visible retry while the first execution is still running. "Run reported cancelled" does not mean "tool stopped".
- **H6.3 (B6-3.1.50, asynchronous tool):** cancellation reaches the coroutine as `CancelledError` and `finally` runs; an effect committed before the await stays, one after it never happens. Two ways to defeat it: a tool that swallows `CancelledError` keeps running, and a blocking call inside an async tool freezes the event loop so no timeout can fire until it returns. Uncertain which way each framework behaves for the swallowing tool.
- **H6.4 (B6-3.1.51, whole run):**
  - Pydantic AI `run(timeout=)` covers model requests, tools and the gaps between steps with one deadline. It raises `RunTimedOut` carrying `all_messages()`, and aborts an in-flight model request or async tool; a sync tool's thread continues.
  - LangGraph has no whole-run timeout; `asyncio.timeout` around the run is application code and behaves like a task cancel. Its `step_timeout` bounds one super-step.
- **H6.5 (B6-3.1.52, cancellation points, four mechanisms):**
  - Task cancel and Pydantic AI `cancel()` abort an in-flight HTTP request (the server sees the disconnect) and async tools.
  - LangGraph `request_drain` does not interrupt anything: it takes effect at the next super-step boundary. An in-flight model request or tool completes first.
  - A requested but not yet started tool does not run after a drain, but the model's pending call is left in state.
  - Sync tools keep running under every mechanism.
  - Pydantic AI `cancel()` is acknowledged promptly as `RunCancelled`; external task cancel stays `CancelledError`.
- **H6.6 (B6-3.1.53, tokens):**
  - LangGraph enforces no token or usage limit. Usage is readable from the messages.
  - Pydantic AI checks provider-reported token usage after each response, before running that response's tool calls, so a response that crosses a token limit stops the run before its tool executes. A limit crossed only by the final answer raises after the answer was received (the answer is discarded).
  - Neither counts provider-client HTTP retries or failed requests, so provider consumption can exceed an application budget without any framework noticing.
  - A context-window rejection is an HTTP 400: one attempt, not retried, raised as a model error with no usage.
  - `max_tokens` is a per-request provider cap on the wire, not a run budget.
- **H6.7 (B6-3.1.54, pending tool call):** after any stop the framework's retained state lists the requested call but cannot say whether it executed. Executed after the stop: Pydantic AI `request_limit`, a running tool at timeout or cancel. Not executed: LangGraph at an odd recursion limit, Pydantic AI `tool_calls_limit`, a token limit crossed by the tool-call response, a drain before the tools node.
- **H6.8 (B6-3.1.55, cancel after a business effect):** an effect committed before the cancel remains, the caller still receives an exception, and the tool's result is not in retained state. The framework does not record the completed operation. The business layer is the only source of the outcome: a lookup by idempotency key shows it; re-running with the same key replays (ledger stays 1), without it duplicates (2). A sync tool cancelled during its pre-commit delay still commits afterwards.
- **H6.9 (B6-3.1.56, state):**
  - LangGraph `invoke` leaves the caller with only the exception; `astream` leaves the last streamed values.
  - Pydantic AI `run` / `run_sync` raise `RunCancelled` / `RunTimedOut` with `all_messages()`; `UsageLimitExceeded` and model errors carry no messages; `iter` keeps the run state.
  - Pending calls are visible in state but ambiguous.
- **H6.10 (B6-3.1.57, timeout x retry):**
  - A timeout triggers a retry only at the SDK layer, and only for HTTP timeouts.
  - A framework-level tool timeout in Pydantic AI becomes a model-visible retry, which can start a second execution while the first is still running (duplicate effect).
  - LangGraph `RetryPolicy` does not retry a `step_timeout`.
- **H6.11 (B6-3.1.58, cumulative deadline):** per-request, per-tool and per-step timeouts provide no bound on total run time: three steps each inside their own timeout can exceed an application deadline. Pydantic AI `run(timeout=)` bounds the total; LangGraph needs an application-level deadline.


## Batch 6 results (2026-11-05)
**Setup.** `phase-03/exp3_batch6.py` (11 groups, 136 recorded result lines), `lab3/slow_server.py` (stdlib scripted HTTP server on 127.0.0.1), tool probes that log their own start / effect / cancel / finish into the run timeline, and a driver that requests a stop when a named event appears (not after a sleep).
- Versions: LangGraph 1.2.18, Pydantic AI 2.62.0, openai SDK 3.39.0, httpx 0.28.1, Python 3.12.10.
- Tests: `tests/phase_03/test_batch6_termination.py` (73 offline, local sockets).
- Data: `phase-03/results/batch6.json` (git-ignored); compact copy in [results/batch6-summary.json](results/batch6-summary.json).
- Gap analysis, mechanism inventory and audit: [batch6-gap-analysis.md](batch6-gap-analysis.md). No dependency added.
- **Live: NOT RUN.** Every question in this batch concerns framework, runtime and provider-client mechanics, which a scripted local server answers exactly; none depends on what a real model would say. The reasoning-echo modes therefore did not arise.
- Times are ms since the run began. The first model request is sent about 150-350 ms in (graph and client set-up); "from 1st request" below removes that.

### Not rerun (already covered)
| Question | Where |
|---|---|
| per-tool timeout matrix for sync and async tools | Phase 2 B2.2 (not repeated; B6-3.1.49 only adds thread liveness and effect timing) |
| request and tool-call limit arithmetic | Batch 3 |
| HTTP 5xx / 429 retry accounting | Batch 3 |

### Mechanisms found in the installed versions (read from source, then exercised)
| Layer | LangGraph | Pydantic AI |
|---|---|---|
| per HTTP attempt | OpenAI SDK `timeout` / `max_retries` via `ChatOpenAI` | `ModelSettings.timeout` / `max_retries` |
| whole model request | none | `ModelSettings.request_timeout` (`ModelRequestTimeout`) |
| tool | none (`step_timeout` covers a super-step) | `Tool(timeout)`, `Agent(tool_timeout)` |
| whole run | none (application `asyncio.timeout`) | `run(timeout=)` / `run_sync(timeout=)` (`RunTimedOut`) |
| first-party stop | `RunControl.request_drain()` (cooperative) | `AgentRun.cancel()`, `CancellationToken` (thread-safe, works with `run_sync`) |
| token / usage | none enforced; `AIMessage.usage_metadata` readable | `UsageLimits` (request, tool calls, input, output, total, per-request input, cost) |

### B6-3.1.48 Model request timeout (real sockets; 0.4 s per attempt, server "works" 1.5 s per request)
| Framework | Case | Exception | Server attempts | Client closed the connection | Response delivered | Returned (ms from 1st request) |
|---|---|---|---:|---:|---:|---:|
| LangGraph | every attempt slow, SDK defaults | `OpenAITimeoutError` | **3** | 3 | 0 | 2597 |
| LangGraph | `max_retries=0` | `OpenAITimeoutError` | 1 | 1 | 0 | 435 |
| LangGraph | first attempt slow, retry fast | none (recovered) | 2 | 1 | 1 | 842 |
| Pydantic AI | every attempt slow, SDK defaults | `ModelTimeoutError` | **3** | 3 | 0 | 2588 |
| Pydantic AI | `max_retries=0` | `ModelTimeoutError` | 1 | 1 | 0 | 413 |
| Pydantic AI | first attempt slow, retry fast | none (recovered) | 2 | 1 | 1 | 883 |
| Pydantic AI | `request_timeout=0.7` over `timeout=0.4` | `ModelRequestTimeout` | 1 | 1 | 0 | 721 |
| Pydantic AI | `request_timeout=3.0` over `timeout=0.4` | `ModelTimeoutError` | 3 | 3 | 0 | 2466 |
- **Layer:** the per-attempt timeout is enforced by the provider client (httpx through the OpenAI SDK) in both frameworks. The SDK retries a timed-out attempt (2 retries by default), so one model call cost 3 HTTP attempts and about 2.6 s for a 0.4 s timeout (6.5x, with SDK backoff). Neither framework counts the extra attempts: one model call, one step.
- **Pydantic AI `request_timeout` is the only cap on the whole request.** At 0.7 s it stopped at 721 ms with one attempt; above the SDK's own exhaustion time (3.0 s) the SDK gave up first.
- **The framework sees the timeout as a model error** (`ModelTimeoutError`, `OpenAITimeoutError`) raised out of the run. There is no retry prompt and the model cannot recover; a first attempt that is slow and a retry that is fast recovers invisibly inside the SDK.
- **Original request:** in every case the server finished the full 1.5 s of work for each abandoned attempt (3 attempts = 4.5 s of provider work for an error) and delivered nothing. The client's timeout stopped waiting, not the provider.
- Retained state after a model error: only the user prompt (no model response existed).

### B6-3.1.49 Synchronous tool (1.0 s of work, effect at the end; timeout or stop at 0.3 s)
| Mechanism | Exception | Caller returned (ms) | Tool finished (ms) | Tool running at return | Effect after return | Model then |
|---|---|---:|---:|---|---:|---|
| PA `Tool(timeout=0.3)` | none | 484 | 1171 | **yes** | 1 | got a retry prompt, answered "Done." |
| PA `run(timeout=0.3)` | `RunTimedOut` | 447 | 1144 | **yes** | 1 | - |
| LG `step_timeout=0.3` | `TimeoutError` | 641 | 1327 | **yes** | 1 | - |
| LG / PA `asyncio.timeout(0.3)` | `TimeoutError` | 631 / 464 | 1339 / 1168 | **yes** | 1 | - |
| PA `AgentRun.cancel()` | `RunCancelled` | 291 | 1186 | **yes** | 1 | - |
| LG / PA `Task.cancel()` | `CancelledError` | 464 / 279 | 1349 / 1168 | **yes** | 1 | - |
| LG `request_drain()` | `GraphDrained` | **1338** | 1336 | no | 0 | - |
Every timeout and cancel returned within about 0.3 s while the tool thread kept running for another 0.7 s, then committed its effect. Only the drain waited for the tool, because it acts at a super-step boundary. "Run reported cancelled" does not mean "tool stopped". The runtime owns this limit: Python has no way to stop a thread.

### B6-3.1.50 Asynchronous tool
| Case | Result |
|---|---|
| normal tool, effect before the await | cancelled, `finally` ran, **effect remained (1)**: Pydantic AI `Tool(timeout)` and LangGraph `step_timeout` |
| normal tool, effect after the await | cancelled, `finally` ran, **no effect** |
| tool that swallows `CancelledError` | the caller was held until the tool finished: returned at 1133-1211 ms (Pydantic AI) and 1211-1366 ms (LangGraph) across two runs, for a 0.3 s timeout; effect committed. **Pydantic AI re-delivered the cancellation tens of thousands of times** (82,288 and 137,015 in two runs) in the tool's loop: its cancel scope keeps cancelling at every await. LangGraph's one-shot task cancel delivered it once |
| blocking `time.sleep(0.8)` inside an `async def` tool | the event loop froze; the timeout fired at 992 ms (PA) / 1142 ms (LG) instead of about 480 / 650 (the normal-tool case) |
| `run(timeout=0.3)` / `asyncio.timeout(0.3)`, normal tool | cancelled, no effect, returned about 0.3 s after the tool began |
Asynchronous cancellation is cooperative. It works when the coroutine awaits and lets `CancelledError` propagate; a swallowing tool or a blocking call defeats it, and the timeout cannot report what the tool refuses to give up.

### B6-3.1.51 Whole-run timeout (deadline 0.6 s)
| Where the run was | Pydantic AI `run(timeout)` | Pydantic AI external `asyncio.timeout` | LangGraph `asyncio.timeout` (application) |
|---|---|---|---|
| model request in flight | `RunTimedOut`, messages carried: 1; provider kept working | `TimeoutError`, no messages; provider kept working | `TimeoutError`; provider kept working |
| async tool in flight | `RunTimedOut`, messages 3; tool cancelled; call pending in state | `TimeoutError`; tool cancelled | `TimeoutError`; tool cancelled |
| sync tool in flight | `RunTimedOut`; thread **still running**; effect after return | same | same |
| 2nd model request in flight | `RunTimedOut`; tool result retained; provider kept working | `TimeoutError` | `TimeoutError` |
- The Pydantic AI deadline covers model requests and tools alike with one clock. `RunTimedOut` is both a `RunCancelled` and a `TimeoutError`, and carries `all_messages()`.
- LangGraph has no whole-run timeout; the application's `asyncio.timeout` behaves like a task cancel and leaves state only in a stream's last chunk.
- The same external deadline around Pydantic AI produces a plain `TimeoutError` with no message snapshot: the native one is what carries state.

### B6-3.1.52 Cancellation propagation (seven points x mechanisms; `LG` = Phase 1 graph with a sync model node, `LG*` = same graph with an async model node)
Latency is request-to-caller-return; "disconnect" = the provider saw the client close.

| Point | LG task | LG drain | LG* task | PA task | PA `cancel()` |
|---|---|---|---|---|---|
| **A** before the request is on the wire | returns at once, **the request is sent anyway 350 ms later and answered** (ghost request) | waits ~350 ms; request sent and answered; stops before the tool | nothing sent | nothing sent | nothing sent |
| **B** during the model request | returns at once; **request not aborted, response delivered about 1.4 s later to nobody** | waits 1.4 s for the response, stops; call left pending | **disconnect**, no response | **disconnect** | **disconnect** |
| **C** tool requested, not started | tool never starts; call pending | waits ~1.5 s; **the tool runs** (its step was already scheduled) | same as LG | tool never starts; call pending | same |
| **D** sync tool running | returns at once; **thread runs on, effect after return** | waits for the tool, effect before return | - | same as LG task | same |
| **E** async tool running | `CancelledError` in the tool, no effect | waits, tool completes, effect | - | `CancelledError` in the tool | same |
| **F** just after the tool finished | effect kept; result lost; call pending | stops at once, state complete | - | same as LG task | same |
| **G** final model response in flight | returns at once; **response delivered later, discarded** | waits; the run **completes normally** (nothing left to drain) | disconnect | disconnect | disconnect |
- All Pydantic AI mechanisms returned within 1-6 ms and tore down the HTTP request. `AgentRun.cancel()` raised `RunCancelled`; external task cancel stayed `CancelledError`, as documented.
- A synchronous Pydantic AI run (`run_sync`) was cancelled from another thread with a `CancellationToken` (`RunCancelled` after 611 ms, async tool cancelled).
- LangGraph's drain is never an interruption: it takes effect at the next super-step boundary, so a long model request or tool finishes first.
- **The Phase 1 LangGraph graph cannot cancel a model request, because its model node is a plain `def` calling `llm.invoke`.** Under `astream` it runs in a worker thread; cancelling the task releases the caller but the HTTP call continues. Switching the node to `async def ... ainvoke` (the optional `async_model` flag) fixes it. This is an application design choice, not a LangGraph limit.

### B6-3.1.53 Token and usage limits (scripted usage: response 1 = 80 tokens, response 2 = 130)
| Case | Result |
|---|---|
| PA `total_tokens_limit=50`, crossed by the **tool-call response** | `UsageLimitExceeded` after response 1; **tool not started**; call pending; usage kept (1 request, 80 tokens) |
| PA total 100 / input 150 / output 40 / per-request input 90, crossed only by the **final answer** | `UsageLimitExceeded` after the answer arrived; **tool had run**; **answer discarded**; both responses were delivered, so the provider had produced them |
| PA `request_limit=1` | the tool ran, the next request was refused (Batch 3) |
| PA `count_tokens_before_request` | `NotImplementedError`: not supported by the OpenAI chat model; 0 requests sent |
| LangGraph | no limit exists; the same script completes; usage readable (80 + 130 = 210) |
| HTTP 400 context-window rejection on request 2 | one attempt, **not retried**; LG `OpenAIInvalidRequestError`, PA `ModelContextWindowExceededError`; usage of request 1 kept |
| HTTP 500, SDK retries | 3 HTTP attempts, **2 counted** (PA `usage.requests`), the failed attempt invisible in usage |
| `max_tokens=25` on the model | sent as `max_completion_tokens: 25` by both; enforced by the provider per request |
- Token limits are checked on provider-reported usage after each response; cumulative limits are re-checked before the next request. A limit is therefore enforced after the tokens were spent.
- Provider-client retries and failed requests are not in the framework's usage, so provider consumption can exceed an application budget unseen.

### B6-3.1.54 Pending tool call at a limit
| Stop | Requested | Started | State: results / pending |
|---|---:|---:|---|
| LG `recursion_limit=3` | 2 | 1 | 1 / **1** (never started) |
| LG `recursion_limit=4` | 2 | 2 | 2 / 0 |
| PA `request_limit=2` | 2 | 2 | 2 / 0 |
| PA `tool_calls_limit=1` | 2 | 1 | 1 / **1** (never started) |
| PA token limit crossed by the tool-call response | 1 | 0 | 0 / **1** (never started) |
| LG drain as the call arrives | 1 | 1 | 1 / 0 |
| PA `cancel()` while the tool is gated | 1 | 0 | 0 / **1** (never started) |
| PA `run(timeout)` with a running sync tool | 1 | 1 | 0 / **1** (running, then committed) |
**The application cannot distinguish a requested operation from an executed one from retained state:** four runs ended with one unanswered call that never started, and one with one that did start and committed. Only a tool-side record separates them.

### B6-3.1.55 Cancellation after a business effect (`Orders`, key `req-77`)
| Case | Ledger at end | Effect after return | Framework recorded the operation | Keyed retry adds | Unkeyed retry adds |
|---|---:|---:|---|---:|---:|
| async: committed, response delayed, cancelled | 1 | 0 | no (no result, 1 pending call) | **0** | 1 |
| async: cancelled before the commit | 0 | 0 | no | **1** | 1 |
| sync: cancelled during the pre-commit delay | 1 | **1** | no | 0 | 1 |
| sync: committed, response delayed, cancelled | 1 | 0 | no | 0 | 1 |
Same in both frameworks. The caller always received an exception; the business layer alone knew the outcome (the key was recorded or not).
**Immediate retry race:** retrying with the same key the moment the error arrived, while the cancelled sync tool was still inside its pre-commit delay, produced **2 orders** against the non-atomic guard (both frameworks) and **1 order plus 1 replay** against the atomic guard.

### B6-3.1.56 Partial state after termination
| Termination | LangGraph `ainvoke` | LangGraph `astream` (last chunk) | Pydantic AI `run` | Pydantic AI `iter` |
|---|---|---|---|---|
| loop limit | exception, no state | history, call pending | `UsageLimitExceeded`, no state | history, usage |
| run timeout | `TimeoutError`, no state | history, call pending | **`RunTimedOut` + `all_messages()`** | history, call pending |
| cancellation | `GraphDrained`, no state | state complete | **`RunCancelled` + `all_messages()`** | history, call pending |
| usage limit | n/a | n/a | `UsageLimitExceeded`, no state | history, usage kept (80 tokens), call pending |
| provider error | `OpenAIInvalidRequestError`, no state | history before the failure | `ModelContextWindowExceededError`, no state | history before the failure |
Only `RunTimedOut` and `RunCancelled` carry a snapshot on the plain call path. **The snapshot is identical whether the sync tool will still commit or the async tool was cancelled** (`user, assistant call`, no result); a second after the snapshot the ledger was `['work']` for the sync tool and `[]` for the async one. Partial state does not say which.

### B6-3.1.57 Timeout and retry
| Case | Result |
|---|---|
| PA `Tool(timeout=0.3)` on a sync tool, model calls again | the retry prompt arrived at 0.3 s while the first thread was running; the model's second call started a **second, overlapping execution**: 2 effects, both after the run returned; 3 model requests |
| LG `step_timeout=0.3` with `RetryPolicy(max_attempts=2)` | **not retried**: `TimeoutError`, 1 execution |
| LG tool-level `TimeoutError` (worker thread, 0.3 s) + `RetryPolicy(max_attempts=2, retry_on=TimeoutError)` | retried with no model call; the second worker started while the first still ran: **2 overlapping executions, 2 effects** |
| HTTP timeout | retried by the SDK only (B6-3.1.48) |
A framework retry after a timeout is a second execution, not a continuation; whether the first one stopped is unknown to the retrier.

### B6-3.1.58 Cumulative deadline (three rounds: model 0.3 s + async tool 0.3 s each)
| Setup | Result | Run duration |
|---|---|---:|
| LG `step_timeout=0.5` only | completed, no error | 2198 ms |
| PA `timeout` / `request_timeout` / tool timeout 0.5 each | completed, no error | 2189 ms |
| PA `run(timeout=1.0)` | `RunTimedOut`, 1 tool done, 1 cancelled | 1011 ms |
| PA / LG `asyncio.timeout(1.0)` | `TimeoutError`, same state | 1018 / 1010 ms |
Every step satisfied its own timeout and the run took twice the application's deadline. Per-step timeouts give no bound on total time; only a run-level deadline does (native in Pydantic AI, application-owned in LangGraph).

### Expected vs actual
| H | Expected | Actual |
|---|---|---|
| H6.1 | provider client enforces and retries; framework sees one call; server keeps working; `request_timeout` is a separate clock | **confirmed** exactly |
| H6.2 | thread alive at return; effect lands later | **confirmed**; only the LangGraph drain waited |
| H6.3 | cancel delivered; swallowing and blocking defeat it (uncertain how) | **confirmed**; swallowing held the caller for the tool's full run; **Pydantic AI re-delivered the cancel tens of thousands of times** (unpredicted) |
| H6.4 | Pydantic AI native deadline covers model and tools; LangGraph has none | **confirmed**; native one carries messages, external does not |
| H6.5 | task cancel and PA `cancel()` abort the HTTP request; drain does not interrupt | **partly wrong:** LangGraph task cancel did **not** abort the request with the Phase 1 sync node (ghost request, even before the wire); an async node does |
| H6.6 | PA tokens checked after each response; LangGraph none; retries invisible | **confirmed**; plus `count_tokens_before_request` unsupported here and a dedicated context-window exception |
| H6.7 | state cannot say whether a call ran | **confirmed** |
| H6.8 | effect remains; only the business layer knows | **confirmed**, plus the immediate-retry race |
| H6.9 | only run-level exceptions carry state | **confirmed** (`RunTimedOut`, `RunCancelled`) |
| H6.10 | PA tool timeout duplicates; LG `step_timeout` not retried | **confirmed**; also LG tool-level timeout + `RetryPolicy` duplicates |
| H6.11 | per-step timeouts bound nothing | **confirmed** |
| not predicted | - | drain at the final response completes the run normally; drain at "tool requested" still runs the tool |

### Ownership (evidence-based)
| Capability | Model | Provider client | Framework | Runtime | Application | Business/Data |
|---|---|---|---|---|---|---|
| Model request timeout | - | **enforces** per attempt; **retries it** | Pydantic AI `request_timeout` caps the whole request; LangGraph only passes the SDK knob | sockets, event loop | chooses values | - |
| Tool timeout | - | - | Pydantic AI turns it into a model-visible retry; LangGraph has none native | cancels coroutines; **cannot stop threads** | sets it; wraps sync tools | the effect itself |
| Whole-run timeout | - | - | Pydantic AI `run(timeout)`; LangGraph none | `asyncio.timeout` | owns it for LangGraph | - |
| Cancellation request | - | - | Pydantic AI `cancel()` / token; LangGraph `request_drain` (cooperative) | `Task.cancel` | **initiates** | - |
| Cancellation propagation | - | closes the connection only if the call is async | Pydantic AI propagates to model and tools; LangGraph does if the **graph's node is async** | delivers `CancelledError` at awaits | decides sync vs async nodes and tools | - |
| Thread termination | - | - | none | **none exists** | design sync tools to be safe if abandoned | idempotency |
| Async task cancellation | - | - | delivers it | cooperative; blocking or swallowing defeats it | tool author | - |
| Token usage tracking | produces the numbers | hides retries and failed attempts | Pydantic AI aggregates; LangGraph leaves per-message metadata | - | sums for LangGraph | - |
| Usage limit enforcement | `max_tokens` per request (provider) | - | Pydantic AI after each response; LangGraph none | - | owns for LangGraph | - |
| Partial-state retention | - | - | `RunTimedOut` / `RunCancelled` carry it; streams keep it; plain paths drop it | - | **must capture** | - |
| Side-effect correctness | - | - | none | none | key generation, retry policy | **owns** |
| Idempotency | - | - | none | - | supplies the key | **owns the atomic check** |

### Answers
1. **Does timeout mean the work stopped?** No. The HTTP client stops waiting while the provider finishes; a sync tool's thread runs on; only a cooperative coroutine stops.
2. **Can sync work continue after the caller gets a timeout?** Yes, with all 8 timeout and cancel mechanisms tried; only the LangGraph drain (a cooperative stop) waited for the tool.
3. **Does async cancellation stop the coroutine?** If it awaits and does not swallow the cancel. A swallowing tool held the caller 0.9 s; a blocking call delayed the timeout.
4. **Can cancellation happen after a business effect committed?** Yes; the effect stays and the framework records nothing about it.
5. **Who owns a whole-agent deadline?** Pydantic AI natively; the application for LangGraph.
6. **Who enforces token limits?** The provider (`max_tokens`, per request, before output), Pydantic AI (after each response), nobody in LangGraph. None sees provider-client retries.
7. **Can an agent terminate while a tool is still executing?** Yes, for sync tools.
8. **What state survives?** See the B6-3.1.56 table: nothing on LangGraph's plain path; a snapshot only in Pydantic AI's run-level exceptions; streams and `iter` keep it.
9. **Can an application safely resume after a timeout without knowing whether the previous operation completed?** Only if the operation is idempotent with an atomic guard; an immediate retry raced the original (2 orders against the non-atomic guard).
10. **What must the business/data layer guarantee?** An atomic idempotency check keyed outside the model, a way to read the outcome by key, and safety when a duplicate or late execution arrives.

### Limits (Batch 6)
- Local server and scripted completions: provider behavior (server-side cancellation, billing of abandoned requests) is modeled, not observed. That an abandoned request is billed is an inference, not a measurement.
- Timings are for this machine; assertions use event order and generous bounds.
- Measured on LangGraph 1.2.18 / Pydantic AI 2.62.0 with the Phase 1 graph (sync model node) and one async-node variant.
- Not tested: streaming responses, `FallbackModel`, durable-execution or checkpoint resume (Batches 8-9), `cost_limit` (needs pricing data), multi-process cancellation, Windows-specific thread or socket behavior beyond this machine.
- Langfuse: NOT AVAILABLE.


---

# Batch 7 - Manual agent stepping (hypotheses written 2026-11-06, before any Batch 7 run)

Batch-local IDs follow the Batch 7 brief (`B7-3.1.59` ... `B7-3.1.68`). Gap analysis, mechanism inventory and audit: [batch7-gap-analysis.md](batch7-gap-analysis.md). Scripted and local; no live run is planned. The reasoning-echo modes do not arise.

Five things are kept apart in every result: **observing** an execution, **intercepting** it (acting before a step runs), **modifying** its state, **approving** an action, **resuming** after a stop.

- **H7.1 (B7-3.1.59, LangGraph `stream`):**
  - `updates` and `values` chunks arrive after the node they describe has finished, so the consumer learns what ran, not what is running.
  - `tasks` / `debug` start events name the node, but in a synchronous stream the consumer is the thread doing the work, so they are probably delivered together with the node's result (after it ran), not before it starts. Uncertain; this is the main thing to measure.
  - A tool execution is told apart from a model request only by the node name (`tools` vs `agent`) or the message type, never by a dedicated event.
  - Stopping consumption between chunks stops everything: nothing is running between chunks.
- **H7.2 (B7-3.1.60, LangGraph `astream`):** same chunks in the same order as `stream`; the event loop stays free between chunks; closing the async iterator after the model step leaves no task running and the tools node never starts.
- **H7.3 (B7-3.1.61, Pydantic AI `iter`):**
  - `iter` yields each node **before** it runs. At `CallToolsNode` the model's proposed calls and arguments are visible and the tool has not started.
  - There is no "tool return" node: tool results first appear as parts of the next `ModelRequestNode.request`.
  - `End` carries the final result.
  - `message_history` grows when the node that produces a message has run.
  - Events from `node.stream()` (`FunctionToolCallEvent`, `FunctionToolResultEvent`) are observation only.
- **H7.4 (B7-3.1.62):** both frameworks expose the proposed call and its arguments at a boundary before the tool executes, and the next model request before it is sent; the developer's continue / stop decision is "advance or not". Observing is not authorizing: nothing makes the decision binding.
- **H7.5 (B7-3.1.63, voluntary stop):** stopping consumption at any of the four boundaries is clean in both frameworks: no tool runs, no request is sent, no task or thread is left. "A tool completed" and "before the next model request" are one boundary in each framework. The same iterator can be advanced later in-process (pause is just not advancing).
- **H7.6 (B7-3.1.64, state):**
  - Pydantic AI node fields are mutable and are the authoritative objects: editing the proposed call's arguments changes what the tool receives and what history records; editing the next request changes the wire.
  - In LangGraph the streamed chunk is a view; whether editing it takes effect is incidental and must not be relied on. The supported route is `update_state` with a checkpointer: edits go through the reducers (by message id) and the next step sees them.
  - Neither framework marks a modification in its history.
- **H7.7 (B7-3.1.65, approval):**
  - Allow and stop are expressible in both with no extra mechanism.
  - Editing arguments: Pydantic AI by mutating the node, LangGraph only through `update_state`.
  - Deny-and-continue (tell the model the call was refused and keep going): Pydantic AI by passing a replacement node to `run.next`; LangGraph only through the state API and a checkpointer.
  - Neither framework records the decision; the application's own log is the only record.
  - `CallToolsNode.tool_call_results` is the deferred-tool resume mechanism (Batch 8); whether it also affects an ordinary tool is probed once, not designed around.
- **H7.8 (B7-3.1.66, after a tool failure):** a failure is a boundary event, not an exception, for the consumer: the original call is visible before it runs, the error at the next boundary (as a tool message / a `retried` return), the model's retry call at the one after, and earlier successes stay in state throughout.
- **H7.9 (B7-3.1.67, limits):**
  - Pausing consumes no LangGraph budget (it counts steps taken).
  - A Pydantic AI run deadline (`timeout`) is wall-clock and keeps running while the consumer pauses, so a slow approver can time the run out.
  - A request-limit error surfaces when the over-limit node is advanced, after the developer has already seen it as the next node.
  - Skipping `CallToolsNode` by passing a replacement node does not count a tool call.
- **H7.10 (B7-3.1.68, comparison):**
  - Pydantic AI offers native observe / intercept / edit / skip at node granularity.
  - LangGraph natively offers observation and stopping; modification and denial need the state API (checkpointer) or hooks inside the graph.
  - Neither records approvals or provides safe resume.
  - Manual stepping gives human approval a place to stand, but not a complete workflow (Batch 8).


## Batch 7 - results (run 2026-11-04; offline, scripted; LangGraph 1.2.18, Pydantic AI 2.62.0)
Full report: [batch7-results.md](batch7-results.md). Raw: `phase-03/results/batch7.json` (git-ignored), summary `results/batch7-summary.json`.

| Hypothesis | Outcome |
|---|---|
| H7.1 LangGraph stream | `updates`/`values` chunks arrive after the node ran; `tasks` start events arrive **before** the work (tool not started at receipt). Prediction "probably with the result" was wrong. |
| H7.2 astream | identical chunks, event loop free, `aclose()` leaves nothing running. Confirmed. |
| H7.3 Pydantic AI iter | nodes yielded before they run; no tool-return node; events observation-only ordering confirmed. |
| H7.4 proposed call visible | confirmed in both; observing is not authorizing. |
| H7.5 voluntary stop | clean at every boundary in both; nothing runs afterwards. Confirmed. |
| H7.6 state | Pydantic AI node edits authoritative (args, request); history edits ignored once the request node exists. LangGraph in-place chunk edit **did** reach the tool (shared object, undocumented); `update_state` works on a closed checkpointed stream and is ignored by a suspended live stream. |
| H7.7 approval | allow/stop/modify/deny all implementable in both; Pydantic AI natively, LangGraph through checkpointer + `update_state` + `stream(None)`. Neither records the decision. `tool_call_results=ToolDenied` also denies an ordinary tool. |
| H7.8 failure | a boundary event, not an exception, in both. Confirmed. |
| H7.9 limits | LangGraph pausing is free; Pydantic AI `iter(timeout=)` keeps running while paused (`RunTimedOut`); request-limit error raised when the over-limit node is advanced; denied call not counted. Confirmed. |
| H7.10 comparison | confirmed; LangGraph modification only on a closed, checkpointed stream. |

Tests: `tests/phase_03/test_batch7_stepping.py`, 47 offline, stable over 3 runs; full suite 593 passed.
