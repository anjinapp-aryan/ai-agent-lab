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
