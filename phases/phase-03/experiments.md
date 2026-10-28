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
