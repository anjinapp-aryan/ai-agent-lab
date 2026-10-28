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
