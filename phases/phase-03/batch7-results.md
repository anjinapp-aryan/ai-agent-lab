# Phase 3.1 Batch 7: manual agent stepping, results

Written for: the lab owner reviewing Batch 7 before Batch 8 (interrupt/resume). Evidence level `[V]` = observed in this batch on the installed versions, offline and scripted; no live model was used.

## 1. Status and verdict
**Batch 7 complete. Verdict: both frameworks give a consumer enough to build a manual approval gate; neither gives a complete one.** Observing, intercepting, modifying, approving and resuming are five different things, and each framework supplies a different subset natively.
- Versions: LangGraph 1.2.18, Pydantic AI 2.62.0, Python 3.12.10. Results are version-specific.
- Nothing committed. Batch 8 not started. No dependency added.

## 2. Repository inspection
Pre-flight: branch `phase-01`, HEAD `a7ce37f`, clean tree, Batches 1-6 present. Existing evidence reused (Batch 1 stream/iter shape; Batch 3 partial state; Batch 5 failure fixtures `e5.sched`; Batch 6 async drivers and cancel matrix; Phase 2 hooks). The new code is `phase-03/exp3_batch7.py`; the Phase 1 agents are unchanged. One experiment-only adaptation: `compiled_graph.checkpointer = InMemorySaver()` is set after `build_graph` so `get_state`/`update_state` work.

## 3. Gap analysis
[batch7-gap-analysis.md](batch7-gap-analysis.md). New: .59, .60, .61, .62, .64, .65, .68. Partial: .63 (voluntary stops only; Batch 6 covered forced cancels), .66 (Batch 5 fixture, boundary view), .67 (no limit arithmetic repeated).

## 4. Reuse audit
Reused the installed iteration APIs (`stream`/`astream`, `Agent.iter`/`AgentRun.next`), `InMemorySaver` (already installed; used only to exercise two calls in-process, no persistence, no interrupts), and the Batch 3-6 harness. The "approval loops" are consumer functions in the experiment file (LangGraph 36 lines; Pydantic AI 53 lines including the generic walker). No stepping engine, event bus or workflow code was built; no BUILD decision record needed.

## 5. Hypotheses
H7.1-H7.10 in [experiments.md](experiments.md), written before any run.

## 6. Results B7-3.1.59 - .68
Raw: `phase-03/results/batch7.json` (git-ignored); compact copy [results/batch7-summary.json](results/batch7-summary.json). Snapshots count what had already happened (model requests sent, tool bodies started/finished) when the consumer received a chunk or node.

**.59 LangGraph `stream` (sync).**
- `updates`, `values`, `tasks`, `debug` work; `checkpoints` and `custom` yield zero chunks without a checkpointer or stream writer; `messages` raised `ValueError: No generations found in stream` because it makes the model stream and the scripted transport is not a streaming response (a limitation of the fixture, not a finding about the mode).
- A tool run produces: `tasks` start event for `tools` (tool not started: `tool_starts=0`), `updates` chunk (tool finished, `tool_ends=1`), `tasks` result. Model steps look the same with node `agent`. Tool vs model is told only by node name (`agent`/`tools`).
- `updates`/`values` chunks arrive **after** the node finished. `tasks` start events arrive **before** the work (the opposite of what H7.1 guessed as likely).
- Closing the generator after the first update: nothing further ran (0.4 s later: same counts).

**.60 LangGraph `astream`.** Chunks identical to `stream` for `updates`, `tasks`, `values`; event loop kept ticking during the run; `aclose()` after the first update left no extra task and the tools node never started.

**.61 Pydantic AI `iter`.** Eight nodes for two tool rounds: `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `ModelRequestNode`, `CallToolsNode`, `ModelRequestNode`, `CallToolsNode`, `End` (the last `CallToolsNode` carries the final text). Each node is yielded **before it runs**: at the first `ModelRequestNode` 0 requests were sent; at `CallToolsNode` the response and its calls are visible and 0 tools have started; at the next `ModelRequestNode` the tool has finished and its `ToolReturnPart` is in the (not yet sent) request. There is no tool-return node. `node.stream()` on `CallToolsNode` gave `FunctionToolCallEvent` (tool not started) then `FunctionToolResultEvent` (tool finished). `ModelRequestNode.stream()` needs a streaming wire response (fixture limitation, not tested).

**.62 Step by step.** Both frameworks show the proposed call and arguments before execution: LangGraph in the `agent` update's `AIMessage.tool_calls` (tool not started), Pydantic AI in `CallToolsNode.model_response`. Pydantic AI shows the next request (`ModelRequestNode.request`) before it is sent (`requests=0,1,2` already sent at nodes 1-3); LangGraph shows a `task_start` for `agent` before the request but not its content. Continue/stop is "advance or not"; nothing makes it binding.

**.63 Voluntary stop.** At every boundary tested (LangGraph: after first response, at `tools` task start, tool completed, before next request, after final answer; Pydantic AI: before first request, after first response, tool completed = before next request, after final answer) nothing ran after the consumer stopped: counts identical 0.4 s later, no thread (sync) or task (async) left. In LangGraph stopping at the `tools` `task_start` means the tool never runs. "Tool completed" and "before next request" are one boundary. Pausing without stopping (0.3 s) also ran nothing: LangGraph's tool had not started at the end of the sleep; Pydantic AI's neither; both completed afterwards.

**.64 State.**
- LangGraph: editing the streamed `AIMessage` in place **did** reach the tool here (1000 instead of 120), because the chunk shares the object with the channel; undocumented, do not rely on it. The supported route works: with a checkpointer, after stopping the stream `get_state` shows `next=('tools',)` and the pending call; `update_state` with the same message id **replaces** it (2 messages, not 3), adds a checkpoint (history 3 to 4, metadata `source="update"`), and `stream(None, cfg)` runs the tool with the edited args. `update_state` while a stream is only suspended (not closed) had **no effect** on that stream (tool received 120).
- Pydantic AI: mutating `CallToolsNode.model_response.parts[0].args` changed what the tool received and what the next request recorded in the assistant message; mutating `ModelRequestNode.request.parts[0].content` changed the wire. Inserting into `run.ctx.state.message_history` did **not** change request 2 (the request was already built from the node).
- Neither marks a consumer edit in the message history; LangGraph does mark it in checkpoint metadata.

**.65 Approval** (allow / stop / modify args / deny and continue, application-side log of decisions):

| Decision | LangGraph | Pydantic AI |
|---|---|---|
| allow | continue the stream | `run.next(node)` |
| stop | close the generator | break; `run` context exits |
| modify args | stop, `update_state` (same id), `stream(None, cfg)` | mutate the node, `run.next(node)` |
| deny and continue | stop, `update_state(ToolMessage, as_node="tools")`, `stream(None, cfg)` | pass a replacement `ModelRequestNode` with a `ToolReturnPart("DENIED")` to `run.next` |

All four behaved identically in outcome (executed calls, request count, model told "DENIED by policy"). Denied calls are not counted in Pydantic AI `usage.tool_calls`. Neither framework records the decision. One probe: setting `CallToolsNode.tool_call_results = {call_id: ToolDenied("no")}` on an **ordinary** tool stopped it from running and the model received the denial text; this is the Batch 8 mechanism and works for non-deferred tools too (one probe, not designed around).

**.66 After a failure** (Batch 5 `sched` fixture, one failure then success).
- LangGraph: the failed call is a `ToolMessage(status="error")` in a normal `tools` update; no exception reached the consumer; the model's retry is the next proposal.
- Pydantic AI: the failure is a `ToolReturnPart(outcome="retried")` in the next `ModelRequestNode.request`; no exception.
- Earlier successes stay in state throughout; 3 executions total in both.

**.67 Limits.**
- LangGraph: pausing consumes no budget (`recursion_limit=6` with 0.3 s pauses completes, 3 requests). With limit 5 (= steps) the consumer receives all 15 chunks including the final answer, then `GraphRecursionError`; with 4 it is raised after the second tool update. (Same arithmetic as Batch 3; not repeated.)
- Pydantic AI: `usage_limits=UsageLimits(request_limit=2)`: the consumer **receives** the third `ModelRequestNode`, and `UsageLimitExceeded` is raised when it is advanced; no third request was sent. The run deadline (`iter(timeout=0.3)`) is wall-clock: a 0.6 s pause at the first `CallToolsNode` raised `RunTimedOut` on advancing and the tool never ran; without the pause the run continued.

**.68 Comparison** (derived matrix in the JSON): see section 9.

## 7. LangGraph findings
- One chunk stream, several modes; `tasks` start events give a pre-work boundary; `updates` give a post-work boundary. Identifying the step is by node name.
- Stopping consumption is clean and total (nothing runs between chunks).
- Native modification needs a checkpointer and the state API, and works on a **closed** stream resumed with `stream(None, cfg)`; editing under a live suspended stream is silently ignored.
- Chunk objects are shared with state: in-place edits can leak into execution. Treat as a hazard rather than a feature.
- Approval/deny/modify loop: 36 lines, all state-API based.

## 8. Pydantic AI findings
- Node granularity with the node **about to run** in hand; fields mutable and authoritative for tool args and the request.
- `run.next(node)` accepts a replacement node: this gave deny-and-continue without a checkpointer or any hook.
- `message_history` edits via `run.ctx.state` are ignored once the request node exists.
- The run deadline and usage limits stay active while the consumer pauses or stops.
- Approval loop: 53 lines including the generic driver (the larger number is the driver, not the policy).

## 9. Framework comparison
| Capability | LangGraph | Pydantic AI |
|---|---|---|
| Observe proposed call before it runs | yes (`agent` update, `tools` task_start) | yes (`CallToolsNode`) |
| See next request before sent | no (event only) | yes (`ModelRequestNode.request`) |
| Voluntary stop clean | yes | yes |
| Edit args natively | via `update_state` + checkpointer + resume | mutate node |
| Deny and continue | via `update_state(as_node)` + checkpointer + resume | replacement node |
| Framework records the decision | no (state edit only in checkpoint metadata) | no |
| Pause costs budget | no | wall-clock deadline yes |
| Needs a checkpointer | yes for any modification | no |

## 10. Expected vs actual
- H7.1: partly wrong. `updates` after, as expected; `tasks` start events arrive **before** the work, not with the result.
- H7.2: confirmed (identical chunks, free loop, clean `aclose`).
- H7.3: confirmed; events observation-only was not tested beyond ordering.
- H7.4: confirmed, including "observing is not authorizing".
- H7.5: confirmed.
- H7.6: partly wrong. Pydantic AI confirmed; LangGraph in-place chunk edit **worked** (predicted incidental; it is, but it took effect), `update_state` confirmed and, newly, ineffective under a live suspended stream. Pydantic AI history edits ignored (not predicted).
- H7.7: confirmed; `tool_call_results` also denies ordinary tools (unknown beforehand).
- H7.8: confirmed.
- H7.9: confirmed (deadline wall-clock; request-limit error on advance; denied call not counted).
- H7.10: confirmed with the nuance that LangGraph modification is possible but only on a closed, checkpointed stream.

## 11. Test results
- New: `tests/phase_03/test_batch7_stepping.py`, 47 tests, all offline; passed 3 of 3 repeated runs (47/47 each, about 11 s).
- Full suite: **593 collected, 593 passed, 0 failed, 0 skipped** (previous 546 + 47). Offline 577 passed; live 16 passed (live tests ran because the active `.env.local` has a provider and used API quota). The two known flaky tests were not touched and did not fail this run.

## 12. Limitations
- Scripted model only: no claim about what a real model does with a denial or an edit.
- Non-streaming fixture: `stream_mode="messages"` and `ModelRequestNode.stream()` were not exercised.
- Sync tools only; async tools, parallel tool batches (several calls in one response) and nested subgraphs were not stepped.
- LangGraph checkpointing only via `InMemorySaver` assigned after compile; behavior with a durable saver, `durability` modes or `interrupt_*` untested.
- Only one policy-style decision per call; "stop" was not followed by a resume (Batch 8).
- No timing assertions; pauses were 0.2-0.6 s.

## 13. Architectural ownership
- Framework: the iteration surface (chunks/nodes), the state objects, usage and deadline counters.
- Application: the policy (what needs approval), the decision, the log of who approved what and when, the identity of the approver, the denial text the model sees, binding of an approval to exact arguments, and what happens to a pending approval when the process ends. None of these is provided by either framework.

## 14. Lessons
1. Observing is free; intercepting needs a boundary **before** the work; both frameworks have one (`task_start`, `CallToolsNode`).
2. A chunk or node object is live state: mutation may take effect, which is power and hazard.
3. Pausing is not free in Pydantic AI (deadline) and is free in LangGraph; a slow approver needs a deadline decision.
4. A framework can resume from a stop only when the state is in the right place: LangGraph in a checkpoint, Pydantic AI in the live `AgentRun` object (in-process only).
5. Approval is not recorded by anyone but the application.

## 15. Remaining unknowns
- Whether `tasks` start events remain before the work for async tools, parallel tool batches and with `durability="exit"`.
- Real-model behavior after a denial text or an edited argument.
- Durable resume of either approach after a process restart (Batch 8).
- `ModelRequestNode.stream()` / `messages` mode behavior with a streaming transport.

## 16. Recommendation for Batch 8
Proceed to interrupt/resume, using what this batch isolated: LangGraph `interrupt_before` with `InMemorySaver` versus Pydantic AI deferred tools (`DeferredToolRequests` / `tool_call_results` / `ToolApproved` / `ToolDenied`). Questions to answer there: can a pending approval survive a process boundary, is the approved argument bound to what was shown, what happens to a decision made twice, and what is lost when the run object (Pydantic AI) or the stream (LangGraph) is gone. Stepping alone is not sufficient for human approval: it provides the place to stand but no record, identity, persistence or binding.

## Answers to the ten architecture questions
1. **Better control:** Pydantic AI at node granularity without extra machinery (mutate, replace, skip); LangGraph gives equal reach only through the checkpointer and state API.
2. **Inspect every request and tool call:** yes in both: LangGraph via `tasks`/`updates`, Pydantic AI via nodes; request content is visible before sending only in Pydantic AI.
3. **Pause before a tool:** yes in both; not advancing is the pause (Pydantic AI deadline keeps running).
4. **Modify args:** yes; Pydantic AI by node mutation, LangGraph by `update_state` on a closed stream (in-place chunk mutation also happened to work).
5. **Inspect/modify state:** Pydantic AI `run.ctx.state`/node fields (history edits ignored late); LangGraph `get_state`/`update_state`/history, checkpointer required.
6. **Does stopping the stream stop the agent:** yes for both, cleanly, because nothing runs between steps (Batch 6's forced cancellation of a running step is different).
7. **Safe to continue after interruption:** in-process yes (same iterator; LangGraph `stream(None, cfg)` on a checkpointed thread); durability across a restart untested.
8. **Lost with `run()`/`invoke()`:** every boundary: no look at proposed calls, no veto, no edit, and (LangGraph, no checkpointer) no state access.
9. **Is stepping enough for human approval:** no. It gives the hook; the record, identity, binding, persistence and timeout policy belong to the application (or to Batch 8 mechanisms).
10. **Framework vs application ownership:** see section 13.
