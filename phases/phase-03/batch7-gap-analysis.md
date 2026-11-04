# Batch 7 gap analysis: manual agent stepping

Written 2026-11-06, before any Batch 7 experiment, from the existing evidence and from the installed sources (LangGraph 1.2.18, Pydantic AI 2.62.0, Python 3.12.10).

Evidence reviewed:
- **Batch 1:** the loop was observed through `graph.stream(stream_mode=["debug","values"])` and `agent.iter()`. It established that these are the views `invoke` / `run` use internally, that LangGraph yields one chunk per super-step (2r+1) and Pydantic AI one per node (2r+4), and that the model is stateless.
- **Batch 2:** the state returned at natural termination, in both shapes.
- **Batch 3:** limit errors raised at boundaries; partial state read from a stream or an `iter` run object.
- **Batch 5:** retained state after failures, pending-call ambiguity, the `lab3/history.py` view.
- **Batch 6:** async drivers (`astream`, `iter`), cancellation at seven points, `RunControl.request_drain`, `AgentRun.cancel()`, state after each termination.
- **Phase 2 Batch 3:** interception hooks (`wrap_tool_call`, `Hooks(before_tool_execute)`) and authorization at them.

## Stepping mechanisms in the installed versions (read from source, `[V-installed]`)
| Need | LangGraph | Pydantic AI |
|---|---|---|
| Iterate execution | `stream` / `astream`; modes `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages`, `custom` | `agent.iter()`: an `AgentRun` that yields **nodes**: `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `End` |
| Unit of a step | a super-step (one node run) | a node |
| What the developer holds | a chunk (state, update or event) | the **node about to run** (its fields are visible and, per the docs, mutable) |
| Drive manually | advance the generator | `await run.next(node)` accepts the node to run, so a node can be edited or replaced |
| Events inside a step | `tasks` / `debug` / `messages` / `custom` modes | `node.stream(ctx)` yields `FunctionToolCallEvent` / `FunctionToolResultEvent` and part events |
| State API | `get_state()`, `update_state()`, `get_state_history()`, **all require a checkpointer** | `run.ctx.state.message_history`, `run.usage`, `run.all_messages()` |
| Native pause | `interrupt_before` / `interrupt_after` (checkpointer; Batch 8) | deferred tools / `tool_call_results` on `CallToolsNode` (Batch 8) |
| Interception hooks (not stepping) | `ToolNode(wrap_tool_call / awrap_tool_call)` | `Hooks(before_tool_execute)` |

| Experiment | Previously tested? | Existing evidence | Remaining gap | Decision |
|---|---|---|---|---|
| B7-3.1.59 LangGraph sync stepping | partly | Batch 1: `debug` + `values` only | the other modes; whether a chunk arrives before or after the work it describes; identifying the executing node; tool vs model; what happens when consumption stops | **NEW** (narrow) |
| B7-3.1.60 LangGraph async stepping | partly | Batch 6 used `astream` for cancellation | ordering parity with sync, event-loop behavior, cleanup when the iterator is closed | **NEW** (narrow; the Batch 6 cancel matrix is not repeated) |
| B7-3.1.61 Pydantic AI manual iteration | partly | Batch 1: node types and counters | what each node exposes, when state changes, the event stream inside a node | **NEW** |
| B7-3.1.62 Step-by-step tool execution | partly | Batches 1, 5: results visible after the fact | proposed call and arguments visible before execution; next request visible before it is sent; a continue / stop decision at each boundary | **NEW** |
| B7-3.1.63 Stop between steps | partly | Batch 6: cancel and drain at points A-G (a stop forced from outside) | a **voluntary** stop (stop consuming) at the boundaries; what runs on afterwards; pausing and continuing in-process | **PARTIAL**: voluntary stops only; Batch 6's mid-step matrix is referenced |
| B7-3.1.64 Inspect and modify state | no | Batch 5 read state, never changed it | which objects are mutable, authoritative, and reflected in the next request | **NEW** |
| B7-3.1.65 Approval before a tool | no | Phase 2 B3: approval-like checks at hooks (interception, not stepping) | whether a stepping loop can allow, deny, edit and record, and which of those the framework itself supports | **NEW** (not the full interrupt/resume design; that is Batch 8) |
| B7-3.1.66 Stepping after tool failure | partly | Batch 5: failure fixtures, final retained history, accounting | what is visible at each boundary (call, error, retry, result, earlier successes) | **PARTIAL**: reuses the Batch 5 fixtures; boundary view only |
| B7-3.1.67 Stepping and limits | partly | Batch 3 arithmetic; Batch 6 run deadline | whether pausing consumes budget, whether a deadline keeps running while paused, whether manual edits or skips change the counters | **PARTIAL**: no limit arithmetic repeated |
| B7-3.1.68 Framework comparison | no | - | a synthesis of the above, with implementation size of the equivalent consumer loop | **NEW** (assembled from the measurements) |

## Reuse audit
- **Reuse:** the installed LangGraph and Pydantic AI iteration APIs, the Batch 3-6 harness (`e3`, `e4`, `e6` probes, scripted transports), the recorder, `lab3/history.py`.
- **Native-API exception:** `get_state` / `update_state` exist only with a checkpointer. `langgraph.checkpoint.memory.InMemorySaver` (installed, MIT, part of the existing stack) is used **only to exercise those two calls in-process**. No persistence, durable resume or interrupt flow is built or evaluated; that is Batch 8.
- **No new dependency.** No stepping engine, event bus, workflow engine or runtime is built: the "stepping loops" are small consumer functions in the experiment file, and their size is reported as a measurement.

## Scope decisions
- Offline and deterministic first; every experiment uses scripted completions through the existing mock transport or the Batch 6 local server. Live validation is not planned: the questions concern what the frameworks expose, not what a model does.
- Observation, interception, modification, approval and resumption are kept as separate columns in every result.
- Not in scope: `interrupt_before` / `interrupt_after` flows, `DeferredToolRequests` resume, checkpoint persistence (Batch 8), and the side-effects-with-resume analysis (Batch 9).
