# Batch 6 gap analysis: timeouts, cancellation and token limits

Written 2026-11-05, before any Batch 6 experiment, from the existing evidence and from the installed source (LangGraph 1.2.18, Pydantic AI 2.62.0, openai SDK 3.39.0, httpx 0.28.1, Python 3.12.10).

Evidence referenced:
- **Phase 1:** `phases/phase-01/failure-analysis.md` F-06 (tool timeout).
- **Phase 2:** `phases/phase-02/timeout.md` and the B2.2 timeout matrix in `experiments.md`; 4.17 cancellation; B2.3 duplicate execution.
- **Phase 3 Batch 2:** natural termination and the empty-reply re-prompt.
- **Phase 3 Batch 3:** loop and request limits, HTTP retries on 5xx/429, "Pydantic AI token limits: not tested".
- **Phase 3 Batch 4:** reasoning echo, repeated failures.
- **Phase 3 Batch 5:** retry exhaustion, retained state, pending calls, the Phase 1 `run()` dropping state.

## Mechanisms that exist in the installed versions (read from source, `[V-installed]`)
| Layer | LangGraph | Pydantic AI |
|---|---|---|
| Provider client (HTTP) | OpenAI SDK `timeout` / `max_retries` via `ChatOpenAI(...)`; per attempt, retried by the SDK | SDK default plus `ModelSettings.timeout` (per attempt) and `ModelSettings.max_retries` |
| Whole model request | none | `ModelSettings.request_timeout`: one clock over every SDK attempt; raises `ModelRequestTimeout` |
| Tool | none native (`step_timeout` covers a whole super-step) | `Tool(timeout=)`, `Agent(tool_timeout=)`: becomes a model-visible retry |
| Super-step / node | `step_timeout` (per super-step) | n/a |
| Whole run | none (application: `asyncio.timeout` around the run) | `run(timeout=)`, `run_sync(timeout=)`: `RunTimedOut` (a `RunCancelled` and a `TimeoutError`) with `all_messages()` |
| First-party stop | `RunControl.request_drain(reason)`: cooperative, honored at a super-step boundary, raises `GraphDrained`, thread-safe | `AgentRun.cancel()`: tears down the in-flight model request and tool tasks, raises `RunCancelled` with `all_messages()`, thread-safe |
| External cancel | `asyncio.Task.cancel()` | `asyncio.Task.cancel()` (stays `CancelledError`, "external cancellation wins") |
| Token / usage | none (provider `max_tokens` per request; `AIMessage.usage_metadata` readable) | `UsageLimits`: `request_limit`, `tool_calls_limit`, `input_tokens_limit`, `output_tokens_limit`, `total_tokens_limit`, `per_request_input_tokens_limit`, `cost_limit`, optional `count_tokens_before_request` |

Documented by Pydantic AI itself: token limits are "checked after each response", cumulative limits are re-checked before the next request, and `request_limit` does not count "provider SDK or HTTP transport retries".

| Capability | Previous evidence | Remaining gap | Decision |
|---|---|---|---|
| Model request timeout | none. Batch 3 retried 5xx/429, never a timeout | which layer enforces it, whether the SDK retries it, what the framework sees, whether the server keeps working. Needs a real socket: `httpx.MockTransport` does not enforce timeouts | **NEW EXPERIMENT** (B6-3.1.48) |
| Provider HTTP timeout | none | same as above; per attempt vs whole request (`request_timeout`) | **NEW EXPERIMENT** (same) |
| Synchronous tool timeout | Phase 1 F-06; Phase 2 B2.2: the worker thread cannot be stopped, finishes at 3.0 s after a 1.0 s timeout, in both frameworks and for `Tool(timeout)`, `Agent(tool_timeout)`, `run_sync(timeout)`, `step_timeout` | thread liveness at the moment the caller sees the error, effect timing relative to that error, the final answer and state | **ALREADY COVERED** for the matrix (not rerun); **PARTIAL** for thread and effect timing, so a minimal re-measure (B6-3.1.49) |
| Asynchronous tool timeout | B2.2 (cancelled at about 1.0 s); Phase 2 4.17 (`finally` runs, 0 effects) | effect committed before the await, a tool that swallows `CancelledError`, a blocking call inside an async tool | **PARTIAL**: variants only (B6-3.1.50) |
| Whole-run timeout | B2.2: `run_sync(timeout)` and `run(timeout)` during one long tool | during a model request and between steps, state kept, LangGraph has no native one | **PARTIAL** (B6-3.1.51) |
| Cancellation | Phase 2 4.17: task cancel during two 5 s tools | the other points (before/during model request, before tool, between steps, final response); the first-party mechanisms (`AgentRun.cancel()`, `request_drain`) were never tested | **PARTIAL** (B6-3.1.52) |
| Cancellation propagation | same | which layers receive it and which keep running at each point | **PARTIAL** (same) |
| Token / usage limits | Batch 3: request limit and `tool_calls_limit` only. Explicitly "not tested: Pydantic AI token limits" | everything token-based, context-window rejection, usage after retries and failed requests | **NEW EXPERIMENT** (B6-3.1.53) |
| Partial state | Batch 3 (loop limit), Batch 5 (exhaustion; Phase 1 `run()` keeps no tool calls after a failed run) | state after timeouts, cancellation and token limits; `RunCancelled.all_messages()` / `RunTimedOut.all_messages()` | **PARTIAL** (B6-3.1.56, built from the other runs) |
| Side effects during termination | Phase 2 4.17 (sync commits after cancel), Batch 3 (no rollback), Batch 5 (pending call may have run) | commit-then-cancel versus cancel-before-commit, and whether the application can learn the outcome | **PARTIAL** (B6-3.1.55) |
| Pending tool call at a limit | Batch 3 B3-3.1.14 (LangGraph parity, Pydantic AI runs the last tool), Batch 5 E2/E3 | one table across every stop type, including token limit, timeout and cancel | **PARTIAL** (B6-3.1.54, assembled from the other runs) |
| Timeout x retry | Batch 3 (HTTP, tool, framework retries); Phase 2 B2.3 (`RetryPolicy` duplicates) | whether a timeout triggers a retry, and whether the timed-out work is still running when it does | **NEW**, finite (B6-3.1.57) |
| Cumulative deadline | none | whether per-step timeouts bound total duration | **NEW** (B6-3.1.58) |

## Reuse audit (focused; health `[V]` via `gh api` and `uv pip list`, 2026-11-05)
| Candidate | License | Maintenance | Already installed? | Decision |
|---|---|---|---|---|
| Python 3.12 `asyncio.timeout`, `asyncio.Task.cancel`, `threading`, `http.server` | PSF | stdlib | yes | **REUSE**: all the timing, cancel and local-server needs |
| agronholm/anyio | MIT | 2567 stars; pushed 2026-11-03 | yes, 4.15.1 (transitive) | **REFERENCE**: structured-concurrency alternative; not needed |
| jd/tenacity | Apache-2.0 | 8.8k stars; pushed 2026-11-01 | yes, 9.2.1 (transitive) | **REFERENCE**: retry is the frameworks' job here |
| openai/tiktoken | MIT | 19.6k stars; pushed 2026-08-17 | yes, 0.14.0 (transitive) | **REFERENCE**: the provider reports usage, so there is no need to count tokens ourselves |
| BerriAI/litellm | NOASSERTION (mixed; has an enterprise directory) | active | no | **REFERENCE**, legal review required if ever considered. It has budgets and timeouts but would replace the provider layer under test |
| pytest-dev/pytest-timeout | MIT | active | no | **REFERENCE**: only guards against a hung test; our tests use bounded waits |
| aio-libs/async-timeout | repository **archived** | — | no | **REJECT**: superseded by `asyncio.timeout` since Python 3.11 |

**Decision:** no new dependency. New code is measurement fixtures only: a local scripted HTTP server on the standard library (needed because the mock transport cannot enforce timeouts), tool probes that log their own start, effect, cancel and finish, and a driver that triggers a stop at a named event.

## Scope decisions
- Offline scripted runs are the primary evidence. Live is limited to what the provider adds (real `max_tokens` behavior, real latency). If quota is unavailable it is reported as BLOCKED, not guessed.
- Not rerun: the B2.2 timeout matrix, the Batch 3 limit arithmetic, the HTTP 5xx/429 retry accounting.
- Not built: a deadline manager, a cancellation helper, a token budgeter, or any retry layer.
- Not in scope (later batches): checkpoint/resume, manual stepping, human interruption. `GraphDrained` and `RunCancelled.all_messages()` are observed here for what they report, not used to resume.
