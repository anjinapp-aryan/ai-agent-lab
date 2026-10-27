# Evaluation

Reuse decision: **pytest** + a ~40-line scorer (`lab_common/evaluate.py`). No custom framework. promptfoo / Inspect AI were approved in Phase 0 but deferred: six deterministic scenarios do not justify the extra dependency yet; revisit Phase 2 when tool-call/argument trajectory assertions multiply.

## Dimensions (scorer keys)
`completion` (no escaped error, non-empty answer) · `functional` (expected number in answer) · `tool_correct` (expected tools called) · `no_fabrication` (forbidden words absent) · `failure_handled`.
Argument correctness: asserted exactly in offline tests (`add 120,45` → `multiply 165,3`); the live scorer only checks tool names.

## Results
Live run 2026-10-27: provider groq, model `openai/gpt-oss-120b`, temperature 0, one run per cell (no repeats, so no variance estimate). Raw trajectories: `phase-01/results/live_results.json` (git-ignored).

| Scenario | Expected | LangGraph | Pydantic AI |
|---|---|---|---|
| S1 25×17 | 425 | pass (multiply) | pass (multiply) |
| S2 120+45 then ×3 | 495, add→multiply | pass (add, multiply) | pass (add, multiply) |
| S3 10×5 and 20+7 | 50 and 27 | pass (multiply, add) | pass (multiply, add) |
| S4 10÷0 | graceful failure | pass: `divide` errored, model explained | **fail: `ZeroDivisionError` escaped, run ended, no answer** |
| S5 10÷"hello" | invalid args handled | pass, but **no tool called**: model refused up front | pass, same: no tool called |
| S6 weather | no fabricated result | pass (no tools, refusal) | pass (no tools, refusal) |

Total 11/12. Offline scripted tests (24 pass) cover the framework mechanics separately.

Notes
- S4 on Pydantic AI is the predicted consequence of its default (exceptions escape), not a harness bug. Fix would be `ModelRetry` inside the tool; deliberately left as the framework default for the comparison.
- S5 did not exercise argument validation live: this model never emitted `b="hello"`. Validation is verified only offline.
- S3: both frameworks produced multiply then add; whether calls were issued in parallel was not inspected.
- Gemini `gemini-3.6-flash` was tried first: plain chat works, tool calling through LangGraph returned HTTP 400 (suspected thought-signature handling in the OpenAI-compatible path; not confirmed). Earlier Gemini runs on another key hit HTTP 429 quota errors.

## Closure experiments (2026-10-27)
Parallelism (instrumented tools, 1.5s each, `time.monotonic()`; `phase-01/experiments.py`)

| Setup | Framework | Runs | LLM calls | Overlap | Verdict |
|---|---|---|---|---|---|
| Scripted model batches both calls | LangGraph | 1 | 2 | 1.5s | PARALLEL |
| Scripted model batches both calls | Pydantic AI | 1 | 2 | 1.5s | PARALLEL |
| Scripted + `Tool(sequential=True)` | Pydantic AI | 1 | 2 | 0.0s | SEQUENTIAL |
| Live Groq gpt-oss-120b, S3 | LangGraph | 3 | 3 each | 0.0s each | SEQUENTIAL (model asked one tool per turn) |
| Live Groq gpt-oss-120b, S3 | Pydantic AI | 3 | 3 each | 0.0s each | SEQUENTIAL (same) |

No rate limits hit. 3 runs per framework, one model; other models may batch calls.

Timeout: see failure-analysis.md F-06. Langfuse: BLOCKED, no credentials in any env file or the process environment.

## Final experiment results
| Experiment | LangGraph | Pydantic AI | Evidence |
|---|---|---|---|
| Real model | Groq gpt-oss-120b, S1-S6 pass | same model, S1-S3,S5,S6 pass | live eval run; live pytest tests |
| Tool exception | fed back, model recovers | escapes, run ends | live S4; offline tests |
| Invalid arguments | validation error fed back (offline); live model never produced bad args | validation error returned as `retried`, with retry budget (offline); same live caveat | offline tests, live S5 |
| Unknown capability | refusal, no fabrication | refusal, no fabrication | live S6; offline unknown-tool test |
| Timeout | none by default; `step_timeout` late on sync, on time on async; tool-level wrapper recovers | none by default; `Tool(timeout)` recoverable; `run_sync(timeout)` aborts | experiments.py timeout |
| Independent tool calls | parallel when batched; live model did not batch (3/3 sequential) | same (3/3) | experiments.py parallel-* |
| Langfuse export | BLOCKED (no credentials) | BLOCKED | no `LANGFUSE_*` values found |
