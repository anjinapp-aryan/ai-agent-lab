# Phase 2 Experiments

Model for any live run: Groq `openai/gpt-oss-120b` (same for both frameworks). Scripted tier first. Hypotheses below were written **before** the experiments were run; results are recorded beside them and any mismatch is called out, never edited away.

## Tier classification (all 15)
| # | Experiment | Tier | Plan |
|---|---|---|---|
| 1 | Simple typed tool | A | scripted |
| 2 | Missing argument | A (C follow-up: does a real model recover from the feedback text?) | scripted first |
| 3 | Wrong type | A (C follow-up as 2) | scripted first |
| 4 | Extra argument | A | scripted |
| 5 | Invalid enum | A (C follow-up as 2) | scripted first |
| 6 | Tool exception | C | scripted, then live if interesting |
| 7 | Timeout | A | scripted (Phase 1 baseline reused) |
| 8 | Duplicate call | A (+B: does a real model duplicate?) | scripted first |
| 9 | Side-effecting create_order | A | scripted |
| 10 | Authorization | A (+B: model reaction to denial) | scripted first |
| 11 | Result validation | A | scripted |
| 12 | Tool-output injection | B | live (simulated) |
| 13 | Parallel and dependent calls | C | scripted, then live |
| 14 | Tool-call limits | A | scripted |
| 15 | Tool selection | B | live, 5 trials per framework |

Note on the "C follow-up" for 2/3/5: a real model almost never emits an invalid call on demand (Phase 1 S5), so the live question is only "given the framework's error text, does the model self-correct?" That needs a replay-then-live hybrid model. It is deferred and will be run only if the scripted results make it informative.

## Batch 1 — pre-registered hypotheses
Tools: `calculate(a: int, b: int) -> int`, `convert(amount: float, currency: Literal["USD","EUR","INR"])`.

**E1 Simple typed tool.** H: both build JSON Schema from type hints and use the docstring as description; Pydantic AI marks the schema `additionalProperties: false`, LangChain does not; args arrive as ints; result handed back to the model: LangGraph stringifies (`"5"`), Pydantic AI keeps a typed value and serializes later.

**E2 Missing argument.** H: both detect before calling the tool (tool never runs) and feed an error to the model. One bad call then a good call: both recover. Persistent bad calls: Pydantic AI stops after its retry budget (`UnexpectedModelBehavior`); LangGraph has no budget and stops only at `recursion_limit` (`GraphRecursionError`).

**E3 Wrong type.** H: Pydantic lax coercion in both: `"10"`→10 and `10.0`→10 accepted; `"hello"`, `10.5`, `None`, `[1]` rejected; `True` accepted as 1. Identical in both frameworks (same validator).

**E4 Extra argument.** H (from audit PoC): LangGraph executes and drops the extra; Pydantic AI rejects. LangChain `args_schema` with `extra="forbid"` makes LangGraph reject. A function with `**kwargs`: extras may be passed through (unknown, to be observed).

**E5 Invalid enum.** H: `"XYZ"` and lowercase `"usd"` rejected by both; the error lists allowed values; a Python `Enum` class behaves like `Literal`.

**E11 Result validation.** H: neither framework validates a return value against its annotation (a function annotated `-> int` that returns `"abc"` is accepted); `None` and dicts are serialized; non-JSON-serializable objects are stringified or raise; a 1,000,000-character result is passed to the model whole with no truncation.

**E14 Tool-call limits.** H: Pydantic AI `UsageLimits.tool_calls_limit` and `request_limit` raise `UsageLimitExceeded`; LangGraph has no tool-call-count limit, only `recursion_limit` (`GraphRecursionError`). With a model that batches 3 calls and a limit of 2, Pydantic AI refuses before executing any or executes only 2 (unknown which). Partial results are lost on error unless messages are captured separately (Pydantic AI `capture_run_messages`).

## Batch 1 results (2026-10-27)
Common setup: LangGraph 1.2.18 and Pydantic AI 2.62.0 via the Phase 1 agents (`lg_agent.run`, `pa_agent.run`); model = Phase 1 **scripted model** (no live model, so no quota used); the wire-level rows use the real `ChatOpenAI` / `OpenAIChatModel` clients against a mock HTTP transport (`lab2/wire.py`) and record the exact request JSON. Tracing: not captured in Batch 1 (the evidence is tool execution logs, framework error text and wire payloads). Timing is negligible for all but E14 (milliseconds). Reproduce: `uv run python phase-02/exp_batch1.py [exp...]`; pinned as `tests/phase_02/test_batch1_tool_mechanics.py` (57 tests).

**Where Expected ≠ Actual** (hypothesis was wrong or incomplete):
1. E1: I expected Pydantic AI to keep a typed tool result while LangGraph stringifies. On the wire both send the string `"5"`; any difference is internal.
2. E1: not predicted — Pydantic AI sends `"strict": true` on the tool definition (the audit had marked strict mode as unknown); LangChain sends no `strict`.
3. E4: for a `**kwargs` function, LangGraph drops the extra and the function receives `{}`; Pydantic AI **passes it through** (`{'secret': 'unexpected'}`). This was listed as unknown.
4. E11: `None` and `NaN` serialize differently (`"null"`/`"NaN"` vs `""`/`"null"`), and an unserializable object **aborts** the Pydantic AI run before the second model request. I had only predicted "stringified or raises".
5. E14: unpredicted exactly how Pydantic AI's limit behaves: the limit is checked before executing, and a batch that would exceed it executes **none** of its calls.

### E1 Simple typed tool — hypothesis mostly confirmed
- Input: `calculate(a:int,b:int)`, call `{a:2,b:3}`. Expected: schema from hints, docstring as description. Actual: both send `{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}`; Pydantic AI adds `"additionalProperties": false` and `"strict": true`. Args arrive as `int`; result `5` reaches the model as `"5"` in both.
- Interpretation: the schema is created by the framework, not the model or the provider; the two frameworks make different choices about closing it and about asking the provider for strict decoding.

### E2 Missing argument — confirmed
- Expected: detected before execution, fed back. Actual: tool body ran 0 times on the bad call; one bad call then a good call recovers in both (3 model calls). Persistent bad calls: LangGraph made 9 model calls and stopped on `GraphRecursionError` (no retry budget); Pydantic AI stopped after 2 model calls (`retries=1`) and 4 (`retries=3`) with `UnexpectedModelBehavior`. Executions: 0.
- Interpretation: detection is the framework wrapper's job in both. Retry ownership differs: Pydantic AI owns a per-tool budget; LangGraph owns only a global step limit.

### E3 Wrong type — confirmed (+1)
- Accepted (coerced to `int`): `"10"`, `10.0`, `True` (→ `1`). Rejected: `"hello"`, `10.5`, `None`, `[1]`, and also `"1e2"` (added). Identical in both frameworks.
- Interpretation: the validator is Pydantic lax mode in both; "wrong type" is only wrong if it cannot be coerced. `True` becoming `1` is a notable surprise to design around.

### E4 Extra argument — confirmed, plus new finding
- LangGraph default: **executed, extra silently dropped**. Pydantic AI default: **rejected** (`extra_forbidden`, error fed back). LangGraph with a Pydantic `args_schema` using `extra="forbid"`: rejected. `**kwargs` function: LangGraph drops, Pydantic AI passes through.
- Interpretation: "extras are ignored" is a LangGraph default, not a Pydantic AI one. A tool that takes `**kwargs` hands the model a channel to inject arbitrary parameters under Pydantic AI.

### E5 Invalid enum — confirmed (+1)
- `"XYZ"`, `"usd"` and `"EUR "` (trailing space) are rejected for both `Literal` and `Enum`; `"USD"` accepted. No case or whitespace normalization. Error lists the allowed values.

### E11 Result validation — confirmed with differences
- A function annotated `-> int` returning `"abc"` is accepted by both (no return validation). Huge result: 1,000,000 characters sent whole by both (no truncation). `None`: LangGraph `"null"`, Pydantic AI `""`. `NaN`: `"NaN"` vs `"null"`. dict: both JSON (spacing differs). Non-serializable object: LangGraph sends `repr()`; Pydantic AI raises `PydanticSerializationError` and ends the run (1 model request instead of 2).
- Interpretation: nobody validates tool output; serialization is the only gate, and the two frameworks have different failure semantics. Truncation and size limits are the tool author's responsibility.

### E14 Tool-call limits — partly predicted
- Pydantic AI `tool_calls_limit=3` on a never-ending model: 3 executions, then `UsageLimitExceeded` ("next tool call(s) would exceed") — the check is **before** executing. `request_limit=4`: 4 executions. Batch of 3 calls with limit 2: **0 executions**. 9 messages remained available via `capture_run_messages`.
- LangGraph: only `recursion_limit` (error `GraphRecursionError`): limit 5→2 executions, 9→4, 17→8; no tool-call-count limit exists (batch of 3 ran all 3); the run result carries no partial state. A checkpointer was not tested.
- Interpretation: stopping runaway tool use is the framework's job but with different units: calls vs requests (Pydantic AI) or graph supersteps (LangGraph). Time limits remain as in Phase 1.

### Not done in Batch 1 (honest scope)
Live-model follow-ups for E2/E3/E5 (does a real model self-correct from the error text?) were deliberately not run: they need a replay-then-live hybrid model and the scripted results were fully deterministic. 0 live trials were used for Batch 1.
