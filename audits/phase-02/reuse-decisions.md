# Reuse Decisions (Phase 2)

Hierarchy: REUSE → ADAPT → COMPOSE → BUILD. 45 GitHub repositories audited (+ PyPI metadata for `langchain`). Health data `[V]` 2026-10-27.

| Capability | Decision | What | Why |
|---|---|---|---|
| A. Schemas | **REUSE** | Pydantic via LangChain tool layer and Pydantic AI | already generate the schemas; PoC shows the difference (`additionalProperties`) |
| B. Tool calling | **REUSE** | the two Phase 1 agents unchanged in structure | all required hooks present in installed versions |
| C. Validation | **REUSE** | framework validation; Pydantic AI `args_validator`/`strict`; LangChain `args_schema` | no extra library needed |
| D. Retries | **REUSE** | Pydantic AI `retries`/`max_retries`/`ModelRetry`; LangGraph `RetryPolicy`; tenacity only if a transient-error case needs it (already installed) | native knobs exist |
| E. Authorization | **REUSE** | Pydantic AI `requires_approval`/`FilteredToolset`/`prepare`; LangGraph `interrupt` + `wrap_tool_call` allowlist | native; no policy engine |
| F. Result handling | **REUSE** | typed returns / `ToolMessage` content; Pydantic models for result checks | truncation not found natively → measured as a finding |
| G. Safety / injection | **REFERENCE** (+ tiny simulated fixture) | AgentDojo, garak, PyRIT, promptfoo red-team for attack patterns | defensive OSS is mostly benchmarks; do not build a security framework |
| H. Idempotency | **BUILD (tiny, concept-learning)** + REFERENCE DBOS/Temporal | ~10-line idempotency-key guard inside the simulated `create_order` fixture | no framework deduplicates; production engines are Phase 6; see record below |
| I. Concurrency | **REUSE** | `UsageLimits.tool_calls_limit`, `Tool(sequential/timeout)`, `recursion_limit`, `step_timeout`, `Send` | Phase 1 harness reused |
| J. Observability | **REUSE** | OTel + OpenInference (Phase 1) with in-memory span assertions | Langfuse still blocked on credentials |
| K. Evaluation | **REUSE** pytest; **COMPOSE-candidate** Inspect AI (Exp 15 only); REFERENCE BFCL taxonomy | pytest sufficient for Exp 1-14 | avoid a second harness unless repeated-trial scoring demands it |

## Per-project verdicts
- **REUSE:** pydantic, langgraph, langchain-core, pydantic-ai, opentelemetry, openinference, tenacity (installed, conditional), pytest.
- **COMPOSE-candidates (optional, need user decision at implementation start):** `langchain` package 1.4.7 (MIT) for `tool_retry`, `tool_call_limit`, `human_in_the_loop`, `tool_selection`, `tool_emulator` middleware, as a third labeled variant "LangChain create_agent + middleware" — NOT mixed into the framework comparison; `inspect-ai` (MIT) for Exp 15.
- **REFERENCE:** jsonschema, instructor, outlines, guardrails-ai, NeMo Guardrails, OpenAI Agents SDK, Google ADK, Microsoft Agent Framework, smolagents, LlamaIndex, OPA, Cerbos, PyCasbin, AgentDojo, garak, microsoft/PyRIT, promptfoo, Invariant, DeepEval, agentevals, BFCL (gorilla), tau-bench, ToolBench, stamina, AnyIO, DBOS, Temporal, Pydantic Logfire, Arize Phoenix.
- **REJECT:** `litl/backoff` (archived), `protectai/llm-guard` (archived), `Azure/PyRIT` (archived; use microsoft/PyRIT as reference).
- **LEGAL REVIEW REQUIRED:** `meta-llama/PurpleLlama` (Llama 3.2 Community License), Restate (BSL 1.1), Phoenix (ELv2), NeMo Guardrails (confirm full LICENSE.md; header says Apache-2.0), Langfuse `ee/`.

## BUILD decision record (H: idempotency-key guard in fixture)
- Capability: prevent duplicate execution of a simulated side-effecting tool.
- Existing OSS considered: DBOS Transact (MIT, 3.3.0), Temporal Python SDK (MIT, 1.34.0), Restate (BSL). All provide durable/exactly-once workflow semantics but require a database/server and change the architecture under test; Phase 6 scope.
- Why not reuse: the experiment needs to *show* the duplicate and the guard side by side in one process; infrastructure would hide the mechanism.
- What we build: a dict-based key->result check in the test fixture.
- Excluded: persistence, concurrency safety, expiry, distributed locks, anything production-grade.

## Fixtures (not frameworks)
Simulated tools (`calculate`, `convert`, `create_order`, `safe_tool/dangerous_tool`, malicious-output tool, weather trio) are test data, not a tool framework. No ToolManager/Registry/Executor/Permission/Retry/Schema framework is permitted.
