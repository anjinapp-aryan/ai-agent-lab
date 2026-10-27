# PHASE 2 DESIGN GATE

Date: 2026-10-27. Phase 1 untouched (4f8e25d). Nothing in Phase 2 is implemented except the audit PoC.

| Item | Result |
|---|---|
| Reuse audit | **PASS WITH LIMITATIONS** |
| Existing OSS sufficient | **YES for all mechanisms; tiny test fixtures and one tiny idempotency guard required** |
| Primary frameworks | LangGraph 1.2.18, Pydantic AI 2.62.0 (unchanged, compared on equal terms) |
| Reusable components | Pydantic validation, LangChain tool layer, `ToolNode`/`wrap_tool_call`/`RetryPolicy`/`interrupt`/`Send`, Pydantic AI `Tool(max_retries, args_validator, strict, sequential, requires_approval, timeout)`/toolsets/`UsageLimits.tool_calls_limit`, OTel + OpenInference, pytest, tenacity (installed) |
| Adaptations required | extend Phase 1 `experiments.py` timestamp harness; extend scorer for tool-selection categories; set `extra="forbid"` on LangChain `args_schema` for Exp 4 |
| New code required | simulated tool fixtures; ~10-line idempotency-key guard (BUILD record in reuse-decisions.md); experiment drivers |
| Must NOT be rebuilt | schema generation, validation, retry/backoff, approval flow, timeout/limit machinery, tracing, evaluation framework, policy engine, security scanner, durable execution |
| Phase 2 experiment count | 15 (two tiers: scripted mechanics, live N=5) |
| Dependencies | **None required.** Optional, MIT, need explicit go-ahead: `langchain` 1.4.7 (middleware variant), `inspect-ai` 0.3.285 (Exp 15) |
| License concerns | PurpleLlama = Llama 3.2 Community License (LEGAL REVIEW REQUIRED, reference only); Phoenix ELv2; Restate BSL; NeMo Guardrails LICENSE.md needs full read (header Apache-2.0); Langfuse `ee/`; archived: backoff, llm-guard, Azure/PyRIT |
| Known limitations | many capability semantics are `[K]`/`[F]` (middleware behavior, provider strict-mode, BFCL/garak contents) until experiments run; contributors/test-coverage/issue-health not measured; live quota (Groq) unverified; Langfuse still blocked; one audited fact needs rerun at implementation time because versions move daily |

## Key audit findings
1. **Extra arguments differ:** LangGraph executes and silently drops them; Pydantic AI rejects (`extra_forbidden`) and sends `additionalProperties:false` (PoC `[V]`).
2. Missing/wrong-type/enum errors are fed to the model by both; both coerce lax values (`"10"`→`10.0`).
3. Pydantic AI has first-class approval, allowlist (`FilteredToolset`), `tool_calls_limit`, `sequential`, per-tool timeout; LangGraph provides `interrupt`, `wrap_tool_call`, `RetryPolicy`, `step_timeout`. The `langchain` package adds tool-level middleware equivalents (files present, behavior unverified).
4. Mature OSS on injection/authorization is mostly test/benchmark tooling or heavy policy engines; defenses are architectural. Several once-popular tools are archived or moved.
5. No framework deduplicates side effects; idempotency belongs to the tool/backend (Experiments 8-9).

## Recommendation: **GO WITH LIMITATIONS**
Proceed to implement Phase 2 experiments after review, starting with Tier A offline groups. Decisions needed from the user at the start: (1) approve or decline the optional `langchain` and `inspect-ai` dependencies, (2) confirm Groq `openai/gpt-oss-120b` for all live trials and N=5, (3) confirm Phase 2 stays free of Langfuse until credentials exist.

## Decisions recorded after review (2026-10-27)
- `langchain`: DECLINED for Phase 2 (revisit only if an experiment shows a capability the current stack cannot study).
- `inspect-ai`: DECLINED for the initial implementation; pytest and the existing scorer stay. A concrete evaluation gap triggers a focused re-audit.
- Model: Groq `openai/gpt-oss-120b` approved for all live comparisons.
- Langfuse: Phase 2 proceeds without it (still BLOCKED on credentials); OTel + OpenInference in-memory tracing continues.
- Live policy: scripted tier first; live trials (up to 5 per framework) only where model behavior matters; actual trial counts are documented; no extra providers or retry infrastructure on rate limits.
- Idempotency-key guard inside simulated `create_order` approved.
