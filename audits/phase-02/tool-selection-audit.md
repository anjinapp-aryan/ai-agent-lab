# Tool-Selection Evaluation — fresh reuse audit (Batch 5, 2026-10-28)

Question: is there a mature, reusable project for evaluating **which tool an agent selects** (and what influences it) that should replace or extend the existing pytest + `RunResult` + scorer infrastructure?
Health data `[V]` via `gh api repos/...` on 2026-10-28; project descriptions `[S]` from web search / papers unless marked.

| Project | Repo | License | Last push | What it offers for selection | Reusable here? | Decision |
|---|---|---|---|---|---|---|
| Berkeley Function Calling Leaderboard (BFCL v3/v4) | ShishirPatil/gorilla | Apache-2.0 | 2026-10-17 | categories simple / multiple / parallel / irrelevance (no call expected); AST argument checking `[S]` | its harness runs its own datasets and model handlers; our variable (name/description/order) is not a knob it exposes | **REFERENCE** (taxonomy: single = 5.1, multiple = 5.2/5.3, irrelevance = 5.10) |
| tau-bench / tau2-bench | sierra-research/tau-bench, tau2-bench | MIT | 2026-03-18 / 2026-10-28 | tool-agent-user simulation with domain policies `[S]` | full simulator with its own agent loop and user LLM; doubles quota | **REFERENCE** (policy-vs-selection idea for 5.17) |
| ToolSandbox | apple-aiml-research/ToolSandbox | NOASSERTION | 2026-09-11 | stateful tool-use evaluation `[S]` | license not machine-detected | **REJECT** (license ambiguous) |
| AgentDojo | ethz-spylab/agentdojo | MIT | 2026-06-02 | injection tasks against tool-using agents `[S]` | own agent pipeline | **REFERENCE** (5.16/5.18 attack shapes) |
| agentevals | langchain-ai/agentevals | MIT | 2026-07-14 | trajectory match evaluators (strict/unordered/subset) `[S]` | we score the first selection only; a trajectory matcher adds nothing | **REFERENCE** |
| DeepEval | confident-ai/deepeval | Apache-2.0 | 2026-10-27 | `ToolCorrectnessMetric` (expected vs called tools) `[S]` | a set comparison we already do in one line; brings a large dependency tree | **REFERENCE** |
| Ragas | vibrantlabsai/ragas (redirect from explodinggradients/ragas) | Apache-2.0 | 2026-02-24 | `ToolCallAccuracy` `[S]` | same as above | **REFERENCE** |
| promptfoo | promptfoo/promptfoo | MIT | 2026-10-28 | YAML matrices, tool-call assertions, red-team plugins `[S]` | Node toolchain | **REFERENCE** |
| Inspect AI | UKGovernmentBEIS/inspect_ai | MIT | 2026-10-28 | tasks, solvers, scorers, epochs (repeated trials) `[S]` | already **DECLINED** for Phase 2 by the user; repeated trials are a `for` loop here | **NOT ADDED** (no capability we cannot reproduce in ~20 lines) |
| EvalScope | modelscope/evalscope | Apache-2.0 | 2026-10-28 | runs BFCL v4 among many benchmarks `[S]` | benchmark runner, not a controlled-variable harness | **REFERENCE** |
| ToolBench | OpenBMB/ToolBench | Apache-2.0 | 2025-05-21 | large API tool-use dataset `[S]` | stale; retrieval-scale, not our question | **REFERENCE** (stale) |
| NESTFUL | IBM/NESTFUL | Apache-2.0 | 2025-09-18 | nested API-call sequences `[S]` | dataset only, 19 stars | **REFERENCE** |
| NexusRaven-V2 | nexusflowai/NexusRaven-V2 | none detected | 2024-02-13 | function-calling model + eval `[S]` | no license, stale | **REJECT** |
| llm-token-harness | SahilAshar/llm-token-harness | none detected | 2026-06-13 | personal tool-selection harness `[S]` | no license, 0 stars | **REJECT** |
| ToolFlood | as1-prog/ToolFlood | none detected | 2026-05-28 | retrieval-layer tool-flooding attack `[S]` | no license; retrieval-based selection is out of scope | **REFERENCE (paper only)** |
| MCPTox, ToolHijacker, MSB, MCPSecBench | papers (arXiv 2508.14925, 2504.19793, 2510.15994, 2508.13220) | — | — | tool-description poisoning / tool-selection injection `[S]` | MCP servers / retrievers; our 5.16 is a one-tool fixture | **REFERENCE** (attack phrasing for 5.16) |

## Decision
**REUSE NOT REQUIRED — existing pytest/scoring infrastructure is sufficient.**
- Selection evaluation here needs: run a case N times, record the first selected tool, map it to one of 8 labels. That is `exp_batch5.live` (a loop) + `lab2.selection.classify` (~25 lines in the style of `lab_common/evaluate.py`).
- The benchmarks above measure models on fixed datasets; this batch varies one tool-metadata variable at a time inside the two frameworks under study, which none of them exposes.
- No new dependency was added. Inspect AI and LangChain remain declined.
- Reused: Phase 1 agents and `RunResult` (+ an optional `call_id` field), Phase 1 scripted models, `lab2.wire` mock transport, Batch 3 `security.check`/`PERMS`/`AUTHZ`, existing provider clients (`ChatOpenAI`, `OpenAIChatModel`, Pydantic AI `WrapperModel` and `AbstractCapability`), `httpx` event hooks for raw capture.
- Built (fixtures only, no decision record needed beyond this note): order-domain tool factory, execution log, first-turn probe (a 6-line subclass per framework that answers the second model call locally to halve quota), classifier.
