# Tool-Call Evaluation (capability K)

Needed metrics: tool-selection accuracy, argument accuracy, unnecessary calls, missing calls, invalid calls, execution correctness.

| Project | Repo | License | Latest | Pushed | What it provides | Decision |
|---|---|---|---|---|---|---|
| pytest (current) | — | MIT | 9.1.1 | — | deterministic assertions on `RunResult.tool_calls` `[V Phase 1]` | REUSE — default for Exp 1-14 |
| Inspect AI | UKGovernmentBEIS/inspect_ai | MIT | 0.3.285 (10-25) | 10-27 | task/solver/scorer framework, tool-calling agents, sandboxes `[K]`; pre-1.0 | COMPOSE-candidate for Exp 15 (selection) if repeated trials + scorers are wanted; adds a dependency |
| promptfoo | promptfoo/promptfoo | MIT | 0.125.1 (10-25) | 10-27 | YAML test matrices, tool-call assertions `[K]`; Node/TypeScript tool | REFERENCE (avoid adding a Node toolchain) |
| DeepEval | confident-ai/deepeval | Apache-2.0 | python-v4.2.4 (09-22) | 10-27 | tool-correctness metric `[K]`; platform upsell | REFERENCE |
| agentevals | langchain-ai/agentevals | MIT | js tag only (03-03) | 2026-07-14 | trajectory match evaluators (strict/unordered/subset) `[V desc: "Readymade evaluators for agent trajectories"]` | REFERENCE; revisit if trajectory-subset matching is needed (check Python packaging first) |
| Berkeley Function Calling Leaderboard | ShishirPatil/gorilla (`berkeley-function-call-leaderboard/`) | Apache-2.0 | v1.3 (2025-07-17) / pushed 10-17 | benchmark: AST-based argument checking, parallel/multiple/irrelevance categories `[K]` | REFERENCE — study the categories (parallel, multiple, irrelevance detection) to shape Exp 13/15; do not run the full benchmark |
| tau-bench | sierra-research/tau-bench | MIT | no release / 2026-03-18 | tool-agent-user policy benchmark `[K]` | REFERENCE |
| ToolBench | OpenBMB/ToolBench | Apache-2.0 | no release / 2025-05-21 | large API-tool benchmark | REFERENCE (stale) |
| AgentDojo | ethz-spylab/agentdojo | MIT | v0.1.35 (2025-10-27) | injection robustness | REFERENCE (Exp 12 design input) |

## Findings
- BFCL's test categories (simple, multiple, parallel, parallel-multiple, irrelevance) map directly onto Exp 1, 15, 13 and the Phase 1 "unknown capability" case, so reuse its *taxonomy*, not its harness.
- Stat reliability: live selection/injection experiments need repeated trials (small N, e.g. 5) since temperature-0 is still not guaranteed deterministic on hosted models. Report counts, not percentages implied as significant.
- Same-model rule: all live trials on Groq `openai/gpt-oss-120b`.

Build: none; scorer extensions stay in the existing ~40-line `evaluate.py` style.
