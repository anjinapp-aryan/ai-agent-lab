# N. Agent Evaluation

| Project | License | Latest | Pushed | Decision |
|---|---|---|---|---|
| Inspect AI (UK AISI) | MIT | 0.3.285 (10-25) | 10-27 | REUSE — agent evals, sandboxes, tool-call scoring `[K]`; pre-1.0, fast-moving |
| promptfoo | MIT | 0.125.1 (10-25) | 10-26 | REUSE — regression / red-team via YAML configs |
| DeepEval | Apache-2.0 | python-v4.2.4 (09-22) | 10-23 | REUSE optional — pytest-style metrics, LLM-as-judge; platform upsell `[K]` |
| Ragas | Apache-2.0 | — | 2026-02-24 | REFERENCE ONLY — RAG-focused, org moved (vibrantlabsai), quiet since Feb |
| Framework trajectory evals (ADK eval, LangSmith) | mixed | | | REFERENCE (LangSmith is hosted/proprietary `[K]`) |

Recommend: **promptfoo or Inspect AI** (pick one for Phase 1; Inspect for agent trajectories, promptfoo for quick regression) + plain pytest for deterministic checks.
Build: none.
