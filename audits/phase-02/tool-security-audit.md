# Authorization (E) and Tool-Call Safety (G)

We study existing approaches only; no security framework is to be built.

## E. Authorization / approval
| Mechanism | Project | License | Evidence | Decision |
|---|---|---|---|---|
| Per-tool `requires_approval`, `ApprovalRequired`, `ApprovalRequiredToolset`, `DeferredToolRequests/Results`, `FilteredToolset`, `PreparedToolset`/`prepare` (dynamic tool visibility) | Pydantic AI | MIT | all names present in installed 2.62.0 `[V]`; semantics `[K]` | REUSE (Exp 10) |
| `interrupt()` + `Command(resume=...)` + checkpointer | LangGraph | MIT | `langgraph.types` imports `[V]`; Phase 0 HITL `[K]` | REUSE (Exp 10) |
| `HumanInTheLoopMiddleware` | langchain `agents/middleware/human_in_the_loop.py` | MIT | file exists `[F]` | COMPOSE-candidate (needs `langchain` package) |
| `ToolNode(wrap_tool_call=...)` | LangGraph | MIT | parameter present `[V]` | REUSE as allowlist interception point |
| Policy engines: Open Policy Agent | open-policy-agent/opa | Apache-2.0, v1.21.4, Go, 12.4k stars | `[V]` | REFERENCE — real policy-as-code; too heavy for a learning experiment |
| Cerbos | cerbos/cerbos | Apache-2.0, v0.57.0, Go | `[V]` | REFERENCE |
| PyCasbin | apache/casbin-pycasbin | Apache-2.0, v2.8.0 | `[V]` | REFERENCE |

Exp 10 uses only framework-native allowlist/approval; no policy engine.

## G. Injection, malicious output, unsafe tools
| Project | Repo | License | Latest / pushed | What it offers | Decision |
|---|---|---|---|---|---|
| AgentDojo | ethz-spylab/agentdojo | MIT | v0.1.35 (2025-10-27) / pushed 2026-06-02 | benchmark of prompt-injection attacks/defenses on tool-using agents `[V desc]` | REFERENCE — study attack patterns; its release is a year old, so do not depend on it |
| garak | NVIDIA/garak | Apache-2.0 | v0.17.1 (10-16) / 10-23 | LLM vulnerability scanner incl. prompt injection `[K]` | REFERENCE |
| PyRIT | **microsoft/PyRIT** | MIT | v1.1.0 (09-04) / 10-27 | red-teaming toolkit. `Azure/PyRIT` is **archived** (moved) `[V]` | REFERENCE |
| promptfoo red team | promptfoo/promptfoo | MIT | 0.125.1 / 10-27 | config-driven injection test suites `[K]` | REFERENCE (maybe COMPOSE later) |
| NeMo Guardrails | NVIDIA-NeMo/Guardrails | LICENSE.md header: Apache-2.0 | v0.24.1 / 10-27 | programmable rails incl. tool/I-O checks `[K]` | REFERENCE |
| Invariant guardrails | invariantlabs-ai/invariant | Apache-2.0 | no release / pushed 2026-01-12 | rule engine for agent traces `[V desc]` | REFERENCE — quiet 9 months |
| LLM Guard | protectai/llm-guard | MIT | **archived** `[V]` | input/output scanners | **REJECT** |
| LlamaFirewall / PurpleLlama | meta-llama/PurpleLlama | top-level LICENSE is the **Llama 3.2 Community License** (GitHub: NOASSERTION) `[V]` | no release / 10-12 | injection/agent-alignment guards | REFERENCE; **LEGAL REVIEW REQUIRED** before any use |

## Findings
- The mature OSS here is mostly *testing/benchmarking* (AgentDojo, garak, PyRIT, promptfoo); defensive controls are mostly *architectural* (least-privilege tool sets, approval gates, treating tool output as data). The framework primitives above are the reusable pieces.
- No library makes untrusted tool output safe; Experiment 12 will measure model susceptibility directly with a simulated malicious result and compare mitigations (delimiting, instruction hierarchy in the prompt, allowlisted follow-up tools) using only framework features and prompts.
- SSRF/filesystem/shell tools are **out of scope**: Phase 2 tools are simulated, in-process, side-effect-free or simulated side effects. `langchain` middleware `shell_tool.py`/`file_search.py` exist `[F]` but are not to be used.
- Credential exposure: simulated secrets only; the controlled injection test must never use real keys or files.
