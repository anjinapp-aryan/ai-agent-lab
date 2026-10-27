# Phase 0 Final Decision

Date: 2026-10-27. Evidence tiers: `[V]` verified via GitHub API / LICENSE text this session, `[S]` secondary web source, `[K]` prior knowledge not re-verified.

## Executive Summary
- Every capability area has mature, permissively licensed OSS. **No BUILD candidates required.**
- Frameworks: LangGraph, Pydantic AI, OpenAI Agents SDK, Google ADK, CrewAI, Microsoft Agent Framework, Agno, Haystack all active (pushed within days) `[V]`. Release numbers are high (Pydantic AI v2.62, LangGraph 1.2.18, ADK v2.13) — APIs have moved a lot; read current docs, not old tutorials.
- MCP spec is at **2026-07-28**, which removes sessions and the initialize handshake `[V][S]`. Older tutorials are wrong for it.
- A2A is at v1.0.1 on GitHub (Linux Foundation) `[V]`; a "v1.2" claim is unconfirmed `[S]`.
- License traps found `[V]`: Restate = BSL 1.1; Arize Phoenix = Elastic License 2.0; Langfuse and LiteLLM = MIT with separately licensed `ee/`/`enterprise/` directories; MCP spec repo mid-transition MIT → Apache-2.0; AutoGen repo reported as CC-BY-4.0.
- Stale/slow: Aider (last release 2025-08, last push 2026-05), SWE-agent (last release 2025-05), Letta (release 05-14), smolagents and Ragas slowing.

## Ecosystem Map
```
Agent Frameworks : LangGraph | Pydantic AI | OpenAI Agents SDK | Google ADK | MS Agent Framework | CrewAI | smolagents | Agno | Haystack
Agent Runtime    : LangGraph checkpointers | DBOS | Temporal | (Restate: BSL)
Tools            : framework-native tools | MCP (Python SDK, spec 2026-07-28) | LiteLLM
Memory           : Mem0 | Graphiti | LangMem | Letta
Planning/Reflect : patterns inside LangGraph / Pydantic AI / ADK (no separate product)
Multi-Agent      : LangGraph supervisor | OpenAI handoffs | ADK hierarchy | CrewAI
Protocols        : MCP (agent↔tool) | A2A (agent↔agent, a2a-python)
Reliability      : LangGraph limits | DBOS | Temporal
Evaluation       : Inspect AI | promptfoo | DeepEval | pytest
Observability    : OpenTelemetry + OpenInference → Langfuse | Phoenix
Browser          : browser-use | Stagehand | Playwright MCP
Coding           : mini-swe-agent | OpenHands | Codex CLI | Claude Agent SDK
```

## Top Candidates per Capability
| Capability | Pick | Runner-up |
|---|---|---|
| Framework / state / HITL | LangGraph | Google ADK |
| Tool calling / typed output | Pydantic AI | OpenAI Agents SDK |
| Minimal loop to read | OpenAI Agents SDK, smolagents, mini-swe-agent | |
| Memory | Mem0 + Graphiti + LangMem (compare) | Letta (reference) |
| Durable execution | DBOS | Temporal |
| MCP | modelcontextprotocol/python-sdk | |
| A2A | a2a-python | a2a-samples |
| Eval | Inspect AI or promptfoo + pytest | DeepEval |
| Observability | OTel + OpenInference + Phoenix or Langfuse | |
| Browser | browser-use | Playwright MCP |
| Coding | mini-swe-agent, OpenHands | Codex CLI |

## License Summary
| Status | Projects |
|---|---|
| MIT `[V]` | LangGraph, LangChain, Pydantic AI, OpenAI Agents SDK, MS Agent Framework, CrewAI, MCP Python SDK, LangMem, Temporal, DBOS, promptfoo, Inspect AI, browser-use, Stagehand, mini-swe-agent, SWE-agent, OpenHands, Claude Agent SDK |
| Apache-2.0 `[V]` | Google ADK, smolagents, Agno, Haystack, A2A + a2a-python + samples, Mem0, Graphiti, Letta, OpenTelemetry, OpenInference, DeepEval, Ragas, Playwright MCP, Aider, Codex CLI |
| Mixed / open-core `[V]` | Langfuse (MIT + `ee/` separate), LiteLLM (MIT + `enterprise/` separate), MCP spec (MIT→Apache-2.0 transition, docs CC-BY-4.0) |
| Source-available | Restate (BSL 1.1), Arize Phoenix (ELv2) |
| Unverified | Inngest, MCP registry (NOASSERTION, LICENSE not read), AutoGen (CC-BY-4.0 label, odd for code) |

Transitive dependency licenses NOT checked. Anything marked above as flagged = `LEGAL REVIEW REQUIRED` before commercial or redistributed use; personal learning use is low risk. No legal advice given.

## Health Summary
Active (pushed ≤ 1 week, release ≤ 1 month) `[V]`: LangGraph, Pydantic AI, CrewAI, OpenAI Agents SDK, ADK, MS Agent Framework, Agno, Haystack, Mem0, Graphiti, a2a-python, MCP SDK (release 10-02), Langfuse, Phoenix, promptfoo, Inspect AI, browser-use, OpenHands, Codex, LiteLLM, DBOS, Temporal.
Slowing: smolagents (release 05-29), Letta (release 05-14), DeepEval (release 09-22), Ragas (push 02-24), mini-swe-agent (release 07-23, push active), Swarm (push 04-15, superseded).
Questionable: Aider, SWE-agent (release stale), AutoGen (superseded).
Pre-1.0 churn risk: OpenAI Agents SDK 0.24, Inspect AI 0.3, browser-use 0.13, Playwright MCP 0.0.83, Claude Agent SDK 0.2.
Stars were not used for decisions.

## Recommended Stack (minimum)
1. **LangGraph** — state, checkpoints, HITL, multi-agent.
2. **Pydantic AI** — typed tools/outputs; second framework for contrast.
3. **MCP Python SDK** (+ Playwright MCP / filesystem server as test tools).
4. **OpenTelemetry + OpenInference + Phoenix** (or Langfuse self-hosted).
5. **promptfoo or Inspect AI + pytest**.
Added by phase: Mem0/Graphiti/LangMem (3), a2a-python (4), DBOS (6), browser-use (8), mini-swe-agent/OpenHands (8). Reading-only: OpenAI Agents SDK, smolagents, ADK, CrewAI.

## Decisions
| Capability | Decision |
|---|---|
| Frameworks | REUSE LangGraph + Pydantic AI; REUSE (compare) OpenAI SDK, ADK, smolagents; REFERENCE CrewAI, MS AF, Agno, Haystack, Swarm |
| Runtime/state | REUSE LangGraph checkpointers; COMPOSE with DBOS in Phase 6; REFERENCE Temporal |
| Tools | REUSE framework tools; COMPOSE MCP |
| Memory | REUSE Mem0, Graphiti, LangMem (compare); REFERENCE Letta |
| Planning / reflection / routing | REUSE + ADAPT framework examples |
| Multi-agent | REUSE LangGraph; COMPOSE with A2A for cross-process |
| MCP / A2A | REUSE official SDKs |
| HITL | REUSE LangGraph interrupt |
| Reliability | COMPOSE LangGraph limits + DBOS |
| Evaluation | COMPOSE Inspect/promptfoo + pytest |
| Observability | COMPOSE OTel + OpenInference + Phoenix/Langfuse |
| Browser / coding | REUSE browser-use, mini-swe-agent, OpenHands |

## Rejected
- AutoGen — superseded by Microsoft Agent Framework; odd license label.
- Restate, Inngest — license not suitable/unverified; Temporal and DBOS cover need.
- Aider, SWE-agent — kept as REFERENCE ONLY, not for running as stack (stale releases).

## Build Candidates
**None required.** Possible later concept-learning exception: a ~50-line planner/executor or a budget guard — only if existing projects hide the behavior; must be documented with the five BUILD fields first.

## Risks
- Fast API churn (major versions, pre-1.0 SDKs): pin versions, re-audit each phase.
- Licensing: Phoenix ELv2, Restate BSL, open-core splits; unread NOASSERTION repos.
- MCP spec just changed (sessions removed); SDK/server support uneven `[S]`.
- A2A adoption evidence is vendor-sourced; interop untested.
- Lock-in: OpenAI/Claude SDKs, hosted LangSmith/Logfire.
- Security: tool poisoning, prompt injection (MCP, browser).
- Framework features cited `[K]` may be stale; verify by running.

## Phase 1 Recommendation
Scope: one tiny agent, built from existing parts, run and inspected — no new infrastructure.
1. Python env with `uv`; pin versions.
2. Same task (e.g. "answer a question using a calculator + file-read tool") in **LangGraph** and **Pydantic AI**; compare code, loop, message trace.
3. Add OTel/OpenInference tracing → Phoenix (local) or Langfuse; inspect spans, tokens, latency.
4. Break it: bad tool args, tool exception, step-limit hit. Record observed behavior.
5. Eval: 5–10 cases with pytest (deterministic) + one promptfoo or Inspect AI run.
6. Read-only: mini-swe-agent and OpenAI Agents SDK loops; write notes in `phases/phase-01/`.
Before starting: pick model provider, confirm API key budget, decide Phoenix (ELv2) vs Langfuse.

## Verdict: **GO WITH LIMITATIONS**
Limitations:
- Contributor counts, test coverage, issue responsiveness, transitive licenses not systematically measured.
- Many capability descriptions are `[K]`, not re-verified this session; Phase 1 runs verify them.
- MCP changelog read only partially; A2A v1.2 claim unresolved; Inngest and MCP registry licenses unread.
- Source-available/open-core projects flagged, not cleared.
No critical license gap, no stack dependency on abandoned projects, no unneeded build. Phase 1 may begin once the user confirms provider and tracing-viewer choice.
