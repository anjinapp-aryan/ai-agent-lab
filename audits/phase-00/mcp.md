# J. MCP (tool/data protocol — not an agent framework, not A2A)

## Spec status
- Latest stable spec revision: **2026-07-28** (GitHub release of modelcontextprotocol/modelcontextprotocol, published 2026-07-28 `[V]`).
- Changes reported by the changelog excerpts `[S]`: protocol-level sessions / `Mcp-Session-Id` removed from Streamable HTTP; `initialize` handshake dropped (version + capabilities carried per request in `_meta`); new `server/discover`; feature lifecycle (Active → Deprecated → Removed, 12-month window); extension model (Tasks, Skills over MCP, MCP Apps). Only first items of the changelog were visible — **read the full changelog at modelcontextprotocol.io/specification/2026-07-28/changelog before Phase 4.**
- Old tutorials (2024-11-05 / 2025 revisions with sessions + initialize) are outdated. Some servers still speak older revisions `[S]`.

## Licensing
- Spec repo LICENSE: transition MIT → Apache-2.0; new contributions Apache-2.0, docs CC-BY-4.0, un-consented legacy contributions remain MIT `[V]`. Mixed, but all permissive. Flag: **LEGAL REVIEW REQUIRED** only if we redistribute spec text; for usage, low concern.
- Python SDK `modelcontextprotocol/python-sdk`: MIT, v2.3.0 (2026-10-02), pushed 10-27 `[V]`.
- Registry `modelcontextprotocol/registry`: license reported NOASSERTION, not inspected → check before use.
- `microsoft/playwright-mcp`: Apache-2.0, v0.0.83 (09-28), pushed 10-22 `[V]`; pre-1.0.

## Decisions
| Item | Decision |
|---|---|
| Python SDK (client + server) | REUSE |
| Reference/other servers (filesystem, git, playwright-mcp) | REUSE (run, inspect) |
| Writing our own MCP server | Allowed only as a thin *tool wrapper using the SDK* (COMPOSE). Not an MCP implementation. |
| Own protocol implementation | BUILD forbidden |

Security topics to test Phase 4: authorization (OAuth), tool poisoning / prompt injection via tool descriptions, remote vs stdio trust.
