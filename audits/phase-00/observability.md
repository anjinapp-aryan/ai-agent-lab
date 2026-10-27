# O. Observability

| Project | License | Latest | Decision |
|---|---|---|---|
| OpenTelemetry (python) | Apache-2.0 `[V]` | v1.45.1 (10-06) | REUSE — vendor-neutral standard |
| OpenInference (Arize) | Apache-2.0 `[V]`, pushed 10-26 | | REUSE — OTel semantic conventions + auto-instrumentation for agent frameworks `[K]` |
| Langfuse | MIT outside `ee/` directories; ClickHouse copyright; `ee/` under separate license `[V]` | v4.70.1 (10-26) | REUSE self-hosted for learning; ee features excluded; LEGAL REVIEW REQUIRED only for commercial hosting |
| Arize Phoenix | **Elastic License 2.0** `[V]` | arize-phoenix-v20.26.1 (10-25) | REUSE for local learning use only (ELv2 forbids offering as managed service); not "open source" by OSI; flag |
| Framework-native tracing (OpenAI Agents SDK, Pydantic AI + Logfire, LangGraph + LangSmith) | mixed | | REFERENCE; Logfire and LangSmith are hosted products `[K]` |

Recommend: OTel + OpenInference instrumentation → Langfuse (self-host) **or** Phoenix (local, simpler) as viewer. Choose Phoenix first for easiest local setup if ELv2 acceptable for personal learning; else Langfuse.
Build: none.
