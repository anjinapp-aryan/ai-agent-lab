# Tool Observability (capability J)

Phase 1 already verified (offline, in-memory) that OpenTelemetry + OpenInference give a parent/child span per tool in both frameworks, with `ERROR` status and an `exception` event on failure/timeout. Langfuse Cloud delivery remains **unverified (no credentials)**.

| Project | Repo | License | Latest | Pushed | Role | Decision |
|---|---|---|---|---|---|---|
| OpenTelemetry Python | open-telemetry/opentelemetry-python | Apache-2.0 | v1.45.1 | 10-27 | tracing API/SDK | REUSE (done) |
| OpenInference | Arize-ai/openinference | Apache-2.0 | (JS pkg tags current; Python instrumentors 0.1.x installed) | 10-27 | semantic conventions, LangChain/Pydantic AI instrumentors | REUSE (done) |
| Langfuse | langfuse/langfuse | MIT outside `ee/` (Phase 0) | v4.71.0 (10-27) | 10-27 | cloud viewer | REUSE; blocked on credentials |
| Arize Phoenix | Arize-ai/phoenix | Elastic License 2.0 (Phase 0) | arize-phoenix-v20.26.1 | 10-27 | local viewer | REFERENCE; ELv2 flagged |
| Pydantic Logfire | pydantic/logfire | MIT SDK | v5.4.0 (10-26) | 10-27 | OTel-based; the hosted backend is a commercial service `[K]` | REFERENCE — avoid backend lock-in; the SDK is not needed since Pydantic AI emits OTel directly `[V Phase 1]` |

## Phase 2 relevance
Experiments should read spans (in-memory exporter, as in Phase 1 `trace_demo.py`) to assert: tool span count per run, retry spans after validation failure, tool span duration vs timeout, approval pauses, and token/tool-call overhead. Token usage attributes on LLM spans: availability `[K]`, verify in Exp 6.

Build: none. No new dependency.
