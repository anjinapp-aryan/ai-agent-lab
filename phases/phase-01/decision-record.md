# Decision Records (Phase 1)

## D1 — Provider: env-driven OpenAI-compatible endpoint (REUSE)
Candidates: LiteLLM (MIT outside `enterprise/`; Phase 0 approved), per-provider SDKs, plain OpenAI-compatible clients.
User supplied Gemini, OpenRouter, Ollama Cloud, Groq keys. All expose OpenAI-compatible chat endpoints `[K]`, and both frameworks already have OpenAI-compatible clients (`ChatOpenAI`, `OpenAIChatModel`). So: `MODEL_PROVIDER/MODEL_NAME/API_KEY[/BASE_URL]` plus a 5-entry base-URL table. No new dependency, no custom abstraction beyond that table. Risk: URLs unverified live; Gemini/Groq compat quirks with tool calls possible.

## D2 — Tracing: Langfuse Cloud Hobby via OTel + OpenInference (REUSE; user decision)
Not self-hosted; no Docker. Spans exported OTLP/HTTP to `<host>/api/public/otel/v1/traces` with Basic auth `public:secret` `[K]` (verify against Langfuse docs on first run). Instrumentation: `openinference-instrumentation-langchain` 0.1.80 (Apache-2.0) for LangGraph; `openinference-instrumentation-pydantic-ai` 0.1.30 (Apache-2.0) + Pydantic AI's native `Agent.instrument_all()` for Pydantic AI. Langfuse core is MIT with separately licensed `ee/` (Phase 0); using the hosted service, no code reuse. Hobby-plan limits/retention not verified. Data (prompts, outputs) leaves the machine: don't put secrets in prompts.
**Separate decision — app deployment (Vercel/Render):** nothing in Phase 1 needs a deployed app (CLI scripts). Deployment deferred; revisit when an API surface exists.

## D3 — Evaluation: pytest + small scorer (REUSE/COMPOSE)
See evaluation.md. promptfoo/Inspect AI deferred, not rejected.

## D4 — Testing offline with scripted models (BUILD, tiny, justified)
Capability: deterministic framework-behavior tests without an API key.
Candidates: Pydantic AI `FunctionModel` (reused as is); LangChain `GenericFakeChatModel` (lacks `bind_tools`, so cannot drive a tool-calling graph unmodified).
Built: ~20-line `ScriptedChatModel` test helper subclassing `BaseChatModel`. Excluded: any production use, any generic fake-LLM library.

## D5 — Dependencies added (all permissive, from Phase 0 list)
langgraph (MIT), langchain-openai (MIT), pydantic-ai-slim[openai] (MIT), python-dotenv (BSD-3), opentelemetry-sdk + OTLP HTTP exporter (Apache-2.0), two OpenInference instrumentors (Apache-2.0), pytest/pytest-asyncio (dev). License of PyPI-reported metadata checked for the direct ones; transitive licenses not audited.
