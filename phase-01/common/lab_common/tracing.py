"""Tracing via OpenTelemetry + OpenInference -> Langfuse Cloud (OTLP/HTTP). No custom tracer.

Env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_OTEL_HOST (default EU cloud).
Without keys, setup_tracing is a no-op unless TRACE_CONSOLE=1 (prints spans to stdout).
"""
import base64
import os

from .envfile import load_env


def setup_tracing(framework: str) -> str:
    """framework: 'langgraph' | 'pydantic-ai'. Returns 'langfuse' | 'console' | 'off'."""
    env = load_env()
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    pub, sec = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    provider = TracerProvider(resource=Resource.create({"service.name": f"ai-agent-lab-{framework}", "deployment.environment.name": env}))
    mode = "off"
    if env == "prod" and not (pub and sec):
        raise RuntimeError("APP_ENV=prod requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    if pub and sec:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        host = os.environ.get("LANGFUSE_OTEL_HOST", "https://cloud.langfuse.com").rstrip("/")
        auth = base64.b64encode(f"{pub}:{sec}".encode()).decode()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"})))
        mode = "langfuse"
    elif os.environ.get("TRACE_CONSOLE") == "1":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        mode = "console"
    trace.set_tracer_provider(provider)

    if framework == "langgraph":
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=provider)
    else:
        from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
        provider.add_span_processor(OpenInferenceSpanProcessor())
        from pydantic_ai import Agent
        Agent.instrument_all()
    return mode
