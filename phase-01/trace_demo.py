"""Offline trace-structure demo: scripted model + real instrumentation + in-memory span capture.

Shows span trees for LangGraph vs Pydantic AI without needing an API key or Langfuse.
Usage: uv run python phase-01/trace_demo.py langgraph|pydantic-ai [seq|zero|badargs]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("common", "langgraph-agent", "pydantic-ai-agent", os.path.join("..", "tests", "phase_01")):
    sys.path.insert(0, os.path.join(HERE, p))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

SCRIPTS = {
    "seq": [("call", [("add", {"a": 120, "b": 45})]), ("call", [("multiply", {"a": 165, "b": 3})]), ("text", "495")],
    "zero": [("call", [("divide", {"a": 10, "b": 0})]), ("text", "Cannot divide by zero.")],
    "badargs": [("call", [("divide", {"a": 10, "b": "hello"})]), ("text", "Cannot divide by text.")],
}


def main():
    fw = sys.argv[1]
    script = SCRIPTS[sys.argv[2] if len(sys.argv) > 2 else "seq"]
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from lab_common.tracing import setup_tracing
    mode = setup_tracing(fw)
    mem = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(mem))
    import conftest
    if fw == "langgraph":
        import lg_agent
        r = lg_agent.run("demo", conftest.lg_model(script))
    else:
        import pa_agent
        r = pa_agent.run("demo", conftest.pa_model(script))
    print(f"exporter={mode}  error={r.error}  answer={r.final_text!r}")
    spans = sorted(mem.get_finished_spans(), key=lambda s: s.start_time)
    children = {}
    for s in spans:
        children.setdefault(s.parent.span_id if s.parent else None, []).append(s)

    def show(parent_id, depth):
        for s in children.get(parent_id, []):
            a = s.attributes or {}
            kind = a.get("openinference.span.kind", "-")
            extra = a.get("tool.name") or a.get("gen_ai.tool.name") or ""
            print(f"{'  ' * depth}{s.name}  [{kind}] {extra} {s.status.status_code.name}")
            show(s.context.span_id, depth + 1)

    show(None, 0)
    print("traces:", len({s.context.trace_id for s in spans}), "spans:", len(spans))


if __name__ == "__main__":
    main()
