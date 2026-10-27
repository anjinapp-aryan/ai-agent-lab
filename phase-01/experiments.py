"""Phase 1 closure experiments: parallel vs sequential tool execution, and tool timeouts.

Not part of the agent. Instrumented tools record time.monotonic() start/end so conclusions
come from measured execution, not from the model's tool-call list.

Usage: uv run python phase-01/experiments.py parallel-offline | parallel-live [N] | timeout
Results append to phase-01/results/experiments.json (git-ignored).
"""
import asyncio
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "langgraph-agent", "pydantic-ai-agent", os.path.join("..", "tests", "phase_01")):
    sys.path.insert(0, os.path.join(HERE, sub))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

DELAY = 1.5
LOG: list[dict] = []   # {tool, start, end, thread}
_lock = threading.Lock()


def _timed(name, fn, delay=DELAY):
    def tool(a: float, b: float) -> float:
        t0 = time.monotonic()
        time.sleep(delay)
        out = fn(a, b)
        with _lock:
            LOG.append({"tool": name, "start": t0, "end": time.monotonic(), "thread": threading.get_ident()})
        return out
    tool.__name__ = name
    return tool


def slow_multiply(delay=None):
    f = _timed("multiply", lambda a, b: a * b, DELAY if delay is None else delay)
    f.__doc__ = "Multiply two numbers and return a * b."
    return f


def slow_add(delay=None):
    f = _timed("add", lambda a, b: a + b, DELAY if delay is None else delay)
    f.__doc__ = "Add two numbers and return a + b."
    return f


def overlap(log, delay=None) -> dict:
    delay = DELAY if delay is None else delay
    """Seconds both tools were running at once, and wall span."""
    if len(log) < 2:
        return {"calls": len(log), "overlap_s": None, "span_s": None, "verdict": "n/a"}
    a, b = sorted(log, key=lambda x: x["start"])[:2]
    ov = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    span = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return {"calls": len(log), "overlap_s": round(ov, 3), "span_s": round(span, 3),
            "same_thread": a["thread"] == b["thread"],
            "verdict": "PARALLEL" if ov > 0.5 * delay else "SEQUENTIAL"}


def _runners():
    import lg_agent, pa_agent
    return {"langgraph": lg_agent.run, "pydantic-ai": pa_agent.run}


def parallel_offline():
    """Scripted model emits BOTH calls in one response -> isolates framework scheduling."""
    import conftest
    script = [("call", [("multiply", {"a": 10, "b": 5}), ("add", {"a": 20, "b": 7})]), ("text", "50 and 27")]
    out = {}
    for fw, run in _runners().items():
        LOG.clear()
        model = conftest.lg_model(script) if fw == "langgraph" else conftest.pa_model(script)
        r = run("q", model, tools=[slow_multiply(), slow_add()])
        out[fw] = {**overlap(LOG), "llm_calls": r.llm_calls, "error": r.error}
        print(fw, out[fw])
    # Pydantic AI native control: Tool(sequential=True) forces serial execution
    from pydantic_ai import Tool
    import pa_agent
    LOG.clear()
    r = pa_agent.run("q", conftest.pa_model(script),
                     tools=[Tool(slow_multiply(), sequential=True), Tool(slow_add(), sequential=True)])
    out["pydantic-ai+sequential_flag"] = {**overlap(LOG), "llm_calls": r.llm_calls, "error": r.error}
    print("pydantic-ai+sequential_flag", out["pydantic-ai+sequential_flag"])
    return out


def parallel_live(n=3):
    """Real model: does it emit both calls at once (llm_calls==2) and do frameworks overlap them?"""
    out = {}
    q = "Calculate 10 * 5 and 20 + 7."
    for fw, run in _runners().items():
        out[fw] = []
        for i in range(n):
            LOG.clear()
            r = run(q, tools=[slow_multiply(), slow_add()])
            row = {**overlap(LOG), "llm_calls": r.llm_calls, "tools": [t.name for t in r.tool_calls],
                   "error": r.error, "answer": r.final_text[:60]}
            out[fw].append(row)
            print(fw, i + 1, row)
    return out


# ---------------- timeout experiments ----------------
def hang_tool(delay: float):
    def slow_op(a: float, b: float) -> float:
        """Add two numbers (deliberately slow)."""
        t0 = time.monotonic()
        time.sleep(delay)
        LOG.append({"tool": "slow_op", "start": t0, "end": time.monotonic(), "thread": threading.get_ident()})
        return a + b
    return slow_op


def thread_timeout_wrapper(fn, seconds):
    """Python-native timeout: run fn in a worker thread, raise TimeoutError if it overruns.
    The thread itself cannot be killed; it keeps running (documented limitation)."""
    import concurrent.futures as cf

    def wrapped(a: float, b: float) -> float:
        """Add two numbers (deliberately slow)."""
        ex = cf.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn, a, b)
        try:
            return fut.result(timeout=seconds)
        except cf.TimeoutError:
            raise TimeoutError(f"slow_op exceeded {seconds}s") from None
        finally:
            ex.shutdown(wait=False)
    wrapped.__name__ = "slow_op"
    return wrapped


def _timed_run(label, fn):
    t0 = time.monotonic()
    try:
        r = fn()
        res = {"error": r.error, "answer": r.final_text[:60], "llm_calls": r.llm_calls,
               "tool_errors": [t.error[:90] for t in r.tool_calls if t.error]}
    except BaseException as e:  # noqa: BLE001
        res = {"raised_outside_run": f"{type(e).__name__}: {str(e)[:120]}"}
    res["wall_s"] = round(time.monotonic() - t0, 2)
    print(label, res)
    return res


def timeout():
    import conftest, lg_agent, pa_agent
    from pydantic_ai import Tool
    call = [("call", [("slow_op", {"a": 1, "b": 2})]), ("text", "done")]
    SLOW, LIMIT = 4.0, 1.0
    out = {}
    # control: no timeout anywhere -> framework just waits
    out["control_lg"] = _timed_run("control langgraph (no timeout)",
        lambda: lg_agent.run("q", conftest.lg_model(call), tools=[hang_tool(SLOW)]))
    out["control_pa"] = _timed_run("control pydantic-ai (no timeout)",
        lambda: pa_agent.run("q", conftest.pa_model(call), tools=[hang_tool(SLOW)]))
    # native: Pydantic AI Tool(timeout=)
    out["native_pa_tool_timeout"] = _timed_run("pydantic-ai Tool(timeout=1)",
        lambda: pa_agent.run("q", conftest.pa_model(call), tools=[Tool(hang_tool(SLOW), timeout=LIMIT)]))
    # native: Pydantic AI whole-run timeout
    out["native_pa_run_timeout"] = _timed_run("pydantic-ai run_sync(timeout=1)",
        lambda: pa_agent.run("q", conftest.pa_model(call), tools=[hang_tool(SLOW)], run_timeout=LIMIT))
    # native: LangGraph step_timeout (sync invoke)
    out["native_lg_step_timeout_sync"] = _timed_run("langgraph step_timeout=1 (sync invoke)",
        lambda: lg_agent.run("q", conftest.lg_model(call), tools=[hang_tool(SLOW)], step_timeout=LIMIT))

    # native: LangGraph step_timeout (async ainvoke)
    async def lg_async():
        g = lg_agent.build_graph(conftest.lg_model(call), tools=[hang_tool(SLOW)], step_timeout=LIMIT)
        from langchain_core.messages import HumanMessage
        t0 = time.monotonic()
        try:
            await g.ainvoke({"messages": [HumanMessage("q")]}, {"recursion_limit": 17})
            return {"completed": True, "wall_s": round(time.monotonic() - t0, 2)}
        except BaseException as e:  # noqa: BLE001
            return {"raised": f"{type(e).__name__}: {str(e)[:100]}", "wall_s": round(time.monotonic() - t0, 2)}
    out["native_lg_step_timeout_async"] = asyncio.run(lg_async())
    print("langgraph step_timeout=1 (async ainvoke)", out["native_lg_step_timeout_async"])
    # python-native wrapper inside the tool -> same TimeoutError to both frameworks
    out["wrapper_lg"] = _timed_run("langgraph + thread-timeout wrapper",
        lambda: lg_agent.run("q", conftest.lg_model(call), tools=[thread_timeout_wrapper(hang_tool(SLOW), LIMIT)]))
    out["wrapper_pa"] = _timed_run("pydantic-ai + thread-timeout wrapper",
        lambda: pa_agent.run("q", conftest.pa_model(call), tools=[thread_timeout_wrapper(hang_tool(SLOW), LIMIT)]))
    return out


def timeout_trace(fw):
    """Span tree for a timed-out tool: LangGraph (thread wrapper -> TimeoutError) / Pydantic AI (Tool(timeout=))."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from lab_common.tracing import setup_tracing
    setup_tracing(fw)
    mem = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(mem))
    import conftest
    call = [("call", [("slow_op", {"a": 1, "b": 2})]), ("text", "done")]
    if fw == "langgraph":
        import lg_agent
        lg_agent.run("q", conftest.lg_model(call), tools=[thread_timeout_wrapper(hang_tool(3.0), 1.0)])
    else:
        import pa_agent
        from pydantic_ai import Tool
        pa_agent.run("q", conftest.pa_model(call), tools=[Tool(hang_tool(3.0), timeout=1.0)])
    spans = sorted(mem.get_finished_spans(), key=lambda x: x.start_time)
    kids = {}
    for sp in spans:
        kids.setdefault(sp.parent.span_id if sp.parent else None, []).append(sp)

    def show(pid, d):
        for sp in kids.get(pid, []):
            dur = (sp.end_time - sp.start_time) / 1e9
            ev = [e.name for e in sp.events]
            print(f"{'  ' * d}{sp.name} [{(sp.attributes or {}).get('openinference.span.kind', '-')}] "
                  f"{sp.status.status_code.name} {dur:.2f}s events={ev}")
            show(sp.context.span_id, d + 1)
    show(None, 0)


def main():
    if sys.argv[1] == "timeout-trace":
        return timeout_trace(sys.argv[2])
    mode = sys.argv[1]
    res = {"parallel-offline": parallel_offline,
           "parallel-live": lambda: parallel_live(int(sys.argv[2]) if len(sys.argv) > 2 else 3),
           "timeout": timeout}[mode]()
    path = os.path.join(HERE, "results", "experiments.json")
    try:
        with open(path, encoding="utf-8") as f:
            allres = json.load(f)
    except FileNotFoundError:
        allres = {}
    allres[mode] = res
    with open(path, "w", encoding="utf-8") as f:
        json.dump(allres, f, indent=2)


if __name__ == "__main__":
    main()
