"""Phase 2 Batch 2 (scripted tier): tool exceptions, retry mechanisms, timeouts, duplicates, idempotency.

Hypotheses are pre-registered in phases/phase-02/experiments.md. Records OBSERVED behavior only.
Reuses Batch 1 helpers (exp_batch1.go etc.), Phase 1 agents and scripted models. No API calls unless `live`.

Usage: uv run python phase-02/exp_batch2.py [exceptions retry timeout duplicates idempotency] | trace <fw> | live
"""
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_batch1 as eb  # noqa: E402  (also sets sys.path for phase-01 + lab2)
from lab2 import fixtures as fx  # noqa: E402
from lab2 import scripted as sc  # noqa: E402

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402

RESULTS: dict = {}
FWS = ["langgraph", "pydantic-ai"]


def call(name, **args):
    return ("call", [(name, args)])


def rec(exp, d):
    RESULTS.setdefault(exp, []).append(d)
    return d


def feedback(seen):
    return eb.short(seen[0][0], 100) if seen else ""


# ================================================================ B2.1 exceptions
KINDS = ["value", "runtime", "timeout", "validation", "tool_exception", "model_retry"]


def exceptions():
    script = lambda k: [call("failing_tool", kind=k), ("text", "handled")]  # noqa: E731
    configs = [("langgraph", "default", {}), ("langgraph", "handle=False", {"handle_tool_errors": False}),
               ("langgraph", "handle=(ValueError,)", {"handle_tool_errors": (ValueError,)}),
               ("pydantic-ai", "default(retries=1)", {})]
    for fw, label, kw in configs:
        for k in KINDS:
            model, r, seen, ex, dt = eb.go(fw, script(k), [fx.failing_tool], **kw)
            d = rec("b21_exceptions", {"framework": fw, "config": label, "kind": k, "tool_runs": len(ex),
                                       "run_error": eb.short(r.error or "", 70), "recovered": r.error is None and r.final_text == "handled",
                                       "model_calls": len(model.seen), "model_saw": feedback(seen)})
            print(f"B2.1 {fw:<12}{label:<22}{k:<15}recovered={d['recovered']!s:<6}model_calls={d['model_calls']} "
                  f"run_error={d['run_error'][:44]!r} model_saw={d['model_saw'][:50]!r}")


# ================================================================ B2.1 retry mechanisms (M / F / T)
def retry():
    from langgraph.types import RetryPolicy
    fast = dict(initial_interval=0.01, backoff_factor=1.0, jitter=False)
    three = [call("flaky"), call("flaky"), call("flaky"), ("text", "done")]
    one = [call("flaky"), ("text", "done")]

    def show(tag, fw, label, model, r, ex, dt):
        d = rec("b21_retry", {"mechanism": tag, "framework": fw, "config": label, "tool_runs": len(ex), "model_calls": len(model.seen),
                              "recovered": r.error is None and r.final_text == "done", "run_error": eb.short(r.error or "", 60), "wall_s": dt})
        print(f"B2.1 [{tag}] {fw:<12}{label:<44}tool_runs={d['tool_runs']} model_calls={d['model_calls']} "
              f"recovered={d['recovered']} err={d['run_error'][:50]!r} {dt}s")

    # (M) model-requested: Pydantic AI ModelRetry, fails twice
    for retries in (1, 3):
        model, r, _, ex, dt = eb.go("pydantic-ai", three, [fx.make_flaky(2, "model_retry")], retries=retries)
        show("M", "pydantic-ai", f"ModelRetry x2, retries={retries}", model, r, ex, dt)
    # (M) LangGraph: error text fed back, model re-calls
    model, r, _, ex, dt = eb.go("langgraph", three, [fx.make_flaky(2, "runtime")])
    show("M", "langgraph", "RuntimeError x2, default handler", model, r, ex, dt)
    # (F) LangGraph RetryPolicy
    cases = [("RetryPolicy(retry_on=Exception) + DEFAULT handler", dict(retry_policy=RetryPolicy(max_attempts=3, retry_on=Exception, **fast))),
             ("RetryPolicy(retry_on=Exception) + handle=False", dict(retry_policy=RetryPolicy(max_attempts=3, retry_on=Exception, **fast), handle_tool_errors=False)),
             ("RetryPolicy(default retry_on) + handle=False", dict(retry_policy=RetryPolicy(max_attempts=3, **fast), handle_tool_errors=False))]
    for label, kw in cases:
        model, r, _, ex, dt = eb.go("langgraph", one, [fx.make_flaky(2, "runtime")], **kw)
        show("F", "langgraph", label, model, r, ex, dt)
    # (F) Pydantic AI: native automatic retry of a plain exception?
    model, r, _, ex, dt = eb.go("pydantic-ai", one, [fx.make_flaky(2, "runtime")], retries=3)
    show("F?", "pydantic-ai", "RuntimeError x2, retries=3 (plain exception)", model, r, ex, dt)
    # (T) tool-internal retry
    internal = [call("flaky_internal"), ("text", "done")]
    for fw in FWS:
        model, r, _, ex, dt = eb.go(fw, internal, [fx.make_flaky_internal_retry(2)])
        show("T", fw, "tool loops internally (fails 2, 3rd ok)", model, r, ex, dt)


# ================================================================ B2.2 timeouts
DUR, LIMIT = 3.0, 1.0
TL: list = []


def _now():
    return time.monotonic()


def sync_slow():
    def slow_op(a: float, b: float) -> float:
        """Add two numbers (deliberately slow)."""
        TL.append(("start", _now()))
        time.sleep(DUR)
        TL.append(("end", _now()))
        return a + b
    return slow_op


def async_slow():
    async def slow_op(a: float, b: float) -> float:
        """Add two numbers (deliberately slow)."""
        TL.append(("start", _now()))
        try:
            await asyncio.sleep(DUR)
            TL.append(("end", _now()))
        except asyncio.CancelledError:
            TL.append(("cancelled", _now()))
            raise
        return a + b
    return slow_op


def _timeline(t0, returned):
    """Wait for the tool to finish or be cancelled, then summarize relative times."""
    deadline = _now() + DUR + 1.0
    while _now() < deadline and not any(e in ("end", "cancelled") for e, _ in TL):
        time.sleep(0.05)
    ev = {e: round(t - t0, 2) for e, t in TL}
    return {"tool_started_s": ev.get("start"), "tool_end_s": ev.get("end"), "tool_cancelled_s": ev.get("cancelled"),
            "run_returned_s": round(returned - t0, 2)}


CALL = [call("slow_op", a=1, b=2), ("text", "done")]


def _row(label, fw, mech, mode, t0, returned, outcome, final, model, seen):
    d = {"framework": fw, "mechanism": mech, "mode": mode, **_timeline(t0, returned), "outcome": eb.short(outcome, 70),
         "model_recovered": final == "done" and not outcome, "model_saw": feedback(seen)}
    rec("b22_timeout", d)
    stopped = "cancelled" if d["tool_cancelled_s"] else ("finished AFTER run returned" if d["tool_end_s"] and d["tool_end_s"] > d["run_returned_s"] + 0.15
                                                          else "finished before return" if d["tool_end_s"] else "?")
    print(f"B2.2 {fw:<12}{mech:<24}{mode:<6}run_returned={d['run_returned_s']}s tool: {stopped} "
          f"(end={d['tool_end_s']} cancelled={d['tool_cancelled_s']}) recovered={d['model_recovered']} outcome={d['outcome'][:40]!r}")


def timeout():
    from langchain_core.messages import HumanMessage
    from pydantic_ai import Agent, Tool, UsageLimits
    from lab_common.prompts import SYSTEM_INSTRUCTIONS

    def sync_case(fw, mech, tools, **kw):
        TL.clear()
        model = (sc.lg_model if fw == "langgraph" else sc.pa_model)(CALL)
        t0 = _now()
        r = (lg_agent.run if fw == "langgraph" else pa_agent.run)("q", model, tools=tools, **kw)
        ret = _now()
        _row(mech, fw, mech, "sync", t0, ret, r.error or "", r.final_text, model, sc.model_sees_lg(model) if fw == "langgraph" else sc.model_sees_pa(model))

    async def async_case(fw, mech, mk):
        TL.clear()
        model = (sc.lg_model if fw == "langgraph" else sc.pa_model)(CALL)
        t0, err, final = _now(), "", ""
        try:
            final = await mk(model)
        except BaseException as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:60]}"
        ret = _now()
        return t0, ret, err, final, model

    sync_case("langgraph", "step_timeout", [sync_slow()], step_timeout=LIMIT)

    async def lg_a(tools):
        async def mk(model):
            g = lg_agent.build_graph(model, tools=tools, step_timeout=LIMIT)
            out = await g.ainvoke({"messages": [HumanMessage("q")]}, {"recursion_limit": 17})
            return out["messages"][-1].content
        return await async_case("langgraph", "step_timeout", mk)

    for tools, mode in (([sync_slow()], "ainv+syncTool"), ([async_slow()], "ainv+asyncTool")):
        t0, ret, err, final, model = asyncio.run(lg_a(tools))
        _row("step_timeout", "langgraph", "step_timeout", mode, t0, ret, err, final, model, sc.model_sees_lg(model))

    sync_case("pydantic-ai", "Tool(timeout=1)", [Tool(sync_slow(), timeout=LIMIT)])
    sync_case("pydantic-ai", "run_sync(timeout=1)", [sync_slow()], run_timeout=LIMIT)

    async def pa_a(tools, **runkw):
        async def mk(model):
            agent = pa_agent.build_agent(model, tools=tools)
            res = await agent.run("q", usage_limits=UsageLimits(request_limit=8), **runkw)
            return str(res.output)
        return await async_case("pydantic-ai", "x", mk)

    for label, tools, runkw in (("Tool(timeout=1)", [Tool(async_slow(), timeout=LIMIT)], {}),
                                ("run(timeout=1)", [async_slow()], {"timeout": LIMIT})):
        try:
            t0, ret, err, final, model = asyncio.run(pa_a(tools, **runkw))
            _row(label, "pydantic-ai", label, "async", t0, ret, err, final, model, sc.model_sees_pa(model))
        except TypeError as e:
            print("B2.2 pydantic-ai", label, "async: not supported ->", e)

    # agent-level default tool_timeout
    TL.clear()
    model = sc.pa_model(CALL)
    agent = Agent(model, instructions=SYSTEM_INSTRUCTIONS, tools=[sync_slow()], tool_timeout=LIMIT)
    t0 = _now()
    try:
        res = agent.run_sync("q")
        err, final = "", str(res.output)
    except BaseException as e:  # noqa: BLE001
        err, final = f"{type(e).__name__}: {str(e)[:60]}", ""
    _row("Agent(tool_timeout=1)", "pydantic-ai", "Agent(tool_timeout=1)", "sync", t0, _now(), err, final, model, sc.model_sees_pa(model))


# ================================================================ B2.3 duplicates
def _call_ids(fw, model):
    if fw == "langgraph":
        from langchain_core.messages import AIMessage
        return [tc["id"] for m in model.seen[-1] if isinstance(m, AIMessage) for tc in m.tool_calls]
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    return [p.tool_call_id for m in model.seen[-1] if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)]


def duplicates():
    from langgraph.types import RetryPolicy
    fast = dict(initial_interval=0.01, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)
    same = lambda: call("record_action", action_id="A-1", payload="same")  # noqa: E731
    for fw in FWS:
        # A: same tool, same args, two turns
        act, log = fx.record_action_factory()
        model, r, _, ex, dt = eb.go(fw, [same(), same(), ("text", "done")], [act])
        d = rec("b23_dup", {"scenario": "A sequential same args", "framework": fw, "executions": len(log), "call_ids": _call_ids(fw, model), "run_error": r.error or ""})
        print(f"B2.3 {fw:<12}A same tool/args, 2 turns      executions={d['executions']} distinct_call_ids={len(set(d['call_ids']))}/{len(d['call_ids'])}")
        # A2: same tool, same args, one batch
        act, log = fx.record_action_factory()
        batch = ("call", [("record_action", {"action_id": "A-1", "payload": "same"})] * 2)
        model, r, _, ex, dt = eb.go(fw, [batch, ("text", "done")], [act])
        d = rec("b23_dup", {"scenario": "A2 same batch same args", "framework": fw, "executions": len(log), "call_ids": _call_ids(fw, model), "run_error": r.error or ""})
        print(f"B2.3 {fw:<12}A2 identical calls in one batch executions={d['executions']} distinct_call_ids={len(set(d['call_ids']))}/{len(d['call_ids'])}")

    # C: failure AFTER side effect (response lost), various retry owners
    def orders_case(label, fw, script, orders, tool, **kw):
        model, r, _, ex, dt = eb.go(fw, script, [tool], **kw)
        d = rec("b23_after_effect", {"scenario": label, "framework": fw, "ledger": len(orders.ledger), "executions": orders.executions,
                                     "replays": orders.replays, "model_calls": len(model.seen), "run_error": eb.short(r.error or "", 60),
                                     "final": eb.short(r.final_text, 30)})
        print(f"B2.3 {fw:<12}{label:<50}orders={d['ledger']} model_calls={d['model_calls']} err={d['run_error'][:45]!r}")
        return d

    co = call("create_order", item="widget", qty=2)
    o = fx.Orders(fail_first=1); orders_case("C1 (M) response lost, model re-calls", "langgraph", [co, co, ("text", "done")], o, o.unguarded())
    o = fx.Orders(fail_first=1, exc=lambda m: __import__("pydantic_ai").ModelRetry(m)); orders_case("C3 (M) ModelRetry after effect, model re-calls", "pydantic-ai", [co, co, ("text", "done")], o, o.unguarded())
    o = fx.Orders(fail_first=1); orders_case("C4 plain exception escapes (run dies)", "pydantic-ai", [co, co, ("text", "done")], o, o.unguarded())
    o = fx.Orders(fail_first=1); orders_case("C4b caller re-runs whole run after failure", "pydantic-ai", [co, ("text", "done")], o, o.unguarded())
    d = orders_case("C4b2 ...second run() by the caller", "pydantic-ai", [co, ("text", "done")], o, o.unguarded())
    o = fx.Orders(fail_first=1); orders_case("C2 (F) RetryPolicy, no model call between", "langgraph", [co, ("text", "done")], o, o.unguarded(),
                                             handle_tool_errors=False, retry_policy=RetryPolicy(**fast))
    # H10: RetryPolicy re-runs the whole node: do already-succeeded calls in the same batch repeat?
    act, log = fx.record_action_factory()
    flaky = fx.make_flaky(1, "runtime")
    batch = ("call", [("record_action", {"action_id": "A-1", "payload": "once"}), ("flaky", {})])
    model, r, _, ex, dt = eb.go("langgraph", [batch, ("text", "done")], [act, flaky], handle_tool_errors=False, retry_policy=RetryPolicy(**fast))
    d = rec("b23_after_effect", {"scenario": "H10 batch [ok effect, failing call] + RetryPolicy", "framework": "langgraph", "record_action_executions": len(log), "model_calls": len(model.seen), "run_error": r.error or ""})
    print(f"B2.3 langgraph   H10 batch [record_action ok, flaky fails once] + RetryPolicy: record_action executed {len(log)}x, model_calls={len(model.seen)}, err={r.error!r}")
    act, log = fx.record_action_factory()
    model, r, _, ex, dt = eb.go("pydantic-ai", [batch, ("text", "done")], [act, fx.make_flaky(1, "runtime")])
    rec("b23_after_effect", {"scenario": "H10b same batch, no retry policy", "framework": "pydantic-ai", "record_action_executions": len(log), "run_error": eb.short(r.error or "", 60)})
    print(f"B2.3 pydantic-ai H10b same batch: record_action executed {len(log)}x, run_error={eb.short(r.error or '', 50)!r}")


# ================================================================ B2.4 idempotency
def idempotency():
    from langgraph.types import RetryPolicy
    from pydantic_ai import ModelRetry, Tool
    fast = dict(initial_interval=0.01, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)

    def go_o(label, fw, script, orders, tool, **kw):
        model, r, _, ex, dt = eb.go(fw, script, [tool], **kw)
        d = rec("b24_idempotency", {"scenario": label, "framework": fw, "orders_created": len(orders.ledger), "executions": orders.executions,
                                    "replays": orders.replays, "model_calls": len(model.seen), "run_error": eb.short(r.error or "", 70),
                                    "tool_feedback": eb.short(next((t.error for t in r.tool_calls if t.error), "") or "", 70)})
        print(f"B2.4 {fw:<12}{label:<56}orders={d['orders_created']} exec={d['executions']} replays={d['replays']} err={d['run_error'][:40]!r} fb={d['tool_feedback'][:40]!r}")
        return d

    def k(key, item="widget", qty=2):
        return call("create_order", item=item, qty=qty, idempotency_key=key)

    for fw in FWS:
        o = fx.Orders(); go_o("first request (key k1)", fw, [k("k1"), ("text", "x")], o, o.model_keyed())
        o = fx.Orders(); go_o("exact duplicate (k1, same args)", fw, [k("k1"), k("k1"), ("text", "x")], o, o.model_keyed())
        o = fx.Orders(); go_o("same key, different args", fw, [k("k1"), k("k1", qty=9), ("text", "x")], o, o.model_keyed())
        o = fx.Orders(); go_o("different key, same args", fw, [k("k1"), k("k2"), ("text", "x")], o, o.model_keyed())
        exc = (lambda m: ModelRetry(m)) if fw == "pydantic-ai" else RuntimeError
        o = fx.Orders(fail_first=1, exc=exc); go_o("lost response, SAME key on retry (model re-calls)", fw, [k("k1"), k("k1"), ("text", "x")], o, o.model_keyed())
        o = fx.Orders(fail_first=1, exc=exc); go_o("lost response, model CHANGES key on retry", fw, [k("k1"), k("k2"), ("text", "x")], o, o.model_keyed())
        o = fx.Orders(fail_first=1, exc=exc); go_o("lost response, key from context (system), model re-calls", fw, [call("create_order", item="widget", qty=2)] * 2 + [("text", "x")], o, o.context_keyed("req-42"))
    o = fx.Orders(fail_first=1); go_o("lost response, (F) RetryPolicy + keyed", "langgraph", [call("create_order", item="widget", qty=2), ("text", "x")], o, o.context_keyed("req-42"),
                                      handle_tool_errors=False, retry_policy=RetryPolicy(**fast))
    # race: identical keyed calls in ONE batch, non-atomic guard (check ... delay ... commit)
    race = ("call", [("create_order", {"item": "widget", "qty": 2})] * 2)
    for fw in FWS:
        o = fx.Orders(delay=0.3); go_o("RACE two identical keyed calls in one batch (parallel)", fw, [race, ("text", "x")], o, o.context_keyed("req-42"))
    o = fx.Orders(delay=0.3)
    go_o("RACE same batch but Tool(sequential=True)", "pydantic-ai", [race, ("text", "x")], o, Tool(o.context_keyed("req-42"), sequential=True))


def live(n=5):
    """H14 (live, <= 5 trials per framework): lost response after the side effect; the MODEL supplies the idempotency key."""
    from pydantic_ai import ModelRetry
    from lab_common.config import load_model_config
    cfg = load_model_config()
    print(f"live model: {cfg.provider}/{cfg.model_name}, trials per framework: {n}")
    instr = ("You are an order assistant. Place orders only with the create_order tool. "
             "If a tool call fails, decide yourself what to do, then tell the user the outcome.")
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = instr   # experiment-local override of the Phase 1 prompt
    for fw in FWS:
        run = lg_agent.run if fw == "langgraph" else pa_agent.run
        for i in range(n):
            o = fx.Orders(fail_first=1, exc=(ModelRetry if fw == "pydantic-ai" else RuntimeError))
            t0 = time.monotonic()
            try:
                r = run("Please place an order for 2 widgets.", tools=[o.model_keyed()])
            except BaseException as e:  # noqa: BLE001
                print(f"LIVE {fw} trial {i + 1}: harness error {type(e).__name__}: {str(e)[:80]}")
                continue
            keys = [t.args.get("idempotency_key") for t in r.tool_calls]
            d = rec("b24_live", {"framework": fw, "trial": i + 1, "tool_calls": len(r.tool_calls), "keys": keys,
                                 "same_key_on_retry": len(set(keys)) == 1 and len(keys) > 1, "orders_created": len(o.ledger),
                                 "replays": o.replays, "run_error": eb.short(r.error or "", 60), "final": eb.short(r.final_text, 70),
                                 "wall_s": round(time.monotonic() - t0, 2)})
            print(f"LIVE {fw:<12}trial {i + 1}: tool_calls={d['tool_calls']} keys={keys} orders={d['orders_created']} replays={d['replays']} "
                  f"err={d['run_error'][:40]!r} final={d['final'][:50]!r}")


def trace(fw):
    """Span tree (OTel + OpenInference, in-memory) of failure-after-side-effect with a model-requested retry."""
    from opentelemetry import trace as ot
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from lab_common.tracing import setup_tracing
    from pydantic_ai import ModelRetry
    setup_tracing(fw)
    mem = InMemorySpanExporter()
    ot.get_tracer_provider().add_span_processor(SimpleSpanProcessor(mem))
    o = fx.Orders(fail_first=1, exc=(ModelRetry if fw == "pydantic-ai" else RuntimeError))
    co = call("create_order", item="widget", qty=2)
    model = (sc.lg_model if fw == "langgraph" else sc.pa_model)([co, co, ("text", "done")])
    (lg_agent.run if fw == "langgraph" else pa_agent.run)("q", model, tools=[o.unguarded()])
    kids = {}
    for sp in sorted(mem.get_finished_spans(), key=lambda x: x.start_time):
        kids.setdefault(sp.parent.span_id if sp.parent else None, []).append(sp)

    def show(pid, d):
        for sp in kids.get(pid, []):
            a = sp.attributes or {}
            print(f"{'  ' * d}{sp.name} [{a.get('openinference.span.kind', '-')}] {sp.status.status_code.name}"
                  f"{' exception-event' if any(e.name == 'exception' for e in sp.events) else ''}")
            show(sp.context.span_id, d + 1)
    show(None, 0)
    print("orders created:", len(o.ledger))


ALL = {"exceptions": exceptions, "retry": retry, "timeout": timeout, "duplicates": duplicates, "idempotency": idempotency, "live": live}


def main():
    if sys.argv[1:2] == ["trace"]:
        return trace(sys.argv[2])
    which = sys.argv[1:] or [k for k in ALL if k != "live"]
    for k in which:
        print(f"\n===== {k} =====")
        ALL[k]()
    path = os.path.join(HERE, "results", "batch2.json")
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except FileNotFoundError:
        old = {}
    old.update(RESULTS)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2, default=str)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
