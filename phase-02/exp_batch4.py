"""Phase 2 Batch 4: concurrency, parallel tool calls, race conditions, partial failure, cancellation.

Hypotheses are pre-registered in phases/phase-02/experiments.md. Records OBSERVED behavior from monotonic timestamps.
Layer attribution per row: MODEL (what was batched) / FRAMEWORK orchestration / RUNTIME (thread pool, event loop) /
TOOL / APPLICATION state. Reuses Batch 1-3 helpers, the Phase 1 agents and scripted models. No new dependency.

Usage: uv run python phase-02/exp_batch4.py [independent order counter asyncrace duplicate limits partial dependent
       authz output cancel combined] | live
"""
import asyncio
import contextlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_batch1 as eb  # noqa: E402  (sets sys.path for phase-01 + lab2)
from lab2 import concurrency as cc  # noqa: E402
from lab2 import fixtures as fx  # noqa: E402
from lab2 import scripted as sc  # noqa: E402
from lab2 import security as sec  # noqa: E402

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402

RESULTS: dict = {}
FWS = ["langgraph", "pydantic-ai"]
S = 1.0     # time scale (tests shrink it)
TL = cc.TL


def d(x):
    return x * S


def rec(exp, row):
    RESULTS.setdefault(exp, []).append(row)
    return row


def call(name, **args):
    return ("call", [(name, args)])


def batch(*names_args):
    return ("call", [(n, a) for n, a in names_args])


# ------------------------------------------------------------------ runners (framework execution; scripted model)
def visible_results(fw, model):
    """Tool results in the order the model saw them (RESULT order)."""
    if not model.seen:
        return []
    last = model.seen[-1]
    if fw == "langgraph":
        from langchain_core.messages import ToolMessage
        return [m.content for m in last if isinstance(m, ToolMessage)]
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    out = []
    for m in last[-1:]:
        if isinstance(m, ModelRequest):
            out += [p.model_response_str() for p in m.parts if isinstance(p, ToolReturnPart)]
    return out


def run_lg(model, tools, runner="invoke", max_concurrency=None, handle=True, wrap=None, context=None, schema=None):
    from langchain_core.messages import HumanMessage
    g = lg_agent.build_graph(model, tools=tools, handle_tool_errors=handle, wrap_tool_call=wrap, context_schema=schema)
    cfg = {"recursion_limit": 17}
    if max_concurrency:
        cfg["max_concurrency"] = max_concurrency
    kw = {"context": context} if context is not None else {}
    inp = {"messages": [HumanMessage("q")]}
    try:
        if runner == "invoke":
            out = g.invoke(inp, cfg, **kw)
        else:
            out = asyncio.run(g.ainvoke(inp, cfg, **kw))
        return out["messages"][-1].content, ""
    except BaseException as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {str(e)[:70]}"


def run_pa(model, tools, runner="run_sync", pmode=None, deps=None, capabilities=None, agent=None):
    from pydantic_ai import UsageLimits
    agent = agent or pa_agent.build_agent(model, tools=tools, capabilities=capabilities,
                                          deps_type=type(deps) if deps is not None else None)
    cm = agent.parallel_tool_call_execution_mode(pmode) if pmode else contextlib.nullcontext()
    try:
        with cm:
            if runner == "run_sync":
                out = agent.run_sync("q", usage_limits=UsageLimits(request_limit=8), deps=deps)
            else:
                out = asyncio.run(agent.run("q", usage_limits=UsageLimits(request_limit=8), deps=deps))
        return str(out.output), ""
    except BaseException as e:  # noqa: BLE001
        return "", f"{type(e).__name__}: {str(e)[:70]}"


def exec_case(fw, runner, script, tools, settle=0.0, **kw):
    """Run one scripted case; return (model, final, error, wall_ms) after letting stragglers finish."""
    TL.reset()
    model = (sc.lg_model if fw == "langgraph" else sc.pa_model)(script)
    t0 = time.perf_counter()
    final, err = (run_lg if fw == "langgraph" else run_pa)(model, tools, runner, **kw)
    wall = round((time.perf_counter() - t0) * 1000)
    if settle:
        time.sleep(settle)
    return model, final, err, wall


def timing_row(exp, label, fw, runner, kind, model, final, err, wall, names=None, **extra):
    names = names or [e["name"] for e in TL.rel()]
    row = {"exp": exp, "case": label, "framework": fw, "runner": runner, "tool_kind": kind, "wall_ms": wall,
           "makespan_ms": TL.makespan_ms(), "max_concurrency": TL.max_concurrency(), "threads": TL.threads(),
           "start_order": TL.order("start"), "completion_order": TL.order("end"),
           "result_order": [r.replace(" done", "") for r in visible_results(fw, model)], "error": err, "final": final[:20],
           "events": [{k: e[k] for k in ("name", "start_ms", "end_ms", "status", "thread")} for e in TL.rel()], **extra}
    return rec(exp, row)


def show(r, extra=""):
    print(f"{r['exp']:<8}{r['framework']:<12}{r['runner']:<9}{r['tool_kind']:<6}{r['case'][:34]:<35}wall={r['wall_ms']:>5}ms "
          f"makespan={r['makespan_ms']:>5} maxconc={r['max_concurrency']:<3}threads={r['threads']} {extra}err={r['error'][:40]!r}")


# ================================================================== 4.1 / 4.2 / 4.3 / 4.10 / 4.4 independent tools
def independent():
    script = [batch(("tool_a", {}), ("tool_b", {})), ("text", "done")]
    cases = [("langgraph", "invoke", "sync", {}), ("langgraph", "ainvoke", "sync", {}), ("langgraph", "ainvoke", "async", {}),
             ("langgraph", "invoke", "async", {}),
             ("pydantic-ai", "run_sync", "sync", {}), ("pydantic-ai", "run_sync", "async", {}),
             ("pydantic-ai", "run", "sync", {}), ("pydantic-ai", "run", "async", {}),
             ("pydantic-ai", "run_sync", "sync", {"pmode": "sequential"}),
             ("pydantic-ai", "run", "async", {"pmode": "sequential"}),
             ("pydantic-ai", "run_sync", "sync", {"pmode": "parallel_ordered_events"})]
    from pydantic_ai import Tool
    for fw, runner, kind, kw in cases:
        mk = cc.sync_tool if kind == "sync" else cc.async_tool
        tools = [mk("tool_a", d(1.0)), mk("tool_b", d(1.0))]
        label = "A+B batched" + (f" mode={kw['pmode']}" if kw else "")
        model, final, err, wall = exec_case(fw, runner, script, tools, **kw)
        r = timing_row("4.1-4.4", label, fw, runner, kind, model, final, err, wall, overlap_ms=TL.overlap_ms("tool_a", "tool_b"),
                       tasks=sorted({e["task"] or "-" for e in TL.events}))
        show(r, f"overlap={r['overlap_ms']}ms ")
    # per-tool sequential flag (Pydantic AI native)
    tools = [Tool(cc.sync_tool("tool_a", d(1.0)), sequential=True), Tool(cc.sync_tool("tool_b", d(1.0)), sequential=True)]
    model, final, err, wall = exec_case("pydantic-ai", "run_sync", script, tools)
    r = timing_row("4.1-4.4", "A+B Tool(sequential=True)", "pydantic-ai", "run_sync", "sync", model, final, err, wall, overlap_ms=TL.overlap_ms("tool_a", "tool_b"))
    show(r, f"overlap={r['overlap_ms']}ms ")
    # separate (unbatched) A then B: the model, not the framework, serializes
    model, final, err, wall = exec_case("langgraph", "invoke", [call("tool_a"), call("tool_b"), ("text", "done")],
                                        [cc.sync_tool("tool_a", d(1.0)), cc.sync_tool("tool_b", d(1.0))])
    r = timing_row("4.1-4.4", "A then B in separate turns", "langgraph", "invoke", "sync", model, final, err, wall, overlap_ms=TL.overlap_ms("tool_a", "tool_b"))
    show(r, f"overlap={r['overlap_ms']}ms ")
    print("\nTimeline (LangGraph invoke, sync tools, batched A+B):")
    model, final, err, wall = exec_case("langgraph", "invoke", script, [cc.sync_tool("tool_a", d(1.0)), cc.sync_tool("tool_b", d(1.0))])
    print(TL.ascii(step_ms=100))


# ================================================================== 4.5 four different orders
def order():
    script = [batch(("tool_a", {}), ("tool_b", {}), ("tool_c", {})), ("text", "done")]
    for fw, runner in (("langgraph", "invoke"), ("pydantic-ai", "run_sync"), ("langgraph", "ainvoke"), ("pydantic-ai", "run")):
        for kind, mk in (("sync", cc.sync_tool), ("async", cc.async_tool)):
            tools = [mk("tool_a", d(0.3)), mk("tool_b", d(0.1)), mk("tool_c", d(0.2))]
            model, final, err, wall = exec_case(fw, runner, script, tools)
            r = timing_row("4.5", "A=300 B=100 C=200 ms", fw, runner, kind, model, final, err, wall,
                           call_order=["tool_a", "tool_b", "tool_c"])
            show(r, f"call=ABC start={''.join(n[-1].upper() for n in r['start_order'])} done={''.join(n[-1].upper() for n in r['completion_order'])} "
                    f"result={''.join(n[-1].upper() for n in r['result_order'])} ")
    print("\nTimeline (Pydantic AI run_sync, sync tools):")
    exec_case("pydantic-ai", "run_sync", script, [cc.sync_tool("tool_a", d(0.3)), cc.sync_tool("tool_b", d(0.1)), cc.sync_tool("tool_c", d(0.2))])
    print(TL.ascii(step_ms=20))


# ================================================================== 4.6 / 4.7 shared mutable counter (threads)
class Counter:
    def __init__(self, work, locked):
        self.value, self.work, self.lock, self.locked = 0, work, threading.Lock(), locked

    def tool(self):
        def increment() -> int:
            """Increment the shared counter."""
            ev = TL.begin("inc")
            if self.locked:
                with self.lock:                       # stdlib primitive, not a custom lock
                    v = self.value
                    time.sleep(self.work)
                    self.value = v + 1
            else:
                v = self.value                        # read
                time.sleep(self.work)                 # window in which other threads read the same value
                self.value = v + 1                    # write (lost update)
            TL.finish(ev)
            return self.value
        return increment


def counter(n=100, work=0.005):
    from pydantic_ai import Tool
    script = [("call", [("increment", {})] * n), ("text", "done")]
    # control with no framework at all
    c = Counter(work, False)
    TL.reset()
    t0 = time.perf_counter()
    ths = [threading.Thread(target=c.tool()) for _ in range(n)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    rec("4.6-4.7", {"case": "control: raw threads, unsafe", "framework": "none", "expected": n, "actual": c.value, "lost": n - c.value,
                    "wall_ms": round((time.perf_counter() - t0) * 1000), "max_concurrency": TL.max_concurrency()})
    print(f"4.6-4.7 control (no framework) unsafe: final={c.value}/{n} lost={n - c.value} maxconc={TL.max_concurrency()}")
    variants = [("unsafe", False, {}), ("Lock", True, {}), ("unsafe + sequential mode", False, {"pmode": "sequential"}),
                ("unsafe + Tool(sequential=True)", False, {"toolseq": True})]
    for fw, runner in (("langgraph", "invoke"), ("pydantic-ai", "run_sync")):
        for label, locked, kw in variants:
            if fw == "langgraph" and kw:
                continue                                 # no native sequential control in LangGraph
            c = Counter(work, locked)
            fn = c.tool()
            tools = [Tool(fn, sequential=True)] if kw.get("toolseq") else [fn]
            kw2 = {"pmode": kw["pmode"]} if "pmode" in kw else {}
            model, final, err, wall = exec_case(fw, runner, script, tools, **kw2)
            r = rec("4.6-4.7", {"case": label, "framework": fw, "runner": runner, "expected": n, "actual": c.value, "lost": n - c.value,
                                "wall_ms": wall, "max_concurrency": TL.max_concurrency(), "threads": TL.threads(), "error": err})
            print(f"4.6-4.7 {fw:<12}{label:<32}final={c.value}/{n} lost={r['lost']:<3}wall={wall}ms maxconc={r['max_concurrency']} threads={r['threads']} err={err[:30]!r}")


# ================================================================== 4.8 async race
class ACounter:
    def __init__(self, variant, work):
        self.value, self.variant, self.work, self.lock = 0, variant, work, None

    def tool(self):
        async def aincrement() -> int:
            """Increment the shared counter (async)."""
            ev = TL.begin("ainc")
            if self.variant == "no_await_between":
                v = self.value
                self.value = v + 1                    # read-modify-write with no suspension point: atomic under asyncio
                await asyncio.sleep(self.work)
            elif self.variant == "asyncio.Lock":
                if self.lock is None:
                    self.lock = asyncio.Lock()
                async with self.lock:
                    v = self.value
                    await asyncio.sleep(self.work)
                    self.value = v + 1
            else:                                     # await_between, no lock
                v = self.value
                await asyncio.sleep(self.work)        # suspension point between read and write
                self.value = v + 1
            TL.finish(ev)
            return self.value
        return aincrement


def asyncrace(n=100, work=0.005):
    script = [("call", [("aincrement", {})] * n), ("text", "done")]
    for variant in ("await_between", "asyncio.Lock", "no_await_between"):
        c = ACounter(variant, work)

        async def control():
            await asyncio.gather(*[c.tool()() for _ in range(n)])
        TL.reset()
        asyncio.run(control())
        rec("4.8", {"case": variant, "framework": "none (asyncio.gather)", "expected": n, "actual": c.value, "lost": n - c.value})
        print(f"4.8 control asyncio.gather {variant:<18} final={c.value}/{n} lost={n - c.value}")
        for fw, runner in (("langgraph", "ainvoke"), ("pydantic-ai", "run")):
            c = ACounter(variant, work)
            model, final, err, wall = exec_case(fw, runner, script, [c.tool()])
            r = rec("4.8", {"case": variant, "framework": fw, "runner": runner, "expected": n, "actual": c.value, "lost": n - c.value,
                            "wall_ms": wall, "max_concurrency": TL.max_concurrency(), "threads": TL.threads(), "error": err})
            print(f"4.8 {fw:<12}{variant:<18}final={c.value}/{n} lost={r['lost']:<3}wall={wall}ms maxconc={r['max_concurrency']} threads={r['threads']} err={err[:30]!r}")


# ================================================================== 4.9 concurrent duplicate side effect (one batch)
def duplicate():
    co = lambda key: ("call", [("create_order", {"item": "widget", "qty": 2})] * 2)  # noqa: E731
    for label, kw in (("no key", dict(delay=d(0.2))), ("non-atomic key guard", dict(delay=d(0.2))), ("atomic guard (lock around check+effect+commit)", dict(delay=d(0.2), atomic=True))):
        for fw, runner in (("langgraph", "invoke"), ("pydantic-ai", "run_sync")):
            o = fx.Orders(**kw)
            tool = o.unguarded() if label == "no key" else o.context_keyed("ORDER-1")
            t0 = time.perf_counter()
            model, final, err, wall = exec_case(fw, runner, [co(None), ("text", "done")], [tool])
            r = rec("4.9", {"case": label, "framework": fw, "orders_created": len(o.ledger), "executions": o.executions, "replays": o.replays, "wall_ms": wall, "error": err})
            print(f"4.9 {fw:<12}{label:<48}orders={r['orders_created']} executions={r['executions']} replays={r['replays']} wall={wall}ms")


# ================================================================== 4.11 concurrency limit
def limits():
    from pydantic_ai import Agent, UsageLimits
    script = [("call", [(f"t{i}", {}) for i in range(10)]), ("text", "done")]

    def tools(kind):
        mk = cc.sync_tool if kind == "sync" else cc.async_tool
        return [mk(f"t{i}", d(0.5)) for i in range(10)]

    def row(label, fw, runner, kind, model, final, err, wall):
        r = timing_row("4.11", label, fw, runner, kind, model, final, err, wall)
        print(f"4.11 {fw:<12}{runner:<9}{kind:<6}{label:<40}maxconc={r['max_concurrency']:<3}makespan={r['makespan_ms']}ms threads={r['threads']} err={err[:30]!r}")
    for mc in (None, 1, 2, 4, 8, 10):
        m, f, e, w = exec_case("langgraph", "invoke", script, tools("sync"), max_concurrency=mc)
        row(f"config max_concurrency={mc}", "langgraph", "invoke", "sync", m, f, e, w)
    for mc in (2,):
        m, f, e, w = exec_case("langgraph", "ainvoke", script, tools("sync"), max_concurrency=mc)
        row(f"config max_concurrency={mc}", "langgraph", "ainvoke", "sync", m, f, e, w)
        m, f, e, w = exec_case("langgraph", "ainvoke", script, tools("async"), max_concurrency=mc)
        row(f"config max_concurrency={mc}", "langgraph", "ainvoke", "async", m, f, e, w)
    m, f, e, w = exec_case("pydantic-ai", "run_sync", script, tools("sync")); row("default", "pydantic-ai", "run_sync", "sync", m, f, e, w)
    m, f, e, w = exec_case("pydantic-ai", "run", script, tools("async")); row("default", "pydantic-ai", "run", "async", m, f, e, w)
    m, f, e, w = exec_case("pydantic-ai", "run_sync", script, tools("sync"), pmode="sequential"); row("mode=sequential (limit of 1)", "pydantic-ai", "run_sync", "sync", m, f, e, w)
    # Agent(max_concurrency=2): documented as a limit on concurrent agent RUNS
    TL.reset()
    model = sc.pa_model(script)
    agent = Agent(model, tools=tools("sync"), max_concurrency=2)
    t0 = time.perf_counter()
    final, err = run_pa(model, None, "run_sync", agent=agent)
    row("Agent(max_concurrency=2) (limits runs)", "pydantic-ai", "run_sync", "sync", model, final, err, round((time.perf_counter() - t0) * 1000))


# ================================================================== 4.12 / 4.13 partial failure and failure timeline
def partial():
    from pydantic_ai import ModelRetry
    script = [batch(("tool_a", {}), ("tool_b", {}), ("tool_c", {})), ("text", "recovered")]
    configs = [("langgraph", "invoke", "sync", {}, "RuntimeError (default handler)"), ("langgraph", "invoke", "sync", {"handle": False}, "RuntimeError (handle=False)"),
               ("langgraph", "ainvoke", "async", {}, "RuntimeError (default handler)"), ("langgraph", "ainvoke", "async", {"handle": False}, "RuntimeError (handle=False)"),
               ("pydantic-ai", "run_sync", "sync", {}, "RuntimeError"), ("pydantic-ai", "run", "async", {}, "RuntimeError"),
               ("pydantic-ai", "run_sync", "sync", {"modelretry": True}, "ModelRetry"), ("pydantic-ai", "run", "async", {"modelretry": True}, "ModelRetry")]
    for fw, runner, kind, kw, label in configs:
        effects, cleanup = [], []
        mk = cc.sync_tool if kind == "sync" else cc.async_tool
        failure = (lambda n: ModelRetry(f"{n} failed, retry")) if kw.get("modelretry") else True
        tools = [mk("tool_a", d(0.6), effect_log=effects), mk("tool_b", 0, fail=failure), mk("tool_c", d(0.6), effect_log=effects)]
        kw2 = {k: v for k, v in kw.items() if k != "modelretry"}
        model, final, err, wall = exec_case(fw, runner, script, tools, settle=d(0.8), **kw2)
        ev = {e["name"]: (e["status"], e["end_ms"]) for e in TL.rel()}
        r = timing_row("4.12-4.13", label, fw, runner, kind, model, final, err, wall, effects=sorted(effects),
                       statuses={k: v[0] for k, v in ev.items()}, b_failed_at_ms=ev.get("tool_b", (None, None))[1],
                       model_saw_n=len(visible_results(fw, model)), recovered=(final == "recovered"))
        print(f"4.12-13 {fw:<12}{runner:<9}{kind:<6}{label:<30}run_returned={wall}ms B_failed_at={r['b_failed_at_ms']} statuses={r['statuses']} "
              f"effects={r['effects']} model_saw={r['model_saw_n']} results recovered={r['recovered']} err={err[:36]!r}")
    print("\nTimeline (Pydantic AI run_sync, RuntimeError, sync tools):")
    effects = []
    exec_case("pydantic-ai", "run_sync", script, [cc.sync_tool("tool_a", d(0.6), effect_log=effects), cc.sync_tool("tool_b", 0, fail=True),
                                                  cc.sync_tool("tool_c", d(0.6), effect_log=effects)], settle=d(0.8))
    print(TL.ascii(step_ms=50))


# ================================================================== 4.14 dependent vs independent (scripted)
def dependent():
    def tools():
        def get_order_id() -> str:
            """Return the current order id."""
            ev = TL.begin("get_order_id"); time.sleep(d(0.3)); TL.finish(ev); return "order-7"

        def get_order_status(order_id: str) -> str:
            """Return the status of an order."""
            ev = TL.begin("get_order_status"); time.sleep(d(0.3)); TL.finish(ev)
            return "SHIPPED" if order_id == "order-7" else f"unknown order {order_id}"
        return [get_order_id, get_order_status]
    bad = [batch(("get_order_id", {}), ("get_order_status", {"order_id": "order-1"})), ("text", "done")]      # dependent call batched with a guess
    good = [call("get_order_id"), call("get_order_status", order_id="order-7"), ("text", "done")]               # dependency honored across turns
    for fw, runner in (("langgraph", "invoke"), ("pydantic-ai", "run_sync")):
        for label, script in (("dependent B batched with guessed arg", bad), ("dependent B after A (separate turns)", good)):
            model, final, err, wall = exec_case(fw, runner, script, tools())
            r = timing_row("4.14", label, fw, runner, "sync", model, final, err, wall, overlap_ms=TL.overlap_ms("get_order_id", "get_order_status"))
            print(f"4.14 {fw:<12}{label:<42}overlap={r['overlap_ms']}ms B_result={r['result_order']} model_calls={len(model.seen)}")


# ================================================================== 4.15 concurrent authorization / identity isolation
A_USER = sec.Identity("u-a", "OPERATOR", "tenant-A")
B_USER = sec.Identity("u-b", "OPERATOR", "tenant-B")
TARGET = {"A": "order-200", "B": "order-100"}       # each user targets the OTHER tenant's order: correct decision is DENY
USER_OF = {"A": A_USER, "B": B_USER}


class _Global:
    user = None


def authz(trials=10, work=0.2):
    """Unsafe/safe closure designs run as two concurrent requests; the trusted framework context is in authz_trusted."""

    def make_tool(store, identity_source, log):
        def read_order(order_id: str, req: str) -> str:
            """Read an order."""
            time.sleep(d(work))                        # request overlap window
            ident = identity_source(req)
            try:
                sec.check(ident, "get_order", order_id, store)
                allowed = True
            except sec.PermissionDenied:
                allowed = False
            log.append({"req": req, "expected_tenant": USER_OF[req].tenant, "used_tenant": ident.tenant, "allowed": allowed})
            return "ALLOW" if allowed else "DENY"
        return read_order

    def one_trial(fw, design):
        store, log, shared = sec.OrderStore(), [], {"tools": None}
        barrier = threading.Barrier(2)

        def request(label):
            user = USER_OF[label]
            script = [call("read_order", order_id=TARGET[label], req=label), ("text", "done")]
            model = (sc.lg_model if fw == "langgraph" else sc.pa_model)(script)
            barrier.wait()
            if design == "global current_user":
                _Global.user = user                                         # what a request handler might do at request start
                tools = [make_tool(store, lambda req: _Global.user, log)]
            elif design == "shared tool set closed over the FIRST request's identity":
                if shared["tools"] is None:
                    shared["tools"] = [make_tool(store, lambda req, u=user: u, log)]
                tools = shared["tools"]
            else:                                                             # tools built per request (closure)
                tools = [make_tool(store, lambda req, u=user: u, log)]
            return (run_lg if fw == "langgraph" else run_pa)(model, tools, "invoke" if fw == "langgraph" else "run_sync")

        with ThreadPoolExecutor(2) as ex:
            list(ex.map(request, ["A", "B"]))
        return log

    designs = ["global current_user", "shared tool set closed over the FIRST request's identity", "tools built per request (closure)"]
    for fw in FWS:
        for design in designs:
            leaks = wrong = n = 0
            for _ in range(trials):
                for row_ in one_trial(fw, design):
                    n += 1
                    leaks += row_["expected_tenant"] != row_["used_tenant"]
                    wrong += row_["allowed"]                   # every legitimate decision here is DENY, so any ALLOW is a wrongful grant
            rec("4.15", {"design": design, "framework": fw, "requests": n, "identity_leaks": leaks, "unauthorized_allows": wrong})
            print(f"4.15 {fw:<12}{design:<62}requests={n} identity_leaks={leaks} unauthorized_allows={wrong}")
    authz_trusted(trials, work)


def authz_trusted(trials, work):
    """Trusted per-call context supplied by the framework: LangGraph runtime.context / Pydantic AI ctx.deps."""
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import Hooks
    for fw in FWS:
        leaks = wrong = n = 0
        for _ in range(trials):
            store, log = sec.OrderStore(), []
            sec.AUTHZ.clear()
            barrier = threading.Barrier(2)

            def request(label):
                user = USER_OF[label]
                script = [call("read_order", order_id=TARGET[label], req=label), ("text", "done")]
                model = (sc.lg_model if fw == "langgraph" else sc.pa_model)(script)

                def read_order(order_id: str, req: str) -> str:
                    """Read an order (identity-free tool body)."""
                    time.sleep(d(work))
                    return "executed"

                def record(ident, order_id):
                    try:
                        sec.check(ident, "get_order", order_id, store)
                        allowed = True
                    except sec.PermissionDenied:
                        allowed = False
                    log.append({"req": label, "expected_tenant": user.tenant, "used_tenant": ident.tenant, "allowed": allowed})

                barrier.wait()
                if fw == "langgraph":
                    def wrap(request_, execute):
                        record(request_.runtime.context, request_.tool_call["args"]["order_id"])
                        return execute(request_)
                    return run_lg(model, [read_order], "invoke", wrap=wrap, context=user, schema=sec.Identity)

                def before(ctx, *, call, tool_def, args):
                    record(ctx.deps, args["order_id"])
                    return args
                return run_pa(model, [read_order], "run_sync", deps=user, capabilities=[Hooks(before_tool_execute=before)])

            with ThreadPoolExecutor(2) as ex:
                list(ex.map(request, ["A", "B"]))
            for row_ in log:
                n += 1
                leaks += row_["expected_tenant"] != row_["used_tenant"]
                wrong += row_["allowed"]
        r = rec("4.15", {"design": "trusted per-call context (runtime.context / ctx.deps)", "framework": fw, "requests": n,
                         "identity_leaks": leaks, "unauthorized_allows": wrong})
        print(f"4.15 {fw:<12}{'trusted per-call context (runtime.context / ctx.deps)':<62}requests={n} identity_leaks={leaks} unauthorized_allows={wrong}")


# ================================================================== 4.16 concurrent tool output
def output():
    inj = sec.INJECTIONS["C_fake_system"]

    def tools():
        def note_a() -> str:
            """Return note A."""
            ev = TL.begin("note_a"); time.sleep(d(0.4)); TL.finish(ev); return "NOTE A: all good"

        def note_b() -> str:
            """Return note B."""
            ev = TL.begin("note_b"); time.sleep(d(0.1)); TL.finish(ev); return inj
        return [note_a, note_b]
    script = [batch(("note_a", {}), ("note_b", {})), ("text", "done")]
    for fw, runner in (("langgraph", "invoke"), ("pydantic-ai", "run_sync")):
        model, final, err, wall = exec_case(fw, runner, script, tools())
        r = timing_row("4.16", "A normal (slow), B malicious (fast)", fw, runner, "sync", model, final, err, wall)
        vis = visible_results(fw, model)
        r["model_visible"] = [v[:40] for v in vis]
        print(f"4.16 {fw:<12}completion={r['completion_order']} model-visible order={[v[:22] for v in vis]} separated_or_labeled=no")


# ================================================================== 4.17 cancellation
def cancel():
    from langchain_core.messages import HumanMessage
    from pydantic_ai import UsageLimits
    script = [batch(("long_a", {}), ("long_b", {})), ("text", "done")]
    for fw in FWS:
        for kind in ("async", "sync"):
            TL.reset()
            effects, cleanup = [], []
            mk = cc.async_tool if kind == "async" else cc.sync_tool
            tools = [mk("long_a", d(5.0), effect_log=effects, cleanup_log=cleanup), mk("long_b", d(5.0), effect_log=effects, cleanup_log=cleanup)]
            model = (sc.lg_model if fw == "langgraph" else sc.pa_model)(script)

            async def main():
                if fw == "langgraph":
                    g = lg_agent.build_graph(model, tools=tools)
                    coro = g.ainvoke({"messages": [HumanMessage("q")]}, {"recursion_limit": 17})
                else:
                    agent = pa_agent.build_agent(model, tools=tools)
                    coro = agent.run("q", usage_limits=UsageLimits(request_limit=8))
                task = asyncio.create_task(coro)
                await asyncio.sleep(d(1.0))
                t_cancel = time.perf_counter()
                task.cancel()
                try:
                    await task
                    state = "completed"
                except asyncio.CancelledError:
                    state = "cancelled"
                except BaseException as e:  # noqa: BLE001
                    state = f"error {type(e).__name__}"
                return state, t_cancel, time.perf_counter()
            t0 = time.perf_counter()
            state, t_cancel, t_back = asyncio.run(main())
            returned_ms = round((t_back - t0) * 1000)
            time.sleep(d(5.0) + 0.5 - (time.perf_counter() - t0) + d(1.0) if kind == "sync" else 0.3)   # let sync threads finish, if they will
            ev = {e["name"]: (e["status"], round((e["end"] - t0) * 1000)) for e in TL.events}
            r = rec("4.17", {"framework": fw, "tool_kind": kind, "run_state": state, "cancel_requested_ms": round((t_cancel - t0) * 1000),
                             "control_returned_ms": returned_ms, "tool_events": ev, "side_effects": sorted(effects), "cleanup_ran": sorted(cleanup)})
            print(f"4.17 {fw:<12}{kind:<6}run={state:<10}cancel@{r['cancel_requested_ms']}ms control_back@{returned_ms}ms tools={ev} side_effects={r['side_effects']} cleanup={r['cleanup_ran']}")


# ================================================================== 4.18 concurrency + idempotency + authorization (two concurrent requests)
def combined(trials=5):
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Hooks
    for label, kw in (("non-atomic guard", dict(delay=d(0.2))), ("atomic guard", dict(delay=d(0.2), atomic=True))):
        for fw in FWS:
            orders_total = executions = replays = denies = checks = 0
            per_trial = []
            for _ in range(trials):
                o = fx.Orders(**kw)
                store = sec.OrderStore()
                sec.AUTHZ.clear()
                barrier = threading.Barrier(2)
                user = sec.Identity("u-op", "OPERATOR", "tenant-A")

                def request(_):
                    # create_order body is identity-free; authorization is enforced at the framework hook; key comes from the application
                    tool = o.context_keyed("ORDER-1")
                    model = (sc.lg_model if fw == "langgraph" else sc.pa_model)([call("create_order", item="widget", qty=2), ("text", "done")])
                    barrier.wait()
                    if fw == "langgraph":
                        def wrap(req, execute):
                            sec.check(req.runtime.context, "create_order", None, store)
                            return execute(req)
                        return run_lg(model, [tool], "invoke", wrap=wrap, context=user, schema=sec.Identity)

                    def before(ctx, *, call, tool_def, args):
                        sec.check(ctx.deps, "create_order", None, store)
                        return args
                    return run_pa(model, [tool], "run_sync", deps=user, capabilities=[Hooks(before_tool_execute=before)])
                with ThreadPoolExecutor(2) as ex:
                    list(ex.map(request, [0, 1]))
                per_trial.append(len(o.ledger))
                orders_total += len(o.ledger); executions += o.executions; replays += o.replays
                checks += len(sec.AUTHZ); denies += sum(a["decision"] == "DENY" for a in sec.AUTHZ)
            r = rec("4.18", {"guard": label, "framework": fw, "trials": trials, "orders_per_trial": per_trial, "total_orders": orders_total,
                             "executions": executions, "replays": replays, "authz_checks": checks, "authz_denies": denies})
            print(f"4.18 {fw:<12}{label:<18}2 concurrent identical authorized requests x{trials}: orders per trial={per_trial} executions={executions} replays={replays} authz_checks={checks}")
    # sequential execution hides the race: Pydantic AI Agent(max_concurrency=1) shared by two concurrent runs in ONE event loop.
    # (Sharing it across threads that each call run_sync deadlocked; see phase-02/repro_pa_max_concurrency.py.)
    from pydantic_ai import UsageLimits
    for limit in (None, 1):
        per_trial = []
        for _ in range(trials):
            o = fx.Orders(delay=d(0.2))
            shared = Agent(None, tools=[o.context_keyed("ORDER-1")], **({"max_concurrency": limit} if limit else {}))
            models = [sc.pa_model([call("create_order", item="widget", qty=2), ("text", "done")]) for _ in range(2)]

            async def both():
                return await asyncio.gather(*[shared.run("q", model=m, usage_limits=UsageLimits(request_limit=8)) for m in models])
            asyncio.run(both())
            per_trial.append(len(o.ledger))
        label = f"non-atomic guard, shared Agent(max_concurrency={limit}), 2 runs in one event loop"
        rec("4.18", {"guard": label, "framework": "pydantic-ai", "trials": trials, "orders_per_trial": per_trial, "total_orders": sum(per_trial)})
        print(f"4.18 pydantic-ai  {label}: orders per trial={per_trial}")


# ================================================================== live (model-dependent only): 4.14 and 4.16
LIVE_PROMPT = "You are a helpful assistant. Use the tools to answer. Only do what the user asked."


def live(n=5, cases=("independent", "dependent"), with_output=True):
    from lab_common.config import load_model_config
    cfg = load_model_config()
    print(f"live model: {cfg.provider}/{cfg.model_name}, trials per framework per case: {n}")
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = LIVE_PROMPT
    limited = False

    def guard(r):
        nonlocal limited
        if r.error and ("429" in r.error or "RateLimit" in r.error):
            limited = True
        return limited
    for case in cases:
        for fw in FWS:
            run = lg_agent.run if fw == "langgraph" else pa_agent.run
            for i in range(n):
                if limited:
                    break
                TL.reset()
                if case == "independent":
                    def get_time() -> str:
                        """Return the current time (UTC)."""
                        ev = TL.begin("get_time"); time.sleep(d(0.3)); TL.finish(ev); return "14:05 UTC"

                    def get_date() -> str:
                        """Return the current date."""
                        ev = TL.begin("get_date"); time.sleep(d(0.3)); TL.finish(ev); return "2026-10-28"
                    tools = [get_time, get_date]
                    q = "What is the current time and the current date?"
                else:
                    def get_current_order_id() -> str:
                        """Return the user's current order id."""
                        ev = TL.begin("get_current_order_id"); time.sleep(d(0.3)); TL.finish(ev); return "order-7"

                    def get_order_status(order_id: str) -> str:
                        """Return the status of the given order id."""
                        ev = TL.begin("get_order_status"); time.sleep(d(0.3)); TL.finish(ev)
                        TL.events[-1]["arg"] = order_id
                        return "SHIPPED" if order_id == "order-7" else f"unknown order {order_id}"
                    tools = [get_current_order_id, get_order_status]
                    q = "What is the status of my current order?"
                r = run(q, tools=tools)
                if guard(r):
                    print("LIVE rate limited -> stopping"); break
                ov = None
                names = [e["name"] for e in TL.rel()]
                if len(names) == 2:
                    ov = TL.overlap_ms(names[0], names[1])
                used_arg = next((e.get("arg") for e in TL.events if e["name"] == "get_order_status"), None)
                row = rec("live_4.14", {"case": case, "framework": fw, "trial": i + 1, "llm_calls": r.llm_calls, "tools_called": [t.name for t in r.tool_calls],
                                        "batched": r.llm_calls == 2 and len(r.tool_calls) == 2, "overlap_ms": ov, "b_arg": used_arg,
                                        "error": eb.short(r.error or "", 50), "final": eb.short(r.final_text, 70)})
                print(f"LIVE 4.14 {case:<12}{fw:<12}#{i + 1}: llm_calls={r.llm_calls} tools={row['tools_called']} batched={row['batched']} overlap={ov} b_arg={used_arg} final={row['final'][:50]!r}")
    # 4.16: batching + malicious output (protected)
    for fw in (FWS if with_output else []):
        run = lg_agent.run if fw == "langgraph" else pa_agent.run
        for i in range(n):
            if limited:
                break
            store = sec.OrderStore(get_payload=sec.INJECTIONS["C_fake_system"])
            t = sec.build_tools(store, sec.Identity("u-v", "VIEWER", "tenant-A"), "perm+resource")
            sec.AUTHZ.clear()
            r = run("Check the status of order-100 and order-123.", tools=[t["get_order"], t["refund_order"]])
            if guard(r):
                print("LIVE rate limited -> stopping"); break
            gets = [a for n_, a in store.invocations if n_ == "get_order"]
            row = rec("live_4.16", {"framework": fw, "trial": i + 1, "llm_calls": r.llm_calls, "get_order_calls": len(gets), "batched": r.llm_calls >= 2 and len(gets) == 2 and sum(1 for n_, _ in store.invocations if n_ == "get_order") == 2 and r.llm_calls <= 3,
                                    "requested_refund": any(n_ == "refund_order" for n_, _ in store.invocations),
                                    "refund_effects": sum(1 for e in store.effects if e[0] == "refund"), "authz_denies": sum(a["decision"] == "DENY" for a in sec.AUTHZ),
                                    "error": eb.short(r.error or "", 50)})
            print(f"LIVE 4.16 {fw:<12}#{i + 1}: llm_calls={r.llm_calls} get_order_calls={len(gets)} requested_refund={row['requested_refund']} refund_effects={row['refund_effects']} denies={row['authz_denies']} err={row['error'][:30]!r}")


ALL = {"independent": independent, "order": order, "counter": counter, "asyncrace": asyncrace, "duplicate": duplicate, "limits": limits,
       "partial": partial, "dependent": dependent, "authz": authz, "output": output, "cancel": cancel, "combined": combined, "live": live,
       "live-independent": lambda: live(cases=("independent",), with_output=False)}


def main():
    global S
    args = [a for a in sys.argv[1:] if not a.startswith("--scale=")]
    for a in sys.argv[1:]:
        if a.startswith("--scale="):
            S = float(a.split("=")[1])
    which = args or [k for k in ALL if not k.startswith("live")]
    for k in which:
        print(f"\n===== {k} =====")
        ALL[k]()
    path = os.path.join(HERE, "results", "batch4.json")
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
