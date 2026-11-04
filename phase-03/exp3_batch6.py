"""Phase 3.1 Batch 6: timeout, cancellation and token limits (B6-3.1.48 ... B6-3.1.58). Local and scripted.

Reuses: the Batch 3-5 harness (`e4`, `e3`, `b2`), the recorder (`Trajectory`), the neutral history view, the Phase 2 `Orders`
fixture, Python `asyncio` / `threading`, and the frameworks' own timeout, cancel and usage features. New code is measurement
only: a stdlib scripted HTTP server (real sockets, so real timeouts), tool probes that log their own start / effect / cancel /
finish, a driver that requests a stop when a named event appears, and a record builder. No deadline manager, no cancellation
helper, no retry layer, no token budgeter.

Counters are kept apart in every record: client wire requests, requests the server saw, framework steps, tool starts / ends /
cancels, effects, and the times the stop was requested, acknowledged and the work actually ended. Times are ms since the run began.
Hypotheses: phases/phase-03/experiments.md (Batch 6). Gap analysis and mechanism table: phases/phase-03/batch6-gap-analysis.md.

Usage: uv run python -u phase-03/exp3_batch6.py [group ...]
       groups: model sync async whole cancel tokens pending effect state retry deadline
"""
import asyncio
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_batch4 as e4  # noqa: E402  (sets sys.path; brings e3 / b1 / b2)

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab2 import fixtures as fx  # noqa: E402
from lab3.history import finish, lg_history, pa_history  # noqa: E402
from lab3.slow_server import SlowServer  # noqa: E402
from lab3.trajectory import Trajectory, lg_model, pa_model  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.runtime import RunControl  # noqa: E402
from pydantic_ai import CancellationToken, Tool, UsageLimits  # noqa: E402
from pydantic_ai.capabilities import Hooks  # noqa: E402

e3, C, FWS = e4.e3, e4.C, e4.FWS
RESULTS: dict = {}
Q = "Handle the request."


def call(name):
    return C(None, [(name, {})])


# ---------------------------------------------------------------- tool probes (log their own start, effect, cancel, finish)
class Probe:
    """Tools that write their own ground truth into the run's timeline, so what the framework reports can be compared with
    what the tool actually did. `ledger` is the tool-side record of committed effects."""

    def __init__(self, traj):
        self.traj, self.ledger = traj, []

    def commit(self, name):
        self.ledger.append(name)
        self.traj.add("effect", "tool", tool=name, thread=threading.get_ident())

    def sync(self, name="slow_sync", dur=1.0, effect="end"):
        probe = self

        def tool() -> str:
            probe.traj.add("tool_start", "tool", tool=name, args={}, thread=threading.get_ident(), flavor="sync")
            try:
                if effect == "start":
                    probe.commit(name)
                time.sleep(dur)
                if effect == "end":
                    probe.commit(name)
                probe.traj.add("tool_end", "tool", tool=name, result="done")
                return f"{name} done"
            finally:
                probe.traj.add("tool_finally", "tool", tool=name)
        tool.__name__, tool.__doc__ = name, f"Run {name}."
        return tool

    def asyn(self, name="slow_async", dur=1.0, effect="end", mode="normal"):
        """mode: normal (awaits, cancellable) | swallow (catches CancelledError and keeps going) | block (time.sleep in a coroutine)."""
        probe = self

        async def tool() -> str:
            probe.traj.add("tool_start", "tool", tool=name, args={}, thread=threading.get_ident(), flavor="async-" + mode)
            try:
                if effect == "start":
                    probe.commit(name)
                if mode == "block":
                    time.sleep(dur)
                elif mode == "swallow":
                    end = time.perf_counter() + dur
                    while time.perf_counter() < end:
                        try:
                            await asyncio.sleep(max(end - time.perf_counter(), 0))
                        except asyncio.CancelledError:
                            probe.traj.add("tool_cancel_swallowed", "tool", tool=name)
                else:
                    await asyncio.sleep(dur)
                if effect == "end":
                    probe.commit(name)
                probe.traj.add("tool_end", "tool", tool=name, result="done")
                return f"{name} done"
            except asyncio.CancelledError:
                probe.traj.add("tool_cancelled", "tool", tool=name)
                raise
            finally:
                probe.traj.add("tool_finally", "tool", tool=name)
        tool.__name__, tool.__doc__ = name, f"Run {name}."
        return tool


# ---------------------------------------------------------------- stop triggers (keyed to named events, not to sleeps)
def when_event(kind, n=None, tool=None):
    def cond(traj):
        return any(e["kind"] == kind and (n is None or e.get("n") == n) and (tool is None or e.get("tool") == tool) for e in list(traj.events))
    return cond


def after_ms(x):
    return lambda traj: (time.perf_counter() - traj.t0) * 1000 >= x


def after_event_ms(kind, ms, n=None, tool=None):
    """True once `ms` milliseconds have passed since the first matching event (a stop requested a moment AFTER it)."""
    def cond(traj):
        hit = next((e["t_ms"] for e in list(traj.events) if e["kind"] == kind and (n is None or e.get("n") == n) and (tool is None or e.get("tool") == tool)), None)
        return hit is not None and (time.perf_counter() - traj.t0) * 1000 >= hit + ms
    return cond


# ---------------------------------------------------------------- drivers (the frameworks' own async iteration APIs)
async def _drive_lg(q, model, tools, traj, holder, o):
    awrap = None
    if o["gate"]:
        async def awrap(request, execute):                            # a window between "tool requested" and "tool started"
            await asyncio.sleep(o["gate"])
            return await execute(request)
    graph = lg_agent.build_graph(model, handle_tool_errors=o["handle_tool_errors"], tools=tools, step_timeout=o["step_timeout"],
                                 retry_policy=o["retry_policy"], awrap_tool_call=awrap, async_model=o["async_model"])

    async def run():
        traj.add("run_started", "app")
        async for mode, chunk in graph.astream({"messages": [HumanMessage(q)]}, {"recursion_limit": o["limit"]},
                                               stream_mode=["debug", "values"], control=holder.get("control")):
            if mode == "values":
                holder["last"] = chunk
            elif chunk["type"] == "task_result":
                traj.add("framework_step", "framework", step=chunk["step"], label=f"node '{chunk['payload']['name']}' finished", state="")
        return holder.get("last")
    if o["app_deadline"]:
        async with asyncio.timeout(o["app_deadline"]):
            last = await run()
    else:
        last = await run()
    msgs = (last or {}).get("messages", [])
    return msgs[-1].content if msgs and not getattr(msgs[-1], "tool_calls", None) else None


async def _drive_pa(q, model, tools, traj, holder, o):
    wrapped = [Tool(t, timeout=o["tool_timeout"]) if o["tool_timeout"] else t for t in tools]
    caps = None
    if o["gate"]:
        async def before(ctx, *, call, tool_def, args):
            await asyncio.sleep(o["gate"])
            return args
        caps = [Hooks(before_tool_execute=before)]
    agent = pa_agent.build_agent(model, retries=o["retries"], tools=wrapped, capabilities=caps)
    limits = o["usage_limits"] or UsageLimits(request_limit=o["limit"])

    async def run():
        traj.add("run_started", "app")
        async with agent.iter(q, usage_limits=limits, timeout=o["run_timeout"], cancellation_token=holder.get("token"),
                              model_settings=o["model_settings"]) as r:
            holder["run"] = r
            i = 0
            async for node in r:
                i += 1
                traj.add("framework_step", "framework", step=i, label=f"about to run {type(node).__name__}", state="")
            return str(r.result.output)
    if o["app_deadline"]:
        async with asyncio.timeout(o["app_deadline"]):
            return await run()
    return await run()


DEFAULTS = dict(limit=25, step_timeout=None, run_timeout=None, app_deadline=None, usage_limits=None, tool_timeout=None, retry_policy=None,
                handle_tool_errors=True, gate=0.0, retries=3, model_settings=None, async_model=False, after_return=None)


def _ms(traj, t):
    return None if t is None else round((t - traj.t0) * 1000)


async def _scenario(fw, label, script, tools_f, *, delays, usages, statuses, mech, when, settle, model_kw, q, o):
    traj = Trajectory()
    probe = Probe(traj)
    tools = tools_f(probe)
    holder = {}
    with SlowServer(script, delays, usages, statuses) as srv:
        model = (lg_model if fw == "langgraph" else pa_model)(traj, base_url=srv.base_url, **(model_kw or {}))
        if mech == "drain":
            holder["control"] = RunControl()
        if mech == "token":
            holder["token"] = CancellationToken()
        out = {"exc": None}

        async def driver():
            try:
                out["final"] = await (_drive_lg if fw == "langgraph" else _drive_pa)(q, model, tools, traj, holder, o)
            except BaseException as e:  # noqa: BLE001  (CancelledError, TimeoutError, RunCancelled, UsageLimitExceeded, ...)
                out["exc"], out["exc_module"] = type(e).__name__, type(e).__module__.split(".")[0]
                out["exc_msg"] = str(e).splitlines()[0][:110] if str(e) else ""
                out["carries_messages"] = len(e.all_messages()) if hasattr(e, "all_messages") else None
                out["mro"] = [c.__name__ for c in type(e).__mro__[:4]]
            finally:
                traj.add("caller_returned", "app", exc=out["exc"])
        task = asyncio.create_task(driver())

        async def watcher():
            end = time.perf_counter() + 12
            while time.perf_counter() < end and not task.done():
                if when(traj):
                    if mech == "task":
                        task.cancel()
                    elif mech == "drain":
                        holder["control"].request_drain("batch6")
                    elif mech == "token":
                        holder["token"].cancel()
                    elif mech == "cancel":
                        while holder.get("run") is None and not task.done():
                            await asyncio.sleep(0.002)
                        if holder.get("run") is not None:
                            holder["run"].cancel()
                    traj.add("stop_requested", "app", mech=mech)
                    return
                await asyncio.sleep(0.003)
        w = asyncio.create_task(watcher()) if mech else None
        await task
        if o["after_return"]:                                             # the application acts the moment it gets the error
            await o["after_return"]()
        await asyncio.sleep(settle)                                       # keep watching: does work continue after the caller returned?
        if w:
            w.cancel()
        log = srv.settle(settle + 3)
    return _record(fw, label, mech, traj, probe, log, holder, out)


def _record(fw, label, mech, traj, probe, log, holder, out):
    ev = list(traj.events)
    first = lambda kind: next((e["t_ms"] for e in ev if e["kind"] == kind), None)  # noqa: E731
    stop_ms, ret_ms = first("stop_requested"), first("caller_returned")
    of = lambda kind: [e for e in ev if e["kind"] == kind]  # noqa: E731
    starts, ends, finals = of("tool_start"), of("tool_end"), of("tool_finally")
    effects, cancelled = of("effect"), of("tool_cancelled")
    if fw == "langgraph":
        msgs = (holder.get("last") or {}).get("messages", [])
        hist = lg_history(msgs)
        meta = [getattr(m, "usage_metadata", None) for m in msgs if type(m).__name__ == "AIMessage"]
        usage = {"source": "AIMessage.usage_metadata (read, never enforced)", "per_message_total": [(m or {}).get("total_tokens") for m in meta],
                 "total_tokens": sum((m or {}).get("total_tokens", 0) for m in meta)}
    else:
        run = holder.get("run")
        hist = finish(pa_history(run.ctx.state.message_history)) if run else []
        u = run.usage if run else None
        usage = {"requests": u.requests, "tool_calls": u.tool_calls, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "total_tokens": u.total_tokens} if u else None
    called = {c[0] for h in hist for c in h.get("calls", [])}
    answered = {h["result_for"] for h in hist if h["role"] == "tool"}
    rec = {
        "framework": fw, "case": label, "mech": mech, "exc": out["exc"], "exc_module": out.get("exc_module"), "exc_msg": out.get("exc_msg"),
        "exc_mro": out.get("mro"), "exc_carries_messages": out.get("carries_messages"), "final": out.get("final"),
        "stop_requested_ms": stop_ms, "caller_returned_ms": ret_ms, "ack_latency_ms": None if stop_ms is None or ret_ms is None else ret_ms - stop_ms,
        "first_request_ms": first("model_request"), "run_started_ms": first("run_started"),
        "run_ms": None if ret_ms is None or first("run_started") is None else round(ret_ms - first("run_started")),
        "client_wire_requests": len(of("model_request")), "client_wire_responses_ok": sum(1 for e in of("model_response") if e.get("status") == 200),
        "calls_requested": sum(len(e["tool_calls"]) for e in of("model_response")),
        "framework_steps": len(of("framework_step")),
        "server_requests": len(log), "server_requests_peer_closed": sum(1 for r in log if r["peer_closed_at"] is not None),
        "server_work_finished_after_return": sum(1 for r in log if ret_ms is not None and _ms(traj, r["work_done_at"]) > ret_ms),
        "server_responses_delivered": sum(1 for r in log if r["responded"]),
        "server_log": [{"n": r["n"], "status": r.get("status"), "arrived": _ms(traj, r["arrived"]), "peer_closed": _ms(traj, r["peer_closed_at"]), "work_done": _ms(traj, r["work_done_at"]),
                        "responded": r["responded"], "max_tokens": r["max_tokens"],
                        "max_completion_tokens": r["max_completion_tokens"]} for r in log],
        "tool_starts": len(starts), "tool_ends": len(ends), "tool_cancelled": len(cancelled), "tool_finally": len(finals),
        "tool_cancel_swallowed": len(of("tool_cancel_swallowed")),
        "tool_running_at_return": bool(ret_ms is not None and any(s["t_ms"] <= ret_ms for s in starts) and
                                       len(starts) > sum(1 for f in finals if f["t_ms"] <= ret_ms)),
        "tool_end_ms": [e["t_ms"] for e in ends], "tool_end_after_return": sum(1 for e in ends if ret_ms is not None and e["t_ms"] > ret_ms),
        "effects": len(effects), "effect_ms": [e["t_ms"] for e in effects], "effects_after_return": sum(1 for e in effects if ret_ms is not None and e["t_ms"] > ret_ms),
        "ledger": list(probe.ledger),
        "state_history": [(h["role"][0], (h["text"] or "")[:22], h.get("outcome")) if h["role"] != "assistant" else ("a", [c[1] for c in h.get("calls", [])])
                          for h in hist],
        "state_pending_calls": len(called - answered), "state_tool_results": sum(1 for h in hist if h["role"] == "tool"), "usage": usage,
        "timeline": [(e["t_ms"], e["kind"]) for e in ev if e["kind"] in ("stop_requested", "caller_returned", "tool_start", "tool_end", "tool_cancelled",
                                                                           "tool_finally", "effect", "model_request", "model_response")],   # swallowed cancels are counted, not listed
    }
    return rec


def scenario(exp, fw, label, script, tools_f, *, delays=None, usages=None, statuses=None, mech=None, when=None, settle=1.3, model_kw=None, q=Q, **opts):
    o = {**DEFAULTS, **opts}
    assert set(opts) <= set(DEFAULTS), set(opts) - set(DEFAULTS)
    rec = asyncio.run(_scenario(fw, label, script, tools_f, delays=delays, usages=usages, statuses=statuses, mech=mech, when=when, settle=settle,
                                model_kw=model_kw, q=q, o=o))
    rec["exp"] = exp
    rec["variant"] = ("async-node" if o["async_model"] else "sync-node (Phase 1 graph)") if fw == "langgraph" else "-"
    RESULTS.setdefault(exp, []).append(rec)
    short = lambda x: "-" if x is None else x  # noqa: E731
    print(f"{exp:<10}{fw[:2]}{'*' if o['async_model'] else ' '}{label[:46]:<47}{(mech or '-'):<7} stop@{short(rec['stop_requested_ms'])!s:<5} ret@{short(rec['caller_returned_ms'])!s:<5} "
          f"exc={rec['exc'] or '-':<22} srv={rec['server_requests']} tools s/e/c={rec['tool_starts']}/{rec['tool_ends']}/{rec['tool_cancelled']} "
          f"running@ret={rec['tool_running_at_return']!s:<5} fx={rec['effects']}(+{rec['effects_after_return']} after) pend={rec['state_pending_calls']}")
    return rec


noop = lambda probe: [probe.sync("noop", 0.0, effect=None)]  # noqa: E731


# ================================================================ B6-3.1.48 model request timeout
def model():
    slow, ok = [1.5] * 4, [C("The answer is 4.")]
    for fw in FWS:
        kw = (lambda **k: k)
        base = dict(timeout=0.4)
        scenario("B6-3.1.48", fw, "timeout=0.4, every attempt slow, SDK default retries", ok, noop, delays=slow, model_kw=kw(**base), settle=0.2)
        scenario("B6-3.1.48", fw, "timeout=0.4, every attempt slow, max_retries=0", ok, noop, delays=slow, model_kw=kw(**base, max_retries=0), settle=0.2)
        scenario("B6-3.1.48", fw, "timeout=0.4, first attempt slow, retry fast", ok, noop, delays=[1.5, 0], model_kw=kw(**base), settle=0.2)
    scenario("B6-3.1.48", "pydantic-ai", "request_timeout=0.7 over timeout=0.4 (SDK retries)", ok, noop, delays=slow, model_kw=dict(timeout=0.4, request_timeout=0.7), settle=0.2)
    scenario("B6-3.1.48", "pydantic-ai", "request_timeout=3.0 over timeout=0.4, all slow", ok, noop, delays=slow, model_kw=dict(timeout=0.4, request_timeout=3.0), settle=0.2)


# ================================================================ B6-3.1.49 synchronous tool timeout / stop
def sync_tool():
    script = [call("slow_sync"), C("Done.")]
    tools = lambda p: [p.sync("slow_sync", 1.0, "end")]  # noqa: E731  (1.0 s of work, effect at the end)
    started = when_event("tool_start")
    scenario("B6-3.1.49", "pydantic-ai", "Tool(timeout=0.3)", script, tools, tool_timeout=0.3, settle=1.6)
    scenario("B6-3.1.49", "pydantic-ai", "run(timeout=0.3)", script, tools, run_timeout=0.3, settle=1.6)
    scenario("B6-3.1.49", "langgraph", "step_timeout=0.3", script, tools, step_timeout=0.3, settle=1.6)
    for fw in FWS:
        scenario("B6-3.1.49", fw, "asyncio.timeout(0.3) around the run", script, tools, app_deadline=0.3, settle=1.6)
    scenario("B6-3.1.49", "pydantic-ai", "AgentRun.cancel() at tool start", script, tools, mech="cancel", when=after_event_ms("tool_start", 100), settle=1.6)
    for fw in FWS:
        scenario("B6-3.1.49", fw, "Task.cancel() at tool start", script, tools, mech="task", when=after_event_ms("tool_start", 100), settle=1.6)
    scenario("B6-3.1.49", "langgraph", "RunControl.request_drain() at tool start", script, tools, mech="drain", when=after_event_ms("tool_start", 100), settle=1.6)
    _ = started


# ================================================================ B6-3.1.50 asynchronous tool timeout / stop
def async_tool():
    script = [call("slow_async"), C("Done.")]
    for effect in ("start", "end"):
        tools = lambda p, e=effect: [p.asyn("slow_async", 1.0, e)]  # noqa: E731
        scenario("B6-3.1.50", "pydantic-ai", f"Tool(timeout=0.3), effect at {effect}", script, tools, tool_timeout=0.3, settle=1.4)
        scenario("B6-3.1.50", "langgraph", f"step_timeout=0.3, effect at {effect}", script, tools, step_timeout=0.3, settle=1.4)
    for mode, dur in (("swallow", 1.0), ("block", 0.8)):
        tools = lambda p, m=mode, d=dur: [p.asyn("slow_async", d, "end", mode=m)]  # noqa: E731
        scenario("B6-3.1.50", "pydantic-ai", f"Tool(timeout=0.3), {mode} tool", script, tools, tool_timeout=0.3, settle=1.4)
        scenario("B6-3.1.50", "langgraph", f"step_timeout=0.3, {mode} tool", script, tools, step_timeout=0.3, settle=1.4)
    tools = lambda p: [p.asyn("slow_async", 1.0, "end")]  # noqa: E731
    scenario("B6-3.1.50", "pydantic-ai", "run(timeout=0.3), normal tool", script, tools, run_timeout=0.3, settle=1.4)
    scenario("B6-3.1.50", "langgraph", "asyncio.timeout(0.3), normal tool", script, tools, app_deadline=0.3, settle=1.4)


# ================================================================ B6-3.1.51 whole-run timeout (deadline 0.6 s)
def whole_run():
    dl = 0.6
    where = [("model request in flight", [C("ok")], noop, [1.5]),
             ("async tool in flight", [call("work"), C("Done.")], lambda p: [p.asyn("work", 1.5, "end")], None),
             ("sync tool in flight", [call("work"), C("Done.")], lambda p: [p.sync("work", 1.5, "end")], None),
             ("2nd model request in flight", [call("work"), C("Done.")], lambda p: [p.asyn("work", 0.05, "end")], [0, 1.5])]
    for label, script, tools, delays in where:
        scenario("B6-3.1.51", "pydantic-ai", f"{label}: run(timeout={dl})", script, tools, delays=delays, run_timeout=dl, settle=1.6)
        scenario("B6-3.1.51", "pydantic-ai", f"{label}: asyncio.timeout({dl})", script, tools, delays=delays, app_deadline=dl, settle=1.6)
        scenario("B6-3.1.51", "langgraph", f"{label}: asyncio.timeout({dl})", script, tools, delays=delays, app_deadline=dl, settle=1.6)


# ================================================================ B6-3.1.52 cancellation propagation (points A-G x four mechanisms)
def cancel_matrix():
    ONE_ROUND = [call("work"), C("Done.")]
    points = [
        ("A before the request reaches the wire", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], dict(delays=None, model_kw=dict(presend=0.5), when=after_event_ms("run_started", 150))),
        ("B during the first model request", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], dict(delays=[1.5, 0], when=after_event_ms("model_request", 100, n=1))),
        ("C requested tool, before it starts", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], dict(gate=0.6, when=after_event_ms("model_response", 50, n=1))),
        ("D during a synchronous tool", ONE_ROUND, lambda p: [p.sync("work", 1.0)], dict(when=after_event_ms("tool_start", 100))),
        ("E during an asynchronous tool", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], dict(when=after_event_ms("tool_start", 100))),
        ("F just after the tool completed", ONE_ROUND, lambda p: [p.asyn("work", 0.2)], dict(delays=[0, 1.0], when=when_event("tool_end"))),
        ("G during the final model response", ONE_ROUND, lambda p: [p.asyn("work", 0.05)], dict(delays=[0, 1.5], when=after_event_ms("model_request", 100, n=2))),
    ]
    mechs = {"langgraph": ["task", "drain"], "pydantic-ai": ["task", "cancel"]}
    for label, script, tools, kw in points:
        for fw in FWS:
            for mech in mechs[fw]:
                scenario("B6-3.1.52", fw, label, script, tools, mech=mech, settle=1.3, **kw)
        if label[0] in "ABCG":                                         # the model-request points: does a coroutine node change anything?
            for mech in mechs["langgraph"]:
                scenario("B6-3.1.52", "langgraph", label, script, tools, mech=mech, settle=1.3, async_model=True, **kw)
    # a sync API cancelled from another thread: Pydantic AI only (CancellationToken); LangGraph's equivalent is the drain
    _token_sync_api()


def _token_sync_api():
    traj = Trajectory()
    probe = Probe(traj)
    tools = [probe.asyn("work", 1.5)]
    token = CancellationToken()
    with SlowServer([call("work"), C("Done.")]) as srv:
        agent = pa_agent.build_agent(pa_model(traj, base_url=srv.base_url), retries=3, tools=tools)
        threading.Timer(0.6, token.cancel).start()
        t0 = time.perf_counter()
        exc = None
        try:
            agent.run_sync(Q, cancellation_token=token)
        except BaseException as e:  # noqa: BLE001
            exc = type(e).__name__
        dt = round((time.perf_counter() - t0) * 1000)
    rec = {"exp": "B6-3.1.52", "framework": "pydantic-ai", "case": "run_sync() cancelled from another thread via CancellationToken", "mech": "token",
           "exc": exc, "returned_after_ms": dt, "tool_cancelled": sum(1 for e in traj.events if e["kind"] == "tool_cancelled")}
    RESULTS.setdefault("B6-3.1.52", []).append(rec)
    print(f"B6-3.1.52 py run_sync + CancellationToken from a thread: exc={exc} returned after {dt} ms, tool cancelled={rec['tool_cancelled']}")


# ================================================================ B6-3.1.53 token and usage limits
def tokens():
    quick = lambda p: [p.asyn("t", 0.05, "end")]  # noqa: E731
    script = [call("t"), C("The answer is 4.")]
    usages = [(60, 20), (100, 30)]                    # response 1 = 80 tokens; response 2 = 130 tokens; cumulative 210
    limits = [("total_tokens_limit=50 (crossed by the tool-call response)", UsageLimits(total_tokens_limit=50)),
              ("total_tokens_limit=100 (crossed by the final answer)", UsageLimits(total_tokens_limit=100)),
              ("input_tokens_limit=150 (cumulative input 60, then 160)", UsageLimits(input_tokens_limit=150)),
              ("output_tokens_limit=40 (cumulative output 20, then 50)", UsageLimits(output_tokens_limit=40)),
              ("per_request_input_tokens_limit=90 (2nd request input is 100)", UsageLimits(per_request_input_tokens_limit=90)),
              ("request_limit=1 (reference)", UsageLimits(request_limit=1)),
              ("count_tokens_before_request + per_request limit 90", UsageLimits(count_tokens_before_request=True, per_request_input_tokens_limit=90))]
    for label, ul in limits:
        scenario("B6-3.1.53", "pydantic-ai", label, script, quick, usages=usages, usage_limits=ul, settle=0.2)
    scenario("B6-3.1.53", "langgraph", "no limit exists; same script and usage", script, quick, usages=usages, settle=0.2)
    for fw in FWS:
        scenario("B6-3.1.53", fw, "context window exceeded on request 2 (HTTP 400)", script, quick, usages=usages, statuses=[200, 400], settle=0.2)
        scenario("B6-3.1.53", fw, "HTTP 500 on the first attempt, SDK retries", script, quick, usages=usages, statuses=[500], settle=0.2)
        scenario("B6-3.1.53", fw, "max_tokens=25 set on the model", script, quick, usages=usages, model_kw=dict(max_tokens=25), settle=0.2)


# ================================================================ B6-3.1.54 pending tool call at a limit
def pending():
    endless = [call("work")] * 40
    quick = lambda p: [p.asyn("work", 0.05, "end")]  # noqa: E731
    scenario("B6-3.1.54", "langgraph", "recursion_limit=3 (odd)", endless, quick, limit=3, settle=0.3)
    scenario("B6-3.1.54", "langgraph", "recursion_limit=4 (even)", endless, quick, limit=4, settle=0.3)
    scenario("B6-3.1.54", "pydantic-ai", "request_limit=2", endless, quick, limit=2, settle=0.3)
    scenario("B6-3.1.54", "pydantic-ai", "tool_calls_limit=1", endless, quick, usage_limits=UsageLimits(tool_calls_limit=1), settle=0.3)
    scenario("B6-3.1.54", "pydantic-ai", "total_tokens_limit=50 crossed by the tool-call response", endless, quick, usages=[(60, 20)] * 40,
             usage_limits=UsageLimits(total_tokens_limit=50), settle=0.3)
    scenario("B6-3.1.54", "langgraph", "drain requested as the tool call arrives", endless, quick, mech="drain", when=after_event_ms("model_response", 5, n=1), settle=0.5)
    scenario("B6-3.1.54", "pydantic-ai", "cancel() while the tool is gated, not started", endless, lambda p: [p.asyn("work", 1.0, "end")], mech="cancel", gate=0.6,
             when=after_event_ms("model_response", 50, n=1), settle=1.0)
    scenario("B6-3.1.54", "pydantic-ai", "run(timeout=0.3) with a running sync tool", [call("work"), C("Done.")], lambda p: [p.sync("work", 1.0, "end")], run_timeout=0.3, settle=1.4)
    rows = RESULTS["B6-3.1.54"]
    for r in rows:
        r["started_without_result_in_state"] = r["tool_starts"] - r["state_tool_results"]
    ambiguous = {(r["state_pending_calls"] >= 1, r["tool_starts"] > r["state_tool_results"]) for r in rows if r["state_pending_calls"] >= 1}
    RESULTS["B6-3.1.54-summary"] = {"pending_call_in_state_but_started": sum(1 for r in rows if r["state_pending_calls"] >= 1 and r["tool_starts"] > r["state_tool_results"]),
                                    "pending_call_in_state_and_never_started": sum(1 for r in rows if r["state_pending_calls"] >= 1 and r["tool_starts"] == r["state_tool_results"]),
                                    "state_alone_can_distinguish": len(ambiguous) < 2}
    print("   pending-in-state summary:", RESULTS["B6-3.1.54-summary"])


# ================================================================ B6-3.1.55 cancellation after a business effect
KEY = "req-77"


def _order_tools(box, kind, pre, post, delay, atomic=False):
    def tools(p):
        o = fx.Orders(delay=delay, atomic=atomic)
        box["o"] = o
        if kind == "async":
            async def create_order(item: str, qty: int) -> str:
                """Create an order."""
                p.traj.add("tool_start", "tool", tool="create_order", args={}, thread=threading.get_ident())
                try:
                    await asyncio.sleep(pre)
                    r = o._execute(item, qty, KEY)
                    p.traj.add("effect", "tool", tool="create_order")
                    await asyncio.sleep(post)
                    p.traj.add("tool_end", "tool", tool="create_order", result=r)
                    return r
                except asyncio.CancelledError:
                    p.traj.add("tool_cancelled", "tool", tool="create_order")
                    raise
                finally:
                    p.traj.add("tool_finally", "tool", tool="create_order")
        else:
            def create_order(item: str, qty: int) -> str:
                """Create an order."""
                p.traj.add("tool_start", "tool", tool="create_order", args={}, thread=threading.get_ident())
                try:
                    r = o._execute(item, qty, KEY)                    # sleeps `delay` BEFORE the ledger append
                    p.traj.add("effect", "tool", tool="create_order")
                    time.sleep(post)
                    p.traj.add("tool_end", "tool", tool="create_order", result=r)
                    return r
                finally:
                    p.traj.add("tool_finally", "tool", tool="create_order")
        return [create_order]
    return tools


def effect_after_cancel():
    script = [C(None, [("create_order", {"item": "widget", "qty": 1})]), C("Order placed.")]
    cases = [("async: effect committed, response delayed", "async", 0.0, 1.5, 0.0, after_event_ms("effect", 50)),
             ("async: cancelled before the commit", "async", 1.0, 0.0, 0.0, after_event_ms("tool_start", 100)),
             ("sync: cancelled during the pre-commit delay", "sync", 0.0, 0.0, 0.8, after_event_ms("tool_start", 100)),
             ("sync: effect committed, response delayed", "sync", 0.0, 1.5, 0.0, after_event_ms("effect", 50))]
    for label, kind, pre, post, delay, when in cases:
        for fw, mech in (("pydantic-ai", "cancel"), ("langgraph", "task")):
            box = {}
            rec = scenario("B6-3.1.55", fw, label, script, _order_tools(box, kind, pre, post, delay), mech=mech, when=when, settle=1.8)
            o = box["o"]
            rec["ledger_at_end"], rec["key_recorded"], rec["replays_before_retry"] = len(o.ledger), KEY in o.seen, o.replays
            before = len(o.ledger)
            o._execute("widget", 1, KEY)                                # the application retries through the business layer with the same key
            rec["keyed_retry_added_orders"] = len(o.ledger) - before
            before = len(o.ledger)
            o._execute("widget", 1, None)                               # the same retry without a key
            rec["unkeyed_retry_added_orders"] = len(o.ledger) - before
            print(f"   ledger at end={rec['ledger_at_end']} key recorded={rec['key_recorded']} keyed retry adds {rec['keyed_retry_added_orders']}, unkeyed adds {rec['unkeyed_retry_added_orders']}")
    # the application retries the moment it receives the error, while a cancelled synchronous tool is still inside its pre-commit delay
    for atomic in (False, True):
        for fw, mech in (("pydantic-ai", "cancel"), ("langgraph", "task")):
            box = {}

            async def retry_now(box=box):
                await asyncio.to_thread(box["o"]._execute, "widget", 1, KEY)

            rec = scenario("B6-3.1.55", fw, f"sync pre-commit cancel, IMMEDIATE keyed retry, atomic guard={atomic}", script,
                           _order_tools(box, "sync", 0.0, 0.0, 0.8, atomic=atomic), mech=mech, when=after_event_ms("tool_start", 100),
                           after_return=retry_now, settle=1.8)
            o = box["o"]
            rec["ledger_at_end"], rec["replays"], rec["immediate_retry"], rec["atomic_guard"] = len(o.ledger), o.replays, True, atomic
            print(f"   immediate keyed retry, atomic={atomic}: ledger at end={rec['ledger_at_end']} replays={rec['replays']}")


# ================================================================ B6-3.1.56 partial state after termination
async def _plain(fw, how):
    """The plain call paths (no stream / iter): what does the caller's exception carry?"""
    traj = Trajectory()
    probe = Probe(traj)
    endless = [call("t")] * 40
    tool = probe.asyn("t", 0.05, "end") if how != "run timeout" else probe.asyn("t", 1.0, "end")
    kw = dict(delays=None, usages=None, statuses=None)
    if how == "model error":
        kw["statuses"] = [200, 400]
    if how == "usage limit":
        kw["usages"] = [(60, 20)] * 40
    holder = {}
    with SlowServer(endless if how != "model error" else [call("t"), C("ok")], **kw) as srv:
        model = (lg_model if fw == "langgraph" else pa_model)(traj, base_url=srv.base_url)
        exc = None
        try:
            if fw == "langgraph":
                graph = lg_agent.build_graph(model, tools=[tool], async_model=True)
                control = RunControl() if how == "cancellation" else None
                if control:
                    threading.Timer(0.5, control.request_drain).start()
                if how == "run timeout":
                    async with asyncio.timeout(0.4):
                        await graph.ainvoke({"messages": [HumanMessage(Q)]}, {"recursion_limit": 25})
                else:
                    await graph.ainvoke({"messages": [HumanMessage(Q)]}, {"recursion_limit": 3 if how == "loop limit" else 25}, control=control)
            else:
                token = CancellationToken() if how == "cancellation" else None
                if token:
                    threading.Timer(0.5, token.cancel).start()
                limits = {"loop limit": UsageLimits(request_limit=2), "usage limit": UsageLimits(total_tokens_limit=50)}.get(how, UsageLimits(request_limit=50))
                agent = pa_agent.build_agent(model, retries=3, tools=[probe.asyn("t", 1.5, "end")] if how == "cancellation" else [tool])
                await agent.run(Q, usage_limits=limits, timeout=0.4 if how == "run timeout" else None, cancellation_token=token)
        except BaseException as e:  # noqa: BLE001
            exc = e
    carried = len(exc.all_messages()) if hasattr(exc, "all_messages") else None
    return {"exp": "B6-3.1.56-plain", "framework": fw, "termination": how, "api": "ainvoke" if fw == "langgraph" else "run", "exc": type(exc).__name__ if exc else None,
            "exc_module": type(exc).__module__.split(".")[0] if exc else None, "exception_carries_messages": carried,
            "exception_attrs_with_state": [a for a in dir(exc) if not a.startswith("_") and a in ("all_messages", "messages", "state", "values", "usage", "history")]}


def state_after_termination():
    terms = ["loop limit", "run timeout", "cancellation", "usage limit", "model error"]
    for how in terms:
        for fw in FWS:
            if how == "usage limit" and fw == "langgraph":
                continue                                                  # LangGraph has no usage limit
            rec = asyncio.run(_plain(fw, how))
            RESULTS.setdefault("B6-3.1.56-plain", []).append(rec)
            print(f"B6-3.1.56 plain {fw[:2]} {how:<13} via {rec['api']:<7} exc={rec['exc']:<22} carries messages={rec['exception_carries_messages']} state attrs={rec['exception_attrs_with_state']}")
    # the streamed / iterated paths, same terminations
    endless = [call("t")] * 40
    quick = lambda p: [p.asyn("t", 0.05, "end")]  # noqa: E731
    slow = lambda p: [p.asyn("t", 1.0, "end")]  # noqa: E731
    scenario("B6-3.1.56", "langgraph", "loop limit via astream", endless, quick, limit=3, settle=0.2)
    scenario("B6-3.1.56", "pydantic-ai", "loop limit via iter", endless, quick, limit=2, settle=0.2)
    scenario("B6-3.1.56", "langgraph", "run timeout via astream", endless, slow, app_deadline=0.4, settle=0.5)
    scenario("B6-3.1.56", "pydantic-ai", "run timeout via iter (run_timeout)", endless, slow, run_timeout=0.4, settle=0.5)
    scenario("B6-3.1.56", "langgraph", "cancellation (drain) via astream", endless, slow, mech="drain", when=after_event_ms("tool_start", 100), settle=0.5)
    scenario("B6-3.1.56", "pydantic-ai", "cancellation via iter (cancel)", endless, slow, mech="cancel", when=after_event_ms("tool_start", 100), settle=0.5)
    scenario("B6-3.1.56", "pydantic-ai", "usage limit via iter", endless, quick, usages=[(60, 20)] * 40, usage_limits=UsageLimits(total_tokens_limit=50), settle=0.2)
    for fw in FWS:
        scenario("B6-3.1.56", fw, "provider error (HTTP 400 on request 2)", [call("t"), C("ok")], quick, statuses=[200, 400], settle=0.2)


# ================================================================ B6-3.1.57 timeout x retry
def timeout_retry():
    from langgraph.types import RetryPolicy
    sync_work = lambda p: [p.sync("slow_sync", 1.0, "end")]  # noqa: E731
    scenario("B6-3.1.57", "pydantic-ai", "Tool(timeout=0.3): the model calls again while the first still runs",
             [call("slow_sync"), call("slow_sync"), C("Done.")], sync_work, tool_timeout=0.3, settle=1.9)
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=2, retry_on=Exception)
    scenario("B6-3.1.57", "langgraph", "step_timeout=0.3 with RetryPolicy(max_attempts=2)", [call("slow_sync"), C("Done.")], sync_work,
             step_timeout=0.3, retry_policy=rp, handle_tool_errors=False, settle=1.6)

    def guarded(p):
        work = p.sync("work", 1.0, "end")

        def guarded_tool() -> str:
            """Run work, giving up after 0.3 s (a thread-based tool-level timeout)."""
            t = threading.Thread(target=work, daemon=True)
            t.start()
            t.join(0.3)
            if t.is_alive():
                raise TimeoutError("work timed out after 0.3 s")
            return "done"
        return [guarded_tool]
    rp2 = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=2, retry_on=TimeoutError)
    scenario("B6-3.1.57", "langgraph", "tool-level timeout raising TimeoutError + RetryPolicy", [call("guarded_tool"), C("Done.")], guarded,
             retry_policy=rp2, handle_tool_errors=False, settle=1.9)
    for rec in RESULTS["B6-3.1.57"]:
        starts = [t for t, k in rec["timeline"] if k == "tool_start"]
        ends = [t for t, k in rec["timeline"] if k == "tool_end"]
        rec["executions_overlapped"] = len(starts) >= 2 and starts[1] < min(ends or [10 ** 9])
        print(f"   {rec['framework'][:2]} {rec['case'][:40]}: tool starts={starts} ends={ends} overlapped={rec['executions_overlapped']}")


# ================================================================ B6-3.1.58 cumulative deadline
def cumulative_deadline():
    three = [call("work")] * 3 + [C("All three done.")]
    work = lambda p: [p.asyn("work", 0.3, "end")]  # noqa: E731
    d = [0.3] * 4
    scenario("B6-3.1.58", "langgraph", "step_timeout=0.5 only (no deadline)", three, work, delays=d, step_timeout=0.5, settle=0.3)
    scenario("B6-3.1.58", "pydantic-ai", "timeout/request_timeout/tool timeout 0.5 each (no deadline)", three, work, delays=d,
             model_kw=dict(timeout=0.5, request_timeout=0.5), tool_timeout=0.5, settle=0.3)
    scenario("B6-3.1.58", "pydantic-ai", "run(timeout=1.0)", three, work, delays=d, run_timeout=1.0, settle=0.8)
    scenario("B6-3.1.58", "pydantic-ai", "asyncio.timeout(1.0) around the run", three, work, delays=d, app_deadline=1.0, settle=0.8)
    scenario("B6-3.1.58", "langgraph", "asyncio.timeout(1.0) around the run", three, work, delays=d, app_deadline=1.0, settle=0.8)


GROUPS = {"model": model, "sync": sync_tool, "async": async_tool, "whole": whole_run, "cancel": cancel_matrix, "tokens": tokens, "pending": pending,
          "effect": effect_after_cancel, "state": state_after_termination, "retry": timeout_retry, "deadline": cumulative_deadline}


def main():
    names = sys.argv[1:] or list(GROUPS)
    for g in names:
        print(f"\n===== {g} =====")
        GROUPS[g]()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "batch6.json")
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except FileNotFoundError:
        old = {}
    old.update(RESULTS)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1, default=str)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
