"""Phase 3.1 Batch 3: iteration limits, retries and loop budget (B3-3.1.11 … B3-3.1.22). Offline (scripted).

Reuses: the Phase 1 graph/agent builders (`lg_agent.build_graph`, `pa_agent.build_agent`), the shared calculator
tools, Phase 2 failure fixtures (`lab2.fixtures.make_flaky`, `Orders`), the Batch 2 scripted transport
(`raw_transport`, `C`) and the Batch 1 recorder. The runners below differ from Batch 1's only in taking the limit and
retry knobs as arguments and in keeping the partial state when a limit exception ends the run.
The only new fixture is `http_flaky`: the same scripted transport, but the first k HTTP attempts return an error status,
so the provider client's own retry becomes visible on the wire.

Hypotheses: phases/phase-03/experiments.md (Batch 3).
Usage: uv run python -u phase-03/exp3_batch3.py [group ...]   groups: limits empty tools http warn effects
"""
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_batch1 as b1  # noqa: E402  (sets sys.path)
import exp3_batch2 as b2  # noqa: E402

import httpx  # noqa: E402

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab2 import fixtures as fx  # noqa: E402
from lab3.trajectory import Trajectory, traced  # noqa: E402
from lab_common.tools import TOOLS  # noqa: E402

C, MUL = b2.C, b2.MUL
ADD = ("add", {"a": 120, "b": 45})
MUL2 = ("multiply", {"a": 165, "b": 3})
FWS = b1.FWS
RESULTS: dict = {}


# ---------------------------------------------------------------- transports
def http_flaky(script, fail=1, status=500):
    """Scripted completions; the first `fail` HTTP attempts get `status` (retry-after-ms: 1 keeps it fast)."""
    inner, log = b2.raw_transport(script)
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] <= fail:
            return httpx.Response(status, json={"error": {"message": f"simulated transient {status}"}}, headers={"retry-after-ms": "1"})
        return inner.handle_request(request)
    return httpx.MockTransport(handler), log, attempts


# ---------------------------------------------------------------- runners (framework iteration APIs, partial state kept)
def run_lg(q, model, tools, traj, recursion_limit, handle_tool_errors=True, retry_policy=None):
    from langchain_core.messages import AIMessage, HumanMessage
    graph = lg_agent.build_graph(model, handle_tool_errors=handle_tool_errors, tools=tools, retry_policy=retry_policy)
    last, err = None, None
    try:
        for mode, chunk in graph.stream({"messages": [HumanMessage(q)]}, {"recursion_limit": recursion_limit}, stream_mode=["debug", "values"]):
            if mode == "values":
                last = chunk
            elif chunk["type"] == "task_result":
                traj.add("framework_step", "framework", step=chunk["step"], label=f"node '{chunk['payload']['name']}' finished", state="")
    except Exception as e:  # noqa: BLE001  (GraphRecursionError, escaped tool error, ...)
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    msgs = (last or {}).get("messages", [])
    ai = [m for m in msgs if isinstance(m, AIMessage)]
    return {"error": err, "framework_model_calls": len(ai), "state": [type(m).__name__ for m in msgs],
            "final_text": ai[-1].content if ai and not ai[-1].tool_calls and err is None else None,
            "pending_tool_calls_in_state": len(ai[-1].tool_calls) if ai and ai[-1].tool_calls and (not msgs or msgs[-1] is ai[-1]) else 0}


def run_pa(q, model, tools, traj, request_limit, tool_calls_limit=None, retries=1):
    from pydantic_ai import UsageLimits
    agent = pa_agent.build_agent(model, retries=retries, tools=tools)
    out = {}

    async def go():
        async with agent.iter(q, usage_limits=UsageLimits(request_limit=request_limit, tool_calls_limit=tool_calls_limit)) as run:
            i = 0
            try:
                async for node in run:
                    i += 1
                    traj.add("framework_step", "framework", step=i, label=f"about to run {type(node).__name__}",
                             state=f"run_step={run.ctx.state.run_step} requests={run.usage.requests}")
                out["error"] = None
                out["final_text"] = str(run.result.output)
            except Exception as e:  # noqa: BLE001  (UsageLimitExceeded, UnexpectedModelBehavior, ...)
                out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                out["final_text"] = None
            st = run.ctx.state
            out.update({"framework_model_calls": run.usage.requests, "usage_tool_calls": run.usage.tool_calls, "run_step": st.run_step,
                        "state": [type(m).__name__ for m in st.message_history], "output_retries_used": getattr(st, "output_retries_used", None),
                        "retries_state": {k: v for k, v in vars(st).items() if "retr" in k.lower()}})
    asyncio.run(go())
    return out


def run(fw, q, script, limit, tools=None, transport=None, **kw):
    traj = Trajectory()
    t = transport or b2.raw_transport(script)[0]
    model = b1.MODELS[fw](traj, transport=t)
    tools = [traced(x, traj) for x in (tools if tools is not None else TOOLS)]   # every tool body logged
    if fw == "langgraph":
        info = run_lg(q, model, tools, traj, limit, **{k: v for k, v in kw.items() if k in ("handle_tool_errors", "retry_policy")})
    else:
        info = run_pa(q, model, tools, traj, limit, **{k: v for k, v in kw.items() if k in ("tool_calls_limit", "retries")})
    a = traj.accounting()
    kinds = [e["kind"] for e in traj.events if e["kind"] != "framework_step"]
    return {"framework": fw, "limit": limit, **{k: v for k, v in kw.items() if not callable(v) and k != "retry_policy"},
            "retry_policy": bool(kw.get("retry_policy")),
            "http_attempts": a["model_requests"], "http_errors": a["http_errors"], "model_responses": a["model_responses_with_tool_calls"] + a["final_responses"],
            "tool_calls_requested": a["tool_calls_requested"], "tool_executions": a["tool_executions"], "framework_steps": a["framework_steps"],
            "last_event": kinds[-1] if kinds else None, "result": "OK" if info["error"] is None else info["error"].split(":")[0], **info,
            "trace": traj.ascii()}


def rec(exp, d):
    RESULTS.setdefault(exp, []).append(d)
    print(f"{exp:<10}{d['framework']:<12}limit={d['limit']!s:<4} {d.get('note', ''):<34} http={d['http_attempts']} model_calls={d['framework_model_calls']} "
          f"responses={d['model_responses']} tools={d['tool_calls_requested']}/{d['tool_executions']}exec steps={d['framework_steps']} "
          f"last={d['last_event']:<14} -> {d['result']}  {(d['error'] or '')[:70]}")
    return d


Q1, Q2 = "Calculate 25 × 17.", "Take 120 + 45. Then multiply the result by 3."


# ---------------------------------------------------------------- B3-3.1.11 – 3.1.14: limits
def limits():
    for fw in FWS:
        for n in (1, 2, 3):
            rec("B3-3.1.11", run(fw, Q1, [C("The final result is 425.")], n))
    for fw in FWS:
        for n in (1, 2, 3, 4, 5):
            rec("B3-3.1.12", run(fw, Q1, [C(None, [MUL]), C("The answer is 425.")], n))
    for fw in FWS:
        for n in range(1, 8):
            rec("B3-3.1.13", run(fw, Q2, [C(None, [ADD]), C(None, [MUL2]), C("The result is 495.")], n))
    endless = [C(None, [MUL])] * 40                                   # never a final response
    for fw in FWS:
        for n in (2, 3, 4, 5, 6):
            d = run(fw, Q1, endless, n)
            d["note"] = "endless tool calls"
            rec("B3-3.1.14", d)
    for n in (0, 1, 2):                                               # Pydantic AI tool_calls_limit: checked before executing?
        d = run("pydantic-ai", Q1, endless, 10, tool_calls_limit=n)
        d["note"] = f"tool_calls_limit={n}"
        rec("B3-3.1.14", d)


# ---------------------------------------------------------------- B3-3.1.15 / 16: empty output re-prompt vs limits
def empty():
    for fw in FWS:
        for n in (1, 2, 3):
            d = run(fw, Q1, [C(""), C("The final result is 425.")], n)
            d["note"] = "empty, then final"
            rec("B3-3.1.15", d)
    for retries in (0, 1, 2, 3):
        for n in (2, 10):
            d = run("pydantic-ai", Q1, [C("")] * 10, n, retries=retries)
            d["note"] = f"persistent empty, retries={retries}"
            rec("B3-3.1.16", d)
    d = run("langgraph", Q1, [C("")] * 10, 10)
    d["note"] = "persistent empty"
    rec("B3-3.1.16", d)


# ---------------------------------------------------------------- B3-3.1.17 / 18: tool retries vs loop budget
def tools():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)
    def calls_for(fail):                                              # model calls flaky until it succeeds, then answers
        return [C(None, [("flaky", {})])] * (fail + 1) + [C("done")]
    # Pydantic AI: ModelRetry (model-visible) with retries 0/1/2; tool fails twice then succeeds
    for retries in (0, 1, 2):
        for fail in (1, 2):
            traj_tools = [fx.make_flaky(fail, "model_retry")]
            d = run("pydantic-ai", "Call flaky.", calls_for(fail), 10, tools=traj_tools, retries=retries)
            d.update(note=f"ModelRetry x{fail}, retries={retries}", flaky_executions=sum(1 for e in fx.EXEC if e[0] == "flaky"))
            fx.EXEC.clear()
            rec("B3-3.1.17", d)
    for fail in (1, 2):                                               # plain exception in Pydantic AI
        d = run("pydantic-ai", "Call flaky.", calls_for(fail), 10, tools=[fx.make_flaky(fail, "runtime")], retries=2)
        d.update(note=f"RuntimeError x{fail}, retries=2", flaky_executions=sum(1 for e in fx.EXEC if e[0] == "flaky"))
        fx.EXEC.clear()
        rec("B3-3.1.17", d)
    # LangGraph: (A) model-visible retry = default error ToolMessage; (B) framework RetryPolicy, no model involvement
    for fail in (1, 2):
        d = run("langgraph", "Call flaky.", calls_for(fail), 25, tools=[fx.make_flaky(fail, "runtime")])
        d.update(note=f"A model-visible, fails x{fail}", flaky_executions=sum(1 for e in fx.EXEC if e[0] == "flaky"))
        fx.EXEC.clear()
        rec("B3-3.1.18", d)
        d = run("langgraph", "Call flaky.", [C(None, [("flaky", {})]), C("done")], 25, tools=[fx.make_flaky(fail, "runtime")],
                handle_tool_errors=False, retry_policy=rp)                 # the model asks once; the framework re-runs
        d.update(note=f"B RetryPolicy(3), fails x{fail}", flaky_executions=sum(1 for e in fx.EXEC if e[0] == "flaky"))
        fx.EXEC.clear()
        rec("B3-3.1.18", d)
    # minimum budget: does the retry consume it?  one round + final needs LG 4 / PA 2 (measured in B3-3.1.12)
    for fw, lim_ok in (("langgraph", 4), ("langgraph", 6), ("pydantic-ai", 2), ("pydantic-ai", 3)):
        how = "runtime" if fw == "langgraph" else "model_retry"
        d = run(fw, "Call flaky.", calls_for(1), lim_ok, tools=[fx.make_flaky(1, how)])
        d.update(note=f"model-visible retry at min limit {lim_ok}", flaky_executions=sum(1 for e in fx.EXEC if e[0] == "flaky"))
        fx.EXEC.clear()
        rec("B3-3.1.18", d)
    d = run("langgraph", "Call flaky.", [C(None, [("flaky", {})]), C("done")], 4, tools=[fx.make_flaky(1, "runtime")],
            handle_tool_errors=False, retry_policy=rp)
    d.update(note="RetryPolicy at min limit 4", flaky_executions=sum(1 for e in fx.EXEC if e[0] == "flaky"))
    fx.EXEC.clear()
    rec("B3-3.1.18", d)


# ---------------------------------------------------------------- B3-3.1.19 / 20: provider-client HTTP retry
def http():
    script = [C(None, [MUL]), C("The answer is 425.")]
    for fw in FWS:
        for fail, status in ((1, 500), (2, 500), (1, 429), (3, 500)):
            t, _, att = http_flaky(script, fail, status)
            d = run(fw, Q1, script, 25, transport=t)
            d.update(note=f"{fail}x HTTP {status} then 200", wire_attempts_seen_by_transport=att["n"])
            rec("B3-3.1.19", d)
    for fw, lim in (("langgraph", 4), ("pydantic-ai", 2)):           # minimum successful limit measured in B3-3.1.12
        for fail in (1, 2):
            t, _, att = http_flaky(script, fail, 500)
            d = run(fw, Q1, script, lim, transport=t)
            d.update(note=f"min limit, {fail}x HTTP 500", wire_attempts_seen_by_transport=att["n"])
            rec("B3-3.1.20", d)


# ---------------------------------------------------------------- B3-3.1.21: behaviour near the limit
def warn():
    endless = [C(None, [MUL])] * 40
    for fw, lim in (("langgraph", 5), ("pydantic-ai", 3)):
        traj = Trajectory()
        model = b1.MODELS[fw](traj, transport=b2.raw_transport(endless)[0])
        tools = [traced(x, traj) for x in TOOLS]
        info = run_lg(Q1, model, tools, traj, lim) if fw == "langgraph" else run_pa(Q1, model, tools, traj, lim)
        reqs = traj.of("model_request")
        shapes = [[(m["role"], (m.get("content") or "")[:60]) for m in r["messages"] if m["role"] in ("system", "user")] for r in reqs]
        d = {"framework": fw, "limit": lim, "requests": len(reqs), "system_and_user_messages_per_request": shapes,
             "identical_instructions_every_request": all(s == shapes[0] for s in shapes), "error": info["error"]}
        RESULTS.setdefault("B3-3.1.21", []).append(d)
        print(f"B3-3.1.21 {fw:<12}limit={lim} requests={len(reqs)} same system/user msgs every request={d['identical_instructions_every_request']} err={info['error']}")
    # reference: the deprecated LangGraph prebuilt agent (remaining_steps)
    import warnings
    from langchain_core.messages import HumanMessage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent
        for lim in (3, 4, 5):
            traj = Trajectory()
            model = b1.MODELS["langgraph"](traj, transport=b2.raw_transport(endless)[0])
            agent = create_react_agent(model, [traced(x, traj) for x in TOOLS])
            err, last = None, None
            try:
                out = agent.invoke({"messages": [HumanMessage(Q1)]}, {"recursion_limit": lim})
                last = out["messages"][-1]
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            d = {"framework": "langgraph prebuilt create_react_agent (deprecated, reference)", "limit": lim,
                 "http_attempts": len(traj.of("model_request")), "tool_executions": len(traj.of("tool_end")), "error": err,
                 "final_message": getattr(last, "content", None), "final_message_written_by": "framework" if last is not None and not traj.of("model_response")[-1]["text"] == getattr(last, "content", None) else "model"}
            RESULTS.setdefault("B3-3.1.21", []).append(d)
            print(f"B3-3.1.21 prebuilt      limit={lim} http={d['http_attempts']} tool_exec={d['tool_executions']} final={d['final_message']!r} by={d['final_message_written_by']} err={err}")


# ---------------------------------------------------------------- B3-3.1.22: side effect before the limit
def effects():
    endless = [C(None, [("create_order", {"item": "widget", "qty": 1})])] * 40
    for fw, lim in (("langgraph", 4), ("langgraph", 5), ("pydantic-ai", 2), ("pydantic-ai", 3)):
        orders = fx.Orders()
        d = run(fw, "Order one widget.", endless, lim, tools=[orders.unguarded()])
        d.update(note="create_order every turn", orders_in_ledger=len(orders.ledger), rolled_back=False if orders.ledger else None)
        rec("B3-3.1.22", d)


def live():
    """Small confirmation of the limit boundary on the real provider (<= 3 runs per framework)."""
    from lab_common.config import load_model_config
    cfg = load_model_config()
    print(f"live model: {cfg.provider}/{cfg.model_name}, temperature 0")
    plan = [("langgraph", 3), ("langgraph", 3), ("langgraph", 4), ("pydantic-ai", 1), ("pydantic-ai", 1), ("pydantic-ai", 2)]
    for i, (fw, lim) in enumerate(plan, 1):
        traj = Trajectory()
        model = b1.MODELS[fw](traj)
        tools = [traced(x, traj) for x in TOOLS]
        info = run_lg(Q1, model, tools, traj, lim) if fw == "langgraph" else run_pa(Q1, model, tools, traj, lim)
        a = traj.accounting()
        last_text = [e["text"] for e in traj.of("model_response") if e.get("status", 200) == 200][-1:] or [None]
        d = {"framework": fw, "limit": lim, "model": cfg.model_name, "http_attempts": a["model_requests"], "http_errors": a["http_errors"],
             "tool_executions": a["tool_executions"], "error": info["error"], "final_text": info["final_text"], "last_model_text": last_text[0],
             "tokens": a["tokens_total"]}
        RESULTS.setdefault("B3-live", []).append(d)
        print(f"LIVE {fw:<12}limit={lim} http={d['http_attempts']} (errors {d['http_errors']}) tool_exec={d['tool_executions']} "
              f"error={(d['error'] or '')[:60]!r} final={d['final_text']!r} last_model_text={(d['last_model_text'] or '')[:50]!r}")
        if info["error"] and "429" in info["error"]:
            print("LIVE STOPPED (quota)")
            return


GROUPS = {"limits": limits, "empty": empty, "tools": tools, "http": http, "warn": warn, "effects": effects, "live": live}


def main():
    for g in sys.argv[1:] or [k for k in GROUPS if k != "live"]:
        print(f"\n===== {g} =====")
        GROUPS[g]()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "batch3.json")
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
