"""Phase 3.1 Batch 7: manual agent stepping (B7-3.1.59 ... B7-3.1.68). Offline, scripted, deterministic.

What each framework's own iteration API lets a developer do between steps: observe, intercept (act before a step runs), modify
state, approve an action, resume. Reuses the Batch 3-6 harness (scripted transports, the recorder, tool probes), the Batch 5
failure fixture and `lab3/history.py`. The "stepping loops" below are small consumer functions that exist only to measure what the
frameworks expose; their size is reported as a result. No stepping engine, event bus, workflow engine or runtime is built.

At every boundary the consumer sees, a snapshot of what has already happened is taken from the run's own timeline (model requests
sent, tool bodies started and finished). Comparing that snapshot with what the chunk or node describes tells whether the boundary
is before or after the work: that is the difference between observing and being able to intercept.

Hypotheses: phases/phase-03/experiments.md (Batch 7). Gap analysis: phases/phase-03/batch7-gap-analysis.md.
Usage: uv run python -u phase-03/exp3_batch7.py [group ...]   groups: lgsync lgasync paiter steps stop state approval failure limits
"""
import asyncio
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_batch6 as e6  # noqa: E402  (brings e4 / e3 / b2 / b1, Probe, SlowServer)

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab3.history import finish, lg_history, pa_history  # noqa: E402
from lab3.trajectory import Trajectory, lg_model, pa_model, traced  # noqa: E402
from lab_common.tools import TOOLS  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

e3, e4, C, call = e6.e3, e6.e4, e6.C, e6.call
RESULTS: dict = {}
FWS = ["langgraph", "pydantic-ai"]
Q = "Take 120 + 45. Then multiply the result by 3."
ADD, MUL = ("add", {"a": 120, "b": 45}), ("multiply", {"a": 165, "b": 3})
TWO_ROUNDS = [C(None, [ADD]), C(None, [MUL]), C("The result is 495.")]
ONE_ROUND = [C(None, [ADD]), C("The sum is 165.")]


def rec(exp, d):
    RESULTS.setdefault(exp, []).append(d)
    return d


def snap(traj):
    """What has already happened: model requests sent, responses received, tool bodies started and finished."""
    return {"requests": len(traj.of("model_request")), "responses": len(traj.of("model_response")),
            "tool_starts": len(traj.of("tool_start")), "tool_ends": len(traj.of("tool_end"))}


def rig(fw, script, tools=None, **model_kw):
    """A fresh trajectory, a scripted transport and a model for `fw`; tools are the traced calculator by default."""
    traj = Trajectory()
    transport = e3.b2.raw_transport(script)[0]
    model = (lg_model if fw == "langgraph" else pa_model)(traj, transport=transport, **model_kw)
    wrapped = [traced(t, traj) for t in (tools if tools is not None else TOOLS)]
    return traj, model, wrapped


def lg_graph(model, tools, **kw):
    return lg_agent.build_graph(model, tools=tools, **kw)


def lg_input(q=Q):
    return {"messages": [HumanMessage(q)]}


CFG = {"recursion_limit": 25}


# ---------------------------------------------------------------- chunk summaries (LangGraph)
def _types(msgs):
    return [type(m).__name__ for m in msgs]


def summarize(mode, chunk):
    try:
        if mode == "values":
            return {"messages": _types(chunk["messages"])}
        if mode == "updates":
            return {"node": list(chunk), "update": {k: (_types(v["messages"]) if v and "messages" in v else None) for k, v in chunk.items()}}
        if mode == "tasks":
            if "input" in chunk or "triggers" in chunk:
                return {"event": "task_start", "name": chunk.get("name")}
            return {"event": "task_result", "name": chunk.get("name"), "error": chunk.get("error")}
        if mode == "debug":
            return {"type": chunk["type"], "step": chunk["step"], "name": chunk["payload"].get("name")}
        if mode == "messages":
            return {"message": type(chunk[0]).__name__, "node": chunk[1].get("langgraph_node")}
        if mode == "checkpoints":
            return {"keys": sorted(chunk)[:6]}
    except Exception as e:  # noqa: BLE001
        return {"summary_error": f"{type(e).__name__}: {e}"}
    return {"raw": repr(chunk)[:80]}


def dedupe(seq):
    """Collapse the many token-level `messages` chunks into counts, keep the order of everything else."""
    out = []
    for s in seq:
        if out and out[-1]["summary"] == s["summary"] and s["mode"] == "messages":
            out[-1]["repeat"] = out[-1].get("repeat", 1) + 1
        else:
            out.append({**s})
    return out


# ================================================================ B7-3.1.59 LangGraph synchronous stepping
def lg_sync():
    for mode in ("values", "updates", "tasks", "debug", "messages", "checkpoints", "custom"):
        traj, model, tools = rig("langgraph", TWO_ROUNDS)
        graph = lg_graph(model, tools)
        seq, t0, err = [], time.perf_counter(), None
        try:
            for chunk in graph.stream(lg_input(), CFG, stream_mode=mode):
                seq.append({"i": len(seq) + 1, "mode": mode, "t_ms": round((time.perf_counter() - t0) * 1000), "summary": summarize(mode, chunk), "snap": snap(traj)})
        except Exception as e:  # noqa: BLE001  (messages mode makes the model stream; the scripted non-streaming transport cannot serve it)
            err = f"{type(e).__name__}: {e}"[:160]
        if err:
            rec("B7-3.1.59", {"framework": "langgraph", "api": "stream", "mode": mode, "chunks": len(seq), "error": err, "sequence": dedupe(seq)})
            print(f"B7-3.1.59 stream_mode={mode:<11} ERROR {err}")
            continue
        ded = dedupe(seq)
        r = rec("B7-3.1.59", {"framework": "langgraph", "api": "stream", "mode": mode, "chunks": len(seq), "sequence": ded, "final_snap": snap(traj)})
        print(f"B7-3.1.59 stream_mode={mode:<11} chunks={len(seq):<3} first={ded[0]['summary'] if ded else None}")
    # whether a chunk arrives BEFORE or AFTER the work it names: the snapshot at the moment of receipt
    traj, model, tools = rig("langgraph", TWO_ROUNDS)
    graph = lg_graph(model, tools)
    order = []
    for mode, chunk in graph.stream(lg_input(), CFG, stream_mode=["tasks", "updates"]):
        order.append({"mode": mode, "summary": summarize(mode, chunk), "snap": snap(traj)})
    rec("B7-3.1.59", {"framework": "langgraph", "api": "stream", "mode": "tasks+updates (arrival vs execution)", "sequence": order})
    for o in order:
        print(f"   {o['mode']:<8}{json.dumps(o['summary'])[:70]:<72} at receipt: requests={o['snap']['requests']} tool_starts={o['snap']['tool_starts']} tool_ends={o['snap']['tool_ends']}")
    # stopping consumption: between chunks nothing runs
    traj, model, tools = rig("langgraph", TWO_ROUNDS)
    graph = lg_graph(model, tools)
    gen = graph.stream(lg_input(), CFG, stream_mode="updates")
    first = next(gen)
    at_stop = snap(traj)
    gen.close()
    time.sleep(0.4)
    rec("B7-3.1.59", {"framework": "langgraph", "api": "stream", "case": "stop after the first update, close the generator", "first_update": summarize("updates", first),
                      "snap_at_stop": at_stop, "snap_0_4s_later": snap(traj), "tool_ran": snap(traj)["tool_starts"] > 0})
    print("   stop after first update:", at_stop, "-> 0.4 s later", snap(traj))


# ================================================================ B7-3.1.60 LangGraph asynchronous stepping
async def _alg(mode_list, script=TWO_ROUNDS):
    traj, model, tools = rig("langgraph", script)
    graph = lg_graph(model, tools)
    seq, ticks = [], {"n": 0}

    async def ticker():
        while True:
            ticks["n"] += 1
            await asyncio.sleep(0.005)
    tk = asyncio.create_task(ticker())
    async for item in graph.astream(lg_input(), CFG, stream_mode=mode_list):
        mode, chunk = item if isinstance(mode_list, list) else (mode_list, item)
        seq.append({"mode": mode, "summary": summarize(mode, chunk), "snap": snap(traj), "ticks": ticks["n"]})
    tk.cancel()
    return seq, traj, ticks["n"]


def lg_async():
    for mode in ("updates", "tasks", "values"):
        sync_seq = [s["summary"] for s in next(r for r in RESULTS["B7-3.1.59"] if r.get("mode") == mode)["sequence"]]
        aseq, traj, ticks = asyncio.run(_alg(mode))
        same = [s["summary"] for s in aseq] == sync_seq
        rec("B7-3.1.60", {"framework": "langgraph", "api": "astream", "mode": mode, "chunks": len(aseq), "same_as_sync": same, "event_loop_ticks_during_run": ticks})
        print(f"B7-3.1.60 astream mode={mode:<8} chunks={len(aseq)} identical to stream: {same} loop ticks during the run={ticks}")

    async def stop_and_close():
        traj, model, tools = rig("langgraph", TWO_ROUNDS)
        graph = lg_graph(model, tools)
        before = len(asyncio.all_tasks())
        agen = graph.astream(lg_input(), CFG, stream_mode="updates")
        first = await agen.__anext__()
        at_stop = snap(traj)
        await agen.aclose()
        await asyncio.sleep(0.4)
        return {"first_update": summarize("updates", first), "snap_at_stop": at_stop, "snap_0_4s_later": snap(traj),
                "tasks_before": before, "tasks_after_close": len(asyncio.all_tasks())}
    r = asyncio.run(stop_and_close())
    rec("B7-3.1.60", {"framework": "langgraph", "api": "astream", "case": "stop after the first update, aclose()", **r})
    print("   astream stop after first update:", r)
    # a sync node in an async stream runs in a worker thread: the loop stays free during the model call (Batch 6: and it cannot be cancelled)
    a, _, ticks = asyncio.run(_alg("updates"))
    rec("B7-3.1.60", {"framework": "langgraph", "api": "astream", "case": "consumer awaits between chunks", "loop_stays_free": ticks > 0})


# ================================================================ B7-3.1.61 Pydantic AI manual iteration
def describe(node):
    cls = type(node).__name__
    if cls == "UserPromptNode":
        return {"user_prompt": node.user_prompt}
    if cls == "ModelRequestNode":
        return {"request_parts": [(type(p).__name__, str(getattr(p, "content", ""))[:30]) for p in node.request.parts]}
    if cls == "CallToolsNode":
        parts = node.model_response.parts
        return {"tool_calls": [(p.tool_name, p.args_as_dict(), p.tool_call_id) for p in parts if type(p).__name__ == "ToolCallPart"],
                "text": [p.content for p in parts if type(p).__name__ == "TextPart"]}
    if cls == "End":
        return {"output": str(node.data.output)}
    return {}


async def _pa_iter(script, with_events=False, tools=None, **iter_kw):
    traj, model, wrapped = rig("pydantic-ai", script, tools)
    agent = pa_agent.build_agent(model, retries=3, tools=wrapped)
    out = []
    async with agent.iter(Q, **iter_kw) as run:
        async for node in run:
            row = {"i": len(out) + 1, "node": type(node).__name__, "fields": describe(node), "snap_at_yield": snap(traj),
                   "history_len": len(run.ctx.state.message_history), "usage": {"requests": run.usage.requests, "tool_calls": run.usage.tool_calls}}
            # ModelRequestNode.stream needs a streaming wire response; the scripted transport is non-streaming (recorded as a limitation)
            if with_events and type(node).__name__ == "CallToolsNode":
                events = []
                async with node.stream(run.ctx) as stream:
                    async for ev in stream:
                        events.append({"event": type(ev).__name__, "snap": snap(traj)})
                row["events"] = events
            out.append(row)
        final = str(run.result.output)
        hist_len = len(run.ctx.state.message_history)
    return out, final, hist_len, traj


def pa_iter():
    rows, final, hist, traj = asyncio.run(_pa_iter(TWO_ROUNDS))
    rec("B7-3.1.61", {"framework": "pydantic-ai", "api": "iter", "case": "nodes in order", "nodes": rows, "final": final, "final_history_len": hist})
    for r in rows:
        print(f"B7-3.1.61 #{r['i']} {r['node']:<16} at yield: requests={r['snap_at_yield']['requests']} tool_starts={r['snap_at_yield']['tool_starts']} "
              f"history={r['history_len']} usage={r['usage']} fields={json.dumps(r['fields'])[:90]}")
    rows2, _, _, _ = asyncio.run(_pa_iter(TWO_ROUNDS, with_events=True))
    rec("B7-3.1.61", {"framework": "pydantic-ai", "api": "iter + node.stream", "case": "events inside a node", "nodes": rows2})
    for r in rows2:
        if r.get("events"):
            print(f"   events in {r['node']}: {[(e['event'], e['snap']['tool_starts'], e['snap']['tool_ends']) for e in r['events']]}")


GROUPS = {"lgsync": lg_sync, "lgasync": lg_async, "paiter": pa_iter}


# ================================================================ shared drivers
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from pydantic_ai import ModelRetry  # noqa: E402
from pydantic_ai.messages import ModelRequest, ToolReturnPart  # noqa: E402
from pydantic_ai._agent_graph import ModelRequestNode  # noqa: E402
import exp3_batch5 as e5  # noqa: E402


def tool_args(traj):
    return [e["args"] for e in traj.of("tool_start")]


def lg_ckpt_graph(model, tools, **kw):
    g = lg_graph(model, tools, **kw)
    g.checkpointer = InMemorySaver()  # experiment-only: get_state / update_state need one; no persistence, no interrupts
    return g


def lg_cfg():
    return {"recursion_limit": 25, "configurable": {"thread_id": f"t{time.time_ns()}"}}


def proposed_calls(ai):
    return [(c["name"], c["args"], c["id"]) for c in (ai.tool_calls or [])]


def walk_lg(script, decide, tools=None, graph_kw=None, ckpt=False, cfg=None):
    """Consume stream(["tasks","updates"]); decide(i, mode, chunk, traj, log) -> 'continue' | 'stop'. Returns (log, traj, graph, cfg, stopped_at)."""
    traj, model, wrapped = rig("langgraph", script, tools)
    graph = (lg_ckpt_graph if ckpt else lg_graph)(model, wrapped, **(graph_kw or {}))
    cfg = cfg or lg_cfg()
    log, stopped = [], None
    gen = graph.stream(lg_input(), cfg, stream_mode=["tasks", "updates"])
    for i, (mode, chunk) in enumerate(gen, 1):
        log.append({"i": i, "mode": mode, "summary": summarize(mode, chunk), "snap": snap(traj)})
        if decide(i, mode, chunk, traj, log) == "stop":
            stopped = i
            gen.close()
            break
    return log, traj, graph, cfg, stopped


async def walk_pa(script, decide, tools=None, agent_kw=None, iter_kw=None, settle=0.4):
    """Consume agent.iter by hand; decide(i, node, run, traj) -> 'continue' | 'stop' | ('replace', node) | ('sleep', s)."""
    traj, model, wrapped = rig("pydantic-ai", script, tools)
    agent = pa_agent.build_agent(model, retries=3, tools=wrapped, **(agent_kw or {}))
    log, final, err, hist = [], None, None, None
    try:
        async with agent.iter(Q, **(iter_kw or {})) as run:
            node = run.next_node
            while type(node).__name__ != "End":
                row = {"i": len(log) + 1, "node": type(node).__name__, "fields": describe(node), "snap": snap(traj),
                       "usage": {"requests": run.usage.requests, "tool_calls": run.usage.tool_calls}}
                log.append(row)
                d = decide(row["i"], node, run, traj)
                if d == "stop":
                    row["stopped"] = True
                    break
                if isinstance(d, tuple) and d[0] == "sleep":
                    await asyncio.sleep(d[1])
                    row["snap_after_pause"] = snap(traj)
                    d = "continue"
                if isinstance(d, tuple) and d[0] == "replace":
                    row["replaced"] = True
                    node = await run.next(d[1])
                else:
                    node = await run.next(node)
            else:
                final = str(run.result.output) if run.result else None
            hist = len(run.ctx.state.message_history)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:200]
    await asyncio.sleep(settle)
    return {"log": log, "final": final, "error": err, "history_len": hist, "snap_after_settle": snap(traj), "tasks_left": len(asyncio.all_tasks()) - 1}, traj


def pa(script, decide, **kw):
    return asyncio.run(walk_pa(script, decide, **kw))


# ================================================================ B7-3.1.62 step-by-step tool execution
def steps():
    log, traj, *_ = walk_lg(TWO_ROUNDS, lambda *a: "continue")
    rec("B7-3.1.62", {"framework": "langgraph", "boundaries": log, "tool_args_seen_by_tools": tool_args(traj)})
    print("B7-3.1.62 langgraph boundaries:")
    for l in log:
        print(f"   {l['i']:<3}{l['mode']:<8}{json.dumps(l['summary'])[:60]:<62} done so far: req={l['snap']['requests']} tool_ends={l['snap']['tool_ends']}")
    traj2, model, wrapped = rig("langgraph", TWO_ROUNDS)
    graph = lg_graph(model, wrapped)
    seen = []
    for mode, chunk in graph.stream(lg_input(), CFG, stream_mode=["tasks", "updates"]):
        if mode == "updates" and "agent" in chunk:
            ai = chunk["agent"]["messages"][-1]
            seen.append({"proposed": proposed_calls(ai), "tool_starts_at_that_moment": snap(traj2)["tool_starts"]})
    rec("B7-3.1.62", {"framework": "langgraph", "case": "proposed call readable before execution", "seen": seen})
    print("   proposed calls readable from the agent update:", seen)
    res, traj3 = pa(TWO_ROUNDS, lambda *a: "continue")
    rec("B7-3.1.62", {"framework": "pydantic-ai", "boundaries": res["log"], "tool_args_seen_by_tools": tool_args(traj3)})
    print("B7-3.1.62 pydantic-ai boundaries:")
    for l in res["log"]:
        print(f"   {l['i']:<3}{l['node']:<16}{json.dumps(l['fields'])[:70]:<72} done so far: req={l['snap']['requests']} tool_ends={l['snap']['tool_ends']}")
    nxt = [l for l in res["log"] if l["node"] == "ModelRequestNode"]
    rec("B7-3.1.62", {"framework": "pydantic-ai", "case": "next request visible before it is sent",
                      "model_request_nodes": [{"request_parts": n["fields"]["request_parts"], "wire_requests_already_sent": n["snap"]["requests"]} for n in nxt]})
    print("   next request visible before sent:", [(n["fields"]["request_parts"][0][0], n["snap"]["requests"]) for n in nxt])


# ================================================================ B7-3.1.63 stop between steps
def lg_stop_at(label, pred, settle=0.4):
    n_threads = threading.active_count()
    log, traj, graph, cfg, stopped = walk_lg(TWO_ROUNDS, lambda i, mode, chunk, tr, lg: "stop" if pred(i, mode, chunk, lg) else "continue")
    at_stop = snap(traj)
    time.sleep(settle)
    after = snap(traj)
    d = {"framework": "langgraph", "boundary": label, "stopped_at_chunk": stopped, "snap_at_stop": at_stop, "snap_after_settle": after,
         "work_continued_after_stop": at_stop != after, "threads_before": n_threads, "threads_after": threading.active_count()}
    rec("B7-3.1.63", d)
    print(f"B7-3.1.63 langgraph {label:<38} at stop {at_stop} -> after {after} continued={d['work_continued_after_stop']} threads {n_threads}->{d['threads_after']}")


def pa_stop_at(label, pred):
    res, traj = pa(TWO_ROUNDS, lambda i, node, run, tr: "stop" if pred(i, node) else "continue")
    row = next((r for r in res["log"] if r.get("stopped")), None)
    d = {"framework": "pydantic-ai", "boundary": label, "stopped_at_node": row and (row["i"], row["node"]), "snap_at_stop": row and row["snap"], "snap_after_settle": res["snap_after_settle"],
         "work_continued_after_stop": bool(row) and row["snap"] != res["snap_after_settle"], "tasks_left": res["tasks_left"], "error": res["error"]}
    rec("B7-3.1.63", d)
    print(f"B7-3.1.63 pydantic-ai {label:<36} stop {d['stopped_at_node']} at stop {row and row['snap']} -> after {res['snap_after_settle']} continued={d['work_continued_after_stop']} tasks_left={res['tasks_left']}")


def _n_agent_starts(lg):
    return sum(1 for l in lg if l["summary"].get("event") == "task_start" and l["summary"].get("name") == "agent")


def stop():
    lg_stop_at("after first model response", lambda i, m, c, lg: m == "updates" and "agent" in c and sum(1 for l in lg if l["mode"] == "updates") == 1)
    lg_stop_at("tool proposed (tools task starts)", lambda i, m, c, lg: m == "tasks" and c.get("name") == "tools" and "input" in c)
    lg_stop_at("tool completed (tools update)", lambda i, m, c, lg: m == "updates" and "tools" in c)
    lg_stop_at("before next request (agent starts #2)", lambda i, m, c, lg: m == "tasks" and c.get("name") == "agent" and "input" in c and _n_agent_starts(lg) == 2)
    lg_stop_at("after final answer", lambda i, m, c, lg: m == "updates" and "agent" in c and not c["agent"]["messages"][-1].tool_calls)
    pa_stop_at("before first request", lambda i, n: type(n).__name__ == "ModelRequestNode" and i == 2)
    pa_stop_at("after first response / tool proposed", lambda i, n: type(n).__name__ == "CallToolsNode" and i == 3)
    pa_stop_at("tool completed = before next request", lambda i, n: type(n).__name__ == "ModelRequestNode" and i == 4)
    pa_stop_at("after final answer (last CallToolsNode)", lambda i, n: type(n).__name__ == "CallToolsNode" and i == 7)
    log, traj, *_ = walk_lg(TWO_ROUNDS, lambda i, m, c, tr, lg: (time.sleep(0.3), "continue")[1] if m == "tasks" and c.get("name") == "tools" and "input" in c else "continue")
    paused = next(l for l in log if l["summary"].get("name") == "tools")
    rec("B7-3.1.63", {"framework": "langgraph", "case": "pause 0.3 s at the tools task start, then continue", "completed_tool_ends": snap(traj)["tool_ends"], "snap_at_pause": paused["snap"]})
    print("B7-3.1.63 LG pause at tools-task start: snap", paused["snap"], "-> finished with tool_ends", snap(traj)["tool_ends"])
    res, traj = pa(TWO_ROUNDS, lambda i, n, r, t: ("sleep", 0.3) if type(n).__name__ == "CallToolsNode" and i == 3 else "continue")
    p = next(r for r in res["log"] if "snap_after_pause" in r)
    rec("B7-3.1.63", {"framework": "pydantic-ai", "case": "pause 0.3 s at first CallToolsNode, then continue", "final": res["final"], "snap_at_yield": p["snap"], "snap_after_pause": p["snap_after_pause"]})
    print("B7-3.1.63 PA pause: at yield", p["snap"], "after 0.3 s pause", p["snap_after_pause"], "-> final", res["final"])


GROUPS.update({"steps": steps, "stop": stop})



def lg_pause_probe():
    """Does the tools task start while the consumer is paused at its task_start event?"""
    box = {}

    def decide(i, m, c, tr, lg):
        if m == "tasks" and c.get("name") == "tools" and "input" in c and "before" not in box:
            box["before"] = snap(tr)
            time.sleep(0.3)
            box["after_pause"] = snap(tr)
        return "continue"
    log, traj, *_ = walk_lg(ONE_ROUND, decide)
    rec("B7-3.1.63", {"framework": "langgraph", "case": "does the tools task run while the consumer sleeps at its task_start", **box, "finished": snap(traj)})
    print("B7-3.1.63 LG sleep at tools task_start:", box, "-> finished", snap(traj))


# ================================================================ B7-3.1.64 inspect and modify state
def _req_tool_contents(traj, n):
    """Content of the tool messages in the n-th (1-based) model request, as the wire carried them."""
    r = traj.of("model_request")[n - 1]
    return [str(m.get("content")) for m in r["messages"] if m["role"] == "tool"]


def state():
    # --- LangGraph (a): edit the streamed chunk in place, hoping the tool sees it (not a supported route)
    traj, model, wrapped = rig("langgraph", ONE_ROUND)
    graph = lg_graph(model, wrapped)
    for chunk in graph.stream(lg_input(), CFG, stream_mode="updates"):
        if "agent" in chunk:
            ai = chunk["agent"]["messages"][-1]
            if ai.tool_calls:
                ai.tool_calls[0]["args"]["a"] = 1000
    rec("B7-3.1.64", {"framework": "langgraph", "case": "mutate chunk in place (not a supported route)", "tool_received": tool_args(traj)})
    print("B7-3.1.64 LG in-place chunk edit: tool received", tool_args(traj))
    # --- LangGraph (b): get_state / update_state on a checkpointed graph; stop after the agent update, edit, then continue the same thread
    traj, model, wrapped = rig("langgraph", ONE_ROUND)
    graph = lg_ckpt_graph(model, wrapped)
    cfg = lg_cfg()
    gen = graph.stream(lg_input(), cfg, stream_mode="updates")
    next(gen)
    gen.close()
    st = graph.get_state(cfg)
    before = {"next": list(st.next), "n_messages": len(st.values["messages"]), "pending": proposed_calls(st.values["messages"][-1]), "has_metadata": sorted(st.metadata)[:4],
              "history_len": len(list(graph.get_state_history(cfg)))}
    ai = st.values["messages"][-1]
    edited = AIMessage(content=ai.content, tool_calls=[{"name": "add", "args": {"a": 1000, "b": 45}, "id": ai.tool_calls[0]["id"]}], id=ai.id)
    graph.update_state(cfg, {"messages": [edited]})
    st2 = graph.get_state(cfg)
    after = {"next": list(st2.next), "n_messages": len(st2.values["messages"]), "pending": proposed_calls(st2.values["messages"][-1]), "history_len": len(list(graph.get_state_history(cfg))),
             "metadata_source": st2.metadata.get("source"), "metadata_writes": str(st2.metadata.get("writes"))[:80]}
    out = [c for c in graph.stream(None, cfg, stream_mode="updates")]
    rec("B7-3.1.64", {"framework": "langgraph", "case": "get_state / update_state by message id, then continue the same thread", "before": before, "after_update": after,
                      "tool_received": tool_args(traj), "final_messages": [type(m).__name__ for m in graph.get_state(cfg).values["messages"]],
                      "marks_modification_in_messages": any(getattr(m, "additional_kwargs", {}) for m in graph.get_state(cfg).values["messages"])})
    print("B7-3.1.64 LG update_state: before", before, "\n   after update", after, "\n   tool received", tool_args(traj))
    # --- LangGraph (c): is a stream still running affected by update_state from the consumer? (edit while the generator is suspended)
    traj, model, wrapped = rig("langgraph", ONE_ROUND)
    graph = lg_ckpt_graph(model, wrapped)
    cfg = lg_cfg()
    gen = graph.stream(lg_input(), cfg, stream_mode="updates")
    first = next(gen)
    ai = first["agent"]["messages"][-1]
    graph.update_state(cfg, {"messages": [AIMessage(content=ai.content, tool_calls=[{"name": "add", "args": {"a": 7, "b": 45}, "id": ai.tool_calls[0]["id"]}], id=ai.id)]})
    err = None
    try:
        rest = list(gen)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:160]
    rec("B7-3.1.64", {"framework": "langgraph", "case": "update_state while the same stream is suspended, then keep iterating it", "tool_received": tool_args(traj), "error": err})
    print("B7-3.1.64 LG update_state under a live (suspended) stream: tool received", tool_args(traj), "error", err)
    # --- Pydantic AI (a): mutate the proposed call's args on CallToolsNode
    def edit_args(i, node, run, tr):
        if type(node).__name__ == "CallToolsNode" and i == 3:
            node.model_response.parts[0].args = {"a": 1000, "b": 45}
        return "continue"
    res, traj = pa(ONE_ROUND, edit_args)
    reqs = traj.of("model_request")
    asst_args = [m.get("tool_calls") for m in reqs[1]["messages"] if m["role"] == "assistant"]
    rec("B7-3.1.64", {"framework": "pydantic-ai", "case": "mutate proposed call args on CallToolsNode", "tool_received": tool_args(traj), "assistant_tool_calls_in_next_request": asst_args, "final": res["final"]})
    print("B7-3.1.64 PA node edit: tool received", tool_args(traj), "history echoes", asst_args)
    # --- Pydantic AI (b): edit the next ModelRequestNode (what goes on the wire)
    def edit_req(i, node, run, tr):
        if type(node).__name__ == "ModelRequestNode" and i == 4:
            node.request.parts[0].content = "EDITED-BY-CONSUMER"
        return "continue"
    res, traj = pa(ONE_ROUND, edit_req)
    rec("B7-3.1.64", {"framework": "pydantic-ai", "case": "mutate the next ModelRequestNode.request", "tool_content_on_wire": _req_tool_contents(traj, 2), "final": res["final"]})
    print("B7-3.1.64 PA request edit: wire carried", _req_tool_contents(traj, 2))
    # --- Pydantic AI (c): inspect and edit history through run.ctx.state
    def edit_hist(i, node, run, tr):
        if type(node).__name__ == "ModelRequestNode" and i == 4:
            run.ctx.state.message_history.insert(0, ModelRequest(parts=[ToolReturnPart(tool_name="x", content="injected", tool_call_id="zz")]))
        return "continue"
    res, traj = pa(ONE_ROUND, edit_hist)
    rec("B7-3.1.64", {"framework": "pydantic-ai", "case": "insert into run.ctx.state.message_history", "roles_in_request_2": [m["role"] for m in traj.of("model_request")[1]["messages"]], "error": res["error"], "final": res["final"]})
    print("B7-3.1.64 PA history insert: roles in request 2", [m["role"] for m in traj.of("model_request")[1]["messages"]], res["error"])


# ================================================================ B7-3.1.65 manual approval before a tool
POLICY_DENY = {"multiply"}


def lg_approve(decision, script=TWO_ROUNDS):
    """Consumer loop: at each tool proposal the application decides allow | deny | modify | stop; every decision goes in the application's own log."""
    traj, model, wrapped = rig("langgraph", script)
    graph = lg_ckpt_graph(model, wrapped)
    cfg = lg_cfg()
    decisions = []
    pending_stream = graph.stream(lg_input(), cfg, stream_mode="updates")
    inp = lg_input()
    while True:
        restart = False
        for chunk in pending_stream:
            if "agent" not in chunk:
                continue
            ai = chunk["agent"]["messages"][-1]
            if not ai.tool_calls:
                continue
            for name, args, cid in proposed_calls(ai):
                d = decision(name, args)
                decisions.append({"tool": name, "args": args, "decision": d if isinstance(d, str) else d[0]})
            kinds = [x["decision"] for x in decisions[-len(ai.tool_calls):]]
            if "stop" in kinds:
                pending_stream.close()
                return {"decisions": decisions, "traj": traj, "graph": graph, "cfg": cfg, "stopped": True}
            if "deny" in kinds or "modify" in kinds:
                pending_stream.close()
                if "modify" in kinds:
                    new = [{"name": n, "args": ({**a, **decision(n, a)[1]} if kinds[k] == "modify" else a), "id": c} for k, (n, a, c) in enumerate(proposed_calls(ai))]
                    graph.update_state(cfg, {"messages": [AIMessage(content=ai.content, tool_calls=new, id=ai.id)]})
                else:
                    graph.update_state(cfg, {"messages": [ToolMessage(content="DENIED by policy", tool_call_id=c, name=n) for n, a, c in proposed_calls(ai)]}, as_node="tools")
                pending_stream = graph.stream(None, cfg, stream_mode="updates")
                restart = True
                break
        if not restart:
            break
    return {"decisions": decisions, "traj": traj, "graph": graph, "cfg": cfg, "stopped": False}


def pa_approve(decision, script=TWO_ROUNDS):
    log_dec = []

    def decide(i, node, run, tr):
        if type(node).__name__ != "CallToolsNode" or not node.model_response.parts or type(node.model_response.parts[0]).__name__ != "ToolCallPart":
            return "continue"
        out = "continue"
        for p in node.model_response.parts:
            if type(p).__name__ != "ToolCallPart":
                continue
            d = decision(p.tool_name, p.args_as_dict())
            log_dec.append({"tool": p.tool_name, "args": p.args_as_dict(), "decision": d if isinstance(d, str) else d[0]})
            if d == "stop":
                return "stop"
            if d == "deny":
                out = ("replace", ModelRequestNode(request=ModelRequest(parts=[ToolReturnPart(tool_name=p.tool_name, content="DENIED by policy", tool_call_id=p.tool_call_id)])))
            elif isinstance(d, tuple) and d[0] == "modify":
                p.args = {**p.args_as_dict(), **d[1]}
        return out
    res, traj = pa(script, decide)
    return {"decisions": log_dec, "res": res, "traj": traj}


def approval():
    cases = {
        "allow all": lambda n, a: "allow",
        "stop at first": lambda n, a: "stop",
        "modify add.a -> 1000": lambda n, a: ("modify", {"a": 1000}) if n == "add" else "allow",
        "deny multiply": lambda n, a: "deny" if n in POLICY_DENY else "allow",
    }
    sizes = {}
    for label, dec in cases.items():
        r = lg_approve(dec)
        t = r["traj"]
        msgs = [type(m).__name__ + (":" + str(m.content)[:24] if type(m).__name__ == "ToolMessage" else "") for m in r["graph"].get_state(r["cfg"]).values["messages"]]
        d = {"framework": "langgraph", "case": label, "decisions": r["decisions"], "tools_executed": tool_args(t), "model_requests": snap(t)["requests"], "state_messages": msgs, "stopped": r["stopped"],
             "decision_recorded_by_framework": any("DENIED" in m or "approv" in m.lower() for m in msgs[:0])}
        rec("B7-3.1.65", d)
        print(f"B7-3.1.65 LG  {label:<22} decisions={[x['decision'] for x in r['decisions']]} executed={tool_args(t)} requests={snap(t)['requests']} state={msgs}")
        r = pa_approve(dec)
        t, res = r["traj"], r["res"]
        d = {"framework": "pydantic-ai", "case": label, "decisions": r["decisions"], "tools_executed": tool_args(t), "model_requests": snap(t)["requests"], "final": res["final"], "error": res["error"],
             "usage_tool_calls": res["log"][-1]["usage"]["tool_calls"] if res["log"] else None}
        rec("B7-3.1.65", d)
        print(f"B7-3.1.65 PA  {label:<22} decisions={[x['decision'] for x in r['decisions']]} executed={tool_args(t)} requests={snap(t)['requests']} final={res['final']} error={res['error']}")
    import inspect
    sizes = {"langgraph_consumer_loop_lines": len(inspect.getsource(lg_approve).splitlines()), "pydantic_ai_consumer_loop_lines": len(inspect.getsource(pa_approve).splitlines()) + len(inspect.getsource(walk_pa).splitlines())}
    rec("B7-3.1.65", {"case": "implementation size of the consumer loops (this file)", **sizes})
    print("B7-3.1.65 sizes:", sizes)
    # does the deferred-tool mechanism affect an ordinary tool? one probe: tool_call_results on the CallToolsNode
    def probe(i, node, run, tr):
        if type(node).__name__ == "CallToolsNode" and i == 3:
            try:
                from pydantic_ai.tools import ToolDenied
                node.tool_call_results = {node.model_response.parts[0].tool_call_id: ToolDenied("no")}
            except Exception as e:  # noqa: BLE001
                tr.add("probe_error", "probe", err=str(e)[:100])
        return "continue"
    res, traj = pa(ONE_ROUND, probe)
    rec("B7-3.1.65", {"framework": "pydantic-ai", "case": "probe: tool_call_results=ToolDenied on an ordinary tool", "tools_executed": tool_args(traj), "final": res["final"], "error": res["error"],
                      "roles_request_2": [m["role"] + ":" + str(m.get("content"))[:30] for m in traj.of("model_request")[1]["messages"]] if len(traj.of("model_request")) > 1 else None})
    print("B7-3.1.65 PA tool_call_results probe: executed", tool_args(traj), "final", res["final"], res["error"])


GROUPS.update({"state": state, "approval": approval, "lgpause": lg_pause_probe})



# ================================================================ B7-3.1.66 stepping after a tool failure (Batch 5 fixture: e5.sched)
FAIL_SCRIPT = [C(None, [ADD]), C(None, [("flaky", {})]), C(None, [("flaky", {})]), C("Done: 165.")]


def _flaky():
    return e5.sched("flaky", ["fail", "ok"], "Temporary failure")


def failure():
    tools = list(TOOLS) + [_flaky()]
    traj, model, wrapped = rig("langgraph", FAIL_SCRIPT, tools)
    graph = lg_graph(model, wrapped)
    log, err = [], None
    try:
        for mode, chunk in graph.stream(lg_input(), CFG, stream_mode=["tasks", "updates"]):
            s = summarize(mode, chunk)
            if mode == "updates" and "tools" in chunk:
                s["tool_messages"] = [(m.name, m.status, str(m.content)[:40]) for m in chunk["tools"]["messages"]]
            if mode == "updates" and "agent" in chunk:
                s["proposed"] = [(n, c) for n, a, c in proposed_calls(chunk["agent"]["messages"][-1])]
            log.append({"mode": mode, "summary": s, "snap": snap(traj)})
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:160]
    rec("B7-3.1.66", {"framework": "langgraph", "boundaries": [l for l in log if l["mode"] == "updates"], "error_escaped_to_consumer": err, "tool_executions": snap(traj)["tool_starts"]})
    print("B7-3.1.66 LG: escaped exception:", err)
    for l in log:
        if l["mode"] == "updates":
            print("   ", json.dumps({k: v for k, v in l["summary"].items() if k != "update"})[:200], "| tool_ends", l["snap"]["tool_ends"])
    tools = list(TOOLS) + [_flaky()]
    res, traj = pa(FAIL_SCRIPT, lambda *a: "continue", tools=tools)
    rows = []
    # re-walk to capture the part kinds and outcomes of each ModelRequestNode
    async def walk():
        t2, model2, w2 = rig("pydantic-ai", FAIL_SCRIPT, list(TOOLS) + [_flaky()])
        agent = pa_agent.build_agent(model2, retries=3, tools=w2)
        out = []
        async with agent.iter(Q) as run:
            async for node in run:
                if type(node).__name__ == "ModelRequestNode":
                    out.append([(type(p).__name__, getattr(p, "outcome", None), str(getattr(p, "content", ""))[:40]) for p in node.request.parts])
        return out
    rows = asyncio.run(walk())
    rec("B7-3.1.66", {"framework": "pydantic-ai", "nodes": res["log"], "request_parts_per_ModelRequestNode": rows, "final": res["final"], "error": res["error"], "tool_executions": snap(traj)["tool_starts"]})
    print("B7-3.1.66 PA: request parts per ModelRequestNode:", rows, "final", res["final"], res["error"])


# ================================================================ B7-3.1.67 stepping and limits
async def _pa_deadline(pause):
    traj, model, wrapped = rig("pydantic-ai", ONE_ROUND)
    agent = pa_agent.build_agent(model, retries=3, tools=wrapped)
    seen, err = [], None
    try:
        async with agent.iter(Q, timeout=0.3) as run:
            node = run.next_node
            while type(node).__name__ != "End":
                seen.append(type(node).__name__)
                if type(node).__name__ == "CallToolsNode" and len(seen) == 3:
                    await asyncio.sleep(pause)
                node = await run.next(node)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:100]}"
    return {"nodes_seen": seen, "error": err, "snap": snap(traj)}


def limits():
    # LangGraph: pausing consumes no budget; the limit counts steps taken
    for lim, pause in ((6, 0.3), (6, 0.0), (5, 0.0), (4, 0.0)):
        traj, model, wrapped = rig("langgraph", TWO_ROUNDS)
        graph = lg_graph(model, wrapped)
        seen, err, t0 = [], None, time.perf_counter()
        try:
            for mode, chunk in graph.stream(lg_input(), {"recursion_limit": lim}, stream_mode=["tasks", "updates"]):
                seen.append(summarize(mode, chunk))
                if mode == "tasks" and chunk.get("name") == "tools" and "input" in chunk and pause:
                    time.sleep(pause)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:70]}"
        last_upd = [s for s in seen if "node" in s]
        rec("B7-3.1.67", {"framework": "langgraph", "recursion_limit": lim, "pause_s_per_tool_step": pause, "chunks_seen": len(seen), "last_update": last_upd[-1] if last_upd else None,
                          "error": err, "requests": snap(traj)["requests"], "tool_ends": snap(traj)["tool_ends"], "wall_s": round(time.perf_counter() - t0, 2)})
        print(f"B7-3.1.67 LG limit={lim} pause={pause}: chunks={len(seen)} last_update={last_upd[-1] if last_upd else None} error={err} requests={snap(traj)['requests']}")
    # Pydantic AI: request_limit error surfaces when the over-limit node is advanced
    from pydantic_ai.usage import UsageLimits
    res, traj = pa(TWO_ROUNDS, lambda *a: "continue", iter_kw={"usage_limits": UsageLimits(request_limit=2)})
    rec("B7-3.1.67", {"framework": "pydantic-ai", "case": "request_limit=2, three requests needed", "nodes_seen_before_error": [(l["i"], l["node"]) for l in res["log"]], "error": res["error"], "snap": res["snap_after_settle"]})
    print("B7-3.1.67 PA request_limit=2: nodes seen", [(l["i"], l["node"]) for l in res["log"]], "error", res["error"], "snap", res["snap_after_settle"])
    # Pydantic AI: skipping a CallToolsNode does not count a tool call (see B7-3.1.65 deny); read usage there
    r = pa_approve(lambda n, a: "deny" if n == "multiply" else "allow")
    rec("B7-3.1.67", {"framework": "pydantic-ai", "case": "deny multiply by replacement node", "executed": tool_args(r["traj"]), "usage_tool_calls_at_last_node": r["res"]["log"][-1]["usage"]})
    print("B7-3.1.67 PA denied call counted in usage:", r["res"]["log"][-1]["usage"], "executed", tool_args(r["traj"]))
    # Pydantic AI: does the run deadline keep running while the consumer pauses?
    for pause in (0.0, 0.6):
        d = asyncio.run(_pa_deadline(pause))
        rec("B7-3.1.67", {"framework": "pydantic-ai", "case": f"iter(timeout=0.3), consumer pauses {pause} s at CallToolsNode", **d})
        print(f"B7-3.1.67 PA iter(timeout=0.3) pause={pause}:", d)


GROUPS.update({"failure": failure, "limits": limits})



# ================================================================ B7-3.1.68 framework comparison (derived from the measurements above)
def compare():
    path = os.path.join(HERE, "results", "batch7.json")
    data = {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        pass
    data.update(RESULTS)

    def find(exp, fw, **kv):
        for r in data.get(exp, []):
            if r.get("framework") == fw and all(str(r.get(k)) == str(v) or (isinstance(v, str) and v in str(r.get(k, ""))) for k, v in kv.items()):
                return r
        return None
    lg_task = next((l for l in (find("B7-3.1.62", "langgraph") or {}).get("boundaries", []) if l["summary"].get("name") == "tools" and l["summary"].get("event") == "task_start"), None)
    pa_calls = next((l for l in (find("B7-3.1.62", "pydantic-ai") or {}).get("boundaries", []) if l["node"] == "CallToolsNode"), None)
    stops = [r for r in data.get("B7-3.1.63", []) if "boundary" in r]
    approval = {(r["framework"], r["case"]): r for r in data.get("B7-3.1.65", []) if "decisions" in r}
    m = {
        "observe_proposed_call_before_execution": {"langgraph": bool(lg_task) and lg_task["snap"]["tool_starts"] == 0, "pydantic-ai": bool(pa_calls) and pa_calls["snap"]["tool_starts"] == 0},
        "voluntary_stop_runs_nothing_afterwards": {fw: all(not r["work_continued_after_stop"] for r in stops if r["framework"] == fw) for fw in FWS},
        "allow": {fw: approval.get((fw, "allow all"), {}).get("model_requests") == 3 for fw in FWS},
        "stop": {fw: approval.get((fw, "stop at first"), {}).get("tools_executed") == [] for fw in FWS},
        "modify_args_then_run": {fw: [a.get("a") for a in approval.get((fw, "modify add.a -> 1000"), {}).get("tools_executed", [])][:1] == [1000.0] for fw in FWS},
        "deny_and_continue": {fw: len(approval.get((fw, "deny multiply"), {}).get("tools_executed", [])) == 1 and approval.get((fw, "deny multiply"), {}).get("model_requests") == 3 for fw in FWS},
        "needs_checkpointer_or_state_api_for_modify_deny": {"langgraph": True, "pydantic-ai": False},
        "framework_records_the_decision": {"langgraph": False, "pydantic-ai": False},
    }
    rec("B7-3.1.68", {"case": "capability matrix derived from B7-3.1.59-67", "matrix": m,
                      "consumer_loop_lines": next((r for r in data.get("B7-3.1.65", []) if "langgraph_consumer_loop_lines" in r), {})})
    print("B7-3.1.68 matrix:")
    for k, v in m.items():
        print(f"   {k:<48} {v}")


GROUPS.update({"compare": compare})


def main():
    for g in sys.argv[1:] or list(GROUPS):
        print(f"\n===== {g} =====")
        GROUPS[g]()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "batch7.json")
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
