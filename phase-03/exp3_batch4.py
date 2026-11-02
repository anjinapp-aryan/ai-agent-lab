"""Phase 3.1 Batch 4: repeated calls, oscillation and no progress (B4-3.1.23 … B4-3.1.38). Offline first.

Reuses the Batch 3 runners and scripted transport, the Phase 2 `Orders` and `make_flaky` fixtures and the recorder.
Nothing here intervenes in a run: each framework runs its own loop, and `lab3.patterns.classify` only reads the record
afterwards. New fixtures are one-line tools with scripted results. Hypotheses: phases/phase-03/experiments.md (Batch 4).

Budgets are model decisions k: LangGraph recursion_limit=2k, Pydantic AI request_limit=k (measured in Batch 3).
Usage: uv run python -u phase-03/exp3_batch4.py [group ...] | live     groups: repeat oscillate progress effects errors empty fallback
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_batch3 as e3  # noqa: E402  (sets sys.path)

import httpx  # noqa: E402

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab2 import fixtures as fx  # noqa: E402
from lab3.patterns import classify  # noqa: E402

C, FWS = e3.C, e3.FWS
RESULTS: dict = {}
Q = "Handle the request."
K = 6


def budget(fw, k=K):
    return 2 * k if fw == "langgraph" else k


# ---------------------------------------------------------------- tiny tools
def get_order(order_id: str) -> str:
    """Get the details of an order."""
    return f"order {order_id}: status=SHIPPED"


def tool_a() -> str:
    """Tool A."""
    return "A done"


def tool_b() -> str:
    """Tool B."""
    return "B done"


def arg_a(n: int) -> str:
    """Tool A with an argument."""
    return f"A({n}) done"


def arg_b(n: int) -> str:
    """Tool B with an argument."""
    return f"B({n}) done"


def search(query: str) -> list:
    """Search for items."""
    return []


def scripted_status(results):
    """check_status returning each result in turn; the last one repeats."""
    state = {"n": 0}

    def check_status(order_id: str) -> str:
        """Check the current status of an order."""
        out = results[min(state["n"], len(results) - 1)]
        state["n"] += 1
        return out
    return check_status


def alternating(how):
    """Fails on odd executions, succeeds on even ones."""
    state = {"n": 0}

    def flaky() -> str:
        """Alternates between failing and succeeding."""
        state["n"] += 1
        if state["n"] % 2:
            if how == "model_retry":
                from pydantic_ai import ModelRetry
                raise ModelRetry("failed, retry")
            raise RuntimeError("failed")
        return "ok"
    return flaky


def call(name, **args):
    return C(None, [(name, args)])


def same_id_transport(script, call_id="call_same"):
    """Scripted completions in which every tool call carries the same call ID."""
    inner, log = e3.b2.raw_transport(script)

    def handler(request):
        r = inner.handle_request(request)
        r.read()
        data = r.json()
        for c in data["choices"][0]["message"].get("tool_calls") or []:
            c["id"] = call_id
        return httpx.Response(200, json=data)
    return httpx.MockTransport(handler), log


def reasoning_transport(script, text="The model's private reasoning."):
    """Scripted completions in which each response carries a `reasoning` field, as gpt-oss-120b's real responses do."""
    inner, log = e3.b2.raw_transport(script)

    def handler(request):
        r = inner.handle_request(request)
        r.read()
        data = r.json()
        data["choices"][0]["message"]["reasoning"] = text
        return httpx.Response(200, json=data)
    return httpx.MockTransport(handler), log


# ---------------------------------------------------------------- trajectory view (logical index kept apart from native counters)
def trajectory(traj):
    calls, rows, starts = [], [], {}
    for i, e in enumerate(traj.events):
        k = e["kind"]
        if k == "model_response":
            calls += [{"request": e["n"], "id": c["id"], "name": c["name"], "args": c["args"]} for c in e["tool_calls"]]
        elif k == "tool_start":
            starts.setdefault(e["tool"], []).append(e["args"])
        elif k == "tool_end":
            pending = next((c for c in calls if c["name"] == e["tool"] and not c.get("done")), None)
            reexecution = pending is None                              # a framework retry: no new model decision
            if reexecution:
                pending = next((c for c in reversed(calls) if c["name"] == e["tool"]), {})
            pending["done"] = True
            step = sum(1 for x in traj.events[:i] if x["kind"] == "framework_step")
            rows.append({"logical": len(rows) + 1, "framework_step_events_before": step, "model_request": pending.get("request"),
                         "tool": e["tool"], "args": starts[e["tool"]].pop(0), "call_id": pending.get("id"), "reexecution": reexecution,
                         "outcome": e.get("result", e.get("error"))})
    return rows


def wire_check(traj):
    """What the framework added to the conversation: anything other than model and tool messages would be a warning."""
    reqs = traj.of("model_request")
    fixed = [[(m["role"], m.get("content")) for m in r["messages"] if m["role"] in ("system", "user")] for r in reqs]
    return {"requests": len(reqs), "system_user_unchanged": all(f == fixed[0] for f in fixed) if fixed else None,
            "last_roles": reqs[-1]["roles"] if reqs else [],
            "last_tool_call_ids": [m.get("tool_call_id") for m in reqs[-1]["messages"] if m["role"] == "tool"] if reqs else [],
            "last_tool_contents": [m.get("content") for m in reqs[-1]["messages"] if m["role"] == "tool"][-3:] if reqs else []}


def case(exp, label, fw, script, limit, tools, transport=None, **kw):
    fx.EXEC.clear()
    d = e3.run(fw, Q, script, limit, tools=tools, transport=transport, **kw)
    traj = e3.LAST_TRAJ
    rows = trajectory(traj)
    w = wire_check(traj)
    stop = "natural (no tool calls)" if d["result"] == "OK" else d["result"]
    row = {"exp": exp, "case": label, "framework": fw, "limit": limit, "model_requests": d["framework_model_calls"], "http_attempts": d["http_attempts"],
           "tool_calls_requested": d["tool_calls_requested"], "tool_executions": d["tool_executions"], "framework_steps": d["framework_steps"],
           "stopped_by": stop, "error": d["error"], "final_text": d["final_text"], "call_ids": [r["call_id"] for r in rows],
           "label": classify([(r["tool"], r["args"], r["outcome"]) for r in rows]), "trajectory": rows, "wire": w,
           "framework_flagged_repetition": _flagged(d, w)}
    RESULTS.setdefault(exp, []).append(row)
    print(f"{exp:<11}{label[:40]:<41}{fw[:2]} lim={limit:<3} req={row['model_requests']} calls={row['tool_calls_requested']} exec={row['tool_executions']} "
          f"steps={row['framework_steps']:<3} label={row['label']:<14} stop={stop[:26]} {(d['error'] or '')[:60]}")
    return row


def _flagged(d, w):
    """True if the error names repetition, or the framework changed the system/user messages during the run."""
    words = ("repeat", "duplicate", "loop", "oscillat", "no progress", "stuck")
    return any(x in (d["error"] or "").lower() for x in words) or w["system_user_unchanged"] is False


# ---------------------------------------------------------------- groups
def repeat():
    for fw in FWS:
        case("B4-3.1.23", "get_order(123) forever", fw, [call("get_order", order_id="123")] * 40, budget(fw), [get_order])
    for fw in FWS:
        case("B4-3.1.24", "3 identical calls, different IDs, then final", fw, [call("get_order", order_id="123")] * 3 + [C("done")], 25, [get_order])
    for fw in FWS:
        t, _ = same_id_transport([call("get_order", order_id="123")] * 3 + [C("done")])
        case("B4-3.1.24", "3 identical calls, SAME ID across turns", fw, [], 25, [get_order], transport=t)
    for fw in FWS:
        t, _ = same_id_transport([C(None, [("get_order", {"order_id": "123"}), ("get_order", {"order_id": "123"})]), C("done")])
        case("B4-3.1.24", "2 identical calls, SAME ID in one response", fw, [], 25, [get_order], transport=t)
    for fw in FWS:
        case("B4-3.1.24", "2 identical calls, 2 IDs in one response", fw, [C(None, [("get_order", {"order_id": "123"})] * 2), C("done")], 25, [get_order])
    for fw in FWS:
        case("B4-3.1.25", "control: 123,124,125,126, then final", fw, [call("get_order", order_id=str(n)) for n in (123, 124, 125, 126)] + [C("done")], 25, [get_order])


def oscillate():
    ab = [call("tool_a"), call("tool_b")] * 20
    for fw in FWS:
        case("B4-3.1.26", "A,B,A,B forever", fw, ab, budget(fw), [tool_a, tool_b])
    args = [s for n in range(1, 21) for s in (call("arg_a", n=n), call("arg_b", n=n))]
    for fw in FWS:
        case("B4-3.1.27", "A(1),B(1),A(2),B(2),... forever", fw, args, budget(fw), [arg_a, arg_b])


def progress():
    st = call("check_status", order_id="123")
    for fw in FWS:
        case("B4-3.1.28", "PENDING forever", fw, [st] * 40, budget(fw), [scripted_status(["PENDING"])])
    texts = ["PENDING", "Still pending.", "Processing has not completed."]
    for fw in FWS:
        case("B4-3.1.29", "reworded PENDING forever", fw, [st] * 40, budget(fw), [scripted_status(texts * 20)])
    for fw in FWS:
        case("B4-3.1.30", "PROCESSING,PROCESSING,SHIPPED, then final", fw, [st] * 3 + [C("Shipped.")], 25, [scripted_status(["PROCESSING", "PROCESSING", "SHIPPED"])])
    for fw in FWS:
        case("B4-3.1.31", "PENDING,PENDING,COMPLETE, then final", fw, [st] * 3 + [C("Complete.")], 25, [scripted_status(["PENDING", "PENDING", "COMPLETE"])])
    for fw in FWS:                                                     # the limit cannot tell legitimate work from a runaway
        case("B4-3.1.31", "same polling, budget of 2 decisions", fw, [st] * 3 + [C("Complete.")], budget(fw, 2), [scripted_status(["PENDING", "PENDING", "COMPLETE"])])


def effects():
    plain = {"item": "widget", "qty": 1}

    def keyed(k):
        return call("create_order", item="widget", qty=1, idempotency_key=k)

    def order_case(exp, label, fw, script, make_tool):
        o = fx.Orders()
        r = case(exp, label, fw, script, 25, [make_tool(o)])
        r["ledger"], r["replays"] = len(o.ledger), o.replays
        print(f"   ledger={r['ledger']} replays={r['replays']}")
    for fw in FWS:
        order_case("B4-3.1.32", "A: no guard, 3 calls", fw, [call("create_order", **plain)] * 3 + [C("done")], lambda o: o.unguarded())
        order_case("B4-3.1.32", "B: guard, same key, 3 calls", fw, [keyed("same-key")] * 3 + [C("done")], lambda o: o.model_keyed())
        order_case("B4-3.1.33", "guard, keys A,B,C", fw, [keyed(k) for k in "ABC"] + [C("done")], lambda o: o.model_keyed())
        order_case("B4-3.1.34", "guard, same key A, 2 call IDs", fw, [keyed("A"), keyed("A"), C("done")], lambda o: o.model_keyed())
        order_case("B4-3.1.34", "guard, same intent, keys A then B", fw, [keyed("A"), keyed("B"), C("done")], lambda o: o.model_keyed())


def errors():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)
    fail = [call("flaky")] * 40
    forever = 10 ** 6
    case("B4-3.1.35", "LG default handler", "langgraph", fail, budget("langgraph"), [fx.make_flaky(forever, "runtime")])
    case("B4-3.1.35", "LG RetryPolicy(3), handle=False", "langgraph", fail, budget("langgraph"), [fx.make_flaky(forever, "runtime")], handle_tool_errors=False, retry_policy=rp)
    case("B4-3.1.35", "LG RetryPolicy(3), default handler", "langgraph", fail, budget("langgraph"), [fx.make_flaky(forever, "runtime")], retry_policy=rp)
    case("B4-3.1.35", "LG default, alternating fail/ok", "langgraph", fail, budget("langgraph"), [alternating("runtime")])
    for retries in (1, 3):
        case("B4-3.1.35", f"PA ModelRetry, retries={retries}", "pydantic-ai", fail, budget("pydantic-ai"), [fx.make_flaky(forever, "model_retry")], retries=retries)
    case("B4-3.1.35", "PA plain exception, retries=3", "pydantic-ai", fail, budget("pydantic-ai"), [fx.make_flaky(forever, "runtime")], retries=3)
    case("B4-3.1.35", "PA ModelRetry alternating, retries=1", "pydantic-ai", fail, budget("pydantic-ai"), [alternating("model_retry")], retries=1)


def empty():
    for fw in FWS:
        case("B4-3.1.36", "search -> [] forever", fw, [call("search", query="x")] * 40, budget(fw), [search])


def fallback():
    """B4-3.1.38: the same three runaway shapes against every native stop available."""
    shapes = {"A,A,A": ([call("get_order", order_id="123")] * 80, lambda: [get_order]),
              "A,B,A,B": ([call("tool_a"), call("tool_b")] * 40, lambda: [tool_a, tool_b]),
              "PENDING x n": ([call("check_status", order_id="123")] * 80, lambda: [scripted_status(["PENDING"])])}
    for name, (script, tools) in shapes.items():
        for fw in FWS:
            case("B4-3.1.38", f"{name}: loop limit", fw, script, budget(fw), tools())
        case("B4-3.1.38", f"{name}: tool_calls_limit=3", "pydantic-ai", script, 25, tools(), tool_calls_limit=3)
        case("B4-3.1.38", f"{name}: default request_limit=50", "pydantic-ai", script, 50, tools())


# ---------------------------------------------------------------- live: model behavior, small
LIVE_PROMPT = "You are an order support assistant. Use the available tools to help the user with their orders."
PING_A = "Step A is complete. Now call ping_b to continue."
PING_B = "Step B is complete. Now call ping_a to continue."


def _run_live(fw, q, limit, tools):
    traj = e3.Trajectory()
    model = e3.b1.MODELS[fw](traj)
    wrapped = [e3.traced(t, traj) for t in tools]
    info = e3.run_lg(q, model, wrapped, traj, limit) if fw == "langgraph" else e3.run_pa(q, model, wrapped, traj, limit)
    return {**info, "traj": traj}


def live(trials=2):
    from lab_common.config import load_model_config
    cfg = load_model_config()
    print(f"live model: {cfg.provider}/{cfg.model_name}, temperature 0, {trials} trials per framework per case, budget {K} decisions")
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = LIVE_PROMPT

    def ping_a() -> str:
        """Run step A of the job."""
        return PING_A

    def ping_b() -> str:
        """Run step B of the job."""
        return PING_B
    cases = [("repeated: PENDING forever", "Check order 123 and keep checking until its status is COMPLETE.", lambda: [scripted_status(["PENDING"])]),
             ("oscillation: tools point at each other", "Run the job, starting with ping_a.", lambda: [ping_a, ping_b]),
             ("polling: PROCESSING,PROCESSING,SHIPPED", "Check order 123 and keep checking until it has shipped, then tell me.", lambda: [scripted_status(["PROCESSING", "PROCESSING", "SHIPPED"])])]
    for label, q, mk in cases:
        for trial in range(1, trials + 1):
            for fw in FWS:
                d = _run_live(fw, q, budget(fw), mk())
                rows = trajectory(d["traj"])
                a = d["traj"].accounting()
                row = {"case": label, "framework": fw, "trial": trial, "model": cfg.model_name, "tools": [r["tool"] for r in rows], "outcomes": [r["outcome"] for r in rows],
                       "label": classify([(r["tool"], r["args"], r["outcome"]) for r in rows]), "model_requests": d["framework_model_calls"], "http_attempts": a["model_requests"],
                       "http_errors": a["http_errors"], "tokens": a["tokens_total"], "error": d["error"], "final_text": (d["final_text"] or "")[:200]}
                RESULTS.setdefault("B4-live", []).append(row)
                line = (f"LIVE {label[:38]:<39}{fw[:2]} #{trial} tools={row['tools']} label={row['label']} req={row['model_requests']} tok={row['tokens']} "
                        f"err={(row['error'] or '')[:50]!r} final={row['final_text'][:60]!r}")
                print(line.encode("ascii", "replace").decode())
                if row["error"] and "429" in row["error"]:
                    print("LIVE STOPPED (quota)")
                    return
                time.sleep(2)


def ablation(trials=3):
    """Why did the same model poll under LangGraph but stop under Pydantic AI? Replay the framework's second request
    (after the first PENDING) directly with httpx, varying only the envelope. Bypasses both frameworks."""
    from lab_common.config import load_model_config
    cfg = load_model_config()
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = LIVE_PROMPT
    question = "Check order 123 and keep checking until its status is COMPLETE."
    captured = {}
    for fw in FWS:
        t, log = e3.b2.raw_transport([call("check_status", order_id="123"), C("done")])
        tools = [scripted_status(["PENDING"])]
        model = e3.b1.MODELS[fw](e3.Trajectory(), transport=t)
        (lg_agent.run if fw == "langgraph" else pa_agent.run)(question, model, tools=tools)
        captured[fw] = {**log[1], "model": cfg.model_name}

    def closed(req, value):
        r = json.loads(json.dumps(req))
        for t in r["tools"]:
            t["function"].pop("strict", None)
            if value:
                t["function"]["parameters"]["additionalProperties"] = False
            else:
                t["function"]["parameters"].pop("additionalProperties", None)
        return r
    variants = {"LG request as sent": captured["langgraph"], "LG + additionalProperties:false": closed(captured["langgraph"], True),
                "PA request as sent": captured["pydantic-ai"], "PA without strict/additionalProperties": closed(captured["pydantic-ai"], False)}
    envelope = {fw: {k: v for k, v in r.items() if k not in ("messages", "tools")} for fw, r in captured.items()}
    print("envelopes:", envelope)
    print("messages identical:", captured["langgraph"]["messages"] == captured["pydantic-ai"]["messages"])
    RESULTS["B4-ablation-envelopes"] = {"envelopes": envelope, "messages": {fw: r["messages"] for fw, r in captured.items()},
                                        "tools": {fw: r["tools"] for fw, r in captured.items()}}
    for name, req in variants.items():
        for i in range(1, trials + 1):
            r = httpx.post(cfg.base_url.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {cfg.api_key}"}, json=req, timeout=60)
            body = r.json()
            msg = body["choices"][0]["message"] if r.status_code == 200 else {}
            outcome = "tool call" if msg.get("tool_calls") else ("text" if r.status_code == 200 else f"HTTP {r.status_code}")
            row = {"variant": name, "trial": i, "model": cfg.model_name, "status": r.status_code, "outcome": outcome, "text": (msg.get("content") or "")[:150],
                   "error": "" if r.status_code == 200 else r.text[:300]}
            RESULTS.setdefault("B4-ablation", []).append(row)
            print(f"ABL {name:<40}#{i} {r.status_code} {outcome:<10} {row['text'][:70]!r} {row['error'][:120]}".encode("ascii", "replace").decode())
            if r.status_code == 429:
                print("ABLATION STOPPED (quota)")
                return
            time.sleep(1.5)


def reasoning(trials=3):
    """The mock replay in ablation() has no `reasoning`; real responses do. Capture each framework's real second request,
    then replay it directly, adding or removing only the assistant message's `reasoning` field."""
    from lab_common.config import load_model_config
    cfg = load_model_config()
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = LIVE_PROMPT
    question = "Check order 123 and keep checking until its status is COMPLETE."
    real = {}
    for fw in FWS:
        d = _run_live(fw, question, budget(fw, 2), [scripted_status(["PENDING"])])
        reqs = d["traj"].of("model_request")
        real[fw] = reqs[1]
        print(fw, "second request roles", reqs[1]["roles"], "assistant has reasoning:", "reasoning" in reqs[1]["messages"][2])
    pa_msgs, lg_msgs = real["pydantic-ai"]["messages"], real["langgraph"]["messages"]
    text = pa_msgs[2].get("reasoning")
    without = json.loads(json.dumps(pa_msgs))
    without[2].pop("reasoning", None)
    added = json.loads(json.dumps(lg_msgs))
    added[2]["reasoning"] = text
    schema = RESULTS.get("B4-ablation-envelopes", {}).get("tools")
    if not schema:
        with open(os.path.join(HERE, "results", "batch4.json"), encoding="utf-8") as f:
            schema = json.load(f)["B4-ablation-envelopes"]["tools"]
    pa_env = {"model": cfg.model_name, "stream": False, "temperature": 0, "tool_choice": "auto", "tools": schema["pydantic-ai"]}
    lg_env = {"model": cfg.model_name, "stream": False, "temperature": 0, "tools": schema["langgraph"]}
    variants = {"PA real request (with reasoning)": {**pa_env, "messages": pa_msgs}, "PA real request, reasoning removed": {**pa_env, "messages": without},
                "LG real request (no reasoning)": {**lg_env, "messages": lg_msgs}, "LG real request + PA's reasoning": {**lg_env, "messages": added}}
    RESULTS["B4-reasoning-text"] = text
    for name, req in variants.items():
        for i in range(1, trials + 1):
            r = httpx.post(cfg.base_url.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {cfg.api_key}"}, json=req, timeout=60)
            msg = r.json()["choices"][0]["message"] if r.status_code == 200 else {}
            outcome = "tool call" if msg.get("tool_calls") else ("text" if r.status_code == 200 else f"HTTP {r.status_code}")
            row = {"variant": name, "trial": i, "model": cfg.model_name, "status": r.status_code, "outcome": outcome, "text": (msg.get("content") or "")[:150], "error": "" if r.status_code == 200 else r.text[:300]}
            RESULTS.setdefault("B4-reasoning", []).append(row)
            print(f"RSN {name:<36}#{i} {r.status_code} {outcome:<10} {row['text'][:70]!r} {row['error'][:120]}".encode("ascii", "replace").decode())
            if r.status_code == 429:
                print("STOPPED (quota)")
                return
            time.sleep(1.5)


GROUPS = {"repeat": repeat, "oscillate": oscillate, "progress": progress, "effects": effects, "errors": errors, "empty": empty,
          "fallback": fallback, "live": live, "ablation": ablation, "reasoning": reasoning}


def main():
    for g in sys.argv[1:] or [k for k in GROUPS if k not in ("live", "ablation", "reasoning")]:
        print(f"\n===== {g} =====")
        GROUPS[g]()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "batch4.json")
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
