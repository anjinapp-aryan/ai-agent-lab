"""Phase 3.1 Batch 5: retry and failure recovery within the agent loop (B5-3.1.39 ... B5-3.1.47).

Scripted offline runs are the primary evidence; a small live set covers only questions where model behavior matters.
Reuses: the Batch 3 runners (`e3.run`, `run_lg`, `run_pa`), the Batch 4 trajectory view and wire check, the Phase 2
`Orders` and `make_flaky_internal_retry` fixtures, the recorder, and the Phase 1 `run()` entry points. New code is
scripted tool fixtures and measurement helpers only; nothing here retries, recovers or decides.
Hypotheses and gap analysis: phases/phase-03/experiments.md, phases/phase-03/batch5-gap-analysis.md.

Live cells (the reasoning-echo correction): LangGraph never echoes the model's reasoning, so its native run is already the
controlled condition. Pydantic AI has two: native (echoes) and controlled (`reasoning="stripped"`). Cells are never
compared across modes as if equivalent.

Usage: uv run python -u phase-03/exp3_batch5.py [group ...] | live [43 42 41]
       offline groups: recoverable permanent success side wording multi natural exhaustion accounting
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_batch4 as e4  # noqa: E402  (sets sys.path; brings e3, b2, b1 helpers)

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab2 import fixtures as fx  # noqa: E402
from lab3.trajectory import Trajectory, traced  # noqa: E402
from pydantic_ai import ModelRetry  # noqa: E402

e3, C, call, FWS = e4.e3, e4.C, e4.call, e4.FWS
RESULTS: dict = {}
Q = "Handle the request."


# ---------------------------------------------------------------- scripted tool fixtures
def sched(name, schedule, msg="Temporary failure"):
    """Tool whose n-th execution is schedule[n] ('ok' | 'fail'); the last entry repeats. A failure raises ModelRetry(msg)."""
    state = {"n": 0}

    def tool() -> str:
        i = state["n"]
        state["n"] += 1
        if schedule[min(i, len(schedule) - 1)] == "fail":
            raise ModelRetry(msg)
        return f"{name} succeeded (execution {i + 1})"
    tool.__name__ = name
    tool.__doc__ = f"Run {name}."
    return tool


def fetch_report_tool(msg, how="raise"):
    """fetch_report(report_id) that always fails with `msg`, either raised (framework template applies) or returned."""
    def fetch_report(report_id: str) -> str:
        """Fetch a report by id."""
        if how == "return":
            return msg
        raise ModelRetry(msg)
    return fetch_report


class LostResponse(ModelRetry):
    """The ambiguous outcome a timeout gives: it does not say whether the order was created."""
    def __init__(self, _ignored="") -> None:
        super().__init__("The order service timed out before returning a response.")


def _counts(rows):
    out = {}
    for r in rows:
        out[r["tool"]] = out.get(r["tool"], 0) + 1
    return out


def _tool_msgs(traj):
    """Per model request: [(tool_call_id, content)] of the tool messages the model was sent."""
    return [[[m.get("tool_call_id"), str(m.get("content"))[:140]] for m in r["messages"] if m["role"] == "tool"] for r in traj.of("model_request")]


def case(exp, label, fw, script, limit, tools, transport=None, q=Q, **kw):
    fx.EXEC.clear()
    d = e3.run(fw, q, script, limit, tools=tools, transport=transport, **kw)
    traj = e3.LAST_TRAJ
    rows = e4.trajectory(traj)
    ends = traj.of("tool_end")
    failed = [e for e in ends if "error" in e]
    fail_idx = [i for i, e in enumerate(ends) if "error" in e]
    visible = sum(1 for i in fail_idx if not rows[i]["reexecution"]
                  and any(rows[j]["tool"] == rows[i]["tool"] and not rows[j]["reexecution"] for j in range(i + 1, len(rows))))
    row = {"exp": exp, "case": label, "framework": fw, "limit": limit, "mode": kw.get("reasoning", "native" if fw == "pydantic-ai" else "native (= controlled)"),
           "http_attempts": d["http_attempts"], "http_errors": d["http_errors"], "model_requests": d["framework_model_calls"],
           "model_responses": d["model_responses"], "tool_calls_requested": d["tool_calls_requested"], "tool_executions": d["tool_executions"],
           "tool_executions_failed": len(failed), "framework_steps": d["framework_steps"], "stopped_by": "natural" if d["result"] == "OK" else d["result"],
           "error": d["error"], "final_text": d["final_text"], "executions_by_tool": _counts(rows), "trajectory": rows,
           "framework_reexecutions": sum(1 for r in rows if r["reexecution"]), "model_visible_retries": visible, "failures": [e["error"][:120] for e in failed],
           "tool_messages_by_request": _tool_msgs(traj), "history": d["history"], "state_types": d["state"],
           "pending_tool_calls_in_state": d.get("pending_tool_calls_in_state"), "wire": e4.wire_check(traj)}
    RESULTS.setdefault(exp, []).append(row)
    print(f"{exp:<10}{label[:44]:<45}{fw[:2]} req={row['model_requests']} exec={row['tool_executions']} (failed {row['tool_executions_failed']}) "
          f"steps={row['framework_steps']:<3} stop={row['stopped_by'][:22]:<22} {(d['error'] or '')[:62]}")
    return row


# ---------------------------------------------------------------- B5-3.1.39: recoverable failure
def recoverable():
    for fw in FWS:
        case("B5-3.1.39", "A fails once, model retries, succeeds, final", fw,
             [call("step_a"), call("step_a"), C("Step A completed after one retry.")], 25, [sched("step_a", ["fail", "ok"])])


# ---------------------------------------------------------------- B5-3.1.40: permanent failure
def permanent():
    give_up = [call("step_a")] * 3 + [C("I could not complete step A: it failed three times with 'Temporary failure'. No result was produced.")]
    for fw in FWS:
        case("B5-3.1.40", "A always fails, model gives up after 3 (retries=3)", fw, give_up, 25, [sched("step_a", ["fail"])], retries=3)
    case("B5-3.1.40", "same script, Phase 1 default retries=1", "pydantic-ai", give_up, 25, [sched("step_a", ["fail"])], retries=1)


# ---------------------------------------------------------------- B5-3.1.41: failure after an earlier success
def success_then_failure():
    script = [call("step_a"), call("step_b"), call("step_c"), C("A and C succeeded. B failed, so I used C instead.")]
    for fw in FWS:
        case("B5-3.1.41", "A ok, B fails, model switches to C, final", fw, script, 25,
             [sched("step_a", ["ok"]), sched("step_b", ["fail"]), sched("step_c", ["ok"])], retries=2)


# ---------------------------------------------------------------- B5-3.1.42: recovery after a partial side effect
def side_effects():
    q = "Place an order for 1 widget."
    create = call("create_order", item="widget", qty=1)

    def orders_case(label, fw, script, make, exc=LostResponse, fail_first=1):
        o = fx.Orders(fail_first=fail_first, exc=exc)
        r = case("B5-3.1.42", label, fw, script, 25, [make(o)], q=q, retries=2)
        r["ledger"], r["replays"] = len(o.ledger), o.replays
        print(f"   ledger={r['ledger']} replays={r['replays']} model sees: {r['tool_messages_by_request'][1][:1]}")
        return r
    for fw in FWS:
        orders_case("A: no guard, lost response, model retries, final claims 1", fw, [create, create, C("Your order has been created.")], lambda o: o.unguarded())
        orders_case("B: context key, lost response, model retries", fw, [create, create, C("Your order has been created.")], lambda o: o.context_keyed("req-1"))
        k = lambda n: call("create_order", item="widget", qty=n, idempotency_key="K1")  # noqa: E731
        orders_case("C: same key, different args (ConflictError)", fw, [k(1), k(2), C("The order could not be changed.")], lambda o: o.model_keyed())


# ---------------------------------------------------------------- B5-3.1.43: error wording
WORDINGS = ["Temporary failure", "Please retry the operation", "Operation unavailable", "Permission denied"]


def wording():
    script = [call("fetch_report", report_id="7"), C("The report could not be fetched.")]
    for text in WORDINGS:
        for fw in FWS:
            case("B5-3.1.43", f"raised: {text}", fw, script, 25, [fetch_report_tool(text)], retries=2)
    for fw in FWS:                                                   # same text delivered as a normal result: no framework template
        case("B5-3.1.43", "returned: Permission denied", fw, script, 25, [fetch_report_tool("Permission denied", "return")], retries=2)


# ---------------------------------------------------------------- B5-3.1.44: multiple failures
def multi():
    a, b, c = sched("step_a", ["ok"]), sched("step_b", ["fail", "ok"]), sched("step_c", ["fail", "ok"])
    script = [C(None, [("step_a", {}), ("step_b", {}), ("step_c", {})]), C(None, [("step_b", {}), ("step_c", {})]), C("A, B and C all completed.")]
    for fw in FWS:
        a, b, c = sched("step_a", ["ok"]), sched("step_b", ["fail", "ok"]), sched("step_c", ["fail", "ok"])
        case("B5-3.1.44", "[A,B,C]: A ok, B+C fail; retry [B,C]; final", fw, script, 25, [a, b, c], retries=2)


# ---------------------------------------------------------------- B5-3.1.45: failure, then natural termination
def natural():
    for fw in FWS:
        case("B5-3.1.45", "A fails, model declines to retry, honest final", fw,
             [call("step_a"), C("Step A failed with 'Temporary failure' and I did not retry it.")], 25, [sched("step_a", ["fail"])], retries=2)
    for fw in FWS:
        case("B5-3.1.45", "A fails, final answer does not mention it", fw, [call("step_a"), C("All done.")], 25, [sched("step_a", ["fail"])], retries=2)
    q = Q
    import lg_agent as lg
    import pa_agent as pa
    for fw in FWS:                                                   # the Phase 1 application-level view of the same run
        traj = Trajectory()
        t = e3.b2.raw_transport([call("step_a"), C("Step A failed and I did not retry it.")])[0]
        model = e3.b1.MODELS[fw](traj, transport=t)
        r = (lg.run if fw == "langgraph" else pa.run)(q, model, tools=[traced(sched("step_a", ["fail"]), traj)])
        row = {"framework": fw, "run_error": r.error, "final_text": r.final_text, "tool_calls": [(x.name, x.result, x.error) for x in r.tool_calls]}
        RESULTS.setdefault("B5-3.1.45-run", []).append(row)
        print(f"   Phase 1 run() {fw[:2]}: error={r.error} tool_calls={[(n, (e or '')[:50]) for n, _, e in row['tool_calls']]} final={r.final_text!r}")


# ---------------------------------------------------------------- B5-3.1.46: retry exhaustion and partial state
def app_run(fw, script, tools, **kw):
    """What the Phase 1 application-level run() returns for the same scripted run."""
    traj = Trajectory()
    t = e3.b2.raw_transport(script)[0]
    model = e3.b1.MODELS[fw](traj, transport=t)
    run = lg_agent.run if fw == "langgraph" else pa_agent.run
    r = run(Q, model, tools=[traced(x, traj) for x in tools], **kw)
    return {"run_error": (r.error or "")[:110], "tool_calls_recorded": len(r.tool_calls), "final_text": r.final_text, "llm_calls": r.llm_calls}


def exhaustion():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)
    seq = [call("step_a")] + [call("step_b")] * 30                  # A succeeds, then B keeps being requested

    def mk():
        return [sched("step_a", ["ok"]), sched("step_b", ["fail"])]

    def plain_b():
        def step_b() -> str:
            """Run step_b."""
            raise RuntimeError("step_b crashed")
        return [sched("step_a", ["ok"]), step_b]
    cases = [("E1 LG default handler, loop limit 6", "langgraph", mk, dict(limit=6), dict(recursion_limit=6)),
             ("E2 LG RetryPolicy(3), handle=False", "langgraph", mk, dict(limit=25, handle_tool_errors=False, retry_policy=rp), dict(handle_tool_errors=False, retry_policy=rp)),
             ("E3 PA ModelRetry, retries=1", "pydantic-ai", mk, dict(limit=25, retries=1), dict(retries=1)),
             ("E4 PA plain RuntimeError", "pydantic-ai", plain_b, dict(limit=25, retries=1), dict(retries=1))]
    for label, fw, make, kw, appkw in cases:
        r = case("B5-3.1.46", label, fw, seq, kw.pop("limit"), make(), **kw)
        r["application_run"] = app_run(fw, seq, make(), **appkw)
        h = r["history"]
        print(f"   native: error={(r['error'] or '')[:60]!r} history={[(x['role'], (x['text'] or '')[:24], x.get('outcome')) for x in h]} pending={r['pending_tool_calls_in_state']}")
        print(f"   Phase 1 run(): {r['application_run']}")
    q = "Place an order for 1 widget."
    create = [call("create_order", item="widget", qty=1)] * 30
    for label, fw, kw in (("E5 LG default, create_order always loses its response, loop limit 6", "langgraph", dict(limit=6)),
                          ("E6 PA ModelRetry, create_order always loses its response, retries=2", "pydantic-ai", dict(limit=25, retries=2))):
        o = fx.Orders(fail_first=10 ** 6, exc=LostResponse)
        r = case("B5-3.1.46", label, fw, create, kw.pop("limit"), [o.unguarded()], q=q, **kw)
        r["ledger"] = len(o.ledger)
        print(f"   ledger={r['ledger']} but the retained state shows {sum(1 for x in r['history'] if x['role'] == 'tool')} tool messages (each one a failure)")


# ---------------------------------------------------------------- B5-3.1.47: recovery accounting (nine separate counters)
def accounting():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)
    table = []

    def add(label, row, internal=0, effects=None):
        t = {"scenario": label, "framework": row["framework"], "http_attempts": row["http_attempts"], "http_errors": row["http_errors"],
             "model_requests": row["model_requests"], "model_responses": row["model_responses"], "tool_executions": row["tool_executions"],
             "framework_steps": row["framework_steps"], "model_visible_retries": row["model_visible_retries"],
             "framework_retries": row["framework_reexecutions"], "tool_internal_retries": internal, "business_effects": effects, "stopped_by": row["stopped_by"]}
        table.append(t)
        print(f"   {label[:40]:<41}{row['framework'][:2]} http={t['http_attempts']} req={t['model_requests']} resp={t['model_responses']} exec={t['tool_executions']} "
              f"steps={t['framework_steps']:<2} mv_retry={t['model_visible_retries']} fw_retry={t['framework_retries']} internal={internal} effects={effects}")
    for fw in FWS:
        add("model-visible retry (39)", case("B5-3.1.47", "model-visible retry", fw, [call("step_a"), call("step_a"), C("done")], 25, [sched("step_a", ["fail", "ok"])]))
    add("framework retry, LG RetryPolicy", case("B5-3.1.47", "framework retry (RetryPolicy)", "langgraph", [call("step_a"), C("done")], 25, [sched("step_a", ["fail", "fail", "ok"])],
                                                  handle_tool_errors=False, retry_policy=rp))
    for fw in FWS:
        fx.EXEC.clear()
        r = case("B5-3.1.47", "tool-internal retry (2 internal failures)", fw, [call("flaky_internal"), C("done")], 25, [fx.make_flaky_internal_retry(2, 3)])
        add("tool-internal retry", r, internal=sum(1 for e in fx.EXEC if e[0] == "flaky_internal") - 1)
    for fw in FWS:
        t, _, att = e3.http_flaky([call("step_a"), C("done")], 1, 500)
        r = case("B5-3.1.47", "HTTP retry (1x 500 then 200)", fw, [], 25, [sched("step_a", ["ok"])], transport=t)
        r["wire_http_attempts_seen_by_transport"] = att["n"]
        add("HTTP retry (SDK)", r)
    for fw in FWS:
        o = fx.Orders(fail_first=1, exc=LostResponse)
        create = call("create_order", item="widget", qty=1)
        r = case("B5-3.1.47", "lost response + model retry, no guard", fw, [create, create, C("done")], 25, [o.unguarded()], q="Place an order.", retries=2)
        add("lost response, model retry, no guard", r, effects=len(o.ledger))
    RESULTS["B5-3.1.47-table"] = table


# ---------------------------------------------------------------- live (model behavior only)
LIVE_SYSTEM = "You are a helpful assistant. Use the available tools to help the user."
CELLS = [("langgraph", "native (= controlled)"), ("pydantic-ai", "native"), ("pydantic-ai", "controlled")]
HONEST = re.compile(r"(unable|cannot|can.t|could not|couldn.t|wasn.t able|not able|failed|failure|unavailable|error|denied|"
                    r"permission|did not work|didn.t work|not available|isn.t available|temporar)", re.I)


def outcome_class(row):
    """One label per live run, from what the run recorded (not from reading the model's mind)."""
    rejected = any("tool_use_failed" in e for e in row.get("provider_errors", [])) or "Tool call validation failed" in (row.get("error") or "")
    if rejected:                      # a 429 the SDK retried is an HTTP retry (Batch 3), not a rejected tool call
        return "PROVIDER_REJECTED_TOOL_CALL"
    if row["stopped_by"] == "natural":
        if not row["final_text"].strip():
            return "EMPTY_FINAL_ACCEPTED"
        return "GAVE_UP_AND_SAID_SO" if HONEST.search(row["final_text"]) else "FINAL_WITHOUT_MENTIONING_FAILURE"
    return "BUDGET_EXHAUSTED_NO_FINAL"


def _quota():
    """After a 429: 'minute' (wait and retry), 'day' (stop), or 'ok'."""
    import httpx
    from lab_common.config import load_model_config
    cfg = load_model_config()
    r = httpx.post(cfg.base_url.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {cfg.api_key}"},
                   json={"model": cfg.model_name, "messages": [{"role": "user", "content": "ok " * 5}], "max_tokens": 5}, timeout=60)
    if r.status_code == 200:
        return "ok"
    return "day" if "per day" in r.text else "minute"


def live_run(cell, q, limit, tools, retries):
    fw, mode = cell
    lg_agent.SYSTEM_INSTRUCTIONS = pa_agent.SYSTEM_INSTRUCTIONS = LIVE_SYSTEM
    traj = Trajectory()
    model = e3.b1.MODELS[fw](traj, **({"reasoning": "stripped"} if mode == "controlled" else {}))
    wrapped = [traced(t, traj) for t in tools]
    info = e3.run_lg(q, model, wrapped, traj, limit) if fw == "langgraph" else e3.run_pa(q, model, wrapped, traj, limit, retries=retries)
    return info, traj


class QuotaStop(Exception):
    pass


def live_one(exp, label, cell, q, budget, make_tools, retries, extra=None):
    for attempt in range(4):
        tools = make_tools()
        info, traj = live_run(cell, q, 2 * budget if cell[0] == "langgraph" else budget, tools, retries)
        err = info["error"] or ""
        if "429" in err or "RateLimit" in err:
            state = _quota()
            if state == "day":
                raise QuotaStop(err[:200])
            print(f"   per-minute limit, waiting 25 s ({cell})")
            time.sleep(25)
            continue
        break
    rows = e4.trajectory(traj)
    a = traj.accounting()
    final = info["final_text"] or ""
    row = {"exp": exp, "case": label, "framework": cell[0], "mode": cell[1], "model": os.environ.get("MODEL_NAME", ""), "model_requests": info["framework_model_calls"],
           "tool_executions": a["tool_executions"], "executions_by_tool": _counts(rows), "stopped_by": "natural" if not info["error"] else info["error"].split(":")[0],
           "error": info["error"], "final_text": final[:400], "tokens": a["tokens_total"], "http_errors": a["http_errors"],
           "provider_errors": [e["error_body"] for e in traj.of("model_response") if e.get("error_body")]}
    row["outcome"] = outcome_class(row)
    if extra:
        row.update(extra(tools))
    RESULTS.setdefault(exp, []).append(row)
    print((f"LIVE {exp} {label[:34]:<35}{cell[0][:2]}/{cell[1][:6]:<7} req={row['model_requests']} exec={row['executions_by_tool']} stop={row['stopped_by'][:18]:<18} "
           f"{row['outcome']} tok={row['tokens']} final={final[:60]!r}").encode("ascii", "replace").decode())
    return row


def live43(trials=3, wordings=None, exp="B5-live-43"):
    for text in wordings or WORDINGS:
        for t in range(trials):
            for cell in CELLS:
                live_one(exp, text, cell, "Fetch report 7 and summarize it for me.", 5, lambda: [fetch_report_tool(text)], 8)
                time.sleep(4)


def live42(trials=3):
    q = "Place an order for 1 widget."
    for label, make in (("A no guard", lambda o: o.unguarded()), ("B context key", lambda o: o.context_keyed("req-1"))):
        for t in range(trials):
            for cell in CELLS:
                holder = {}

                def mk():
                    holder["o"] = fx.Orders(fail_first=1, exc=LostResponse)
                    return [make(holder["o"])]
                live_one("B5-live-42", label, cell, q, 6, mk, 3, extra=lambda _t: {"ledger": len(holder["o"].ledger), "replays": holder["o"].replays})
                time.sleep(4)


def live41(trials=3):
    q = "Look up customer 42, then send them a receipt. If the receipt cannot be sent, leave a note on their account instead."

    def mk():
        def lookup_customer(customer_id: str) -> str:
            """Look up a customer by id."""
            return "Customer 42: Alex Rivera"

        def send_receipt(customer_id: str) -> str:
            """Send the latest receipt to a customer."""
            raise ModelRetry("Operation unavailable")

        def log_note(customer_id: str, note: str) -> str:
            """Add a note to a customer's account."""
            return "note saved"
        return [lookup_customer, send_receipt, log_note]
    for t in range(trials):
        for cell in CELLS:
            live_one("B5-live-41", "A ok, B unavailable, C available", cell, q, 6, mk, 4)
            time.sleep(4)


def live(which=None):
    from lab_common.config import load_model_config
    cfg = load_model_config()
    os.environ["MODEL_NAME"] = cfg.model_name
    print(f"live model: {cfg.provider}/{cfg.model_name}, temperature 0; cells: {[c[0][:2] + '/' + c[1][:6] for c in CELLS]}")
    try:
        for name in which or ["43", "42", "41"]:
            if name == "43recapture":    # one wording rerun only to capture the provider's error bodies and check reproducibility
                live43(wordings=["Please retry the operation"], exp="B5-live-43-recapture")
            else:
                {"43": live43, "42": live42, "41": live41}[name]()
    except QuotaStop as e:
        print(f"LIVE STOPPED (daily quota): {e}")
        RESULTS.setdefault("live_stopped", []).append(str(e))


GROUPS = {"recoverable": recoverable, "permanent": permanent, "success": success_then_failure, "side": side_effects, "wording": wording,
          "multi": multi, "natural": natural, "exhaustion": exhaustion, "accounting": accounting}


def main():
    args = sys.argv[1:]
    if args and args[0] == "live":
        live(args[1:] or None)
        mode = "live"
    else:
        mode = "offline"
        for g in args or list(GROUPS):
            print(f"\n===== {g} =====")
            GROUPS[g]()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "batch5.json")
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
