"""Phase 2 Batch 1 (scripted tier): Experiments 1,2,3,4,5,11,14. No API calls.

Hypotheses are pre-registered in phases/phase-02/experiments.md. This script records OBSERVED behavior only;
interpretation lives in the docs. Reuses Phase 1 agents (lg_agent.run / pa_agent.run), Phase 1 scripted models
(via lab2.scripted) and the Phase 1 RunResult.

Usage: uv run python phase-02/exp_batch1.py [1|2|3|4|5|11|14 ...]   -> phase-02/results/batch1.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", os.path.join("..", "phase-01", "common"), os.path.join("..", "phase-01", "langgraph-agent"),
            os.path.join("..", "phase-01", "pydantic-ai-agent")):
    sys.path.insert(0, os.path.join(HERE, sub))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab2 import fixtures as fx  # noqa: E402
from lab2 import scripted as sc  # noqa: E402
from lab2 import wire  # noqa: E402

FW = {
    "langgraph": (lg_agent.run, sc.lg_model, sc.model_sees_lg),
    "pydantic-ai": (pa_agent.run, sc.pa_model, sc.model_sees_pa),
}
RESULTS: dict = {}


def go(fw, script, tools, **kw):
    run, mk, sees = FW[fw]
    fx.EXEC.clear()
    model = mk(script)
    t0 = time.monotonic()
    r = run("q", model, tools=tools, **kw)
    return model, r, sees(model), list(fx.EXEC), round(time.monotonic() - t0, 3)


def short(x, n=150):
    return str(x).replace("\n", " ")[:n]


def row(exp, case, fw, model, r, seen, ex, dt, **extra):
    d = {"exp": exp, "case": case, "framework": fw, "executed": len(ex), "exec": [[repr(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for v in e] for e in ex],
         "tool_error": short(next((t.error for t in r.tool_calls if t.error), None) or ""),
         "llm_calls": r.llm_calls, "model_calls": len(model.seen), "error": short(r.error or ""), "final": short(r.final_text, 40),
         "seen_by_model": [[short(t, 120), s] for t, s in seen], "wall_s": dt, **extra}
    RESULTS.setdefault(str(exp), []).append(d)
    return d


def show(d):
    print(f"E{d['exp']:<3}{d['case']:<28}{d['framework']:<12}exec={d["executed"]} model_calls={d["model_calls"]} "
          f"err={d['error'][:48]!r} tool_err={d['tool_error'][:70]!r}")


def good(a=2, b=3):
    return ("call", [("calculate", {"a": a, "b": b})])


# ---------------------------------------------------------------- E1
def e1():
    for fw in FW:
        model, r, seen, ex, dt = go(fw, [good(2, 3), ("text", "5")], [fx.calculate])
        schema = model.bound[0] if fw == "langgraph" else {"name": model.tools[0][0], "description": model.tools[0][1],
                                                           "parameters": model.tools[0][2]}
        d = row(1, "simple typed tool", fw, model, r, seen, ex, dt, schema=schema,
                arg_types=[type(v).__name__ for v in ex[0][1:]] if ex else None, tool_result=r.tool_calls[0].result)
        show(d)
        print("      schema sent:", json.dumps(schema)[:330])
        print("      model sees :", d["seen_by_model"])


# ---------------------------------------------------------------- E2
def e2():
    bad = ("call", [("calculate", {"a": 1})])
    for fw in FW:
        d = row(2, "bad then good", fw, *go(fw, [bad, good(1, 2), ("text", "3")], [fx.calculate])); show(d)
    for fw in FW:
        d = row(2, "persistent bad (default)", fw, *go(fw, [bad], [fx.calculate])); show(d)
    d = row(2, "persistent bad (retries=3)", "pydantic-ai", *go("pydantic-ai", [bad], [fx.calculate], retries=3)); show(d)


# ---------------------------------------------------------------- E3
def e3():
    for val in ["hello", "10", 10.5, 10.0, True, None, [1], "1e2"]:
        for fw in FW:
            d = row(3, f"a={val!r}", fw, *go(fw, [("call", [("calculate", {"a": val, "b": 2})]), ("text", "x")], [fx.calculate]))
            d["accepted"] = d["executed"] == 1
            d["received"] = d["exec"][0][1:] if d["exec"] else None
            d["received_types"] = [type(v).__name__ for v in fx.EXEC[0][1:]] if fx.EXEC else None
            print(f"E3  a={val!r:<9}{fw:<12}accepted={d['accepted']} tool received={d['received']} types={d['received_types']}")


# ---------------------------------------------------------------- E4
def e4():
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int
        b: int

    forbid = StructuredTool.from_function(func=fx.calculate, name="calculate", description=fx.calculate.__doc__, args_schema=Strict)
    extra = ("call", [("calculate", {"a": 10, "b": 20, "secret": "unexpected"})])
    for fw in FW:
        d = row(4, "extra arg, default tool", fw, *go(fw, [extra, ("text", "x")], [fx.calculate])); show(d)
    d = row(4, "extra arg, args_schema forbid", "langgraph", *go("langgraph", [extra, ("text", "x")], [forbid])); show(d)
    kw = ("call", [("calc_kwargs", {"a": 10, "b": 20, "secret": "unexpected"})])
    for fw in FW:
        d = row(4, "extra arg, **kwargs func", fw, *go(fw, [kw, ("text", "x")], [fx.calc_kwargs])); show(d)
        print("      tool received:", d["exec"], "| model sees:", d["seen_by_model"])


# ---------------------------------------------------------------- E5
def e5():
    for tool, label in ((fx.convert, "Literal"), (fx.convert_enum, "Enum")):
        for val in ["XYZ", "usd", "USD", "EUR "]:
            for fw in FW:
                script = [("call", [(tool.__name__, {"amount": 5, "currency": val})]), ("text", "x")]
                d = row(5, f"{label} currency={val!r}", fw, *go(fw, script, [tool]))
                print(f"E5  {label:<8}{val!r:<8}{fw:<12}executed={d['executed']} feedback={d['tool_error'][:95]!r}")


# ---------------------------------------------------------------- E11
def e11():
    for fn in (fx.r_valid, fx.r_dict, fx.r_none, fx.r_wrong_type, fx.r_object, fx.r_nan, fx.r_huge):
        for fw in FW:
            d = row(11, fn.__name__, fw, *go(fw, [("call", [(fn.__name__, {})]), ("text", "done")], [fn]))
            seen = d["seen_by_model"]
            full = FW[fw][2](None) if False else None  # (seen already truncated for display)
            # recompute true length of what the model received
            model = None
            print(f"E11 {fn.__name__:<14}{fw:<12}executed={d['executed']} run_error={d['error'][:60]!r} "
                  f"tool_err={d['tool_error'][:50]!r} model_sees={d['seen_by_model'][:1]}")


def e11_sizes():
    """True size (chars) of the huge result as handed to the model, per framework."""
    out = {}
    for fw in FW:
        run, mk, sees = FW[fw]
        fx.EXEC.clear()
        m = mk([("call", [("r_huge", {})]), ("text", "done")])
        run("q", m, tools=[fx.r_huge])
        if fw == "langgraph":
            from langchain_core.messages import ToolMessage
            n = [len(x.content) for x in m.seen[-1] if isinstance(x, ToolMessage)]
        else:
            from pydantic_ai.messages import ModelRequest, ToolReturnPart
            n = [len(p.model_response_str()) for msg in m.seen[-1][-1:] if isinstance(msg, ModelRequest)
                 for p in msg.parts if isinstance(p, ToolReturnPart)]
        out[fw] = n
        print(f"E11 huge result chars delivered to model: {fw}: {n} (tool returned {fx.HUGE_CHARS})")
    RESULTS["11"].append({"huge_chars_delivered": out})


# ---------------------------------------------------------------- E14
def e14():
    from pydantic_ai import UsageLimits, capture_run_messages
    loop = [good(1, 1)]   # model never stops: same valid call forever
    for fw, kw, label in (
        ("pydantic-ai", dict(usage_limits=UsageLimits(tool_calls_limit=3)), "tool_calls_limit=3"),
        ("pydantic-ai", dict(usage_limits=UsageLimits(request_limit=4)), "request_limit=4"),
        ("pydantic-ai", dict(), "default (request_limit=8)"),
        ("langgraph", dict(recursion_limit=5), "recursion_limit=5"),
        ("langgraph", dict(recursion_limit=9), "recursion_limit=9"),
        ("langgraph", dict(), "default (recursion 17)"),
    ):
        if fw == "pydantic-ai":
            with capture_run_messages() as cap:
                res = go(fw, loop, [fx.calculate], **kw)
            d = row(14, label, fw, *res, captured_messages=len(cap))
        else:
            d = row(14, label, fw, *go(fw, loop, [fx.calculate], **kw), captured_messages=0)
        print(f"E14 {label:<26}{fw:<12}tool executions={d['executed']} model_calls={d['model_calls']} "
              f"error={d['error'][:70]!r} messages_retained_after_error={d['captured_messages']}")
    batch = ("call", [("calculate", {"a": i, "b": i}) for i in (1, 2, 3)])
    d = row(14, "batch of 3, tool_calls_limit=2", "pydantic-ai",
            *go("pydantic-ai", [batch, ("text", "x")], [fx.calculate], usage_limits=UsageLimits(tool_calls_limit=2)))
    print(f"E14 batch of 3 + tool_calls_limit=2 pydantic-ai: executions={d['executed']} error={d['error'][:80]!r}")
    d = row(14, "batch of 3 (no tool-call limit)", "langgraph", *go("langgraph", [batch, ("text", "x")], [fx.calculate]))
    print(f"E14 batch of 3, langgraph (no such limit exists): executions={d['executed']} error={d['error'][:60]!r}")


def wire_run(fw, script, tools):
    """Run the real provider client against the mock transport; return (RunResult, request bodies)."""
    fx.EXEC.clear()
    model, log = (wire.lg_wire_model if fw == "langgraph" else wire.pa_wire_model)(script)
    r = FW[fw][0]("q", model, tools=tools)
    return r, log


def e1w():
    """E1/E11 on the wire: exact request JSON sent by the real clients."""
    for fw in FW:
        r, log = wire_run(fw, [good(2, 3), ("text", "5")], [fx.calculate])
        t = log[0].get("tools")
        msgs2 = [m for m in log[1]["messages"] if m["role"] == "tool"]
        d = {"exp": "1w", "framework": fw, "tools_on_wire": t, "tool_msg_on_wire": msgs2, "requests": len(log), "error": r.error}
        RESULTS.setdefault("1w", []).append(d)
        print(f"E1w {fw:<12}requests={len(log)} error={r.error}")
        print("    tools on wire:", json.dumps(t))
        print("    tool message on wire:", json.dumps(msgs2))


def e11w():
    for fn in (fx.r_none, fx.r_object, fx.r_nan, fx.r_huge):
        for fw in FW:
            r, log = wire_run(fw, [("call", [(fn.__name__, {})]), ("text", "done")], [fn])
            tool_msgs = [m for m in (log[1]["messages"] if len(log) > 1 else []) if m["role"] == "tool"]
            content = tool_msgs[0]["content"] if tool_msgs else None
            shown = (content if isinstance(content, str) else json.dumps(content))
            d = {"exp": "11w", "case": fn.__name__, "framework": fw, "requests": len(log), "run_error": short(r.error or ""),
                 "tool_content_len": len(shown) if shown is not None else None, "tool_content": short(shown, 90)}
            RESULTS.setdefault("11w", []).append(d)
            print(f"E11w {fn.__name__:<10}{fw:<12}requests={len(log)} run_error={d['run_error'][:75]!r} "
                  f"tool content on wire: len={d['tool_content_len']} {d['tool_content'][:50]!r}")


ALL = {"1": e1, "1w": e1w, "11w": e11w, "2": e2, "3": e3, "4": e4, "5": e5, "11": lambda: (e11(), e11_sizes()), "14": e14}


def main():
    which = sys.argv[1:] or list(ALL)
    for k in which:
        print(f"\n===== Experiment {k} =====")
        ALL[k]()
    path = os.path.join(HERE, "results", "batch1.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
