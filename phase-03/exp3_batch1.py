"""Phase 3.1 Batch 1: trace the agent loop (3.1.1 one loop, 3.1.2 dependency chain, 3.1.3 independent calls).

The Phase 1 agents' graph/agent are reused unchanged (`lg_agent.build_graph`, `pa_agent.build_agent`, shared
calculator tools, same system prompt and step cap). The loop is driven by the frameworks' own iteration APIs
(`graph.stream`, `agent.iter`), which their `invoke`/`run` call internally. Hypotheses: phases/phase-03/experiments.md.

Modes:
  offline  real provider clients + mock HTTP transport replaying a script (lab2.wire), no API calls
  live     Groq (model from .env.<APP_ENV>), 3 trials per framework per task
Usage: uv run python -u phase-03/exp3_batch1.py [offline] | live [3.1.1 3.1.2 3.1.3]
       -> phase-03/results/batch1.json (git-ignored), summary -> phases/phase-03/results/batch1-summary.json
"""
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in ("phase-03/common", "phase-02/common", "phase-01/common", "phase-01/langgraph-agent", "phase-01/pydantic-ai-agent"):
    sys.path.insert(0, os.path.join(ROOT, p))
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import lg_agent  # noqa: E402
import pa_agent  # noqa: E402
from lab2 import wire  # noqa: E402
from lab3.trajectory import Trajectory, lg_model, pa_model, traced  # noqa: E402
from lab_common.prompts import MAX_STEPS  # noqa: E402
from lab_common.tools import TOOLS  # noqa: E402

FWS = ["langgraph", "pydantic-ai"]
RESULTS: dict = {}

TASKS = {
    "3.1.1": ("Calculate 25 × 17.", [("call", [("multiply", {"a": 25, "b": 17})]), ("text", "25 × 17 = 425")]),
    "3.1.2": ("Take 120 + 45. Then multiply the result by 3.",
              [("call", [("add", {"a": 120, "b": 45})]), ("call", [("multiply", {"a": 165, "b": 3})]), ("text", "The result is 495.")]),
    "3.1.3": ("Calculate 10 × 5 and 20 + 7.",
              [("call", [("multiply", {"a": 10, "b": 5}), ("add", {"a": 20, "b": 7})]), ("text", "10 × 5 = 50 and 20 + 7 = 27.")]),
    "3.1.3-seq": ("Calculate 10 × 5 and 20 + 7.",
                  [("call", [("multiply", {"a": 10, "b": 5})]), ("call", [("add", {"a": 20, "b": 7})]), ("text", "50 and 27.")]),
}


# ---------------------------------------------------------------- one run through each framework's iteration API
def run_langgraph(question, model, traj):
    from langchain_core.messages import HumanMessage
    graph = lg_agent.build_graph(model, tools=[traced(t, traj) for t in TOOLS])
    cfg = {"recursion_limit": 2 * MAX_STEPS + 1}                  # same as lg_agent.run
    final = None
    for mode, chunk in graph.stream({"messages": [HumanMessage(question)]}, cfg, stream_mode=["debug", "values"]):
        if mode == "values":                                       # state after each super-step (channel values)
            final = chunk
            steps = traj.of("framework_step")
            if steps:
                steps[-1]["state"] = f"messages_in_state={len(chunk['messages'])} (system prompt added per call, not stored)"
        elif chunk["type"] == "task_result":
            traj.add("framework_step", "framework", step=chunk["step"], label=f"node '{chunk['payload']['name']}' finished", state="")
    msgs = final["messages"]
    return {"final_text": msgs[-1].content, "state_messages": [type(m).__name__ for m in msgs], "super_steps": max(e["step"] for e in traj.of("framework_step"))}


def run_pydantic_ai(question, model, traj):
    from pydantic_ai import UsageLimits
    agent = pa_agent.build_agent(model, tools=[traced(t, traj) for t in TOOLS])

    async def go():
        async with agent.iter(question, usage_limits=UsageLimits(request_limit=MAX_STEPS)) as run:   # same as pa_agent.run
            i = 0
            async for node in run:
                i += 1
                st = run.ctx.state
                traj.add("framework_step", "framework", step=i, label=f"about to run {type(node).__name__}",
                         state=f"messages={len(st.message_history)} run_step={st.run_step} requests={run.usage.requests}")
            return run
    run = asyncio.run(go())
    return {"final_text": str(run.result.output), "state_messages": [type(m).__name__ for m in run.result.all_messages()],
            "run_step": run.ctx.state.run_step, "usage_requests": run.usage.requests, "usage_tool_calls": run.usage.tool_calls}


RUNNERS = {"langgraph": run_langgraph, "pydantic-ai": run_pydantic_ai}
MODELS = {"langgraph": lg_model, "pydantic-ai": pa_model}


def context_check(traj):
    """Does each tool result appear in the NEXT model request? (wire evidence)"""
    reqs = traj.of("model_request")
    ends = traj.of("tool_end")
    out = []
    for e in ends:
        later = [r for r in reqs if r["t_ms"] > e["t_ms"]]
        nxt = later[0] if later else None
        seen = bool(nxt) and any(m.get("role") == "tool" and str(e.get("result")) in str(m.get("content")) for m in nxt["messages"])
        out.append({"tool": e["tool"], "result": e.get("result"), "in_next_request": seen, "next_request": nxt["n"] if nxt else None})
    return out


def one(exp, fw, question, transport, trial=1, mode="offline"):
    traj = Trajectory()
    model = MODELS[fw](traj, transport=transport)
    t0 = time.perf_counter()
    err = None
    try:
        info = RUNNERS[fw](question, model, traj)
    except Exception as e:  # noqa: BLE001
        info, err = {}, f"{type(e).__name__}: {e}"
    d = {"exp": exp, "framework": fw, "mode": mode, "trial": trial, "question": question, "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
         "accounting": traj.accounting(), "context": context_check(traj), "info": info, "error": err,
         "temperature_sent": [r["temperature"] for r in traj.of("model_request")][:1],
         "tool_threads": len({e["thread"] for e in traj.of("tool_start")}),
         "events": [{k: v for k, v in e.items() if k != "messages"} for e in traj.events], "trace": traj.ascii(),
         "requests": [{"n": r["n"], "roles": r["roles"], "messages": r["messages"]} for r in traj.of("model_request")]}
    RESULTS.setdefault(exp, []).append(d)
    return d, traj


def show(d):
    a = d["accounting"]
    print(f"\n--- {d['exp']} {d['framework']} [{d['mode']} #{d['trial']}] err={d['error']}")
    print(d["trace"])
    print(f"  ACCOUNTING {a}")
    print(f"  CONTEXT    {[(c['tool'], c['result'], c['in_next_request']) for c in d['context']]}")
    print(f"  STATE      {d['info']}")


# ---------------------------------------------------------------- offline (scripted completions over the real clients)
def offline():
    for exp, (q, script) in TASKS.items():
        for fw in FWS:
            t, _ = wire.transport(script)
            d, _ = one(exp, fw, q, t)
            show(d)
    # cross-check: the plain Phase 1 run() (invoke / run_sync) produces the same wire and tool accounting
    for fw in FWS:
        q, script = TASKS["3.1.2"]
        traj = Trajectory()
        t, _ = wire.transport(script)
        model = MODELS[fw](traj, transport=t)
        r = (lg_agent.run if fw == "langgraph" else pa_agent.run)(q, model, tools=[traced(x, traj) for x in TOOLS])
        acc = traj.accounting()
        RESULTS.setdefault("crosscheck", []).append({"framework": fw, "accounting": acc, "llm_calls": r.llm_calls,
                                                      "tool_calls": len(r.tool_calls), "error": r.error})
        print(f"\nCROSSCHECK {fw}: run() accounting {acc}, RunResult.llm_calls={r.llm_calls}, tool_calls={len(r.tool_calls)}")


# ---------------------------------------------------------------- live
def live(which=None, n=3):
    from lab_common.config import load_model_config
    cfg = load_model_config()
    print(f"live model: {cfg.provider}/{cfg.model_name}, temperature 0, {n} trials per framework per task")
    for exp in (which or ["3.1.1", "3.1.2", "3.1.3"]):
        q = TASKS[exp][0]
        for trial in range(1, n + 1):
            for fw in FWS:
                d, _ = one(exp, fw, q, None, trial, "live")
                d["model"] = cfg.model_name
                if d["error"] and ("429" in d["error"] or "rate limit" in d["error"].lower()):
                    print(f"LIVE STOPPED (quota): {d['error'][:200]}")
                    RESULTS.setdefault("live_stopped", []).append(d["error"][:300])
                    return
                show(d)
                time.sleep(1.5)


def main():
    args = sys.argv[1:]
    if args and args[0] == "live":
        live(args[1:] or None)
    else:
        offline()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "batch1.json")
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except FileNotFoundError:
        old = {}
    mode = "live" if args and args[0] == "live" else "offline"
    for k, v in RESULTS.items():
        old[f"{mode}:{k}"] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1, default=str)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
