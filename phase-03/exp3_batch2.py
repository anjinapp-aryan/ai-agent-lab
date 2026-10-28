"""Phase 3.1 Batch 2: natural termination (B2-3.1.4 … B2-3.1.10). Offline first; one optional small live check.

Reuses the Batch 1 runners (`graph.stream` / `agent.iter` over the Phase 1 graph/agent) and the trajectory recorder.
The only new fixture is `raw_transport`: like lab2.wire's mock, but each scripted completion can set `content`,
`tool_calls` and `finish_reason` independently, so the termination signal can be isolated. It produces only
payloads the OpenAI chat-completions schema allows. Hypotheses: phases/phase-03/experiments.md (Batch 2).

Usage: uv run python -u phase-03/exp3_batch2.py [offline] | live
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_batch1 as b1  # noqa: E402  (sets sys.path)

import httpx  # noqa: E402

from lab3.trajectory import Trajectory  # noqa: E402

FWS = b1.FWS
RESULTS: dict = {}
SENTINEL = "SENTINEL: this completion must never be requested"


def C(content=None, calls=(), finish=None):
    """One scripted completion. finish defaults to what a provider sends: tool_calls if calls else stop."""
    return {"content": content, "calls": list(calls), "finish": finish or ("tool_calls" if calls else "stop")}


def raw_transport(script):
    """Mock HTTP transport replaying `script`; after the script it serves SENTINEL so an extra request is visible."""
    log = []

    def handler(request):
        log.append(json.loads(request.content))
        n = len(log)
        step = script[n - 1] if n <= len(script) else C(SENTINEL)
        msg = {"role": "assistant", "content": step["content"]}
        if step["calls"]:
            msg["tool_calls"] = [{"id": f"call_{n}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                                 for i, (name, args) in enumerate(step["calls"])]
        body = {"id": f"cmpl-{n}", "object": "chat.completion", "created": 0, "model": "mock",
                "choices": [{"index": 0, "message": msg, "finish_reason": step["finish"]}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler), log


MUL = ("multiply", {"a": 25, "b": 17})
CASES = {
    "B2-3.1.4 no tool call": ("What is 2 + 2?", [C("Final answer: 4")]),
    "B2-3.1.5 tool then final": ("Calculate 25 × 17.", [C(None, [MUL]), C("No further tools are required. The answer is 425.")]),
    "B2-3.1.6 final, finish=stop": ("Calculate 25 × 17.", [C("The final result is 425.", finish="stop")]),
    "B2-3.1.6 final, finish=length": ("Calculate 25 × 17.", [C("The final result is 425.", finish="length")]),
    "B2-3.1.6 final, finish=tool_calls (no calls)": ("Calculate 25 × 17.", [C("The final result is 425.", finish="tool_calls")]),
    "B2-3.1.6 final, finish=content_filter": ("Calculate 25 × 17.", [C("The final result is 425.", finish="content_filter")]),
    "B2-3.1.6 tool call, finish=stop": ("Calculate 25 × 17.", [C(None, [MUL], finish="stop"), C("The answer is 425.")]),
    "B2-3.1.7 empty content": ("Calculate 25 × 17.", [C("")]),
    "B2-3.1.7 null content": ("Calculate 25 × 17.", [C(None)]),
    "B2-3.1.7 empty content, finish=length": ("Calculate 25 × 17.", [C("", finish="length")]),
    "B2-3.1.7 empty content, finish=content_filter": ("Calculate 25 × 17.", [C("", finish="content_filter")]),
    "B2-3.1.7 empty content twice": ("Calculate 25 × 17.", [C(""), C("")]),
    "B2-3.1.8 tool then final (no 3rd request)": ("Calculate 25 × 17.", [C(None, [MUL]), C("The answer is 425.")]),
    "B2-3.1.10 final text with whitespace/markdown": ("Calculate 25 × 17.", [C("  **The answer is 425.**\n\n")]),
}


def snapshot_lg(state):
    out = []
    for m in state["messages"]:
        out.append({"type": type(m).__name__, "content": m.content, "tool_calls": [(c["name"], c["args"]) for c in getattr(m, "tool_calls", [])],
                    "finish_reason": (getattr(m, "response_metadata", {}) or {}).get("finish_reason")})
    return out


def snapshot_pa(run):
    from pydantic_ai.messages import ModelRequest, ModelResponse
    out = []
    for m in run.result.all_messages():
        parts = [{"part": type(p).__name__, "content": getattr(p, "content", None) if not hasattr(p, "tool_name") or hasattr(p, "content") else None,
                  "tool": getattr(p, "tool_name", None)} for p in m.parts]
        out.append({"type": type(m).__name__, "parts": parts,
                    "finish_reason": getattr(m, "finish_reason", None) if isinstance(m, ModelResponse) else None,
                    "instructions": bool(getattr(m, "instructions", None)) if isinstance(m, ModelRequest) else None})
    return out


def one(case, fw):
    q, script = CASES[case]
    t, log = raw_transport(script)
    traj = Trajectory()
    model = b1.MODELS[fw](traj, transport=t)
    err, info, snap = None, {}, None
    try:
        info = b1.RUNNERS[fw](q, model, traj)
        raw = info.pop("_raw")
        snap = snapshot_lg(raw) if fw == "langgraph" else snapshot_pa(raw)
        if fw == "pydantic-ai":
            info["output_type"] = type(raw.result.output).__name__
            info["output_repr"] = repr(raw.result.output)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    resp = traj.of("model_response")
    wire_final = resp[-1]["text"] if resp else None
    d = {"case": case, "framework": fw, "accounting": traj.accounting(), "requests_served": len(log), "scripted": len(script),
         "extra_request": len(log) > len(script), "sentinel_requested": any(r.get("text") == SENTINEL for r in resp),
         "final_text": info.get("final_text"), "wire_final_content": wire_final,
         "final_identical_to_wire": info.get("final_text") == wire_final if info else None,
         "steps": [e["label"] for e in traj.of("framework_step")],
         "last_request_tail": (traj.of("model_request")[-1]["messages"][-1] if traj.of("model_request") else None), "snapshot": snap, "info": info, "error": err, "trace": traj.ascii()}
    RESULTS.setdefault("offline", []).append(d)
    return d, traj


def show(d):
    a = d["accounting"]
    print(f"\n--- {d['case']} | {d['framework']} | err={d['error']}")
    print(d["trace"])
    print(f"  requests={a['model_requests']} (scripted {d['scripted']}, extra={d['extra_request']}) tool_calls={a['tool_calls_requested']} "
          f"executions={a['tool_executions']} steps={a['framework_steps']}")
    print(f"  final_text={d['final_text']!r} wire={d['wire_final_content']!r} identical={d['final_identical_to_wire']}")
    if d["snapshot"]:
        print(f"  snapshot: {[(s['type'], s.get('finish_reason')) for s in d['snapshot']]}")


def offline():
    for case in CASES:
        for fw in FWS:
            d, _ = one(case, fw)
            show(d)
    # the Phase 1 entry points (application layer) on the same final response
    import lg_agent
    import pa_agent
    for fw in FWS:
        q, script = CASES["B2-3.1.10 final text with whitespace/markdown"]
        t, _ = raw_transport(script)
        traj = Trajectory()
        r = (lg_agent.run if fw == "langgraph" else pa_agent.run)(q, b1.MODELS[fw](traj, transport=t))
        RESULTS.setdefault("phase1_run", []).append({"framework": fw, "final_text": r.final_text, "llm_calls": r.llm_calls, "error": r.error})
        print(f"\nPhase 1 run() {fw}: RunResult.final_text={r.final_text!r} llm_calls={r.llm_calls} error={r.error}")


def live(n=3):
    from lab_common.config import load_model_config
    cfg = load_model_config()
    print(f"live model: {cfg.provider}/{cfg.model_name}, temperature 0, {n} trials per framework")
    for trial in range(1, n + 1):
        for fw in FWS:
            d, _ = b1.one("B2-live 'What is 2 + 2?'", fw, "What is 2 + 2?", None, trial, "live")
            d["model"] = cfg.model_name
            if d["error"] and ("429" in d["error"] or "rate limit" in d["error"].lower()):
                print(f"LIVE STOPPED (quota): {d['error'][:400]}")
                RESULTS.setdefault("live_stopped", []).append(d["error"][:600])
                RESULTS["live"] = [r for r in b1.RESULTS.get("B2-live 'What is 2 + 2?'", []) if not r["error"]]
                return
            b1.show(d)
            time.sleep(2)
    RESULTS["live"] = b1.RESULTS.get("B2-live 'What is 2 + 2?'", [])


def main():
    args = sys.argv[1:]
    mode = "live" if args and args[0] == "live" else "offline"
    live() if mode == "live" else offline()
    path = os.path.join(HERE, "results", "batch2.json")
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except FileNotFoundError:
        old = {}
    old.update({k if k.startswith(mode) else f"{mode}:{k}": v for k, v in RESULTS.items()})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1, default=str)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
