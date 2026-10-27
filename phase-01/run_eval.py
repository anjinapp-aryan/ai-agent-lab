"""Run all shared scenarios through both agents live; print table; save JSON trajectories.

Usage:  uv run python phase-01/run_eval.py [--trace]
Needs MODEL_PROVIDER / MODEL_NAME / API_KEY in .env (see .env.example).
"""
import json
import os
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "langgraph-agent", "pydantic-ai-agent"):
    sys.path.insert(0, os.path.join(HERE, sub))

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from lab_common.config import load_model_config  # noqa: E402
from lab_common.dataset import SCENARIOS  # noqa: E402
from lab_common.evaluate import passed, score  # noqa: E402


def main():
    cfg = load_model_config()
    print(f"provider={cfg.provider} model={cfg.model_name}")
    import lg_agent, pa_agent  # noqa: E401
    # Tracing: one provider per process, so --trace instruments whichever framework is chosen via --only.
    only = next((a.split("=")[1] for a in sys.argv if a.startswith("--only=")), None)
    runners = {"langgraph": lg_agent.run, "pydantic-ai": pa_agent.run}
    if only:
        runners = {only: runners[only]}
    if "--trace" in sys.argv:
        from lab_common.tracing import setup_tracing
        setup_tracing(only or "langgraph")
    out = []
    print(f"{'scenario':<5}{'framework':<13}{'pass':<6}{'tools':<40}answer")
    for fw, run in runners.items():
        for s in SCENARIOS:
            r = run(s.question)
            sc = score(s, r)
            tools = ",".join(f"{t.name}{'!' if t.error else ''}" for t in r.tool_calls) or "-"
            print(f"{s.id:<5}{fw:<13}{str(passed(sc)):<6}{tools:<40}{(r.error or r.final_text)[:70]!r}")
            out.append({"framework": fw, "scenario": s.id, "scores": sc, "result": asdict(r)})
    path = os.path.join(HERE, "results", "live_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"provider": cfg.provider, "model": cfg.model_name, "runs": out}, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
