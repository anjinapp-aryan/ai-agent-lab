"""Repro (Batch 4 finding): Pydantic AI Agent(max_concurrency=N) shared by threads that each call run_sync.

Variants: loop / loop-nolimit -> two agent.run() in ONE event loop (asyncio.gather): the limit serializes the runs.
          threads -> two threads, each agent.run_sync() (one event loop per thread): observed to HANG with max_concurrency=1.
          threads-nolimit -> same without the limit: completes.
Usage: uv run python phase-02/repro_pa_max_concurrency.py loop|threads|threads-nolimit   (wrap `threads` in a timeout)
"""
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "common"), os.path.join(HERE, "..", "phase-01", "common"), os.path.join(HERE, "..", "tests", "phase_01")]
from pydantic_ai import Agent  # noqa: E402
from lab2 import scripted as sc  # noqa: E402

SCRIPT = [("call", [("tool", {})]), ("text", "done")]


SPANS = []


async def tool() -> str:
    """Sleep 0.2 s (async), recording its span."""
    t = time.perf_counter()
    await asyncio.sleep(0.2)
    SPANS.append((t, time.perf_counter()))
    return "ok"


def main(variant):
    kw = {} if variant.endswith("nolimit") else {"max_concurrency": 1}
    agent = Agent(None, tools=[tool], **kw)                    # one shared agent; each run gets its own scripted model
    m = [sc.pa_model(SCRIPT), sc.pa_model(SCRIPT)]
    t0 = time.perf_counter()
    if variant.startswith("loop"):
        async def both():
            return await asyncio.gather(agent.run("q", model=m[0]), agent.run("q", model=m[1]))
        out = [r.output for r in asyncio.run(both())]
    else:
        with ThreadPoolExecutor(2) as ex:
            out = list(ex.map(lambda i: agent.run_sync("q", model=m[i]).output, [0, 1]))
    (a0, a1), (b0, b1) = sorted(SPANS)[:2] if len(SPANS) >= 2 else ((0, 0), (0, 0))
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    print(variant, out, f"wall={time.perf_counter() - t0:.2f}s tool_overlap={overlap:.2f}s")


if __name__ == "__main__":
    main(sys.argv[1])
