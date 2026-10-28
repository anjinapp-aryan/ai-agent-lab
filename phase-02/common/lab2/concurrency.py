"""Batch 4 fixtures: a monotonic timeline recorder and timed tools. Measurement code, not a scheduler.

Every tool body records (name, start, end, thread id, asyncio task name). Overlap, concurrency level and the four orders
(call / start / completion / result) are computed from these timestamps, never from the model's tool-call list.
"""
import asyncio
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Timeline:
    events: list = field(default_factory=list)
    t0: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self):
        with self._lock:
            self.events.clear()
            self.t0 = time.perf_counter()

    def begin(self, name):
        try:
            task = asyncio.current_task()
            task_name = task.get_name() if task else None
        except RuntimeError:
            task_name = None
        return {"name": name, "start": time.perf_counter(), "end": None, "thread": threading.get_ident(), "task": task_name,
                "status": "running"}

    def finish(self, ev, status="ok"):
        ev["end"], ev["status"] = time.perf_counter(), status
        with self._lock:
            self.events.append(ev)

    # ---- analysis (relative milliseconds)
    def rel(self):
        return [{**e, "start_ms": round((e["start"] - self.t0) * 1000), "end_ms": round((e["end"] - self.t0) * 1000)}
                for e in sorted(self.events, key=lambda e: e["start"])]

    def makespan_ms(self):
        r = self.rel()
        return (max(e["end_ms"] for e in r) - min(e["start_ms"] for e in r)) if r else 0

    def max_concurrency(self):
        pts = []
        for e in self.events:
            pts += [(e["start"], 1), (e["end"], -1)]
        cur = best = 0
        for _, d in sorted(pts, key=lambda p: (p[0], p[1])):
            cur += d
            best = max(best, cur)
        return best

    def overlap_ms(self, a, b):
        ea = [e for e in self.events if e["name"] == a]
        eb = [e for e in self.events if e["name"] == b]
        if not ea or not eb:
            return None
        return round(max(0.0, min(ea[0]["end"], eb[0]["end"]) - max(ea[0]["start"], eb[0]["start"])) * 1000)

    def order(self, key):
        return [e["name"] for e in sorted(self.events, key=lambda e: e[key])]

    def threads(self):
        return len({e["thread"] for e in self.events})

    def ascii(self, step_ms=50, width=60):
        r = self.rel()
        lines = []
        for e in r:
            a, b = e["start_ms"] // step_ms, max(e["end_ms"] // step_ms, e["start_ms"] // step_ms + 1)
            lines.append(f"{e['name']:<8}|" + " " * a + "#" * (b - a) + f"  {e['start_ms']}-{e['end_ms']}ms [{e['status']}]")
        return "\n".join(lines)


TL = Timeline()


def sync_tool(name, dur, fail=False, effect_log=None, tl=None, cleanup_log=None):
    tl = tl or TL

    def tool() -> str:
        ev = tl.begin(name)
        try:
            time.sleep(dur)
            if fail:
                raise (fail(name) if callable(fail) else RuntimeError(f"{name} failed"))
            if effect_log is not None:
                effect_log.append(name)
            tl.finish(ev)
            return f"{name} done"
        except BaseException:
            tl.finish(ev, "failed")
            raise
        finally:
            if cleanup_log is not None:
                cleanup_log.append(name)
    tool.__name__ = name
    tool.__doc__ = f"Run {name}."
    return tool


def async_tool(name, dur, fail=False, effect_log=None, tl=None, cleanup_log=None):
    tl = tl or TL

    async def tool() -> str:
        ev = tl.begin(name)
        try:
            await asyncio.sleep(dur)
            if fail:
                raise (fail(name) if callable(fail) else RuntimeError(f"{name} failed"))
            if effect_log is not None:
                effect_log.append(name)
            tl.finish(ev)
            return f"{name} done"
        except asyncio.CancelledError:
            tl.finish(ev, "cancelled")
            raise
        except BaseException:
            tl.finish(ev, "failed")
            raise
        finally:
            if cleanup_log is not None:
                cleanup_log.append(name)
    tool.__name__ = name
    tool.__doc__ = f"Run {name}."
    return tool
