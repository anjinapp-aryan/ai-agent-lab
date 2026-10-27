"""Deterministic scoring of a RunResult against a Scenario. Plain functions, used by pytest.

Dimensions: functional correctness, tool correctness, completion, failure handling.
(Argument correctness is checked per-scenario in tests where exact args are known.)
"""
import re

from .dataset import RunResult, Scenario


def _numbers(text: str) -> set[str]:
    out = set()
    for n in re.findall(r"-?\d[\d,]*\.?\d*", text):
        n = n.replace(",", "").rstrip(".")
        out.add(n[:-2] if n.endswith(".0") else n)
    return out


def score(s: Scenario, r: RunResult) -> dict[str, bool]:
    text = r.final_text.lower()
    called = {t.name for t in r.tool_calls}
    return {
        "completion": r.error is None and bool(r.final_text.strip()),
        "functional": all(e in _numbers(r.final_text) or e.lower() in text for e in s.expect_answer),
        "tool_correct": set(s.expect_tools) <= called,
        "no_fabrication": not any(f in text for f in s.forbid_answer),
        # failure scenarios: any tool error must not be silently turned into a numeric result
        "failure_handled": s.kind != "failure" or not any(t.error is None and t.name == "divide" and s.id in ("S4", "S5") for t in r.tool_calls),
    }


def passed(scores: dict[str, bool]) -> bool:
    return all(scores.values())
