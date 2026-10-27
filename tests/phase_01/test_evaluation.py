"""Evaluation: scorer unit tests (offline) + shared-scenario runs against both frameworks (live).

Reuse: plain pytest; no custom eval framework. promptfoo/Inspect deferred (see evaluation.md).
"""
import pytest

from conftest import live
from lab_common.dataset import SCENARIOS, RunResult, ToolCallRecord
from lab_common.evaluate import passed, score

S = {s.id: s for s in SCENARIOS}


def test_scorer_accepts_correct_answer():
    r = RunResult("The answer is 425.", [ToolCallRecord("multiply", {"a": 25, "b": 17}, "425.0")], 2)
    assert passed(score(S["S1"], r))


def test_scorer_rejects_wrong_answer():
    r = RunResult("It is 424.", [ToolCallRecord("multiply", {}, "425.0")], 2)
    assert not score(S["S1"], r)["functional"]


def test_scorer_rejects_answer_without_tool():
    r = RunResult("425", [], 1)
    assert not score(S["S1"], r)["tool_correct"]


def test_scorer_flags_fabricated_weather():
    r = RunResult("It is 24 degrees and sunny.", [], 1)
    assert not score(S["S6"], r)["no_fabrication"]


def test_scorer_flags_error_escape_as_incomplete():
    assert not score(S["S4"], RunResult("", [], 1, error="ZeroDivisionError"))["completion"]


@live
@pytest.mark.live
@pytest.mark.parametrize("sid", list(S))
def test_live_scenario(framework, sid):
    name, run, _ = framework
    r = run(S[sid].question)
    sc = score(S[sid], r)
    assert passed(sc), f"{name} {sid} failed {sc}; calls={r.tool_calls}; error={r.error}; answer={r.final_text!r}"
