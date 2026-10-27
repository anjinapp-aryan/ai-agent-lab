"""Phase 2 Batch 1 regression tests (Experiments 1-5, 11, 14). Scripted model, offline, no API calls.

Each assertion pins an OBSERVED behavior recorded in phases/phase-02/experiments.md.
"""
import pytest
from pydantic_ai import UsageLimits

import exp_batch1 as eb
from lab2 import fixtures as fx

FWS = ["langgraph", "pydantic-ai"]
BAD = ("call", [("calculate", {"a": 1})])      # missing b


def run(fw, script, tools, **kw):
    model, r, seen, ex, _ = eb.go(fw, script, tools, **kw)
    return model, r, seen, ex


# ---------------- E1: schema and wire payload
def test_e1_schema_sent_to_model_langgraph_open_pydantic_ai_closed():
    lg, _, _, _ = run("langgraph", [eb.good(), ("text", "5")], [fx.calculate])
    pa, _, _, _ = run("pydantic-ai", [eb.good(), ("text", "5")], [fx.calculate])
    lg_params = lg.bound[0]["function"]["parameters"]
    pa_params = pa.tools[0][2]
    assert "additionalProperties" not in lg_params and pa_params["additionalProperties"] is False
    assert lg.bound[0]["function"]["description"] == pa.tools[0][1] == "Add two integers and return a + b."


@pytest.mark.parametrize("fw,strict", [("langgraph", False), ("pydantic-ai", True)])
def test_e1_wire_strict_flag(fw, strict):
    _, log = eb.wire_run(fw, [eb.good(2, 3), ("text", "5")], [fx.calculate])
    fn = log[0]["tools"][0]["function"]
    assert ("strict" in fn and fn["strict"] is True) == strict
    assert [m["content"] for m in log[1]["messages"] if m["role"] == "tool"] == ["5"]


# ---------------- E2: missing argument
@pytest.mark.parametrize("fw", FWS)
def test_e2_missing_arg_detected_before_execution_then_recovers(fw):
    _, r, _, ex = run(fw, [BAD, eb.good(1, 2), ("text", "3")], [fx.calculate])
    assert len(ex) == 1 and r.error is None and r.llm_calls == 3 and r.tool_calls[0].error


def test_e2_persistent_bad_calls_langgraph_unbounded_until_step_limit():
    model, r, _, ex = run("langgraph", [BAD], [fx.calculate])
    assert not ex and "GraphRecursionError" in r.error and len(model.seen) == 9


@pytest.mark.parametrize("retries,model_calls", [(1, 2), (3, 4)])
def test_e2_persistent_bad_calls_pydantic_ai_bounded_by_retry_budget(retries, model_calls):
    model, r, _, ex = run("pydantic-ai", [BAD], [fx.calculate], retries=retries)
    assert not ex and "UnexpectedModelBehavior" in r.error and len(model.seen) == model_calls


# ---------------- E3: wrong type / coercion
ACCEPTED = {"10": 10, 10.0: 10, True: 1}
REJECTED = ["hello", 10.5, None, [1], "1e2"]


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("val,expected", list(ACCEPTED.items()))
def test_e3_lax_coercion_accepted(fw, val, expected):
    _, _, _, ex = run(fw, [("call", [("calculate", {"a": val, "b": 2})]), ("text", "x")], [fx.calculate])
    assert ex == [("calculate", expected, 2)] and type(ex[0][1]) is int


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("val", REJECTED)
def test_e3_uncoercible_rejected_before_execution(fw, val):
    _, r, _, ex = run(fw, [("call", [("calculate", {"a": val, "b": 2})]), ("text", "x")], [fx.calculate])
    assert not ex and r.tool_calls[0].error


# ---------------- E4: extra arguments
EXTRA = ("call", [("calculate", {"a": 10, "b": 20, "secret": "unexpected"})])


def test_e4_langgraph_default_drops_extra_and_executes():
    _, r, _, ex = run("langgraph", [EXTRA, ("text", "x")], [fx.calculate])
    assert ex == [("calculate", 10, 20)] and r.tool_calls[0].error is None


def test_e4_pydantic_ai_default_rejects_extra():
    _, r, _, ex = run("pydantic-ai", [EXTRA, ("text", "x")], [fx.calculate])
    assert not ex and "extra_forbidden" in r.tool_calls[0].error


def test_e4_langgraph_args_schema_forbid_rejects_extra():
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict

    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int
        b: int

    t = StructuredTool.from_function(func=fx.calculate, name="calculate", description="d", args_schema=Strict)
    _, r, _, ex = run("langgraph", [EXTRA, ("text", "x")], [t])
    assert not ex and r.tool_calls[0].error


def test_e4_kwargs_function_langgraph_drops_pydantic_ai_passes_through():
    script = [("call", [("calc_kwargs", {"a": 10, "b": 20, "secret": "unexpected"})]), ("text", "x")]
    _, _, _, lg = run("langgraph", script, [fx.calc_kwargs])
    _, _, _, pa = run("pydantic-ai", script, [fx.calc_kwargs])
    assert lg[0][3] == {} and pa[0][3] == {"secret": "unexpected"}


# ---------------- E5: enum
@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("tool", [fx.convert, fx.convert_enum])
@pytest.mark.parametrize("val,ok", [("USD", True), ("XYZ", False), ("usd", False), ("EUR ", False)])
def test_e5_enum_exact_match_only(fw, tool, val, ok):
    _, r, _, ex = run(fw, [("call", [(tool.__name__, {"amount": 5, "currency": val})]), ("text", "x")], [tool])
    assert bool(ex) == ok and bool(r.tool_calls[0].error) != ok


# ---------------- E11: result validation (what actually goes on the wire)
def _wire_content(fw, fn):
    r, log = eb.wire_run(fw, [("call", [(fn.__name__, {})]), ("text", "done")], [fn])
    tool = [m for m in (log[1]["messages"] if len(log) > 1 else []) if m["role"] == "tool"]
    return r, len(log), (tool[0]["content"] if tool else None)


@pytest.mark.parametrize("fw", FWS)
def test_e11_annotation_not_enforced_on_returns(fw):
    r, _, content = _wire_content(fw, fx.r_wrong_type)    # annotated -> int, returns "abc"
    assert r.error is None and content == "abc"


@pytest.mark.parametrize("fw", FWS)
def test_e11_huge_result_not_truncated(fw):
    r, _, content = _wire_content(fw, fx.r_huge)
    assert r.error is None and len(content) == fx.HUGE_CHARS


def test_e11_none_and_nan_serialization_differs():
    assert _wire_content("langgraph", fx.r_none)[2] == "null" and _wire_content("pydantic-ai", fx.r_none)[2] == ""
    assert _wire_content("langgraph", fx.r_nan)[2] == "NaN" and _wire_content("pydantic-ai", fx.r_nan)[2] == "null"


def test_e11_unserializable_object_langgraph_stringifies_pydantic_ai_aborts():
    r_lg, n_lg, c_lg = _wire_content("langgraph", fx.r_object)
    r_pa, n_pa, _ = _wire_content("pydantic-ai", fx.r_object)
    assert r_lg.error is None and n_lg == 2 and "Opaque" in c_lg
    assert "PydanticSerializationError" in r_pa.error and n_pa == 1   # aborted before the second model request


# ---------------- E14: limits
LOOP = [eb.good(1, 1)]
BATCH3 = ("call", [("calculate", {"a": i, "b": i}) for i in (1, 2, 3)])


def test_e14_pydantic_ai_tool_calls_limit_checked_before_executing():
    model, r, _, ex = run("pydantic-ai", LOOP, [fx.calculate], usage_limits=UsageLimits(tool_calls_limit=3))
    assert len(ex) == 3 and "UsageLimitExceeded" in r.error and "tool_calls_limit" in r.error


def test_e14_pydantic_ai_batch_over_limit_executes_none():
    _, r, _, ex = run("pydantic-ai", [BATCH3, ("text", "x")], [fx.calculate], usage_limits=UsageLimits(tool_calls_limit=2))
    assert not ex and "UsageLimitExceeded" in r.error


@pytest.mark.parametrize("limit,executions", [(5, 2), (9, 4), (17, 8)])
def test_e14_langgraph_only_has_step_limit(limit, executions):
    _, r, _, ex = run("langgraph", LOOP, [fx.calculate], recursion_limit=limit)
    assert len(ex) == executions and "GraphRecursionError" in r.error


def test_e14_langgraph_has_no_tool_call_count_limit():
    _, r, _, ex = run("langgraph", [BATCH3, ("text", "x")], [fx.calculate])
    assert len(ex) == 3 and r.error is None


def test_e14_pydantic_ai_messages_retained_after_limit_error_with_capture():
    from pydantic_ai import capture_run_messages
    with capture_run_messages() as cap:
        run("pydantic-ai", LOOP, [fx.calculate], usage_limits=UsageLimits(tool_calls_limit=3))
    assert len(cap) > 0
