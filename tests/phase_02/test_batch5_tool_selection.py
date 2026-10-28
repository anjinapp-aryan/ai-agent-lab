"""Phase 2 Batch 5 regression tests: tool selection.

Offline tests pin FRAMEWORK mechanics and what each framework SENDS to the model (scripted model + mock HTTP transport).
They say nothing about what a real model selects; model behavior is recorded in phases/phase-02/tool-selection.md.
One small live test asserts the security INVARIANT (selection is not authorization), not model behavior.
"""
import pytest

import exp_batch5 as e5
from lab2 import security as sec
from lab2 import selection as sel
from lab2 import wire

import lg_agent
import pa_agent

FWS = e5.FWS


@pytest.fixture(autouse=True)
def _prompt():
    e5.set_prompt(e5.PROMPT)
    sel.reset()
    yield


def call(name, **args):
    return ("call", [(name, args)])


def rows(fn, key):
    e5.RESULTS.clear()
    fn()
    return list(e5.RESULTS[key])


def find(rs, fw, part):
    hits = [r for r in rs if r["framework"] == fw and part in r["case"]]
    assert len(hits) == 1, f"{fw} / {part!r} matched {len(hits)}"
    return hits[0]


# ------------------------------------------------------------ classifier (Phase 1 scorer style, one label per run)
def _rr(calls=(), text=""):
    from lab_common.dataset import RunResult, ToolCallRecord
    return RunResult(final_text=text, tool_calls=[ToolCallRecord(n, a) for n, a in calls])


@pytest.mark.parametrize("calls,text,kw,label", [
    ([("get_order", {"order_id": "123"})], "", {}, "CORRECT"),
    ([("get_order", {"order_id": "#123"})], "", {}, "INVALID_ARGUMENTS"),
    ([("cancel_order", {"order_id": "123"})], "", {}, "WRONG_TOOL"),
    ([("track_order", {"order_id": "123"})], "", {"alternatives": ["track_order"]}, "VALID_ALTERNATIVE"),
    ([("track_order", {"order_id": "123"})], "", {"alternatives": ["track_order"], "ambiguous": True}, "AMBIGUOUS"),
    ([("update_shipping_address", {})], "", {}, "INVALID_TOOL"),
    ([], "Which order do you mean?", {}, "NO_TOOL"),
    ([], "I can't change the delivery address with the tools I have.", {}, "REFUSAL"),
    ([("get_order", {"order_id": "456"})], "", {"order_id": None}, "INVALID_ARGUMENTS"),
])
def test_classify(calls, text, kw, label):
    avail = ["get_order", "track_order", "cancel_order"]
    assert sel.classify(_rr(calls, text), expect=["get_order"], available=avail, **kw) == label
    assert label in sel.ALLOWED


def test_fixture_is_a_plain_function_and_renaming_keeps_the_body():
    f = sel.make_tool("order_op_a", "Gets an order.", kind="get_order")
    assert f.__name__ == "order_op_a" and f.__doc__ == "Gets an order."
    out = f(order_id="123")
    assert out.startswith("TOOL_EXECUTED:order_op_a#1") and "wireless headphones" in out
    assert sel.EXEC[0]["tool"] == "order_op_a" and sel.EXEC[0]["kind"] == "get_order" and not sel.EFFECTS


# ------------------------------------------------------------ mechanics: the framework executes what the model selected
@pytest.fixture(scope="module")
def mech():
    return rows(e5.mechanics, "M")


@pytest.mark.parametrize("fw", FWS)
def test_framework_executes_exactly_the_selected_tool(mech, fw):
    for part, tool in (("pass-through get_order", "get_order"), ("pass-through cancel_order", "cancel_order")):
        r = find(mech, fw, part)
        assert r["requested"] == [tool] and r["executed"] == [tool]       # no re-ranking, no veto of a wrong-but-valid choice


@pytest.mark.parametrize("fw,msg", [("langgraph", "is not a valid tool"), ("pydantic-ai", "Unknown tool name")])
def test_invented_tool_name_is_rejected_and_fed_back_not_executed(mech, fw, msg):
    r = find(mech, fw, "invented tool")
    assert r["executed"] == [] and r["class"] == "INVALID_TOOL" and msg in r["model_saw"] and r["run_error"] == ""
    rec = find(mech, fw, "unknown tool then recovers")
    assert rec["requested"] == ["change_address", "get_order"] and rec["executed"] == ["get_order"]


@pytest.mark.parametrize("fw", FWS)
def test_missing_required_argument_is_rejected_before_the_body(mech, fw):
    assert find(mech, fw, "missing required arg")["executed"] == []


@pytest.mark.parametrize("fw", FWS)
def test_schema_pattern_is_enforced_by_the_framework_not_the_model(mech, fw):
    plain = find(mech, fw, "schema=plain")
    rich = find(mech, fw, "schema=rich")
    assert plain["executed"] == ["get_order"] and plain["executed_args"] == [{"order_id": "#123"}]
    assert rich["executed"] == []
    assert ("string_pattern_mismatch" if fw == "pydantic-ai" else "ToolInvocationError") in rich["model_saw"]


# ------------------------------------------------------------ what the model receives (5.14)
@pytest.fixture(scope="module")
def wire_rows():
    return rows(e5.wire_payloads, "5.14")


def test_request_envelopes_differ_between_frameworks(wire_rows):
    lg = find(wire_rows, "langgraph", "request envelope")
    pa = find(wire_rows, "pydantic-ai", "request envelope")
    assert lg["tool_names"] == pa["tool_names"] == e5.S5                  # same tools, same order
    assert lg["temperature"] == pa["temperature"] == 0                    # equalized for this batch
    assert lg["tool_choice"] == "(absent)" and pa["tool_choice"] == "auto"
    assert lg["strict"] == ["(absent)"] and pa["strict"] == [True]
    assert lg["additionalProperties"] == ["(absent)"] and pa["additionalProperties"] == [False]
    assert lg["tools_sha"] != pa["tools_sha"]


def test_phase1_pydantic_ai_default_sends_no_temperature(wire_rows):
    assert find(wire_rows, "pydantic-ai", "without settings")["temperature"] == "(absent)"


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("var,changed", [("name", ["retrieve_order_information.name"]), ("description", ["get_order.description"]),
                                         ("schema", ["get_order.parameters"]),
                                         ("malicious description", ["check_order_promotions.description"])])
def test_one_variable_at_a_time_on_the_wire(wire_rows, fw, var, changed):
    r = find(wire_rows, fw, f"variable={var}")
    assert r["changed"] == changed and r["rest_of_request_identical"]


@pytest.mark.parametrize("fw", FWS)
def test_tool_order_is_passed_through_unchanged(wire_rows, fw):
    r = find(wire_rows, fw, "variable=order")
    assert r["changed"][0].startswith("same definitions") and r["rest_of_request_identical"]


@pytest.mark.parametrize("fw", FWS)
def test_malicious_description_reaches_the_model_verbatim(wire_rows, fw):
    assert find(wire_rows, fw, "variable=malicious description")["b_example"]["description"] == e5.PROMO_EVIL


# ------------------------------------------------------------ tool_choice (5.15)
@pytest.fixture(scope="module")
def choice_rows():
    return rows(e5.choice, "5.15")


@pytest.mark.parametrize("part,wire_value", [("tool_choice=None", "(absent)"), ("tool_choice='auto'", "auto"), ("tool_choice='none'", "none"),
                                             ("tool_choice='required'", "required"),
                                             ("tool_choice='get_order'", {"type": "function", "function": {"name": "get_order"}})])
def test_langgraph_sends_tool_choice_only_when_bound(choice_rows, part, wire_value):
    assert find(choice_rows, "langgraph", f"LG bind_tools({part})")["wire_tool_choice"] == wire_value


@pytest.mark.parametrize("part,wire_value", [("'auto'", "auto"), ("'none'", "none"), ("'required'", "required"),
                                             ("['get_order']", {"type": "function", "function": {"name": "get_order"}})])
def test_pydantic_ai_capability_sends_tool_choice(choice_rows, part, wire_value):
    assert find(choice_rows, "pydantic-ai", f"PA capability tool_choice={part}")["wire_tool_choice"] == wire_value


def test_pydantic_ai_rejects_static_required_before_any_request(choice_rows):
    r = find(choice_rows, "pydantic-ai", "static model_settings tool_choice='required'")
    assert r["wire_tool_choice"] == "(no request)" and "UserError" in r["run_error"]


@pytest.mark.parametrize("fw,err", [("langgraph", "GraphRecursionError"), ("pydantic-ai", "UsageLimitExceeded")])
def test_required_on_every_step_never_finishes(choice_rows, fw, err):
    r = find(choice_rows, fw, "required on every step")
    assert err in r["run_error"] and r["requests"] >= 8


# ------------------------------------------------------------ selection != authorization != execution (5.17 / 5.18 offline)
@pytest.fixture(scope="module")
def sec_rows():
    e5.RESULTS.clear()
    e5.security()
    return e5.RESULTS["5.17-offline"] + e5.RESULTS["5.18-offline"]


@pytest.mark.parametrize("fw", FWS)
def test_selected_restricted_tool_executes_without_application_check(sec_rows, fw):
    r = [x for x in sec_rows if x["exp"] == "5.17" and x["framework"] == fw and x["case"] == "no application check"][0]
    assert r["bodies_started"] == ["refund_order", "delete_order"] and r["effects"] == ["refund_order", "delete_order"]


@pytest.mark.parametrize("fw", FWS)
def test_application_check_blocks_effects_after_selection(sec_rows, fw):
    for exp in ("5.17", "5.18"):
        r = [x for x in sec_rows if x["exp"] == exp and x["framework"] == fw and "VIEWER" in x["case"]][0]
        assert r["effects"] == [] and r["authz_denies"] >= 1 and "refund_order" in r["bodies_started"]


@pytest.mark.parametrize("fw", FWS)
def test_injected_next_selection_executes_when_unprotected(sec_rows, fw):
    r = [x for x in sec_rows if x["exp"] == "5.18" and x["framework"] == fw and x["case"] == "unprotected"][0]
    assert r["effects"] == ["refund_order"]


# ------------------------------------------------------------ first-turn probe: tool still executes, 2nd model call is local
@pytest.mark.parametrize("fw", FWS)
def test_probe_executes_selected_tool_then_ends_without_second_request(fw):
    t, log = wire.transport([call("get_order", order_id="123"), ("text", "should not be requested")])
    m = (sel.lg_model if fw == "langgraph" else sel.pa_model)(probe=True, transport=t)
    r = (lg_agent.run if fw == "langgraph" else pa_agent.run)("Show me order 123.", m, tools=sel.tools("get_order", "track_order"))
    assert len(log) == 1 and r.final_text == sel.PROBE_END and [e["tool"] for e in sel.EXEC] == ["get_order"]
    assert r.tool_calls[0].call_id == "call_1_0"


def test_relabel_recovers_selection_from_execution_log_when_the_run_ended_by_exception():
    d = {"requested": [], "executed": [(1, "refund_order", "failed")], "expect": ["refund_order"], "alternatives": [], "ambiguous": False,
         "tools": ["get_order", "refund_order"], "class": "NO_TOOL", "selected": None, "run_error": "PermissionDenied: ..."}
    e5.relabel(d)
    assert d["selected"] == "refund_order" and d["class"] == "CORRECT" and "execution log" in d["selected_from"]
    assert not d["provider_rejected"]
    assert e5.relabel({**d, "executed": [], "requested": [], "run_error": "400 - Tool choice is required, but model did not call a tool"})["provider_rejected"]


# ------------------------------------------------------------ live: invariant, not model behavior
from lab_common.config import model_configured  # noqa: E402

live = pytest.mark.skipif(not model_configured(), reason="no model provider configured")


@live
@pytest.mark.live
@pytest.mark.parametrize("fw", FWS)
def test_live_restricted_selection_never_produces_a_side_effect_when_authorization_is_enforced(fw):
    c = [x for x in e5.live_plan() if x["case"] == "authz info absent from prompt"][0]
    try:
        d = e5.live_one(c, fw, 0)
    except e5.DailyLimit:
        d = None
    if d is None:
        pytest.skip("provider rate limit")
    assert d["effects"] == []                                              # whatever the model selected
    assert not ({"refund_order", "delete_order"} & set(d["requested"])) or d["authz_denies"] >= 1
