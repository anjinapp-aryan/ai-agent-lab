"""Phase 3.1 Batch 5 regression tests: retry and failure recovery within the loop (offline, scripted completions).

Pins what each framework keeps, shows the model, and leaves to other layers when a tool fails. Not model behavior:
live observations are in phases/phase-03/experiments.md.
"""
import pytest

import exp3_batch5 as e5
from lab2 import fixtures as fx

C, call, FWS = e5.C, e5.call, e5.FWS
LG_TEMPLATE = "Error: ModelRetry('{}')\n Please fix your mistakes."
PA_TEMPLATE = '{{"error":"{}"}}'
STEP_A_SCRIPT = [call("step_a"), call("step_a"), C("Step A completed after one retry.")]


def tmpl(fw, text):
    return (LG_TEMPLATE if fw == "langgraph" else PA_TEMPLATE).format(text)


def run(fw, script, tools, limit=25, **kw):
    return e5.case("t", "t", fw, script, limit, tools, **kw)


def ids_and_outcomes(row):
    return [(h["result_for"], h["outcome"]) for h in row["history"] if h["role"] == "tool"]


# ------------------------------------------------------------ B5-3.1.39 recoverable failure
@pytest.mark.parametrize("fw", FWS)
def test_failed_call_stays_in_history_and_the_retry_is_a_new_model_decision(fw):
    r = run(fw, STEP_A_SCRIPT, [e5.sched("step_a", ["fail", "ok"])])
    assert (r["model_requests"], r["tool_executions"], r["tool_executions_failed"], r["stopped_by"]) == (3, 2, 1, "natural")
    assert [x["call_id"] for x in r["trajectory"]] == ["call_1_0", "call_2_0"] and r["framework_reexecutions"] == 0
    assert ids_and_outcomes(r) == [("call_1_0", "error" if fw == "langgraph" else "retried"), ("call_2_0", "success")]
    assert r["tool_messages_by_request"][1] == [["call_1_0", tmpl(fw, "Temporary failure")]]            # the model sees the error next
    assert [m[0] for m in r["tool_messages_by_request"][2]] == ["call_1_0", "call_2_0"]                  # and still sees it afterwards
    assert r["final_text"] == "Step A completed after one retry."


# ------------------------------------------------------------ B5-3.1.40 permanent failure
@pytest.mark.parametrize("fw", FWS)
def test_a_models_truthful_give_up_is_accepted_and_the_record_keeps_every_failure(fw):
    script = [call("step_a")] * 3 + [C("I could not complete step A: it failed three times.")]
    r = run(fw, script, [e5.sched("step_a", ["fail"])], retries=3)
    assert r["stopped_by"] == "natural" and r["tool_executions_failed"] == 3 and len(r["failures"]) == 3
    assert r["final_text"].startswith("I could not complete step A")


def test_the_frameworks_retry_budget_can_pre_empt_the_models_decision_to_give_up():
    script = [call("step_a")] * 3 + [C("I could not complete step A.")]
    r = run("pydantic-ai", script, [e5.sched("step_a", ["fail"])], retries=1)                          # the Phase 1 agent default
    assert r["model_requests"] == 2 and "exceeded max retries count of 1" in r["error"] and r["final_text"] is None


# ------------------------------------------------------------ B5-3.1.41 failure after an earlier success
@pytest.mark.parametrize("fw", FWS)
def test_earlier_success_and_later_failure_stay_in_every_later_request_and_a_is_not_rerun(fw):
    script = [call("step_a"), call("step_b"), call("step_c"), C("A and C succeeded.")]
    r = run(fw, script, [e5.sched("step_a", ["ok"]), e5.sched("step_b", ["fail"]), e5.sched("step_c", ["ok"])], retries=2)
    assert r["executions_by_tool"] == {"step_a": 1, "step_b": 1, "step_c": 1} and r["stopped_by"] == "natural"
    last = r["tool_messages_by_request"][3]
    assert [m[0] for m in last] == ["call_1_0", "call_2_0", "call_3_0"]
    assert last[0][1].startswith("step_a succeeded") and last[1][1] == tmpl(fw, "Temporary failure") and last[2][1].startswith("step_c succeeded")


# ------------------------------------------------------------ B5-3.1.42 recovery after a partial side effect
def _orders(fw, script, make, fail_first=1):
    o = fx.Orders(fail_first=fail_first, exc=e5.LostResponse)
    r = run(fw, script, [make(o)], q="Place an order for 1 widget.", retries=2)
    return r, o


CREATE = call("create_order", item="widget", qty=1)


@pytest.mark.parametrize("fw", FWS)
def test_without_a_guard_the_retry_makes_a_second_order_and_the_unverified_final_answer_is_accepted(fw):
    r, o = _orders(fw, [CREATE, CREATE, C("Your order has been created.")], lambda x: x.unguarded())
    assert len(o.ledger) == 2 and r["tool_executions"] == 2 and r["stopped_by"] == "natural"
    assert r["final_text"] == "Your order has been created."                                              # nothing checks it against the ledger
    assert "timed out" in r["tool_messages_by_request"][1][0][1] and "created" not in r["tool_messages_by_request"][1][0][1]


@pytest.mark.parametrize("fw", FWS)
def test_with_a_context_key_the_retry_replays_and_the_model_cannot_tell(fw):
    r, o = _orders(fw, [CREATE, CREATE, C("Your order has been created.")], lambda x: x.context_keyed("req-1"))
    assert len(o.ledger) == 1 and o.replays == 1
    assert r["tool_messages_by_request"][2][1][1].startswith("order #1 created")                           # a replay reads like a fresh result


def test_same_key_different_args_conflict_is_recoverable_in_langgraph_and_ends_the_pydantic_ai_run():
    def keyed(n):
        return call("create_order", item="widget", qty=n, idempotency_key="K1")
    lg, o = _orders("langgraph", [keyed(1), keyed(2), C("The order could not be changed.")], lambda x: x.model_keyed())
    assert lg["stopped_by"] == "natural" and len(o.ledger) == 1 and lg["tool_executions_failed"] == 2
    pa, o = _orders("pydantic-ai", [keyed(1), keyed(2), C("The order could not be changed.")], lambda x: x.model_keyed())
    assert pa["stopped_by"] == "ConflictError" and len(o.ledger) == 1 and pa["final_text"] is None


# ------------------------------------------------------------ B5-3.1.43 error wording
@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("text", e5.WORDINGS)
def test_wording_reaches_the_model_inside_a_framework_template_and_changes_nothing_else(fw, text):
    script = [call("fetch_report", report_id="7"), C("The report could not be fetched.")]
    r = run(fw, script, [e5.fetch_report_tool(text)], retries=2)
    assert r["tool_messages_by_request"][1] == [["call_1_0", tmpl(fw, text)]]
    assert (r["model_requests"], r["tool_executions"], r["stopped_by"]) == (2, 1, "natural")             # no wording is classified


@pytest.mark.parametrize("fw", FWS)
def test_a_returned_error_string_reaches_the_model_verbatim_without_a_template(fw):
    r = run(fw, [call("fetch_report", report_id="7"), C("done")], [e5.fetch_report_tool("Permission denied", "return")], retries=2)
    assert r["tool_messages_by_request"][1] == [["call_1_0", "Permission denied"]] and r["tool_executions_failed"] == 0


# ------------------------------------------------------------ B5-3.1.44 multiple failures
@pytest.mark.parametrize("fw", FWS)
def test_failed_tools_retry_independently_successes_are_not_rerun_and_ids_stay_paired(fw):
    script = [C(None, [("step_a", {}), ("step_b", {}), ("step_c", {})]), C(None, [("step_b", {}), ("step_c", {})]), C("A, B and C all completed.")]
    r = run(fw, script, [e5.sched("step_a", ["ok"]), e5.sched("step_b", ["fail", "ok"]), e5.sched("step_c", ["fail", "ok"])], retries=2)
    assert r["executions_by_tool"] == {"step_a": 1, "step_b": 2, "step_c": 2} and r["tool_executions_failed"] == 2
    second, third = r["tool_messages_by_request"][1], r["tool_messages_by_request"][2]
    assert [m[0] for m in second] == ["call_1_0", "call_1_1", "call_1_2"]
    assert second[0][1].startswith("step_a succeeded") and second[1][1] == tmpl(fw, "Temporary failure") == second[2][1]
    assert [m[0] for m in third] == ["call_1_0", "call_1_1", "call_1_2", "call_2_0", "call_2_1"]
    assert third[3][1].startswith("step_b succeeded") and third[4][1].startswith("step_c succeeded")


# ------------------------------------------------------------ B5-3.1.45 failure, then natural termination
@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("final", ["Step A failed and I did not retry it.", "All done."])
def test_any_final_answer_after_a_failure_is_accepted_and_the_failure_stays_in_the_record(fw, final):
    r = run(fw, [call("step_a"), C(final)], [e5.sched("step_a", ["fail"])], retries=2)
    assert r["stopped_by"] == "natural" and r["final_text"] == final                                       # honest or not, it is accepted
    assert r["failures"] == ["ModelRetry: Temporary failure"] and ids_and_outcomes(r)[0][1] in ("error", "retried")


@pytest.mark.parametrize("fw", FWS)
def test_phase1_run_keeps_the_error_when_the_run_completes(fw):
    e5.RESULTS.clear()
    e5.natural()
    r = [x for x in e5.RESULTS["B5-3.1.45-run"] if x["framework"] == fw]
    assert r[0]["run_error"] is None and r[0]["tool_calls"][0][2] and "Temporary failure" in r[0]["tool_calls"][0][2]


# ------------------------------------------------------------ B5-3.1.46 retry exhaustion and partial state
@pytest.fixture(scope="module")
def exhaustion_rows():
    e5.RESULTS.clear()
    e5.exhaustion()
    return {r["case"][:2]: r for r in e5.RESULTS["B5-3.1.46"]}


def test_failed_runs_keep_state_in_the_native_view(exhaustion_rows):
    for key in ("E1", "E2", "E3", "E4"):
        r = exhaustion_rows[key]
        assert r["error"] and any(h["role"] == "tool" and h["text"].startswith("step_a succeeded") for h in r["history"])   # the earlier success survives


def test_exhaustion_leaves_a_pending_call_in_langgraph_state_when_the_policy_raises(exhaustion_rows):
    e2 = exhaustion_rows["E2"]
    assert e2["tool_executions"] == 4 and e2["model_requests"] == 2 and e2["pending_tool_calls_in_state"] == 1 and e2["stopped_by"] == "ModelRetry"
    assert not any("step_b" in str(h.get("text")) for h in e2["history"] if h["role"] == "tool")             # three executions, none recorded in state


def test_exhaustion_kinds(exhaustion_rows):
    assert exhaustion_rows["E1"]["stopped_by"] == "GraphRecursionError" and exhaustion_rows["E1"]["tool_executions_failed"] == 2
    assert exhaustion_rows["E3"]["stopped_by"] == "UnexpectedModelBehavior" and exhaustion_rows["E3"]["tool_executions_failed"] == 2
    assert exhaustion_rows["E4"]["stopped_by"] == "RuntimeError" and exhaustion_rows["E4"]["tool_executions_failed"] == 1


def test_the_phase1_application_level_run_returns_no_tool_calls_after_any_failed_run(exhaustion_rows):
    for key in ("E1", "E2", "E3", "E4"):
        a = exhaustion_rows[key]["application_run"]
        assert a["run_error"] and a["tool_calls_recorded"] == 0 and a["final_text"] == "" and a["llm_calls"] == 0


def test_committed_side_effects_are_missing_from_or_under_reported_by_retained_state(exhaustion_rows):
    e5_, e6 = exhaustion_rows["E5"], exhaustion_rows["E6"]
    assert e5_["ledger"] == 3 and sum(1 for h in e5_["history"] if h["role"] == "tool") == 3
    assert e6["ledger"] == 3 and sum(1 for h in e6["history"] if h["role"] == "tool") == 2                   # the last commit is in no retained message
    assert not any("created" in h["text"] for h in e5_["history"] + e6["history"] if h["role"] == "tool")


# ------------------------------------------------------------ B5-3.1.47 accounting
def test_accounting_columns_stay_separate():
    e5.RESULTS.clear()
    e5.accounting()
    t = {(x["scenario"], x["framework"]): x for x in e5.RESULTS["B5-3.1.47-table"]}
    for fw in FWS:
        mv = t[("model-visible retry (39)", fw)]
        assert (mv["model_requests"], mv["tool_executions"], mv["model_visible_retries"], mv["framework_retries"], mv["tool_internal_retries"]) == (3, 2, 1, 0, 0)
        ti = t[("tool-internal retry", fw)]
        assert (ti["tool_executions"], ti["tool_internal_retries"], ti["model_visible_retries"], ti["framework_retries"]) == (1, 2, 0, 0)
        h = t[("HTTP retry (SDK)", fw)]
        assert (h["http_attempts"], h["http_errors"], h["model_requests"], h["model_responses"]) == (3, 1, 2, 2)
        lost = t[("lost response, model retry, no guard", fw)]
        assert (lost["tool_executions"], lost["business_effects"]) == (2, 2)
    fr = t[("framework retry, LG RetryPolicy", "langgraph")]
    assert (fr["model_requests"], fr["tool_executions"], fr["framework_retries"], fr["model_visible_retries"], fr["framework_steps"]) == (2, 3, 2, 0, 3)


# ------------------------------------------------------------ live-result classifier and provider-error capture
def _live_row(**kw):
    return {"stopped_by": "natural", "final_text": "I could not fetch it.", "error": None, "provider_errors": [], **kw}


@pytest.mark.parametrize("kw,label", [
    ({}, "GAVE_UP_AND_SAID_SO"),
    ({"final_text": "  "}, "EMPTY_FINAL_ACCEPTED"),
    ({"final_text": "Here is the summary."}, "FINAL_WITHOUT_MENTIONING_FAILURE"),
    ({"final_text": "I’m sorry—I wasn’t able to retrieve it."}, "GAVE_UP_AND_SAID_SO"),
    ({"stopped_by": "UsageLimitExceeded", "final_text": ""}, "BUDGET_EXHAUSTED_NO_FINAL"),
    ({"stopped_by": "OpenAIInvalidRequestError", "error": "Error code: 400 - Tool call validation failed: ...", "final_text": ""}, "PROVIDER_REJECTED_TOOL_CALL"),
    ({"stopped_by": "UsageLimitExceeded", "final_text": "", "provider_errors": ['{"error":{"code":"tool_use_failed"}}']}, "PROVIDER_REJECTED_TOOL_CALL"),
    ({"stopped_by": "UsageLimitExceeded", "final_text": "", "provider_errors": ['{"error":{"message":"Rate limit reached ... requests per minute"}}']}, "BUDGET_EXHAUSTED_NO_FINAL"),
])
def test_outcome_class(kw, label):
    assert e5.outcome_class(_live_row(**kw)) == label


@pytest.mark.parametrize("fw", FWS)
def test_recorder_keeps_the_providers_error_body_for_a_non_200_response(fw):
    t, _, _ = e5.e3.http_flaky([call("step_a"), C("done")], 3, 500)             # the SDK gives up after 2 retries
    r = run(fw, [], [e5.sched("step_a", ["ok"])], transport=t)
    bodies = [e["error_body"] for e in e5.e3.LAST_TRAJ.of("model_response") if e.get("error_body")]
    assert r["stopped_by"] != "natural" and len(bodies) == 3 and all("simulated transient 500" in b for b in bodies)


# ------------------------------------------------------------ reasoning-echo controlled parity (Mode B)
def _assistant_with_reasoning(fw, **kw):
    t, _ = e5.e4.reasoning_transport([call("step_a"), call("step_a"), C("done")])
    e5.e3.run(fw, "q", [], 25, tools=[e5.sched("step_a", ["fail", "ok"])], transport=t, retries=2, **kw)
    reqs = e5.e3.LAST_TRAJ.of("model_request")
    return [m for r in reqs for m in r["messages"] if m["role"] == "assistant"], reqs


def test_native_pydantic_ai_echoes_reasoning_langgraph_does_not_and_controlled_mode_removes_the_echo():
    pa, _ = _assistant_with_reasoning("pydantic-ai")
    lg, _ = _assistant_with_reasoning("langgraph")
    ctl, _ = _assistant_with_reasoning("pydantic-ai", reasoning="stripped")
    assert pa and all("reasoning" in m for m in pa)
    assert lg and not any("reasoning" in m for m in lg)
    assert ctl and not any("reasoning" in m for m in ctl)


def test_controlled_mode_changes_only_the_reasoning_field():
    _, native = _assistant_with_reasoning("pydantic-ai")
    _, controlled = _assistant_with_reasoning("pydantic-ai", reasoning="stripped")

    def strip(msgs):
        return [{k: v for k, v in m.items() if k != "reasoning"} for m in msgs]
    assert [strip(r["messages"]) for r in native] == [strip(r["messages"]) for r in controlled]
    assert native[1]["messages"] != controlled[1]["messages"]
