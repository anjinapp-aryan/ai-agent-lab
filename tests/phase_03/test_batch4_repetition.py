"""Phase 3.1 Batch 4 regression tests: repeated calls, oscillation and no progress (offline, scripted completions).

Pins what the frameworks do NOT detect, what stops each trajectory, and that business effects belong to the data layer.
Not model behavior: live observations are in phases/phase-03/experiments.md.
"""
import ast
import inspect

import pytest

import exp3_batch4 as e4
from lab2 import fixtures as fx
from lab3 import patterns

C, FWS, call, case, budget = e4.C, e4.FWS, e4.call, e4.case, e4.budget
LIMIT_ERROR = {"langgraph": "GraphRecursionError", "pydantic-ai": "UsageLimitExceeded"}
ST = call("check_status", order_id="123")


def repeated_order(n=40):
    return [call("get_order", order_id="123")] * n


def runaway(fw, script, tools, k=e4.K):
    return case("t", "t", fw, script, budget(fw, k), tools)


# ------------------------------------------------------------ repetition is not detected
@pytest.mark.parametrize("fw", FWS)
def test_identical_calls_all_execute_until_the_loop_limit(fw):
    r = runaway(fw, repeated_order(), [e4.get_order])
    assert r["model_requests"] == r["tool_executions"] == e4.K and r["stopped_by"] == LIMIT_ERROR[fw]
    assert not r["framework_flagged_repetition"] and r["wire"]["system_user_unchanged"]


@pytest.mark.parametrize("fw", FWS)
def test_different_call_ids_each_get_their_own_tool_message(fw):
    r = case("t", "t", fw, [call("get_order", order_id="123")] * 3 + [C("done")], 25, [e4.get_order])
    assert r["call_ids"] == ["call_1_0", "call_2_0", "call_3_0"] == r["wire"]["last_tool_call_ids"]
    assert r["stopped_by"].startswith("natural")


@pytest.mark.parametrize("fw", FWS)
def test_call_id_reused_across_turns_still_executes_every_time(fw):
    t, _ = e4.same_id_transport([call("get_order", order_id="123")] * 3 + [C("done")])
    r = case("t", "t", fw, [], 25, [e4.get_order], transport=t)
    assert r["tool_executions"] == 3 and r["wire"]["last_tool_call_ids"] == ["call_same"] * 3


def test_duplicate_call_id_in_one_response_is_rejected_only_by_pydantic_ai():
    script = [C(None, [("get_order", {"order_id": "123"})] * 2), C("done")]
    t, _ = e4.same_id_transport(script)
    lg = case("t", "t", "langgraph", [], 25, [e4.get_order], transport=t)
    assert lg["tool_executions"] == 2 and lg["stopped_by"].startswith("natural")
    t, _ = e4.same_id_transport(script)
    pa = case("t", "t", "pydantic-ai", [], 25, [e4.get_order], transport=t)
    assert pa["tool_executions"] == 0 and "must have unique `tool_call_id`" in pa["error"]


@pytest.mark.parametrize("fw", FWS)
def test_identical_calls_with_distinct_ids_in_one_response_both_execute(fw):
    r = case("t", "t", fw, [C(None, [("get_order", {"order_id": "123"})] * 2), C("done")], 25, [e4.get_order])
    assert r["tool_executions"] == 2 and r["model_requests"] == 2


@pytest.mark.parametrize("fw", FWS)
def test_same_tool_different_arguments_is_a_normal_run(fw):
    r = case("t", "t", fw, [call("get_order", order_id=str(n)) for n in range(123, 127)] + [C("done")], 25, [e4.get_order])
    assert r["tool_executions"] == 4 and r["stopped_by"].startswith("natural") and r["label"] == "NONE"


# ------------------------------------------------------------ oscillation and no progress are not detected
@pytest.mark.parametrize("fw", FWS)
def test_exact_oscillation_runs_to_the_limit(fw):
    r = runaway(fw, [call("tool_a"), call("tool_b")] * 20, [e4.tool_a, e4.tool_b])
    assert [x["tool"] for x in r["trajectory"]] == ["tool_a", "tool_b"] * 3 and r["stopped_by"] == LIMIT_ERROR[fw]
    assert r["label"] == "OSCILLATION" and not r["framework_flagged_repetition"]


@pytest.mark.parametrize("fw", FWS)
def test_oscillation_with_changing_arguments_runs_to_the_limit_and_is_not_labelled(fw):
    script = [s for n in range(1, 21) for s in (call("arg_a", n=n), call("arg_b", n=n))]
    r = runaway(fw, script, [e4.arg_a, e4.arg_b])
    assert r["stopped_by"] == LIMIT_ERROR[fw] and r["label"] == "NONE"


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("results,label", [(["PENDING"], "NO_PROGRESS"), (["PENDING", "Still pending.", "Processing has not completed."] * 20, "REPEATED_CALL")])
def test_no_progress_is_invisible_to_the_framework_and_rewording_defeats_string_equality(fw, results, label):
    r = runaway(fw, [ST] * 40, [e4.scripted_status(results)])
    assert r["stopped_by"] == LIMIT_ERROR[fw] and r["wire"]["system_user_unchanged"] and r["label"] == label


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("results", [["PROCESSING", "PROCESSING", "SHIPPED"], ["PENDING", "PENDING", "COMPLETE"]])
def test_legitimate_polling_finishes_naturally_but_the_syntactic_label_is_a_false_alarm(fw, results):
    r = case("t", "t", fw, [ST] * 3 + [C("Finished.")], 25, [e4.scripted_status(results)])
    assert r["stopped_by"].startswith("natural") and r["final_text"] == "Finished." and r["label"] == "REPEATED_CALL"


@pytest.mark.parametrize("fw", FWS)
def test_the_limit_cannot_tell_legitimate_polling_from_a_runaway(fw):
    r = case("t", "t", fw, [ST] * 3 + [C("Finished.")], budget(fw, 2), [e4.scripted_status(["PENDING", "PENDING", "COMPLETE"])])
    assert r["stopped_by"] == LIMIT_ERROR[fw] and r["tool_executions"] == 2


# ------------------------------------------------------------ business effects belong to the data layer
def _orders(fw, script, mk):
    o = fx.Orders()
    r = case("t", "t", fw, script + [C("done")], 25, [mk(o)])
    return r, o


def _keyed(k):
    return call("create_order", item="widget", qty=1, idempotency_key=k)


@pytest.mark.parametrize("fw", FWS)
def test_framework_executes_duplicates_and_only_the_guard_protects_the_ledger(fw):
    r, o = _orders(fw, [call("create_order", item="widget", qty=1)] * 3, lambda x: x.unguarded())
    assert r["tool_executions"] == 3 and len(o.ledger) == 3
    r, o = _orders(fw, [_keyed("same-key")] * 3, lambda x: x.model_keyed())
    assert r["tool_executions"] == 3 and len(o.ledger) == 1 and o.replays == 2


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("keys,ledger", [("ABC", 3), ("AA", 1), ("AB", 2)])
def test_the_key_defines_the_business_operation_not_the_framework(fw, keys, ledger):
    r, o = _orders(fw, [_keyed(k) for k in keys], lambda x: x.model_keyed())
    assert r["tool_executions"] == len(keys) and len(o.ledger) == ledger and len(set(r["call_ids"])) == len(keys)


# ------------------------------------------------------------ repetition after errors and empty results
FAIL = [call("flaky")] * 40
FOREVER = 10 ** 6


def _rp():
    from langgraph.types import RetryPolicy
    return RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)


def test_langgraph_default_error_loop_is_stopped_only_by_the_limit():
    r = case("t", "t", "langgraph", FAIL, 12, [fx.make_flaky(FOREVER, "runtime")])
    assert r["tool_executions"] == 6 and r["stopped_by"] == "GraphRecursionError"


def test_langgraph_retry_policy_is_bounded_and_makes_one_model_decision():
    r = case("t", "t", "langgraph", FAIL, 12, [fx.make_flaky(FOREVER, "runtime")], handle_tool_errors=False, retry_policy=_rp())
    assert r["model_requests"] == 1 and r["tool_executions"] == 3 and r["stopped_by"] == "RuntimeError"
    assert [x["reexecution"] for x in r["trajectory"]] == [False, True, True]
    assert r["call_ids"] == ["call_1_0"] * 3


def test_langgraph_default_handler_consumes_the_error_before_retry_policy_sees_it():
    r = case("t", "t", "langgraph", FAIL, 12, [fx.make_flaky(FOREVER, "runtime")], retry_policy=_rp())
    assert r["model_requests"] == r["tool_executions"] == 6 and not any(x["reexecution"] for x in r["trajectory"])


@pytest.mark.parametrize("retries,requests", [(1, 2), (3, 4)])
def test_pydantic_ai_retry_budget_stops_repeated_failure(retries, requests):
    r = case("t", "t", "pydantic-ai", FAIL, 6, [fx.make_flaky(FOREVER, "model_retry")], retries=retries)
    assert r["model_requests"] == requests and f"exceeded max retries count of {retries}" in r["error"]


def test_pydantic_ai_retry_budget_resets_on_success_so_alternating_failure_reaches_the_limit():
    r = case("t", "t", "pydantic-ai", FAIL, 6, [e4.alternating("model_retry")], retries=1)
    assert r["tool_executions"] == 6 and r["stopped_by"] == "UsageLimitExceeded"


def test_pydantic_ai_plain_exception_escapes_at_once():
    r = case("t", "t", "pydantic-ai", FAIL, 6, [fx.make_flaky(FOREVER, "runtime")], retries=3)
    assert r["model_requests"] == r["tool_executions"] == 1 and r["stopped_by"] == "RuntimeError"


@pytest.mark.parametrize("fw,content", [("langgraph", []), ("pydantic-ai", "")])
def test_empty_result_has_no_loop_semantics_and_is_not_an_error(fw, content):
    r = runaway(fw, [call("search", query="x")] * 40, [e4.search])
    assert r["stopped_by"] == LIMIT_ERROR[fw] and r["tool_executions"] == e4.K
    assert r["wire"]["last_tool_contents"][-1] == content


# ------------------------------------------------------------ fallback stops
@pytest.mark.parametrize("script,tools", [(repeated_order(), [e4.get_order]), ([call("tool_a"), call("tool_b")] * 20, [e4.tool_a, e4.tool_b])])
def test_pydantic_ai_tool_calls_limit_is_count_based_and_refuses_before_the_next_execution(script, tools):
    r = case("t", "t", "pydantic-ai", script, 25, tools, tool_calls_limit=3)
    assert r["tool_executions"] == 3 and r["tool_calls_requested"] == 4 and "tool_calls_limit" in r["error"]


def test_pydantic_ai_default_request_limit_lets_a_runaway_make_50_decisions():
    r = case("t", "t", "pydantic-ai", repeated_order(80), 50, [e4.get_order])
    assert r["model_requests"] == r["tool_executions"] == 50 and r["stopped_by"] == "UsageLimitExceeded"


# ------------------------------------------------------------ reasoning echo (found live, pinned offline)
def test_pydantic_ai_echoes_the_models_reasoning_and_langgraph_drops_it():
    seen = {}
    for fw in FWS:
        t, _ = e4.reasoning_transport([ST, C("done")])
        case("t", "t", fw, [], 25, [e4.scripted_status(["PENDING"])], transport=t)
        assistant = e4.e3.LAST_TRAJ.of("model_request")[1]["messages"][2]
        seen[fw] = assistant.get("reasoning")
    assert seen == {"langgraph": None, "pydantic-ai": "The model's private reasoning."}


# ------------------------------------------------------------ the classifier is a pure, offline function
@pytest.mark.parametrize("executions,label", [
    ([], "NONE"),
    ([("a", {}, 1)] * 2, "NONE"),
    ([("a", {}, 1)] * 3, "NO_PROGRESS"),
    ([("a", {}, 1), ("a", {}, 2), ("a", {}, 3)], "REPEATED_CALL"),
    ([("a", {}, 1), ("b", {}, 1)] * 2, "OSCILLATION"),
    ([("a", {}, 1), ("b", {}, 1), ("a", {}, 1)], "NONE"),
    ([("a", {"n": 1}, 1), ("b", {"n": 1}, 1), ("a", {"n": 2}, 1), ("b", {"n": 2}, 1)], "NONE"),
    ([("a", {"n": i}, 1) for i in range(5)], "NONE"),
])
def test_classify(executions, label):
    assert patterns.classify(executions) == label


def test_classifier_imports_no_framework_and_exposes_one_function():
    tree = ast.parse(inspect.getsource(patterns))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert names == {"json"} and not froms
    assert [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))] == ["classify"]
