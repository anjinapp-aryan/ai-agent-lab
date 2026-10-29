"""Phase 3.1 Batch 3 regression tests: iteration limits, retries and loop budget (offline, scripted completions).

Pins which event increments which counter in each framework. Not model behavior.
"""
import pytest

import exp3_batch3 as e3
from lab2 import fixtures as fx

C, MUL, ADD, MUL2, Q1, Q2 = e3.C, e3.MUL, e3.ADD, e3.MUL2, e3.Q1, e3.Q2
ONE_ROUND = [C(None, [MUL]), C("The answer is 425.")]
TWO_ROUNDS = [C(None, [ADD]), C(None, [MUL2]), C("The result is 495.")]
ENDLESS = [C(None, [MUL])] * 40


def ok(d):
    return d["result"] == "OK"


# ------------------------------------------------------------ what each limit counts
@pytest.mark.parametrize("script,steps,requests,q", [([C("final")], 1, 1, Q1), (ONE_ROUND, 3, 2, Q1), (TWO_ROUNDS, 5, 3, Q2)])
def test_langgraph_recursion_limit_must_exceed_super_steps(script, steps, requests, q):
    assert not ok(e3.run("langgraph", q, script, steps))                       # limit == super-steps -> GraphRecursionError
    d = e3.run("langgraph", q, script, steps + 1)
    assert ok(d) and d["framework_steps"] == steps and d["framework_model_calls"] == requests


@pytest.mark.parametrize("script,requests,q", [([C("final")], 1, Q1), (ONE_ROUND, 2, Q1), (TWO_ROUNDS, 3, Q2)])
def test_pydantic_ai_request_limit_counts_model_requests(script, requests, q):
    assert ok(e3.run("pydantic-ai", q, script, requests))
    if requests > 1:
        d = e3.run("pydantic-ai", q, script, requests - 1)
        assert d["error"].startswith("UsageLimitExceeded") and d["framework_model_calls"] == requests - 1


def test_langgraph_discards_a_completed_final_answer_when_the_last_step_hits_the_limit():
    d = e3.run("langgraph", Q1, ONE_ROUND, 3)
    assert d["error"].startswith("GraphRecursionError")
    assert d["state"] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]   # the final answer is in state, run still fails
    assert d["model_responses"] == 2 and d["last_event"] == "model_response"


# ------------------------------------------------------------ final-step safety (endless tool calls)
@pytest.mark.parametrize("limit,requested,executed,pending", [(2, 1, 1, 0), (3, 2, 1, 1), (4, 2, 2, 0), (5, 3, 2, 1), (6, 3, 3, 0)])
def test_langgraph_odd_limits_leave_the_last_requested_tool_unexecuted(limit, requested, executed, pending):
    d = e3.run("langgraph", Q1, ENDLESS, limit)
    assert (d["tool_calls_requested"], d["tool_executions"], d["pending_tool_calls_in_state"]) == (requested, executed, pending)


@pytest.mark.parametrize("limit", [2, 3, 4])
def test_pydantic_ai_request_limit_runs_the_last_tool_then_refuses_the_next_request(limit):
    d = e3.run("pydantic-ai", Q1, ENDLESS, limit)
    assert d["tool_calls_requested"] == d["tool_executions"] == limit and d["last_event"] == "tool_end"
    assert d["http_attempts"] == limit and d["error"].startswith("UsageLimitExceeded")


@pytest.mark.parametrize("n", [0, 1, 2])
def test_pydantic_ai_tool_calls_limit_refuses_before_executing(n):
    d = e3.run("pydantic-ai", Q1, ENDLESS, 10, tool_calls_limit=n)
    assert d["tool_calls_requested"] == n + 1 and d["tool_executions"] == n and d["last_event"] == "model_response"
    assert "tool_calls_limit" in d["error"]


# ------------------------------------------------------------ output re-prompt vs budget
def test_pydantic_ai_reprompt_is_a_model_request_that_consumes_request_limit():
    script = [C(""), C("The final result is 425.")]
    assert e3.run("pydantic-ai", Q1, script, 1)["error"].startswith("UsageLimitExceeded")
    d = e3.run("pydantic-ai", Q1, script, 2)
    assert ok(d) and d["http_attempts"] == 2 and d["final_text"] == "The final result is 425."
    lg = e3.run("langgraph", Q1, script, 2)
    assert ok(lg) and lg["http_attempts"] == 1 and lg["final_text"] == ""   # LangGraph accepts the empty response


@pytest.mark.parametrize("retries,limit,requests,exc", [(0, 10, 1, "UnexpectedModelBehavior"), (1, 10, 2, "UnexpectedModelBehavior"),
                                                        (2, 10, 3, "UnexpectedModelBehavior"), (3, 10, 4, "UnexpectedModelBehavior"),
                                                        (2, 2, 2, "UsageLimitExceeded"), (3, 2, 2, "UsageLimitExceeded")])
def test_persistent_empty_output_is_stopped_by_whichever_budget_runs_out_first(retries, limit, requests, exc):
    d = e3.run("pydantic-ai", Q1, [C("")] * 10, limit, retries=retries)
    assert d["http_attempts"] == requests and d["error"].startswith(exc)


# ------------------------------------------------------------ tool retries vs budget
def _flaky(fw, fail, limit, **kw):
    fx.EXEC.clear()
    script = [C(None, [("flaky", {})])] * (fail + 1) + [C("done")]
    how = "model_retry" if fw == "pydantic-ai" else "runtime"
    d = e3.run(fw, "Call flaky.", script, limit, tools=[fx.make_flaky(fail, how)], **kw)
    d["flaky_executions"] = sum(1 for e in fx.EXEC if e[0] == "flaky")
    return d


def test_pydantic_ai_model_retry_costs_one_model_request_per_retry():
    d = _flaky("pydantic-ai", 1, 10, retries=1)
    assert ok(d) and d["http_attempts"] == 3 and d["flaky_executions"] == 2
    assert _flaky("pydantic-ai", 1, 2, retries=1)["error"].startswith("UsageLimitExceeded")   # the retry needed a 3rd request
    assert "exceeded max retries count of 0" in _flaky("pydantic-ai", 1, 10, retries=0)["error"]


def test_pydantic_ai_plain_exception_is_not_retried():
    fx.EXEC.clear()
    d = e3.run("pydantic-ai", "Call flaky.", [C(None, [("flaky", {})]), C("done")], 10, tools=[fx.make_flaky(1, "runtime")], retries=2)
    assert d["error"].startswith("RuntimeError") and d["http_attempts"] == 1


def test_langgraph_model_visible_retry_costs_two_super_steps():
    d = _flaky("langgraph", 1, 25)
    assert ok(d) and d["framework_steps"] == 5 and d["http_attempts"] == 3
    assert not ok(_flaky("langgraph", 1, 4)) and ok(_flaky("langgraph", 1, 6))


def test_langgraph_retry_policy_reexecutes_inside_one_super_step_without_the_model():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=3, retry_on=Exception)
    for limit in (25, 4):
        fx.EXEC.clear()
        d = e3.run("langgraph", "Call flaky.", [C(None, [("flaky", {})]), C("done")], limit, tools=[fx.make_flaky(2, "runtime")],
                   handle_tool_errors=False, retry_policy=rp)
        assert ok(d) and d["http_attempts"] == 2 and d["framework_steps"] == 3
        assert sum(1 for e in fx.EXEC if e[0] == "flaky") == 3              # 3 executions, 1 model decision, 1 super-step


# ------------------------------------------------------------ provider-client HTTP retry
@pytest.mark.parametrize("fw,limit", [("langgraph", 4), ("pydantic-ai", 2)])
@pytest.mark.parametrize("fail", [1, 2])
def test_http_retry_is_invisible_to_the_loop_budget(fw, limit, fail):
    t, _, att = e3.http_flaky(ONE_ROUND, fail, 500)
    d = e3.run(fw, Q1, ONE_ROUND, limit, transport=t)
    assert ok(d) and att["n"] == 2 + fail and d["framework_model_calls"] == 2
    assert d["framework_steps"] == (3 if fw == "langgraph" else 6)


@pytest.mark.parametrize("fw", e3.FWS)
def test_http_retry_gives_up_after_the_sdk_default_of_two_retries(fw):
    t, _, att = e3.http_flaky(ONE_ROUND, 3, 500)
    d = e3.run(fw, Q1, ONE_ROUND, 25, transport=t)
    assert att["n"] == 3 and d["result"] != "OK" and d["framework_model_calls"] == 0


# ------------------------------------------------------------ near-limit behavior and side effects
def test_only_the_deprecated_prebuilt_writes_a_final_answer_at_the_limit():
    e3.RESULTS.clear()
    e3.warn()
    rows = e3.RESULTS["B3-3.1.21"]
    for fw in ("langgraph", "pydantic-ai"):
        r = [x for x in rows if x["framework"] == fw][0]
        assert r["identical_instructions_every_request"] and r["error"]           # no warning injected, the run just fails
    pre = {x["limit"]: x for x in rows if "prebuilt" in x["framework"]}
    assert pre[4]["error"] is None and pre[4]["final_message"] == "Sorry, need more steps to process this request."
    assert pre[3]["error"] and pre[5]["error"]


@pytest.mark.parametrize("fw,limit,orders", [("langgraph", 4, 2), ("langgraph", 5, 2), ("pydantic-ai", 2, 2), ("pydantic-ai", 3, 3)])
def test_side_effects_before_the_limit_are_not_rolled_back(fw, limit, orders):
    o = fx.Orders()
    d = e3.run(fw, "Order one widget.", [C(None, [("create_order", {"item": "widget", "qty": 1})])] * 40, limit, tools=[o.unguarded()])
    assert d["result"] != "OK" and len(o.ledger) == orders == d["tool_executions"]
