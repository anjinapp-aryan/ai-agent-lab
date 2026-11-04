"""Phase 3.1 Batch 6 regression tests: timeouts, cancellation and token limits.

Local only: a stdlib HTTP server on 127.0.0.1 plays a slow provider (real sockets, so real client timeouts), tools log their own
start / effect / cancel / finish, and scripted completions carry the model's decisions. Durations are short and assertions use
event ordering and generous bounds, not tight timings. They pin framework and runtime mechanics for these versions; they say
nothing about a real model.
"""
import threading
import time

import pytest

import exp3_batch6 as e6
from lab3.slow_server import SlowServer
from pydantic_ai import UsageLimits

C, call, FWS, noop = e6.C, e6.call, e6.FWS, e6.noop
OK = [C("The answer is 4.")]
ONE_ROUND = [call("work"), C("Done.")]


def run(fw, label, script, tools, **kw):
    return e6.scenario("t", fw, label, script, tools, **kw)


# ------------------------------------------------------------ B6-3.1.48 model request timeout (real sockets)
@pytest.mark.parametrize("fw", FWS)
def test_the_sdk_retries_a_timed_out_attempt_and_the_framework_sees_one_failed_model_call(fw):
    r = run(fw, "timeout", OK, noop, delays=[0.6] * 4, model_kw=dict(timeout=0.15), settle=0.2)
    assert r["exc"] in ("OpenAITimeoutError", "ModelTimeoutError") and r["state_tool_results"] == 0
    assert r["server_requests"] == 3 == r["client_wire_requests"]                  # 1 attempt + the SDK's 2 default retries
    assert r["server_requests_peer_closed"] == 3 and r["server_responses_delivered"] == 0
    assert r["exc_module"] in ("langchain_openai", "pydantic_ai")                  # raised by the model adapter layer, not by the agent loop


@pytest.mark.parametrize("fw", FWS)
def test_disabling_sdk_retries_gives_one_attempt_and_a_slow_first_attempt_can_recover_inside_one_model_call(fw):
    one = run(fw, "max_retries=0", OK, noop, delays=[0.6] * 4, model_kw=dict(timeout=0.15, max_retries=0), settle=0.2)
    assert one["server_requests"] == 1 and one["exc"] in ("OpenAITimeoutError", "ModelTimeoutError")
    rec = run(fw, "retry recovers", OK, noop, delays=[0.6, 0], model_kw=dict(timeout=0.15), settle=0.2)
    assert rec["exc"] is None and rec["server_requests"] == 2 and rec["final"] is not None


def test_pydantic_ai_request_timeout_is_one_clock_over_all_attempts():
    r = run("pydantic-ai", "request_timeout", OK, noop, delays=[0.6] * 4, model_kw=dict(timeout=0.15, request_timeout=0.3), settle=0.2)
    assert r["exc"] == "ModelRequestTimeout" and r["server_requests"] <= 2
    assert "ModelAPIError" in r["exc_mro"]                                         # a model-adapter error, not a RunTimedOut


def test_the_server_keeps_working_after_the_client_gave_up():
    r = run("pydantic-ai", "background work", OK, noop, delays=[0.7], model_kw=dict(timeout=0.15, max_retries=0), settle=0.2)
    log = r["server_log"][0]
    assert log["peer_closed"] is not None and log["responded"] is False
    assert log["work_done"] - log["arrived"] >= 600 and r["server_work_finished_after_return"] == 1


# ------------------------------------------------------------ B6-3.1.49 synchronous tool: the caller's error is not the tool stopping
SYNC = lambda p: [p.sync("work", 0.8, "end")]  # noqa: E731


@pytest.mark.parametrize("fw,opts", [("pydantic-ai", dict(tool_timeout=0.2)), ("pydantic-ai", dict(run_timeout=0.2)), ("langgraph", dict(step_timeout=0.2)),
                                     ("langgraph", dict(app_deadline=0.2)), ("pydantic-ai", dict(app_deadline=0.2))])
def test_a_timeout_returns_while_a_synchronous_tool_is_still_running_and_its_effect_lands_afterwards(fw, opts):
    r = run(fw, "sync timeout", ONE_ROUND, SYNC, settle=1.1, **opts)
    assert r["tool_running_at_return"] and r["effects_after_return"] == 1 and r["ledger"] == ["work"]


@pytest.mark.parametrize("fw,mech", [("pydantic-ai", "cancel"), ("pydantic-ai", "task"), ("langgraph", "task")])
def test_cancelling_a_run_does_not_stop_a_synchronous_tool_thread(fw, mech):
    r = run(fw, "sync cancel", ONE_ROUND, SYNC, mech=mech, when=e6.after_event_ms("tool_start", 80), settle=1.1)
    assert r["tool_running_at_return"] and r["effects_after_return"] == 1 and r["ack_latency_ms"] < 400
    assert r["exc"] in ("RunCancelled", "CancelledError")


def test_langgraph_drain_waits_for_the_running_tool_instead_of_interrupting_it():
    r = run("langgraph", "sync drain", ONE_ROUND, SYNC, mech="drain", when=e6.after_event_ms("tool_start", 80), settle=0.3)
    assert r["exc"] == "GraphDrained" and not r["tool_running_at_return"] and r["effects_after_return"] == 0
    assert r["caller_returned_ms"] >= r["stop_requested_ms"] + 500                  # the caller waited for the step boundary


def test_pydantic_ai_tool_timeout_is_a_model_visible_retry_while_the_thread_runs_on():
    r = run("pydantic-ai", "tool timeout retry prompt", ONE_ROUND, SYNC, tool_timeout=0.2, settle=1.1)
    assert r["exc"] is None and r["final"] == "Done." and r["server_requests"] == 2   # the model was asked again
    assert r["tool_running_at_return"] and r["state_tool_results"] == 1                # a retry prompt is the only "result"


# ------------------------------------------------------------ B6-3.1.50 asynchronous tool
@pytest.mark.parametrize("fw,opts", [("pydantic-ai", dict(tool_timeout=0.2)), ("langgraph", dict(step_timeout=0.2))])
def test_an_async_tool_is_cancelled_finally_runs_and_only_an_effect_before_the_await_survives(fw, opts):
    early = run(fw, "effect first", ONE_ROUND, lambda p: [p.asyn("work", 0.8, "start")], settle=1.0, **opts)
    late = run(fw, "effect last", ONE_ROUND, lambda p: [p.asyn("work", 0.8, "end")], settle=1.0, **opts)
    for r in (early, late):
        assert r["tool_cancelled"] == 1 and r["tool_finally"] == 1 and not r["tool_running_at_return"]
    assert early["effects"] == 1 and late["effects"] == 0


@pytest.mark.parametrize("fw,opts", [("pydantic-ai", dict(tool_timeout=0.2)), ("langgraph", dict(step_timeout=0.2))])
def test_a_tool_that_swallows_cancellation_holds_the_caller_until_it_finishes(fw, opts):
    r = run(fw, "swallow", ONE_ROUND, lambda p: [p.asyn("work", 0.9, "end", mode="swallow")], settle=0.5, **opts)
    assert r["tool_cancel_swallowed"] >= 1 and r["tool_ends"] == 1 and r["tool_cancelled"] == 0
    assert r["caller_returned_ms"] - r["first_request_ms"] >= 800                      # a 0.2 s timeout became ~0.9 s


@pytest.mark.parametrize("fw,opts", [("pydantic-ai", dict(tool_timeout=0.2)), ("langgraph", dict(step_timeout=0.2))])
def test_a_blocking_call_inside_an_async_tool_freezes_the_loop_so_the_timeout_fires_late(fw, opts):
    r = run(fw, "block", ONE_ROUND, lambda p: [p.asyn("work", 0.7, "end", mode="block")], settle=0.3, **opts)
    assert r["tool_ends"] == 1 and r["caller_returned_ms"] - r["first_request_ms"] >= 650


# ------------------------------------------------------------ B6-3.1.51 whole-run timeout
def test_pydantic_ai_native_run_timeout_carries_a_message_snapshot_and_an_external_deadline_does_not():
    native = run("pydantic-ai", "native", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], run_timeout=0.3, settle=0.3)
    external = run("pydantic-ai", "external", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], app_deadline=0.3, settle=0.3)
    assert native["exc"] == "RunTimedOut" and {"RunCancelled", "TimeoutError"} <= set(native["exc_mro"] + ["TimeoutError"])
    assert native["exc_carries_messages"] and not external["exc_carries_messages"]
    assert external["exc"] == "TimeoutError"


def test_langgraph_has_no_native_run_timeout_the_application_deadline_is_what_fires():
    r = run("langgraph", "app deadline", ONE_ROUND, lambda p: [p.asyn("work", 1.0)], app_deadline=0.3, settle=0.3)
    assert r["exc"] == "TimeoutError" and r["tool_cancelled"] == 1 and r["state_pending_calls"] == 1


@pytest.mark.parametrize("fw,opts", [("pydantic-ai", dict(run_timeout=0.3)), ("langgraph", dict(app_deadline=0.3))])
def test_a_deadline_interrupts_a_model_request_but_the_provider_keeps_working(fw, opts):
    r = run(fw, "model in flight", OK, noop, delays=[0.9], settle=0.3, **opts)
    assert r["exc"] in ("RunTimedOut", "TimeoutError") and r["server_requests"] == 1 and r["server_work_finished_after_return"] == 1


# ------------------------------------------------------------ B6-3.1.52 cancellation propagation
IN_FLIGHT = dict(delays=[0.9, 0], when=e6.after_event_ms("model_request", 80, n=1))


@pytest.mark.parametrize("fw,mech", [("pydantic-ai", "task"), ("pydantic-ai", "cancel")])
def test_pydantic_ai_cancel_tears_down_the_in_flight_http_request(fw, mech):
    r = run(fw, "B", ONE_ROUND, lambda p: [p.asyn("work", 0.5)], mech=mech, settle=0.2, **IN_FLIGHT)
    assert r["server_requests_peer_closed"] == 1 and r["server_responses_delivered"] == 0 and r["ack_latency_ms"] < 400


def test_the_phase1_langgraph_graph_cannot_interrupt_a_model_request_but_an_async_node_can():
    sync_node = run("langgraph", "B sync node", ONE_ROUND, lambda p: [p.asyn("work", 0.5)], mech="task", settle=0.2, **IN_FLIGHT)
    async_node = run("langgraph", "B async node", ONE_ROUND, lambda p: [p.asyn("work", 0.5)], mech="task", settle=0.2, async_model=True, **IN_FLIGHT)
    assert sync_node["ack_latency_ms"] < 400                                          # the caller is released at once ...
    assert sync_node["server_requests_peer_closed"] == 0 and sync_node["server_responses_delivered"] == 1      # ... the request is not
    assert any(k == "model_response" and t > sync_node["caller_returned_ms"] for t, k in sync_node["timeline"])
    assert async_node["server_requests_peer_closed"] == 1 and async_node["server_responses_delivered"] == 0


def test_langgraph_drain_does_not_interrupt_a_model_request_and_leaves_the_call_pending():
    r = run("langgraph", "B drain", ONE_ROUND, lambda p: [p.asyn("work", 0.5)], mech="drain", settle=0.2, **IN_FLIGHT)
    assert r["exc"] == "GraphDrained" and r["caller_returned_ms"] >= r["stop_requested_ms"] + 600
    assert r["tool_starts"] == 0 and r["state_pending_calls"] == 1


@pytest.mark.parametrize("fw,mech,asyncnode", [("pydantic-ai", "cancel", False), ("pydantic-ai", "task", False), ("langgraph", "task", True)])
def test_a_stop_before_the_request_reaches_the_wire_sends_nothing(fw, mech, asyncnode):
    r = run(fw, "A", ONE_ROUND, lambda p: [p.asyn("work", 0.5)], mech=mech, when=e6.after_event_ms("run_started", 120),
            model_kw=dict(presend=0.5), async_model=asyncnode, settle=0.8)
    assert r["server_requests"] == 0 and r["client_wire_requests"] == 0 and r["tool_starts"] == 0


def test_the_phase1_langgraph_sync_node_still_sends_the_request_after_the_caller_was_cancelled():
    r = run("langgraph", "A sync node", ONE_ROUND, lambda p: [p.asyn("work", 0.5)], mech="task", when=e6.after_event_ms("run_started", 120),
            model_kw=dict(presend=0.5), settle=0.8)
    assert r["exc"] == "CancelledError" and r["client_wire_requests"] == 1 and r["server_requests"] == 1       # a ghost request
    sent_at = next(t for t, k in r["timeline"] if k == "model_request")
    assert sent_at > r["caller_returned_ms"] and r["tool_starts"] == 0                                         # sent after the caller had its error


@pytest.mark.parametrize("fw,mech", [("pydantic-ai", "cancel"), ("langgraph", "task")])
def test_cancelling_an_async_tool_delivers_cancellederror_but_a_langgraph_drain_lets_it_finish(fw, mech):
    r = run(fw, "E", ONE_ROUND, lambda p: [p.asyn("work", 0.8)], mech=mech, when=e6.after_event_ms("tool_start", 80), settle=0.3)
    assert r["tool_cancelled"] == 1 and r["effects"] == 0
    d = run("langgraph", "E drain", ONE_ROUND, lambda p: [p.asyn("work", 0.8)], mech="drain", when=e6.after_event_ms("tool_start", 80), settle=0.3)
    assert d["tool_cancelled"] == 0 and d["effects"] == 1 and d["exc"] == "GraphDrained"


def test_a_requested_tool_that_has_not_started_does_not_start_after_a_cancel_but_a_drain_still_runs_it():
    gate = dict(gate=0.5, when=e6.after_event_ms("model_response", 50, n=1), settle=0.9)
    pa = run("pydantic-ai", "C cancel", ONE_ROUND, lambda p: [p.asyn("work", 0.3)], mech="cancel", **gate)
    lg = run("langgraph", "C task", ONE_ROUND, lambda p: [p.asyn("work", 0.3)], mech="task", **gate)
    drain = run("langgraph", "C drain", ONE_ROUND, lambda p: [p.asyn("work", 0.3)], mech="drain", **gate)
    assert pa["tool_starts"] == 0 == lg["tool_starts"] and pa["state_pending_calls"] == 1 == lg["state_pending_calls"]
    assert drain["tool_starts"] == 1 and drain["exc"] == "GraphDrained"                # the tools super-step was already scheduled


def test_a_synchronous_pydantic_ai_run_can_be_cancelled_from_another_thread_with_a_token():
    e6.RESULTS.pop("B6-3.1.52", None)
    e6._token_sync_api()
    rec = e6.RESULTS["B6-3.1.52"][-1]
    assert rec["exc"] == "RunCancelled" and rec["tool_cancelled"] == 1


# ------------------------------------------------------------ B6-3.1.53 token and usage limits
USAGES = [(60, 20), (100, 30)]                      # response 1 = 80 tokens, response 2 = 130, cumulative 210
QUICK = lambda p: [p.asyn("t", 0.05, "end")]  # noqa: E731
TOOL_SCRIPT = [call("t"), C("The answer is 4.")]


def tok(limits, **kw):
    return run("pydantic-ai", "tokens", TOOL_SCRIPT, QUICK, usages=USAGES, usage_limits=limits, settle=0.2, **kw)


def test_a_token_limit_crossed_by_the_tool_call_response_stops_before_the_tool_runs():
    r = tok(UsageLimits(total_tokens_limit=50))
    assert r["exc"] == "UsageLimitExceeded" and r["tool_starts"] == 0 and r["state_pending_calls"] == 1
    assert r["usage"]["total_tokens"] == 80 and r["usage"]["requests"] == 1 and r["usage"]["tool_calls"] == 0


@pytest.mark.parametrize("limits", [UsageLimits(total_tokens_limit=100), UsageLimits(input_tokens_limit=150), UsageLimits(output_tokens_limit=40),
                                    UsageLimits(per_request_input_tokens_limit=90)])
def test_a_limit_crossed_only_by_the_final_answer_raises_after_the_answer_arrived_and_the_tool_ran(limits):
    r = tok(limits)
    assert r["exc"] == "UsageLimitExceeded" and r["final"] is None and r["tool_ends"] == 1
    assert r["server_requests"] == 2 and r["server_responses_delivered"] == 2           # the provider answered and billed


def test_request_limit_does_not_stop_the_last_requested_tool_unlike_a_token_limit():
    assert tok(UsageLimits(request_limit=1))["tool_ends"] == 1


def test_counting_tokens_before_the_request_is_not_available_for_this_provider_model():
    r = tok(UsageLimits(count_tokens_before_request=True, per_request_input_tokens_limit=90))
    assert r["exc"] == "NotImplementedError" and r["server_requests"] == 0


def test_langgraph_enforces_no_token_limit_but_the_usage_is_readable():
    r = run("langgraph", "no limit", TOOL_SCRIPT, QUICK, usages=USAGES, settle=0.2)
    assert r["exc"] is None and r["usage"]["total_tokens"] == 210 and r["usage"]["per_message_total"] == [80, 130]


@pytest.mark.parametrize("fw,name", [("langgraph", "OpenAIInvalidRequestError"), ("pydantic-ai", "ModelContextWindowExceededError")])
def test_a_context_window_rejection_is_one_attempt_not_retried_and_keeps_earlier_usage(fw, name):
    r = run(fw, "400", TOOL_SCRIPT, QUICK, usages=USAGES, statuses=[200, 400], settle=0.2)
    assert r["exc"] == name and r["server_requests"] == 2 and r["server_log"][1]["status"] == 400
    assert (r["usage"]["total_tokens"] == 80)


@pytest.mark.parametrize("fw", FWS)
def test_http_retries_are_invisible_to_usage_so_provider_consumption_can_exceed_the_application_count(fw):
    r = run(fw, "500 then ok", TOOL_SCRIPT, QUICK, usages=USAGES, statuses=[500], settle=0.2)
    assert r["exc"] is None and r["server_requests"] == 3 and r["server_log"][0]["status"] == 500
    assert r["usage"]["total_tokens"] == 210
    if fw == "pydantic-ai":
        assert r["usage"]["requests"] == 2                                                # 3 HTTP attempts, 2 counted


@pytest.mark.parametrize("fw", FWS)
def test_max_tokens_is_a_provider_side_per_request_cap_sent_on_the_wire(fw):
    r = run(fw, "max_tokens", TOOL_SCRIPT, QUICK, usages=USAGES, model_kw=dict(max_tokens=25), settle=0.2)
    assert [x["max_completion_tokens"] for x in r["server_log"]] == [25, 25] and r["exc"] is None


# ------------------------------------------------------------ B6-3.1.54 pending tool call at a limit
def test_a_pending_call_in_retained_state_cannot_tell_started_from_never_started():
    e6.RESULTS.pop("B6-3.1.54", None)
    e6.pending()
    summary = e6.RESULTS["B6-3.1.54-summary"]
    assert summary["pending_call_in_state_but_started"] >= 1 and summary["pending_call_in_state_and_never_started"] >= 1
    assert summary["state_alone_can_distinguish"] is False
    by = {(r["framework"], r["case"]): r for r in e6.RESULTS["B6-3.1.54"]}
    assert by[("langgraph", "recursion_limit=3 (odd)")]["state_pending_calls"] == 1 and by[("langgraph", "recursion_limit=4 (even)")]["state_pending_calls"] == 0
    assert by[("pydantic-ai", "request_limit=2")]["tool_ends"] == 2 and by[("pydantic-ai", "tool_calls_limit=1")]["tool_ends"] == 1


# ------------------------------------------------------------ B6-3.1.55 cancellation after a business effect
@pytest.fixture(scope="module")
def effect_rows():
    e6.RESULTS.pop("B6-3.1.55", None)
    e6.effect_after_cancel()
    return e6.RESULTS["B6-3.1.55"]


def _row(rows, text, fw):
    hits = [r for r in rows if text in r["case"] and r["framework"] == fw and "IMMEDIATE" not in r["case"]]
    assert len(hits) == 1, (text, fw, len(hits))
    return hits[0]


@pytest.mark.parametrize("fw", FWS)
def test_an_effect_committed_before_the_cancel_remains_and_a_keyed_retry_replays(effect_rows, fw):
    r = _row(effect_rows, "async: effect committed", fw)
    assert r["ledger_at_end"] == 1 and r["key_recorded"] and r["keyed_retry_added_orders"] == 0 and r["unkeyed_retry_added_orders"] == 1
    assert r["state_tool_results"] == 0 and r["state_pending_calls"] == 1                  # the framework recorded no result for the completed operation


@pytest.mark.parametrize("fw", FWS)
def test_cancelling_before_the_commit_leaves_no_effect_and_a_keyed_retry_creates_exactly_one(effect_rows, fw):
    r = _row(effect_rows, "async: cancelled before", fw)
    assert r["ledger_at_end"] == 0 and not r["key_recorded"] and r["keyed_retry_added_orders"] == 1


@pytest.mark.parametrize("fw", FWS)
def test_a_synchronous_tool_cancelled_before_its_commit_still_commits_after_the_caller_returned(effect_rows, fw):
    r = _row(effect_rows, "sync: cancelled during", fw)
    assert r["tool_running_at_return"] and r["effects_after_return"] == 1 and r["ledger_at_end"] == 1 and r["keyed_retry_added_orders"] == 0


@pytest.mark.parametrize("fw", FWS)
def test_an_immediate_keyed_retry_races_the_cancelled_original_unless_the_guard_is_atomic(effect_rows, fw):
    rows = [r for r in effect_rows if "IMMEDIATE" in r["case"] and r["framework"] == fw]
    plain, atomic = [r for r in rows if not r["atomic_guard"]][0], [r for r in rows if r["atomic_guard"]][0]
    assert plain["ledger_at_end"] == 2 and plain["replays"] == 0
    assert atomic["ledger_at_end"] == 1 and atomic["replays"] == 1


# ------------------------------------------------------------ B6-3.1.56 partial state after termination
@pytest.fixture(scope="module")
def plain_rows():
    e6.RESULTS.pop("B6-3.1.56-plain", None)
    e6.RESULTS.pop("B6-3.1.56", None)
    e6.state_after_termination()
    return {(r["framework"], r["termination"]): r for r in e6.RESULTS["B6-3.1.56-plain"]}


def test_langgraph_exceptions_carry_no_state_on_the_plain_call_path(plain_rows):
    for how in ("loop limit", "run timeout", "cancellation", "model error"):
        assert plain_rows[("langgraph", how)]["exception_carries_messages"] is None


def test_only_pydantic_ai_timeout_and_cancel_exceptions_carry_a_message_snapshot(plain_rows):
    assert plain_rows[("pydantic-ai", "run timeout")]["exception_carries_messages"] >= 2
    assert plain_rows[("pydantic-ai", "cancellation")]["exception_carries_messages"] >= 2
    for how in ("loop limit", "usage limit", "model error"):
        assert plain_rows[("pydantic-ai", how)]["exception_carries_messages"] is None


def test_the_streamed_paths_keep_state_after_every_termination(plain_rows):
    rows = {(r["framework"], r["case"]): r for r in e6.RESULTS["B6-3.1.56"]}
    assert rows[("langgraph", "loop limit via astream")]["state_pending_calls"] == 1
    assert rows[("pydantic-ai", "usage limit via iter")]["usage"]["total_tokens"] == 80
    assert rows[("pydantic-ai", "run timeout via iter (run_timeout)")]["state_pending_calls"] == 1


def test_a_snapshot_taken_at_a_timeout_looks_the_same_whether_or_not_the_tool_will_still_commit():
    snaps = {}
    for kind in ("sync", "async"):
        tools = (lambda p: [p.sync("work", 0.8, "end")]) if kind == "sync" else (lambda p: [p.asyn("work", 0.8, "end")])
        r = run("pydantic-ai", kind, ONE_ROUND, tools, run_timeout=0.3, settle=1.0)
        snaps[kind] = (r["state_history"], r["state_pending_calls"], r["effects"])
    assert snaps["sync"][:2] == snaps["async"][:2]                                         # identical retained state ...
    assert (snaps["sync"][2], snaps["async"][2]) == (1, 0)                                 # ... different business outcome


# ------------------------------------------------------------ B6-3.1.57 timeout x retry
def test_a_pydantic_ai_tool_timeout_can_start_a_second_execution_while_the_first_still_runs():
    r = run("pydantic-ai", "retry overlap", [call("work"), call("work"), C("Done.")], lambda p: [p.sync("work", 0.8, "end")], tool_timeout=0.2, settle=1.4)
    starts = [t for t, k in r["timeline"] if k == "tool_start"]
    ends = [t for t, k in r["timeline"] if k == "tool_end"]
    assert len(starts) == 2 and starts[1] < min(ends) and r["effects"] == 2 and r["effects_after_return"] == 2


def test_a_langgraph_step_timeout_is_not_retried_by_retry_policy():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=2, retry_on=Exception)
    r = run("langgraph", "step timeout + policy", ONE_ROUND, lambda p: [p.sync("work", 0.8, "end")], step_timeout=0.2, retry_policy=rp,
            handle_tool_errors=False, settle=1.0)
    assert r["exc"] == "TimeoutError" and r["tool_starts"] == 1


def test_a_tool_level_timeout_error_is_retried_by_retry_policy_while_the_first_worker_runs_on():
    from langgraph.types import RetryPolicy
    rp = RetryPolicy(initial_interval=0.001, backoff_factor=1.0, jitter=False, max_attempts=2, retry_on=TimeoutError)

    def guarded(p):
        work = p.sync("work", 0.8, "end")

        def guarded_tool() -> str:
            """Run work, giving up after 0.2 s."""
            t = threading.Thread(target=work, daemon=True)
            t.start()
            t.join(0.2)
            if t.is_alive():
                raise TimeoutError("timed out")
            return "done"
        return [guarded_tool]
    r = run("langgraph", "tool-level timeout + policy", [call("guarded_tool"), C("Done.")], guarded, retry_policy=rp, handle_tool_errors=False, settle=1.4)
    assert r["tool_starts"] == 2 and r["effects"] == 2 and r["exc"] == "TimeoutError"


# ------------------------------------------------------------ B6-3.1.58 cumulative deadline
THREE = [call("work")] * 3 + [C("All three done.")]
WORK = lambda p: [p.asyn("work", 0.2, "end")]  # noqa: E731


def test_per_step_timeouts_bound_each_step_but_not_the_total_run():
    lg = run("langgraph", "step only", THREE, WORK, delays=[0.2] * 4, step_timeout=0.45, settle=0.2)
    pa = run("pydantic-ai", "step only", THREE, WORK, delays=[0.2] * 4, model_kw=dict(timeout=0.45, request_timeout=0.45), tool_timeout=0.45, settle=0.2)
    for r in (lg, pa):
        assert r["exc"] is None and r["tool_ends"] == 3 and r["run_ms"] >= 1200          # every step inside its own limit, total far over 0.7 s


@pytest.mark.parametrize("fw,opts", [("pydantic-ai", dict(run_timeout=0.7)), ("pydantic-ai", dict(app_deadline=0.7)), ("langgraph", dict(app_deadline=0.7))])
def test_only_a_whole_run_deadline_bounds_the_total(fw, opts):
    r = run(fw, "deadline", THREE, WORK, delays=[0.2] * 4, settle=0.5, **opts)
    assert r["exc"] in ("RunTimedOut", "TimeoutError") and r["run_ms"] < 1000 and r["tool_ends"] < 3


# ------------------------------------------------------------ the server fixture itself
def test_the_slow_server_logs_client_disconnects_and_completed_work():
    import httpx
    with SlowServer([C("hi")], delays=[0.5]) as srv:
        t0 = time.perf_counter()
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(srv.base_url + "/chat/completions", json={"messages": []}, timeout=0.1)
        log = srv.settle(2.0)
    assert len(log) == 1 and log[0]["peer_closed_at"] is not None and log[0]["responded"] is False
    assert log[0]["work_done_at"] - t0 >= 0.45
