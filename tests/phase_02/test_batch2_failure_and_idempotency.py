"""Phase 2 Batch 2 regression tests: exceptions, retry mechanisms, timeouts, duplicates, idempotency.

Scripted model + simulated tools only (no API calls, no real side effects). Each assertion pins an OBSERVED behavior
recorded in phases/phase-02/experiments.md. Timeouts use short durations (DUR/LIMIT patched) to keep the suite fast.
"""
import pytest

import exp_batch2 as e2


def _run(fn, key):
    e2.RESULTS.clear()
    fn()
    return list(e2.RESULTS[key])


def pick(rows, **kw):
    hits = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    assert len(hits) == 1, f"{kw} matched {len(hits)} rows"
    return hits[0]


# ---------------------------------------------------------------- B2.1 exceptions
@pytest.fixture(scope="module")
def exc_rows():
    return _run(e2.exceptions, "b21_exceptions")


@pytest.mark.parametrize("kind", e2.KINDS)
def test_langgraph_default_feeds_every_exception_class_to_the_model(exc_rows, kind):
    r = pick(exc_rows, framework="langgraph", config="default", kind=kind)
    assert r["recovered"] and r["model_calls"] == 2


@pytest.mark.parametrize("kind", e2.KINDS)
def test_langgraph_handle_false_every_exception_terminates_run(exc_rows, kind):
    assert not pick(exc_rows, framework="langgraph", config="handle=False", kind=kind)["recovered"]


def test_langgraph_handle_tuple_handles_only_listed_class(exc_rows):
    rec = {r["kind"]: r["recovered"] for r in exc_rows if r["config"] == "handle=(ValueError,)"}
    assert rec == {"value": True, "runtime": False, "timeout": False, "validation": False,
                   "tool_exception": False, "model_retry": False}


def test_pydantic_ai_only_validation_error_and_modelretry_are_model_visible(exc_rows):
    rec = {r["kind"]: r["recovered"] for r in exc_rows if r["framework"] == "pydantic-ai"}
    assert rec == {"value": False, "runtime": False, "timeout": False, "validation": True,
                   "tool_exception": False, "model_retry": True}


# ---------------------------------------------------------------- B2.1 retry mechanisms
@pytest.fixture(scope="module")
def retry_rows():
    return _run(e2.retry, "b21_retry")


def test_M_pydantic_ai_modelretry_budget(retry_rows):
    low = pick(retry_rows, mechanism="M", config="ModelRetry x2, retries=1")
    hi = pick(retry_rows, mechanism="M", config="ModelRetry x2, retries=3")
    assert not low["recovered"] and "UnexpectedModelBehavior" in low["run_error"] and low["tool_runs"] == 2
    assert hi["recovered"] and hi["tool_runs"] == 3 and hi["model_calls"] == 4


def test_M_langgraph_model_re_issues_call(retry_rows):
    r = pick(retry_rows, mechanism="M", framework="langgraph")
    assert r["recovered"] and r["tool_runs"] == 3 and r["model_calls"] == 4


def test_F_langgraph_retry_policy_inert_under_default_error_handler(retry_rows):
    r = pick(retry_rows, config="RetryPolicy(retry_on=Exception) + DEFAULT handler")
    assert r["tool_runs"] == 1 and r["model_calls"] == 2


def test_F_langgraph_retry_policy_reruns_without_model_call(retry_rows):
    r = pick(retry_rows, config="RetryPolicy(retry_on=Exception) + handle=False")
    assert r["recovered"] and r["tool_runs"] == 3 and r["model_calls"] == 2


def test_F_langgraph_default_retry_on_skips_runtimeerror(retry_rows):
    r = pick(retry_rows, config="RetryPolicy(default retry_on) + handle=False")
    assert r["tool_runs"] == 1 and "RuntimeError" in r["run_error"]


def test_F_pydantic_ai_has_no_automatic_retry_for_plain_exceptions(retry_rows):
    r = pick(retry_rows, mechanism="F?")
    assert r["tool_runs"] == 1 and "RuntimeError" in r["run_error"]


@pytest.mark.parametrize("fw", e2.FWS)
def test_T_tool_internal_retry_invisible_to_framework(retry_rows, fw):
    r = pick(retry_rows, mechanism="T", framework=fw)
    assert r["recovered"] and r["tool_runs"] == 3 and r["model_calls"] == 2


# ---------------------------------------------------------------- B2.2 timeouts
@pytest.fixture(scope="module")
def to_rows():
    old = (e2.DUR, e2.LIMIT)
    e2.DUR, e2.LIMIT = 1.2, 0.4
    try:
        rows = _run(e2.timeout, "b22_timeout")
    finally:
        e2.DUR, e2.LIMIT = old
    return rows


def test_langgraph_sync_step_timeout_fires_late(to_rows):
    r = pick(to_rows, framework="langgraph", mechanism="step_timeout", mode="sync")
    assert r["run_returned_s"] >= 1.1 and "TimeoutError" in r["outcome"]


def test_langgraph_ainvoke_sync_tool_returns_on_time_but_tool_keeps_running(to_rows):
    r = pick(to_rows, framework="langgraph", mode="ainv+syncTool")
    assert r["run_returned_s"] < 0.9 and r["tool_end_s"] > r["run_returned_s"] + 0.3 and not r["model_recovered"]


def test_langgraph_ainvoke_async_tool_is_cancelled(to_rows):
    r = pick(to_rows, framework="langgraph", mode="ainv+asyncTool")
    assert r["run_returned_s"] < 0.9 and r["tool_cancelled_s"] is not None and r["tool_end_s"] is None


def test_pydantic_ai_tool_timeout_sync_model_recovers_worker_continues(to_rows):
    r = pick(to_rows, framework="pydantic-ai", mechanism="Tool(timeout=1)", mode="sync")
    assert r["run_returned_s"] < 0.9 and r["model_recovered"] and r["tool_end_s"] > r["run_returned_s"] + 0.3


def test_pydantic_ai_run_timeout_aborts_run_worker_continues(to_rows):
    r = pick(to_rows, framework="pydantic-ai", mechanism="run_sync(timeout=1)")
    assert "RunTimedOut" in r["outcome"] and r["tool_end_s"] > r["run_returned_s"] + 0.3


def test_pydantic_ai_async_tool_cancelled_for_both_mechanisms(to_rows):
    for mech in ("Tool(timeout=1)", "run(timeout=1)"):
        r = pick(to_rows, framework="pydantic-ai", mechanism=mech, mode="async")
        assert r["tool_cancelled_s"] is not None and r["tool_end_s"] is None


def test_pydantic_ai_agent_level_tool_timeout_behaves_like_per_tool(to_rows):
    r = pick(to_rows, framework="pydantic-ai", mechanism="Agent(tool_timeout=1)")
    assert r["model_recovered"] and r["run_returned_s"] < 0.9


# ---------------------------------------------------------------- B2.3 duplicates
@pytest.fixture(scope="module")
def dup_rows():
    e2.RESULTS.clear()
    e2.duplicates()
    return list(e2.RESULTS["b23_dup"]), list(e2.RESULTS["b23_after_effect"])


@pytest.mark.parametrize("fw", e2.FWS)
def test_no_deduplication_by_args_or_call_id(dup_rows, fw):
    dup, _ = dup_rows
    for sc in ("A sequential same args", "A2 same batch same args"):
        r = pick(dup, framework=fw, scenario=sc)
        assert r["executions"] == 2 and len(set(r["call_ids"])) == 2


def test_failure_after_side_effect_model_retry_duplicates_in_both(dup_rows):
    _, ae = dup_rows
    assert pick(ae, scenario="C1 (M) response lost, model re-calls")["ledger"] == 2
    assert pick(ae, scenario="C3 (M) ModelRetry after effect, model re-calls")["ledger"] == 2


def test_failure_after_side_effect_langgraph_retrypolicy_duplicates_without_model(dup_rows):
    r = pick(dup_rows[1], scenario="C2 (F) RetryPolicy, no model call between")
    assert r["ledger"] == 2 and r["model_calls"] == 2


def test_pydantic_ai_plain_exception_leaves_one_effect_and_dead_run_caller_retry_duplicates(dup_rows):
    ae = dup_rows[1]
    first = pick(ae, scenario="C4 plain exception escapes (run dies)")
    assert first["ledger"] == 1 and "RuntimeError" in first["run_error"]
    assert pick(ae, scenario="C4b2 ...second run() by the caller")["ledger"] == 2


def test_langgraph_retrypolicy_reruns_already_succeeded_calls_in_the_batch(dup_rows):
    ae = dup_rows[1]
    assert pick(ae, scenario="H10 batch [ok effect, failing call] + RetryPolicy")["record_action_executions"] == 2
    assert pick(ae, scenario="H10b same batch, no retry policy")["record_action_executions"] == 1


# ---------------------------------------------------------------- B2.4 idempotency
@pytest.fixture(scope="module")
def idem_rows():
    return _run(e2.idempotency, "b24_idempotency")


@pytest.mark.parametrize("fw", e2.FWS)
def test_idempotency_matrix(idem_rows, fw):
    P = lambda s: pick(idem_rows, framework=fw, scenario=s)  # noqa: E731
    assert P("first request (key k1)")["orders_created"] == 1
    assert (P("exact duplicate (k1, same args)")["orders_created"], P("exact duplicate (k1, same args)")["replays"]) == (1, 1)
    assert P("different key, same args")["orders_created"] == 2
    assert P("lost response, SAME key on retry (model re-calls)")["orders_created"] == 1
    assert P("lost response, model CHANGES key on retry")["orders_created"] == 2
    assert P("lost response, key from context (system), model re-calls")["orders_created"] == 1


def test_same_key_different_args_conflict_surfaces_per_framework_error_policy(idem_rows):
    lg = pick(idem_rows, framework="langgraph", scenario="same key, different args")
    pa = pick(idem_rows, framework="pydantic-ai", scenario="same key, different args")
    assert lg["orders_created"] == pa["orders_created"] == 1
    assert "ConflictError" in lg["tool_feedback"] and lg["run_error"] == ""      # fed back to the model
    assert "ConflictError" in pa["run_error"]                                     # escapes the run


def test_retrypolicy_with_keyed_tool_does_not_duplicate(idem_rows):
    assert pick(idem_rows, scenario="lost response, (F) RetryPolicy + keyed")["orders_created"] == 1


def test_non_atomic_guard_loses_race_in_parallel_batch_but_not_when_sequential(idem_rows):
    for fw in e2.FWS:
        assert pick(idem_rows, framework=fw, scenario="RACE two identical keyed calls in one batch (parallel)")["orders_created"] == 2
    seq = pick(idem_rows, framework="pydantic-ai", scenario="RACE same batch but Tool(sequential=True)")
    assert seq["orders_created"] == 1 and seq["replays"] == 1
