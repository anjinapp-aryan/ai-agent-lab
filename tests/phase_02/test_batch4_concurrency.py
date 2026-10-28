"""Phase 2 Batch 4 regression tests: concurrency, ordering, races, partial failure, cancellation, isolation.

Scripted model + local fixtures only (no network). Durations are scaled by S=0.3 to keep the suite fast; assertions use
ratios of the scaled duration, never absolute wall-clock numbers. Each assertion pins an OBSERVED behavior recorded in
phases/phase-02/experiments.md (Batch 4).
"""
import pytest

import exp_batch4 as e4

S = 0.3


@pytest.fixture(scope="module", autouse=True)
def scaled():
    old = e4.S
    e4.S = S
    yield
    e4.S = old


def rows_of(fn, key, *args):
    e4.RESULTS.clear()
    fn(*args)
    return list(e4.RESULTS[key])


def one(rows, **kw):
    hits = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    assert len(hits) == 1, f"{kw} matched {len(hits)}"
    return hits[0]


DUR_MS = 1000 * S


# ------------------------------------------------------------ 4.1-4.4, 4.10 independent tools
@pytest.fixture(scope="module")
def ind():
    return rows_of(e4.independent, "4.1-4.4")


@pytest.mark.parametrize("fw,runner,kind,threads", [("langgraph", "invoke", "sync", 2), ("langgraph", "ainvoke", "sync", 2), ("langgraph", "ainvoke", "async", 1),
                                                    ("pydantic-ai", "run_sync", "sync", 2), ("pydantic-ai", "run_sync", "async", 1),
                                                    ("pydantic-ai", "run", "sync", 2), ("pydantic-ai", "run", "async", 1)])
def test_batched_independent_tools_overlap(ind, fw, runner, kind, threads):
    r = one(ind, framework=fw, runner=runner, tool_kind=kind, case="A+B batched")
    assert r["overlap_ms"] >= 0.8 * DUR_MS and r["makespan_ms"] < 1.5 * DUR_MS and r["threads"] == threads


def test_langgraph_invoke_cannot_run_async_only_tools(ind):
    r = one(ind, framework="langgraph", runner="invoke", tool_kind="async", case="A+B batched")
    assert r["events"] == [] and all("does not support sync invocation" in x for x in r["result_order"])


@pytest.mark.parametrize("case", ["A+B batched mode=sequential", "A+B Tool(sequential=True)"])
def test_pydantic_ai_sequential_controls_serialize(ind, case):
    for r in [x for x in ind if x["case"] == case]:
        assert r["overlap_ms"] == 0 and r["makespan_ms"] >= 1.8 * DUR_MS and r["max_concurrency"] == 1


def test_model_unbatched_calls_are_sequential_regardless_of_framework(ind):
    r = one(ind, case="A then B in separate turns")
    assert r["overlap_ms"] == 0 and r["makespan_ms"] >= 1.8 * DUR_MS


# ------------------------------------------------------------ 4.5 four orders
@pytest.fixture(scope="module")
def orders():
    return rows_of(e4.order, "4.5")


def test_call_start_completion_result_orders(orders):
    for r in orders:
        if r["framework"] == "langgraph" and r["runner"] == "invoke" and r["tool_kind"] == "async":
            continue                                                  # async-only tools do not run under invoke
        assert r["start_order"] == ["tool_a", "tool_b", "tool_c"], r      # start follows call order
        assert r["completion_order"] == ["tool_b", "tool_c", "tool_a"], r  # completion follows duration
        assert r["result_order"] == ["tool_a", "tool_b", "tool_c"], r     # model sees call order, not completion order


# ------------------------------------------------------------ 4.6-4.8 races
@pytest.fixture(scope="module")
def counter_rows():
    return rows_of(e4.counter, "4.6-4.7")


def test_unsafe_counter_loses_updates_with_and_without_framework(counter_rows):
    for r in [x for x in counter_rows if "unsafe" in x["case"] and "sequential" not in x["case"]]:
        assert r["lost"] > 0, r


def test_threading_lock_fixes_count(counter_rows):
    for r in [x for x in counter_rows if x["case"] == "Lock"]:
        assert r["actual"] == 100


def test_sequential_execution_hides_the_race_without_fixing_the_tool(counter_rows):
    for case in ("unsafe + sequential mode", "unsafe + Tool(sequential=True)"):
        r = one(counter_rows, framework="pydantic-ai", case=case)
        assert r["actual"] == 100 and r["max_concurrency"] == 1


@pytest.fixture(scope="module")
def async_rows():
    return rows_of(e4.asyncrace, "4.8")


def test_async_is_not_race_free(async_rows):
    for r in async_rows:
        if r["case"] == "await_between":
            assert r["lost"] > 0, r
        else:                                       # asyncio.Lock, or no suspension point between read and write
            assert r["actual"] == 100, r


# ------------------------------------------------------------ 4.9 concurrent duplicate side effect
@pytest.fixture(scope="module")
def dup():
    return rows_of(e4.duplicate, "4.9")


@pytest.mark.parametrize("fw", e4.FWS)
def test_concurrent_duplicates_need_an_atomic_guard(dup, fw):
    assert one(dup, framework=fw, case="no key")["orders_created"] == 2
    assert one(dup, framework=fw, case="non-atomic key guard")["orders_created"] == 2
    atomic = one(dup, framework=fw, case="atomic guard (lock around check+effect+commit)")
    assert atomic["orders_created"] == 1 and atomic["replays"] == 1


# ------------------------------------------------------------ 4.11 concurrency limits
@pytest.fixture(scope="module")
def lim():
    return rows_of(e4.limits, "4.11")


@pytest.mark.parametrize("n", [1, 2, 4])
def test_langgraph_invoke_honors_config_max_concurrency(lim, n):
    assert one(lim, framework="langgraph", runner="invoke", case=f"config max_concurrency={n}")["max_concurrency"] == n


def test_langgraph_ainvoke_ignores_config_max_concurrency(lim):
    for kind in ("sync", "async"):
        assert one(lim, framework="langgraph", runner="ainvoke", tool_kind=kind)["max_concurrency"] == 10


def test_pydantic_ai_has_no_per_run_tool_limit_only_modes(lim):
    assert one(lim, framework="pydantic-ai", case="default", runner="run_sync")["max_concurrency"] == 10
    assert one(lim, framework="pydantic-ai", case="mode=sequential (limit of 1)")["max_concurrency"] == 1
    assert one(lim, framework="pydantic-ai", case="Agent(max_concurrency=2) (limits runs)")["max_concurrency"] == 10


# ------------------------------------------------------------ 4.12-4.13 partial failure
@pytest.fixture(scope="module")
def part():
    return rows_of(e4.partial, "4.12-4.13")


def _p(part, fw, runner, label):
    return one(part, framework=fw, runner=runner, case=label)


def test_failure_does_not_cancel_sync_siblings_and_nothing_rolls_back(part):
    for fw, runner, label in (("langgraph", "invoke", "RuntimeError (default handler)"), ("langgraph", "invoke", "RuntimeError (handle=False)"),
                              ("pydantic-ai", "run_sync", "RuntimeError")):
        r = _p(part, fw, runner, label)
        assert r["effects"] == ["tool_a", "tool_c"] and r["statuses"]["tool_b"] == "failed", r


def test_pydantic_ai_sync_returns_before_siblings_finish(part):
    r = _p(part, "pydantic-ai", "run_sync", "RuntimeError")
    assert r["wall_ms"] < 0.5 * 0.6 * DUR_MS and "RuntimeError" in r["error"] and r["model_saw_n"] == 0


def test_langgraph_sync_waits_for_siblings_before_raising(part):
    r = _p(part, "langgraph", "invoke", "RuntimeError (handle=False)")
    assert r["wall_ms"] >= 0.8 * 0.6 * DUR_MS and "RuntimeError" in r["error"]


@pytest.mark.parametrize("fw,runner,label", [("langgraph", "ainvoke", "RuntimeError (handle=False)"), ("pydantic-ai", "run", "RuntimeError")])
def test_escaping_failure_cancels_async_siblings(part, fw, runner, label):
    r = _p(part, fw, runner, label)
    assert r["statuses"]["tool_a"] == r["statuses"]["tool_c"] == "cancelled" and r["effects"] == []


@pytest.mark.parametrize("fw,runner,label", [("langgraph", "invoke", "RuntimeError (default handler)"), ("langgraph", "ainvoke", "RuntimeError (default handler)"),
                                             ("pydantic-ai", "run_sync", "ModelRetry"), ("pydantic-ai", "run", "ModelRetry")])
def test_model_visible_failure_keeps_partial_success_visible(part, fw, runner, label):
    r = _p(part, fw, runner, label)
    assert r["model_saw_n"] == 3 and r["recovered"] and r["effects"] == ["tool_a", "tool_c"]


# ------------------------------------------------------------ 4.14 dependency
@pytest.fixture(scope="module")
def dep():
    return rows_of(e4.dependent, "4.14")


@pytest.mark.parametrize("fw", e4.FWS)
def test_frameworks_do_not_enforce_dependencies(dep, fw):
    bad = one(dep, framework=fw, case="dependent B batched with guessed arg")
    good = one(dep, framework=fw, case="dependent B after A (separate turns)")
    assert bad["overlap_ms"] > 0 and "unknown order order-1" in bad["result_order"]
    assert good["overlap_ms"] == 0 and "SHIPPED" in good["result_order"]


# ------------------------------------------------------------ 4.15 identity isolation
@pytest.fixture(scope="module")
def iso():
    return rows_of(e4.authz, "4.15", 5)


@pytest.mark.parametrize("fw", e4.FWS)
def test_shared_identity_designs_leak_request_scoped_ones_do_not(iso, fw):
    for design in ("global current_user", "shared tool set closed over the FIRST request's identity"):
        r = one(iso, framework=fw, design=design)
        assert r["identity_leaks"] > 0 and r["unauthorized_allows"] > 0, r
    for design in ("tools built per request (closure)", "trusted per-call context (runtime.context / ctx.deps)"):
        r = one(iso, framework=fw, design=design)
        assert r["identity_leaks"] == 0 and r["unauthorized_allows"] == 0, r


# ------------------------------------------------------------ 4.16 concurrent output
@pytest.fixture(scope="module")
def out():
    return rows_of(e4.output, "4.16")


@pytest.mark.parametrize("fw", e4.FWS)
def test_results_reach_model_in_call_order_without_separation(out, fw):
    r = one(out, framework=fw)
    assert r["completion_order"] == ["note_b", "note_a"]
    assert r["model_visible"][0].startswith("NOTE A")                  # slow normal result first (call order)
    assert r["model_visible"][1].startswith("Order status")            # fast malicious result second, unlabeled


# ------------------------------------------------------------ 4.17 cancellation
@pytest.fixture(scope="module")
def canc():
    return rows_of(e4.cancel, "4.17")


@pytest.mark.parametrize("fw", e4.FWS)
def test_cancel_stops_async_tools_but_sync_threads_commit_after_cancel(canc, fw):
    a = one(canc, framework=fw, tool_kind="async")
    s = one(canc, framework=fw, tool_kind="sync")
    assert a["run_state"] == "cancelled" and a["side_effects"] == [] and all(v[0] == "cancelled" for v in a["tool_events"].values())
    assert s["run_state"] == "cancelled" and s["side_effects"] == ["long_a", "long_b"]
    assert all(v[1] > s["cancel_requested_ms"] for v in s["tool_events"].values())       # effects committed after the cancel
    assert a["cleanup_ran"] == s["cleanup_ran"] == ["long_a", "long_b"]


# ------------------------------------------------------------ 4.18 concurrency + idempotency + authorization
@pytest.fixture(scope="module")
def comb():
    return rows_of(e4.combined, "4.18", 2)


@pytest.mark.parametrize("fw", e4.FWS)
def test_two_concurrent_authorized_requests(comb, fw):
    na = one(comb, framework=fw, guard="non-atomic guard")
    at = one(comb, framework=fw, guard="atomic guard")
    assert na["orders_per_trial"] == [2, 2] and at["orders_per_trial"] == [1, 1]
    assert na["authz_checks"] == at["authz_checks"] == 4 and na["authz_denies"] == 0


def test_run_serialization_hides_the_race(comb):
    free = one(comb, guard="non-atomic guard, shared Agent(max_concurrency=None), 2 runs in one event loop")
    serial = one(comb, guard="non-atomic guard, shared Agent(max_concurrency=1), 2 runs in one event loop")
    assert free["orders_per_trial"] == [2, 2] and serial["orders_per_trial"] == [1, 1]
