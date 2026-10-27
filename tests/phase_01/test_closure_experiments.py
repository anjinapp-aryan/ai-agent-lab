"""Deterministic regression tests for the Phase 1 closure experiments (scripted model, short delays).

They pin FRAMEWORK scheduling/timeout behavior; whether a real model batches tool calls is a separate,
live measurement (phase-01/experiments.py parallel-live).
"""
import pytest
from pydantic_ai import Tool

import experiments as ex
import lg_agent
import pa_agent
from conftest import lg_model, pa_model

D = 0.4
BATCH = [("call", [("multiply", {"a": 10, "b": 5}), ("add", {"a": 20, "b": 7})]), ("text", "50 and 27")]
SLOW_CALL = [("call", [("slow_op", {"a": 1, "b": 2})]), ("text", "done")]


def _overlap(run, model, tools):
    ex.LOG.clear()
    run("q", model, tools=tools)
    return ex.overlap(ex.LOG, D)


def test_langgraph_runs_batched_tool_calls_in_parallel():
    assert _overlap(lg_agent.run, lg_model(BATCH), [ex.slow_multiply(D), ex.slow_add(D)])["verdict"] == "PARALLEL"


def test_pydantic_ai_runs_batched_tool_calls_in_parallel():
    assert _overlap(pa_agent.run, pa_model(BATCH), [ex.slow_multiply(D), ex.slow_add(D)])["verdict"] == "PARALLEL"


def test_pydantic_ai_sequential_flag_serializes():
    tools = [Tool(ex.slow_multiply(D), sequential=True), Tool(ex.slow_add(D), sequential=True)]
    assert _overlap(pa_agent.run, pa_model(BATCH), tools)["verdict"] == "SEQUENTIAL"


def test_pydantic_ai_tool_timeout_becomes_model_visible_retry():
    r = pa_agent.run("q", pa_model(SLOW_CALL), tools=[Tool(ex.hang_tool(2.0), timeout=0.3)])
    assert r.error is None and r.final_text == "done" and "Timed out" in r.tool_calls[0].error


def test_pydantic_ai_run_timeout_aborts_run():
    r = pa_agent.run("q", pa_model(SLOW_CALL), tools=[ex.hang_tool(2.0)], run_timeout=0.3)
    assert r.error and "RunTimedOut" in r.error


def test_langgraph_has_no_tool_level_timeout_sync_step_timeout_is_late():
    import time
    t0 = time.monotonic()
    r = lg_agent.run("q", lg_model(SLOW_CALL), tools=[ex.hang_tool(1.2)], step_timeout=0.3)
    assert r.error and "TimeoutError" in r.error
    assert time.monotonic() - t0 >= 1.1  # sync invoke waited for the tool to finish before reporting


def test_tool_raised_timeout_langgraph_feeds_back_pydantic_ai_aborts():
    wrapped = lambda: ex.thread_timeout_wrapper(ex.hang_tool(1.0), 0.3)  # noqa: E731
    lg = lg_agent.run("q", lg_model(SLOW_CALL), tools=[wrapped()])
    pa = pa_agent.run("q", pa_model(SLOW_CALL), tools=[wrapped()])
    assert lg.error is None and lg.final_text == "done" and "TimeoutError" in lg.tool_calls[0].error
    assert pa.error and "TimeoutError" in pa.error
