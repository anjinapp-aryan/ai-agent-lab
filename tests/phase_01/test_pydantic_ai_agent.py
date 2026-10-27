"""Offline Pydantic AI mechanics. Behavior observed here is recorded in failure-analysis.md."""
import pa_agent
from conftest import pa_model


def test_happy_single_tool():
    r = pa_agent.run("q", pa_model([("call", [("multiply", {"a": 25, "b": 17})]), ("text", "425")]))
    assert [t.name for t in r.tool_calls] == ["multiply"] and r.tool_calls[0].result == "425.0"
    assert r.llm_calls == 2 and r.final_text == "425" and r.error is None


def test_sequential_tools_result_feeds_next_call():
    script = [("call", [("add", {"a": 120, "b": 45})]),
              ("call", [("multiply", {"a": 165, "b": 3})]), ("text", "495")]
    r = pa_agent.run("q", pa_model(script))
    assert [t.name for t in r.tool_calls] == ["add", "multiply"]
    assert r.llm_calls == 3 and r.final_text == "495"


def test_tool_exception_escapes_agent_by_default():
    script = [("call", [("divide", {"a": 10, "b": 0})]), ("text", "unreachable")]
    r = pa_agent.run("q", pa_model(script))
    assert r.error and "ZeroDivisionError" in r.error and r.final_text == ""


def test_invalid_arguments_become_retry_prompt_then_recover():
    script = [("call", [("divide", {"a": 10, "b": "hello"})]),
              ("text", "Cannot divide by text.")]
    r = pa_agent.run("q", pa_model(script))
    assert r.tool_calls[0].error  # RetryPromptPart shown to the model
    assert r.error is None and r.final_text


def test_invalid_arguments_exhaust_retries():
    r = pa_agent.run("q", pa_model([("call", [("divide", {"a": 10, "b": "hello"})])]), retries=1)
    assert r.error and "UnexpectedModelBehavior" in r.error


def test_step_limit_stops_infinite_tool_loop():
    r = pa_agent.run("q", pa_model([("call", [("add", {"a": 1, "b": 1})])]))
    assert r.error and "UsageLimitExceeded" in r.error


def test_no_tool_needed_returns_text_directly():
    r = pa_agent.run("q", pa_model([("text", "I can't check weather.")]))
    assert r.tool_calls == [] and r.llm_calls == 1


def test_unknown_tool_name_fed_back_not_crash():  # FAILURE-05 model/tool mismatch
    r = pa_agent.run("q", pa_model([("call", [("sqrt", {"a": 9})]), ("text", "I cannot do sqrt.")]))
    assert "Unknown tool name" in r.tool_calls[0].error and r.error is None
