"""Offline LangGraph mechanics. Behavior observed here is recorded in failure-analysis.md."""
import lg_agent
from conftest import lg_model


def test_happy_single_tool():
    r = lg_agent.run("q", lg_model([("call", [("multiply", {"a": 25, "b": 17})]), ("text", "425")]))
    assert [t.name for t in r.tool_calls] == ["multiply"] and r.tool_calls[0].result == "425.0"
    assert r.llm_calls == 2 and r.final_text == "425" and r.error is None


def test_sequential_tools_result_feeds_next_call():
    script = [("call", [("add", {"a": 120, "b": 45})]),
              ("call", [("multiply", {"a": 165, "b": 3})]), ("text", "495")]
    r = lg_agent.run("q", lg_model(script))
    assert [t.name for t in r.tool_calls] == ["add", "multiply"]
    assert r.llm_calls == 3 and r.final_text == "495"


def test_tool_exception_default_is_fed_back_to_model():
    script = [("call", [("divide", {"a": 10, "b": 0})]), ("text", "Cannot divide by zero.")]
    r = lg_agent.run("q", lg_model(script))
    assert r.tool_calls[0].error and "ZeroDivisionError" in r.tool_calls[0].error or "division" in r.tool_calls[0].error
    assert r.error is None and "zero" in r.final_text.lower()


def test_tool_exception_with_handling_disabled_terminates_run():
    script = [("call", [("divide", {"a": 10, "b": 0})]), ("text", "unreachable")]
    r = lg_agent.run("q", lg_model(script), handle_tool_errors=False)
    assert r.error and "ZeroDivisionError" in r.error and r.final_text == ""


def test_invalid_arguments_reported_to_model():
    script = [("call", [("divide", {"a": 10, "b": "hello"})]), ("text", "Cannot divide by text.")]
    r = lg_agent.run("q", lg_model(script))
    # Observed: ToolNode catches the pydantic validation failure and wraps it as ToolInvocationError
    assert r.tool_calls[0].error and "ToolInvocationError" in r.tool_calls[0].error
    assert "valid number" in r.tool_calls[0].error
    assert r.error is None


def test_step_limit_stops_infinite_tool_loop():
    r = lg_agent.run("q", lg_model([("call", [("add", {"a": 1, "b": 1})])]))  # model never stops
    assert r.error and "GraphRecursionError" in r.error


def test_no_tool_needed_returns_text_directly():
    r = lg_agent.run("q", lg_model([("text", "I can't check weather.")]))
    assert r.tool_calls == [] and r.llm_calls == 1


def test_unknown_tool_name_fed_back_not_crash():  # FAILURE-05 model/tool mismatch
    r = lg_agent.run("q", lg_model([("call", [("sqrt", {"a": 9})]), ("text", "I cannot do sqrt.")]))
    assert "not a valid tool" in r.tool_calls[0].error and r.error is None
