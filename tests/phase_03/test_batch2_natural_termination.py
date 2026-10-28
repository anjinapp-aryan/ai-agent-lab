"""Phase 3.1 Batch 2 regression tests: natural termination (offline, scripted completions over the real clients).

A sentinel completion is served if a framework ever asks for more completions than were scripted, so a hidden extra
model request is visible. These pin FRAMEWORK termination mechanics, not model behavior.
"""
import pytest

import exp3_batch2 as e2

FWS = e2.FWS
STEPS = {"langgraph": ["node 'agent' finished"],
         "pydantic-ai": ["about to run UserPromptNode", "about to run ModelRequestNode", "about to run CallToolsNode", "about to run End"]}


def run(case, fw):
    d, traj = e2.one(case, fw)
    return d, traj


@pytest.mark.parametrize("fw", FWS)
def test_no_tool_call_is_one_request_and_the_models_text_is_the_result(fw):
    d, _ = run("B2-3.1.4 no tool call", fw)
    a = d["accounting"]
    assert (a["model_requests"], a["tool_calls_requested"], a["tool_executions"]) == (1, 0, 0)
    assert not d["extra_request"] and d["error"] is None
    assert d["steps"] == STEPS[fw]
    assert d["final_text"] == "Final answer: 4" and d["final_identical_to_wire"]


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("case", ["B2-3.1.5 tool then final", "B2-3.1.8 tool then final (no 3rd request)"])
def test_tool_then_final_makes_no_third_request(case, fw):
    d, traj = run(case, fw)
    a = d["accounting"]
    assert (a["model_requests"], a["tool_executions"], a["final_responses"]) == (2, 1, 1)
    assert d["requests_served"] == 2 and not d["sentinel_requested"]
    assert traj.events[-1]["kind"] != "model_request"                       # nothing is requested after the final response


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("finish", ["stop", "length", "tool_calls (no calls)", "content_filter"])
def test_text_without_tool_calls_terminates_whatever_the_finish_reason(fw, finish):
    d, _ = run(f"B2-3.1.6 final, finish={finish}", fw)
    assert d["accounting"]["model_requests"] == 1 and d["error"] is None and d["final_text"] == "The final result is 425."


@pytest.mark.parametrize("fw", FWS)
def test_tool_calls_execute_even_when_finish_reason_says_stop(fw):
    d, _ = run("B2-3.1.6 tool call, finish=stop", fw)
    assert d["accounting"]["tool_executions"] == 1 and d["accounting"]["model_requests"] == 2


def test_finish_reason_is_recorded_but_not_used_to_route():
    lg, _ = run("B2-3.1.6 final, finish=tool_calls (no calls)", "langgraph")
    pa, _ = run("B2-3.1.6 final, finish=tool_calls (no calls)", "pydantic-ai")
    assert lg["snapshot"][-1]["finish_reason"] == "tool_calls"              # kept as metadata, loop still ended
    assert pa["snapshot"][-1]["finish_reason"] == "tool_call"               # Pydantic AI normalizes the value


@pytest.mark.parametrize("case", ["B2-3.1.7 empty content", "B2-3.1.7 null content"])
def test_empty_response_langgraph_terminates_pydantic_ai_reprompts(case):
    lg, _ = run(case, "langgraph")
    assert lg["accounting"]["model_requests"] == 1 and lg["final_text"] == "" and lg["error"] is None
    pa, _ = run(case, "pydantic-ai")
    assert pa["accounting"]["model_requests"] == 2 and pa["extra_request"]               # framework-initiated request
    assert pa["last_request_tail"] == {"role": "system", "content": "Please return text or call a tool."}


@pytest.mark.parametrize("finish,exc", [("length", "UnexpectedModelBehavior"), ("content_filter", "ContentFilterError")])
def test_empty_response_with_finish_reason_raises_in_pydantic_ai_only(finish, exc):
    lg, _ = run(f"B2-3.1.7 empty content, finish={finish}", "langgraph")
    pa, _ = run(f"B2-3.1.7 empty content, finish={finish}", "pydantic-ai")
    assert lg["error"] is None and lg["final_text"] == ""
    assert pa["error"].startswith(exc) and pa["accounting"]["model_requests"] == 1


def test_repeated_empty_response_exhausts_pydantic_ai_output_retries():
    pa, _ = run("B2-3.1.7 empty content twice", "pydantic-ai")
    assert pa["accounting"]["model_requests"] == 2 and "Exceeded maximum output retries (1)" in pa["error"]


@pytest.mark.parametrize("fw", FWS)
def test_final_text_is_returned_unchanged(fw):
    d, _ = run("B2-3.1.10 final text with whitespace/markdown", fw)
    assert d["final_text"] == "  **The answer is 425.**\n\n" and d["final_identical_to_wire"]
    if fw == "pydantic-ai":
        assert d["info"]["output_type"] == "str"


@pytest.mark.parametrize("fw", FWS)
def test_state_returned_at_natural_termination(fw):
    d, _ = run("B2-3.1.5 tool then final", fw)
    snap = d["snapshot"]
    if fw == "langgraph":
        assert [s["type"] for s in snap] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]     # no SystemMessage stored
        assert snap[-1]["tool_calls"] == [] and snap[-1]["content"] == "No further tools are required. The answer is 425."
    else:
        assert [s["type"] for s in snap] == ["ModelRequest", "ModelResponse", "ModelRequest", "ModelResponse"]
        assert d["info"]["run_step"] == 2 and d["info"]["usage_tool_calls"] == 1
