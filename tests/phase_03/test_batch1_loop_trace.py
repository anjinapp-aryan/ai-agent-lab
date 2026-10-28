"""Phase 3.1 Batch 1 regression tests: the agent loop as observed on the wire, in the tools and in the framework.

Offline only: the real provider clients talk to a mock HTTP transport replaying scripted completions, so these pin
FRAMEWORK loop mechanics (what is sent, executed and counted), not model behavior. Live observations are in
phases/phase-03/experiments.md.
"""
import pytest

import exp3_batch1 as e1
from lab2 import wire
from lab3.trajectory import Trajectory, traced

FWS = e1.FWS


def run(exp, fw):
    q, script = e1.TASKS[exp]
    t, _ = wire.transport(script)
    d, traj = e1.one(exp, fw, q, t)
    assert d["error"] is None, d["error"]
    return d, traj


# framework_steps: LangGraph = super-steps (one per node execution); Pydantic AI = nodes yielded by agent.iter()
EXPECTED = {  # exp: (model_requests, tool_calls, tool_executions, langgraph_steps, pydantic_ai_steps)
    "3.1.1": (2, 1, 1, 3, 6),
    "3.1.2": (3, 2, 2, 5, 8),
    "3.1.3": (2, 2, 2, 3, 6),
    "3.1.3-seq": (3, 2, 2, 5, 8),
}


@pytest.mark.parametrize("fw", FWS)
@pytest.mark.parametrize("exp", list(EXPECTED))
def test_loop_accounting(exp, fw):
    d, _ = run(exp, fw)
    a = d["accounting"]
    req, calls, execs, lg_steps, pa_steps = EXPECTED[exp]
    assert a["model_requests"] == req and a["tool_calls_requested"] == calls and a["tool_executions"] == execs
    assert a["final_responses"] == 1 and a["model_responses_with_tool_calls"] == req - 1
    assert a["framework_steps"] == (lg_steps if fw == "langgraph" else pa_steps)


@pytest.mark.parametrize("fw", FWS)
def test_langgraph_steps_are_two_per_tool_round_plus_one_and_pydantic_ai_steps_track_requests(fw):
    d, traj = run("3.1.2", fw)
    if fw == "langgraph":
        assert [e["label"] for e in traj.of("framework_step")] == [f"node '{n}' finished" for n in ("agent", "tools", "agent", "tools", "agent")]
        assert d["info"]["super_steps"] == 5
    else:
        labels = [e["label"].split()[-1] for e in traj.of("framework_step")]
        assert labels == ["UserPromptNode", "ModelRequestNode", "CallToolsNode", "ModelRequestNode", "CallToolsNode",
                          "ModelRequestNode", "CallToolsNode", "End"]
        assert d["info"]["run_step"] == d["info"]["usage_requests"] == 3 and d["info"]["usage_tool_calls"] == 2


@pytest.mark.parametrize("fw", FWS)
def test_each_tool_result_is_in_the_next_model_request_and_not_before(fw):
    d, traj = run("3.1.2", fw)
    assert [(c["tool"], c["result"], c["in_next_request"], c["next_request"]) for c in d["context"]] == \
        [("add", 165.0, True, 2), ("multiply", 495.0, True, 3)]
    reqs = traj.of("model_request")
    assert reqs[0]["roles"] == ["system", "user"]
    assert reqs[1]["roles"] == ["system", "user", "assistant", "tool"]
    assert reqs[2]["roles"] == ["system", "user", "assistant", "tool", "assistant", "tool"]   # full history re-sent each time
    assert not any("165" in str(m.get("content")) for m in reqs[0]["messages"])


@pytest.mark.parametrize("fw", FWS)
def test_wire_tool_and_framework_events_interleave_in_loop_order(fw):
    _, traj = run("3.1.1", fw)
    order = [e["kind"] for e in traj.events if e["kind"] != "framework_step"]
    assert order == ["model_request", "model_response", "tool_start", "tool_end", "model_request", "model_response"]


@pytest.mark.parametrize("fw", FWS)
def test_batched_calls_are_one_iteration_with_two_observations(fw):
    _, traj = run("3.1.3", fw)
    resp = traj.of("model_response")
    assert [c["name"] for c in resp[0]["tool_calls"]] == ["multiply", "add"]
    assert traj.of("model_request")[1]["roles"] == ["system", "user", "assistant", "tool", "tool"]
    assert [e["tool"] for e in traj.of("tool_end")] == ["multiply", "add"]


@pytest.mark.parametrize("fw", FWS)
def test_loop_ends_on_a_response_without_tool_calls_and_makes_no_further_request(fw):
    _, traj = run("3.1.1", fw)
    last = traj.of("model_response")[-1]
    assert last["tool_calls"] == [] and last["finish_reason"] == "stop"
    assert traj.events.index(last) > traj.events.index(traj.of("model_request")[-1])


@pytest.mark.parametrize("fw", FWS)
def test_iteration_api_matches_the_plain_phase1_run(fw):
    """graph.stream / agent.iter show the same loop that lg_agent.run (invoke) / pa_agent.run (run_sync) execute."""
    import lg_agent
    import pa_agent
    q, script = e1.TASKS["3.1.2"]
    traj = Trajectory()
    t, _ = wire.transport(script)
    r = (lg_agent.run if fw == "langgraph" else pa_agent.run)(q, e1.MODELS[fw](traj, transport=t),
                                                               tools=[traced(x, traj) for x in e1.TOOLS])
    stream_d, _ = run("3.1.2", fw)
    a = traj.accounting()
    a.pop("framework_steps")
    s = dict(stream_d["accounting"])
    s.pop("framework_steps")
    assert a == s and r.llm_calls == 3 and len(r.tool_calls) == 2 and r.error is None


def test_traced_wrapper_keeps_the_tool_schema():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from lab_common.tools import multiply
    assert convert_to_openai_tool(traced(multiply, Trajectory())) == convert_to_openai_tool(multiply)
