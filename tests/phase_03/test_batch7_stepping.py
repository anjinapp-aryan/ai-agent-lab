"""Phase 3.1 Batch 7 regression tests: manual agent stepping.

Offline and scripted (no model, no network). They pin what the installed LangGraph and Pydantic AI expose between steps:
what a consumer sees at each boundary, what has already happened by then, what stopping, editing and denying do. Assertions
use event counts and ordering, never timings. They say nothing about a real model.
"""
import asyncio
import threading
import time

import pytest

import exp3_batch7 as e7
from langchain_core.messages import AIMessage
from pydantic_ai.usage import UsageLimits

FWS = e7.FWS
TWO, ONE = e7.TWO_ROUNDS, e7.ONE_ROUND


def lg_updates(script=TWO):
    traj, model, tools = e7.rig("langgraph", script)
    g = e7.lg_graph(model, tools)
    return traj, [(m, c, e7.snap(traj)) for m, c in g.stream(e7.lg_input(), e7.CFG, stream_mode=["tasks", "updates"])]


# ------------------------------------------------------------ B7-3.1.59 / .60 LangGraph stepping
def test_lg_update_chunk_arrives_after_the_node_finished():
    traj, items = lg_updates()
    agent_updates = [s for m, c, s in items if m == "updates" and "agent" in c]
    tool_updates = [s for m, c, s in items if m == "updates" and "tools" in c]
    assert [s["requests"] for s in agent_updates] == [1, 2, 3]
    assert [(s["tool_starts"], s["tool_ends"]) for s in tool_updates] == [(1, 1), (2, 2)]


def test_lg_tasks_start_event_arrives_before_the_work_it_names():
    traj, items = lg_updates()
    starts = [s for m, c, s in items if m == "tasks" and c.get("name") == "tools" and "input" in c]
    assert [s["tool_starts"] for s in starts] == [0, 1]  # the tool of that round has not started when its task_start is received
    agent_starts = [s["requests"] for m, c, s in items if m == "tasks" and c.get("name") == "agent" and "input" in c]
    assert agent_starts == [0, 1, 2]


def test_lg_tool_vs_model_step_is_told_only_by_node_name():
    traj, items = lg_updates()
    names = [list(c)[0] for m, c, s in items if m == "updates"]
    assert names == ["agent", "tools", "agent", "tools", "agent"]


@pytest.mark.parametrize("mode", ["values", "updates", "tasks", "debug"])
def test_lg_astream_yields_the_same_chunks_as_stream(mode):
    traj, model, tools = e7.rig("langgraph", TWO)
    sync = [e7.summarize(mode, c) for c in e7.lg_graph(model, tools).stream(e7.lg_input(), e7.CFG, stream_mode=mode)]
    aseq, _, _ = asyncio.run(e7._alg(mode))
    assert [s["summary"] for s in aseq] == sync


def test_lg_astream_keeps_the_event_loop_free():
    aseq, traj, ticks = asyncio.run(e7._alg("updates"))
    assert ticks > 0 and len(aseq) == 5


def test_lg_messages_mode_needs_a_streaming_model_response():
    traj, model, tools = e7.rig("langgraph", ONE)
    with pytest.raises(ValueError, match="No generations"):
        list(e7.lg_graph(model, tools).stream(e7.lg_input(), e7.CFG, stream_mode="messages"))


def test_lg_checkpoints_and_custom_modes_are_silent_without_a_checkpointer_or_writer():
    for mode in ("checkpoints", "custom"):
        traj, model, tools = e7.rig("langgraph", ONE)
        assert list(e7.lg_graph(model, tools).stream(e7.lg_input(), e7.CFG, stream_mode=mode)) == []


# ------------------------------------------------------------ B7-3.1.61 Pydantic AI manual iteration
def test_pa_iter_yields_nodes_before_they_run():
    rows, final, hist, traj = asyncio.run(e7._pa_iter(TWO))
    assert [r["node"] for r in rows] == ["UserPromptNode", "ModelRequestNode", "CallToolsNode", "ModelRequestNode", "CallToolsNode", "ModelRequestNode", "CallToolsNode", "End"]
    assert rows[1]["snap_at_yield"]["requests"] == 0 and rows[2]["snap_at_yield"]["tool_starts"] == 0
    assert rows[3]["snap_at_yield"]["tool_ends"] == 1 and rows[3]["snap_at_yield"]["requests"] == 1
    assert final == "The result is 495." and hist == 6


def test_pa_proposed_call_and_arguments_are_visible_at_calltools_node():
    rows, *_ = asyncio.run(e7._pa_iter(TWO))
    calls = [r["fields"]["tool_calls"] for r in rows if r["node"] == "CallToolsNode"]
    assert list(calls[0][0][:2]) == ["add", {"a": 120, "b": 45}] and list(calls[1][0][:2]) == ["multiply", {"a": 165, "b": 3}] and calls[2] == []


def test_pa_has_no_tool_return_node_results_appear_in_the_next_request():
    rows, *_ = asyncio.run(e7._pa_iter(TWO))
    assert "ToolReturnNode" not in {r["node"] for r in rows}
    assert rows[3]["fields"]["request_parts"][0][0] == "ToolReturnPart"


def test_pa_tool_events_inside_calltools_node_are_observed_around_the_tool_run():
    rows, *_ = asyncio.run(e7._pa_iter(TWO, with_events=True))
    ev = [e for r in rows if r.get("events") for e in r["events"]]
    assert [e["event"] for e in ev] == ["FunctionToolCallEvent", "FunctionToolResultEvent"] * 2
    assert [(e["snap"]["tool_starts"], e["snap"]["tool_ends"]) for e in ev][:2] == [(0, 0), (1, 1)]


# ------------------------------------------------------------ B7-3.1.62 / .63 stepping and voluntary stops
@pytest.mark.parametrize("label,pred", [
    ("after first response", lambda i, m, c, lg: m == "updates" and "agent" in c),
    ("tools task starts", lambda i, m, c, lg: m == "tasks" and c.get("name") == "tools" and "input" in c),
    ("tool completed", lambda i, m, c, lg: m == "updates" and "tools" in c),
])
def test_lg_voluntary_stop_runs_nothing_afterwards(label, pred):
    before = threading.active_count()
    log, traj, graph, cfg, stopped = e7.walk_lg(TWO, lambda i, m, c, tr, lg: "stop" if pred(i, m, c, lg) else "continue")
    at = e7.snap(traj)
    time.sleep(0.2)
    assert e7.snap(traj) == at and stopped is not None and threading.active_count() <= before


def test_lg_stopping_at_the_tools_task_start_means_the_tool_never_runs():
    log, traj, *_ = e7.walk_lg(TWO, lambda i, m, c, tr, lg: "stop" if m == "tasks" and c.get("name") == "tools" and "input" in c else "continue")
    time.sleep(0.2)
    assert e7.snap(traj)["tool_starts"] == 0


@pytest.mark.parametrize("idx,node", [(2, "ModelRequestNode"), (3, "CallToolsNode"), (4, "ModelRequestNode"), (7, "CallToolsNode")])
def test_pa_voluntary_stop_runs_nothing_afterwards(idx, node):
    res, traj = e7.pa(TWO, lambda i, n, r, t: "stop" if i == idx else "continue")
    stopped = next(r for r in res["log"] if r.get("stopped"))
    assert stopped["node"] == node and stopped["snap"] == res["snap_after_settle"] and res["tasks_left"] == 0 and res["final"] is None


def test_pa_tool_completed_and_before_next_request_are_one_boundary():
    res, traj = e7.pa(TWO, lambda i, n, r, t: "stop" if i == 4 else "continue")
    s = res["log"][-1]["snap"]
    assert (s["tool_ends"], s["requests"]) == (1, 1)  # the tool ran, the second request is not yet sent


def test_pausing_is_just_not_advancing_and_the_run_completes_afterwards():
    res, traj = e7.pa(TWO, lambda i, n, r, t: ("sleep", 0.2) if i == 3 else "continue")
    p = next(r for r in res["log"] if "snap_after_pause" in r)
    assert p["snap"] == p["snap_after_pause"] and res["final"] == "The result is 495."
    box = {}

    def decide(i, m, c, tr, lg):
        if m == "tasks" and c.get("name") == "tools" and "input" in c and not box:
            box["before"] = e7.snap(tr)
            time.sleep(0.2)
            box["after"] = e7.snap(tr)
        return "continue"
    log, traj, *_ = e7.walk_lg(ONE, decide)
    assert box["before"] == box["after"] and e7.snap(traj)["tool_ends"] == 1


# ------------------------------------------------------------ B7-3.1.64 state
def test_lg_in_place_chunk_edit_reaches_the_tool_in_this_version():
    traj, model, tools = e7.rig("langgraph", ONE)
    for chunk in e7.lg_graph(model, tools).stream(e7.lg_input(), e7.CFG, stream_mode="updates"):
        if "agent" in chunk and chunk["agent"]["messages"][-1].tool_calls:
            chunk["agent"]["messages"][-1].tool_calls[0]["args"]["a"] = 1000
    assert e7.tool_args(traj)[0]["a"] == 1000  # incidental (shared object), not a documented route


def test_lg_update_state_edits_the_checkpoint_and_the_next_stream_uses_it():
    traj, model, tools = e7.rig("langgraph", ONE)
    g, cfg = e7.lg_ckpt_graph(model, tools), e7.lg_cfg()
    gen = g.stream(e7.lg_input(), cfg, stream_mode="updates")
    next(gen)
    gen.close()
    st = g.get_state(cfg)
    assert list(st.next) == ["tools"] and e7.snap(traj)["tool_starts"] == 0
    ai = st.values["messages"][-1]
    g.update_state(cfg, {"messages": [AIMessage(content=ai.content, id=ai.id, tool_calls=[{"name": "add", "args": {"a": 7, "b": 45}, "id": ai.tool_calls[0]["id"]}])]})
    assert len(g.get_state(cfg).values["messages"]) == 2  # replaced by id, not appended
    list(g.stream(None, cfg, stream_mode="updates"))
    assert e7.tool_args(traj) == [{"a": 7.0, "b": 45.0}]


def test_lg_update_state_does_not_affect_a_stream_that_is_suspended_mid_run():
    traj, model, tools = e7.rig("langgraph", ONE)
    g, cfg = e7.lg_ckpt_graph(model, tools), e7.lg_cfg()
    gen = g.stream(e7.lg_input(), cfg, stream_mode="updates")
    ai = next(gen)["agent"]["messages"][-1]
    g.update_state(cfg, {"messages": [AIMessage(content=ai.content, id=ai.id, tool_calls=[{"name": "add", "args": {"a": 7, "b": 45}, "id": ai.tool_calls[0]["id"]}])]})
    list(gen)
    assert e7.tool_args(traj) == [{"a": 120.0, "b": 45.0}]


def test_pa_node_edit_changes_what_the_tool_receives_and_what_history_records():
    def edit(i, node, run, tr):
        if type(node).__name__ == "CallToolsNode" and i == 3:
            node.model_response.parts[0].args = {"a": 1000, "b": 45}
        return "continue"
    res, traj = e7.pa(ONE, edit)
    assert e7.tool_args(traj) == [{"a": 1000.0, "b": 45.0}]
    assert "1000" in str(traj.of("model_request")[1]["messages"])


def test_pa_request_node_edit_changes_the_wire():
    def edit(i, node, run, tr):
        if type(node).__name__ == "ModelRequestNode" and i == 4:
            node.request.parts[0].content = "EDITED"
        return "continue"
    res, traj = e7.pa(ONE, edit)
    assert e7._req_tool_contents(traj, 2) == ["EDITED"]


# ------------------------------------------------------------ B7-3.1.65 approval
CASES = {
    "allow": (lambda n, a: "allow", 3, [{"a": 120.0, "b": 45.0}, {"a": 165.0, "b": 3.0}]),
    "stop": (lambda n, a: "stop", 1, []),
    "modify": (lambda n, a: ("modify", {"a": 1000}) if n == "add" else "allow", 3, [{"a": 1000.0, "b": 45.0}, {"a": 165.0, "b": 3.0}]),
    "deny": (lambda n, a: "deny" if n == "multiply" else "allow", 3, [{"a": 120.0, "b": 45.0}]),
}


@pytest.mark.parametrize("case", list(CASES))
def test_lg_approval_loop(case):
    dec, requests, executed = CASES[case]
    r = e7.lg_approve(dec)
    assert e7.tool_args(r["traj"]) == executed and e7.snap(r["traj"])["requests"] == requests


@pytest.mark.parametrize("case", list(CASES))
def test_pa_approval_loop(case):
    dec, requests, executed = CASES[case]
    r = e7.pa_approve(dec)
    assert e7.tool_args(r["traj"]) == executed and e7.snap(r["traj"])["requests"] == requests and r["res"]["error"] is None


def test_the_model_is_told_a_call_was_denied_in_both_frameworks():
    lg = e7.lg_approve(CASES["deny"][0])
    pa = e7.pa_approve(CASES["deny"][0])
    assert "DENIED by policy" in str(lg["traj"].of("model_request")[-1]["messages"])
    assert "DENIED by policy" in str(pa["traj"].of("model_request")[-1]["messages"])


def test_denied_call_is_not_counted_as_a_tool_call_in_pa_usage():
    r = e7.pa_approve(CASES["deny"][0])
    assert r["res"]["log"][-1]["usage"]["tool_calls"] == 1


def test_no_framework_records_the_approval_decision():
    r = e7.lg_approve(CASES["allow"][0])
    msgs = r["graph"].get_state(r["cfg"]).values["messages"]
    assert not any("approv" in str(m.content).lower() or "allow" in str(m.content).lower() for m in msgs)
    p = e7.pa_approve(CASES["allow"][0])
    assert all("approv" not in str(m).lower() for m in p["traj"].of("model_request")[-1]["messages"])


def test_pa_tool_call_results_denies_an_ordinary_tool_without_running_it():
    def probe(i, node, run, tr):
        if type(node).__name__ == "CallToolsNode" and i == 3:
            from pydantic_ai.tools import ToolDenied
            node.tool_call_results = {node.model_response.parts[0].tool_call_id: ToolDenied("no")}
        return "continue"
    res, traj = e7.pa(ONE, probe)
    assert e7.tool_args(traj) == [] and res["error"] is None


# ------------------------------------------------------------ B7-3.1.66 failure
def test_lg_failure_is_a_tool_message_boundary_and_retry_is_the_next_proposal():
    tools = list(e7.TOOLS) + [e7._flaky()]
    traj, model, wrapped = e7.rig("langgraph", e7.FAIL_SCRIPT, tools)
    msgs = []
    for chunk in e7.lg_graph(model, wrapped).stream(e7.lg_input(), e7.CFG, stream_mode="updates"):
        if "tools" in chunk:
            msgs += [m.status for m in chunk["tools"]["messages"]]
    assert msgs == ["success", "error", "success"] and e7.snap(traj)["tool_starts"] == 3


def test_pa_failure_is_a_retried_tool_return_in_the_next_request_node():
    async def walk():
        traj, model, wrapped = e7.rig("pydantic-ai", e7.FAIL_SCRIPT, list(e7.TOOLS) + [e7._flaky()])
        agent = e7.pa_agent.build_agent(model, retries=3, tools=wrapped)
        out = []
        async with agent.iter(e7.Q) as run:
            async for node in run:
                if type(node).__name__ == "ModelRequestNode":
                    out.append([getattr(p, "outcome", None) for p in node.request.parts])
        return out
    assert asyncio.run(walk()) == [[None], ["success"], ["retried"], ["success"]]


# ------------------------------------------------------------ B7-3.1.67 limits
def test_lg_pausing_consumes_no_recursion_budget():
    traj, model, wrapped = e7.rig("langgraph", TWO)
    n = 0
    for mode, chunk in e7.lg_graph(model, wrapped).stream(e7.lg_input(), {"recursion_limit": 6}, stream_mode=["tasks"]):
        time.sleep(0.01)
        n += 1
    assert e7.snap(traj)["requests"] == 3


def test_lg_limit_equal_to_steps_raises_after_the_final_answer_was_streamed():
    traj, model, wrapped = e7.rig("langgraph", TWO)
    seen = []
    with pytest.raises(Exception, match="Recursion limit of 5"):
        for chunk in e7.lg_graph(model, wrapped).stream(e7.lg_input(), {"recursion_limit": 5}, stream_mode="updates"):
            seen.append(list(chunk)[0])
    assert seen[-1] == "agent" and e7.snap(traj)["requests"] == 3


def test_pa_request_limit_error_surfaces_on_advancing_the_over_limit_node():
    res, traj = e7.pa(TWO, lambda *a: "continue", iter_kw={"usage_limits": UsageLimits(request_limit=2)})
    assert [l["node"] for l in res["log"]][-1] == "ModelRequestNode" and "UsageLimitExceeded" in res["error"] and e7.snap(traj)["requests"] == 2


def test_pa_run_deadline_keeps_running_while_the_consumer_pauses():
    fast = asyncio.run(e7._pa_deadline(0.0))
    slow = asyncio.run(e7._pa_deadline(0.6))
    assert fast["error"] is None and "RunTimedOut" in slow["error"] and slow["snap"]["tool_starts"] == 0
