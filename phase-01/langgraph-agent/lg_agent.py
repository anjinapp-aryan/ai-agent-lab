"""Calculator agent on LangGraph, built as an explicit StateGraph.

    START -> agent(LLM) -- tool_calls? yes --> tools(ToolNode) -> agent ...
                        -- no tool calls  --> END

The loop is the graph cycle agent <-> tools. The stop condition is the conditional edge
(tools_condition): no tool_calls on the last AI message -> END.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # for direct script runs

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lab_common.config import load_model_config
from lab_common.dataset import RunResult, ToolCallRecord
from lab_common.prompts import MAX_STEPS, SYSTEM_INSTRUCTIONS
from lab_common.tools import TOOLS


def default_model():
    cfg = load_model_config()
    return ChatOpenAI(model=cfg.model_name, api_key=cfg.api_key, base_url=cfg.base_url, temperature=0)


def build_graph(model=None, handle_tool_errors=True, tools=None, step_timeout=None):
    """handle_tool_errors is passed to ToolNode; see failure-analysis.md for what it changes."""
    tools = tools or TOOLS
    llm = (model or default_model()).bind_tools(tools)

    def agent_node(state: MessagesState):
        msgs = [SystemMessage(SYSTEM_INSTRUCTIONS)] + state["messages"]
        return {"messages": [llm.invoke(msgs)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_errors))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    g.add_edge("tools", "agent")
    compiled = g.compile()
    compiled.step_timeout = step_timeout  # native LangGraph per-step timeout (experiments only)
    return compiled


def run(question: str, model=None, handle_tool_errors=True, tools=None, step_timeout=None,
        recursion_limit=None) -> RunResult:
    graph = build_graph(model, handle_tool_errors, tools, step_timeout)
    result = RunResult(final_text="")
    messages = []
    try:
        out = graph.invoke({"messages": [HumanMessage(question)]},
                           {"recursion_limit": recursion_limit or 2 * MAX_STEPS + 1})
        messages = out["messages"]
    except GraphRecursionError as e:
        result.error = f"GraphRecursionError: {e}"
    except Exception as e:  # exception escaped the graph (e.g. tool error with handling disabled)
        result.error = f"{type(e).__name__}: {e}"
    if messages:
        _fill(result, messages)
    return result


def _fill(result: RunResult, messages) -> None:
    by_id = {}
    for m in messages:
        if isinstance(m, AIMessage):
            result.llm_calls += 1
            for tc in m.tool_calls:
                rec = ToolCallRecord(tc["name"], tc["args"])
                by_id[tc["id"]] = rec
                result.tool_calls.append(rec)
        elif isinstance(m, ToolMessage) and m.tool_call_id in by_id:
            rec = by_id[m.tool_call_id]
            if m.status == "error":
                rec.error = str(m.content)
            else:
                rec.result = str(m.content)
    last = messages[-1]
    if isinstance(last, AIMessage):
        result.final_text = last.content if isinstance(last.content, str) else str(last.content)


if __name__ == "__main__":
    from lab_common.tracing import setup_tracing
    setup_tracing("langgraph")
    q = " ".join(sys.argv[1:]) or "Calculate 120 + 45 and then multiply the result by 3."
    r = run(q)
    for t in r.tool_calls:
        print("TOOL", t.name, t.args, "->", t.result if t.error is None else f"ERROR {t.error}")
    print("LLM calls:", r.llm_calls, "| error:", r.error)
    print("ANSWER:", r.final_text)
