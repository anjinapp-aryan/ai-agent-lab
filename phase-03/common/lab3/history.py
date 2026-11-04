"""Framework-neutral view of the conversation a run left behind (measurement only).

Turns LangGraph messages or Pydantic AI messages into the same list of plain dicts, so retained state can be compared:
{role, text, calls: [(id, name, args)], result_for: call_id, outcome}. It reads state; it never produces or changes it.
"""


def lg_history(messages):
    out = []
    for m in messages:
        kind = type(m).__name__
        if kind == "HumanMessage":
            out.append({"role": "user", "text": m.content})
        elif kind == "AIMessage":
            out.append({"role": "assistant", "text": m.content if isinstance(m.content, str) else str(m.content),
                        "calls": [(c["id"], c["name"], c["args"]) for c in m.tool_calls]})
        elif kind == "ToolMessage":
            out.append({"role": "tool", "text": m.content if isinstance(m.content, str) else str(m.content),
                        "result_for": m.tool_call_id, "outcome": getattr(m, "status", None)})
    return out


def pa_history(messages):
    out = []
    for m in messages:
        for p in m.parts:
            kind = type(p).__name__
            if kind == "UserPromptPart":
                out.append({"role": "user", "text": p.content if isinstance(p.content, str) else str(p.content)})
            elif kind == "ToolReturnPart":
                out.append({"role": "tool", "text": p.model_response_str(), "result_for": p.tool_call_id, "outcome": p.outcome})
            elif kind == "RetryPromptPart":
                out.append({"role": "tool", "text": p.model_response(), "result_for": p.tool_call_id, "outcome": "retry_prompt"})
            elif kind == "ToolCallPart":
                _merge_call(out, p)
            elif kind == "TextPart":
                out.append({"role": "assistant", "text": p.content, "calls": []})
    return out


def _merge_call(out, part):
    """Consecutive tool-call parts of one response become one assistant entry, like LangGraph's single AIMessage."""
    call = (part.tool_call_id, part.tool_name, part.args_as_dict())
    if out and out[-1]["role"] == "assistant" and out[-1].get("_open"):
        out[-1]["calls"].append(call)
    else:
        out.append({"role": "assistant", "text": "", "calls": [call], "_open": True})


def finish(history):
    for h in history:
        h.pop("_open", None)
    return history
