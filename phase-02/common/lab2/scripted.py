"""Phase 2 scripted models = Phase 1 scripted models (REUSED by path) + recording of what the model is sent.

Phase 1's tests/phase_01/conftest.py is loaded as an isolated module so nothing is copied or modified
and the `conftest` module name never collides.
"""
import importlib.util
import os

_P1 = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "phase_01", "conftest.py")
_spec = importlib.util.spec_from_file_location("phase01_scripted", _P1)
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)


class CapturingLG(p1.ScriptedChatModel):
    """Phase 1 ScriptedChatModel + records bound tool schemas and every message list it receives."""
    seen: list = []
    bound: list = []

    def bind_tools(self, tools, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        self.bound = [convert_to_openai_tool(t) for t in tools]
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def lg_model(script):
    return CapturingLG(script=script, seen=[], bound=[])


def pa_model(script):
    """Phase 1 pa_model, wrapped (not reimplemented) to record messages and tool definitions."""
    m = p1.pa_model(script)
    inner = m.function
    m.seen, m.tools = [], []

    def recording(messages, info):
        m.seen.append(list(messages))
        m.tools = [(t.name, t.description, t.parameters_json_schema) for t in info.function_tools]
        return inner(messages, info)

    m.function = recording
    return m


def model_sees_lg(model):
    """Tool-result content the model received on its LAST call (LangGraph): list of (text, status)."""
    from langchain_core.messages import ToolMessage
    last = model.seen[-1] if model.seen else []
    return [(m.content, m.status) for m in last if isinstance(m, ToolMessage)][-3:]


def model_sees_pa(model):
    """Same for Pydantic AI: (provider-serialized text, outcome) of tool returns in the last request."""
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    last = model.seen[-1] if model.seen else []
    out = []
    for msg in last[-1:]:
        if isinstance(msg, ModelRequest):
            for p in msg.parts:
                if isinstance(p, ToolReturnPart):
                    try:
                        out.append((p.model_response_str(), p.outcome))
                    except Exception as e:  # serialization failure surfaces when the result is mapped for the provider
                        out.append((f"SERIALIZATION ERROR {type(e).__name__}: {str(e)[:70]}", p.outcome))
    return out[-3:]
