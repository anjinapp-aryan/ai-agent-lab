"""Scripted (offline) models for both frameworks.

A script is a list of steps; each step is either
  ("text", "final answer")  or  ("call", [(tool_name, args_dict), ...]).
One step is consumed per LLM call. These test the FRAMEWORK mechanics deterministically;
live tests (marker `live`) test real model behavior.
"""
import uuid

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from lab_common.config import model_configured


class ScriptedChatModel(BaseChatModel):
    script: list
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kind, payload = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if kind == "text":
            msg = AIMessage(content=payload)
        else:
            msg = AIMessage(content="", tool_calls=[
                {"name": n, "args": a, "id": uuid.uuid4().hex[:8], "type": "tool_call"} for n, a in payload])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def lg_model(script):
    return ScriptedChatModel(script=script)


def pa_model(script):
    state = {"i": 0}

    def fn(messages, info):
        kind, payload = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        if kind == "text":
            return ModelResponse(parts=[TextPart(payload)])
        return ModelResponse(parts=[ToolCallPart(n, a) for n, a in payload])

    return FunctionModel(fn)


@pytest.fixture(params=["langgraph", "pydantic_ai"])
def framework(request):
    """Yields (name, run_fn, model_factory) so one test body runs against both frameworks."""
    if request.param == "langgraph":
        import lg_agent
        return "langgraph", lg_agent.run, lg_model
    import pa_agent
    return "pydantic_ai", pa_agent.run, pa_model


live = pytest.mark.skipif(not model_configured(), reason="no model provider configured in .env")
