"""Calculator agent on Pydantic AI.

The loop lives inside Agent.run (a graph of UserPrompt -> ModelRequest -> CallTools -> ...
that you do not write). The stop condition: the model returns a plain text response
(no tool calls) -> that text is the output.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # for direct script runs

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from lab_common.config import load_model_config
from lab_common.dataset import RunResult, ToolCallRecord
from lab_common.prompts import MAX_STEPS, SYSTEM_INSTRUCTIONS
from lab_common.tools import TOOLS


def default_model():
    cfg = load_model_config()
    return OpenAIChatModel(cfg.model_name, provider=OpenAIProvider(base_url=cfg.base_url, api_key=cfg.api_key))


def build_agent(model=None, retries=1, tools=None):
    """retries = per-tool retry budget for validation errors / ModelRetry."""
    return Agent(model or default_model(), instructions=SYSTEM_INSTRUCTIONS, tools=tools or TOOLS, retries=retries)


def run(question: str, model=None, retries=1, tools=None, run_timeout=None) -> RunResult:
    agent = build_agent(model, retries, tools)
    result = RunResult(final_text="")
    try:
        out = agent.run_sync(question, usage_limits=UsageLimits(request_limit=MAX_STEPS), timeout=run_timeout)
        result.final_text = str(out.output)
        _fill(result, out.all_messages())
    except Exception as e:  # tool exception / exhausted retries / usage limit escape the agent
        result.error = f"{type(e).__name__}: {e}"
    return result


def _fill(result: RunResult, messages) -> None:
    by_id = {}
    for m in messages:
        if isinstance(m, ModelResponse):
            result.llm_calls += 1
            for p in m.parts:
                if isinstance(p, ToolCallPart):
                    rec = ToolCallRecord(p.tool_name, p.args_as_dict())
                    by_id[p.tool_call_id] = rec
                    result.tool_calls.append(rec)
        elif isinstance(m, ModelRequest):
            for p in m.parts:
                # Observed in v2.62: validation failures come back as ToolReturnPart with
                # outcome="retried" (not a RetryPromptPart); success is outcome="success".
                if isinstance(p, ToolReturnPart) and p.tool_call_id in by_id:
                    if p.outcome == "success":
                        by_id[p.tool_call_id].result = str(p.content)
                    else:
                        by_id[p.tool_call_id].error = f"{p.outcome}: {p.content}"
                elif isinstance(p, RetryPromptPart) and p.tool_call_id in by_id:
                    by_id[p.tool_call_id].error = p.model_response()


if __name__ == "__main__":
    from lab_common.tracing import setup_tracing
    setup_tracing("pydantic-ai")
    q = " ".join(sys.argv[1:]) or "Calculate 120 + 45 and then multiply the result by 3."
    r = run(q)
    for t in r.tool_calls:
        print("TOOL", t.name, t.args, "->", t.result if t.error is None else f"ERROR {t.error}")
    print("LLM calls:", r.llm_calls, "| error:", r.error)
    print("ANSWER:", r.final_text)
