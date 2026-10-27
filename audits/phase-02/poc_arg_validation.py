"""Phase 2 audit proof-of-concept (NOT the Phase 2 experiments): verify three claims offline.

1. What JSON schema does each framework send to the model for the same typed tool?
2. How is a missing / extra / enum-invalid argument handled (rejected, ignored, coerced)?
Scripted model only; no API calls. Run: uv run python audits/phase-02/poc_arg_validation.py
"""
import json
import os
import sys
from typing import Literal

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "langgraph-agent", "pydantic-ai-agent"):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "phase-01", sub))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "phase_01"))
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

import conftest  # noqa: E402
import lg_agent  # noqa: E402
import pa_agent  # noqa: E402


def convert(amount: float, currency: Literal["USD", "EUR", "INR"]) -> str:
    """Convert an amount into the given currency (simulated)."""
    return f"{amount} {currency}"


CASES = {
    "valid": {"amount": 10, "currency": "USD"},
    "missing_arg": {"amount": 10},
    "wrong_type": {"amount": "hello", "currency": "USD"},
    "extra_arg": {"amount": 10, "currency": "USD", "secret": "unexpected"},
    "invalid_enum": {"amount": 10, "currency": "XYZ"},
    "coercible_string": {"amount": "10", "currency": "USD"},
}


def schemas():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai import Agent
    print("LangChain schema:", json.dumps(convert_to_openai_tool(convert)["function"]["parameters"]))
    seen = {}

    def fn(messages, info):
        seen["p"] = info.function_tools[0].parameters_json_schema
        return ModelResponse(parts=[TextPart("x")])
    Agent(FunctionModel(fn), tools=[convert]).run_sync("q")
    print("PydanticAI schema:", json.dumps(seen["p"]))


def main():
    schemas()
    print()
    for name, args in CASES.items():
        script = [("call", [("convert", args)]), ("text", "end")]
        lg = lg_agent.run("q", conftest.lg_model(script), tools=[convert])
        pa = pa_agent.run("q", conftest.pa_model(script), tools=[convert])
        for fw, r in (("langgraph", lg), ("pydantic-ai", pa)):
            t = r.tool_calls[0] if r.tool_calls else None
            if t is None:
                print(f"{name:17}{fw:12}no tool record; error={r.error}")
                continue
            outcome = "EXECUTED -> " + str(t.result) if t.error is None else "REJECTED -> " + t.error.replace("\n", " ")[:110]
            print(f"{name:17}{fw:12}{outcome}")


if __name__ == "__main__":
    main()
