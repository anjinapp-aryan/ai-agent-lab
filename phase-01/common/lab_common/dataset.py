"""Shared scenarios and the framework-neutral result type used by both agents."""
from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: str | None = None   # stringified tool output
    error: str | None = None    # tool error / validation error text shown to the model
    call_id: str | None = None  # provider tool-call id (links the model's request to the tool result)


@dataclass
class RunResult:
    final_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    error: str | None = None    # exception that escaped the framework (agent terminated)


@dataclass
class Scenario:
    id: str
    question: str
    kind: str                                   # happy | failure
    expect_answer: list[str] = field(default_factory=list)   # substrings, any formatting of number
    expect_tools: list[str] = field(default_factory=list)    # tool names that must be called
    forbid_answer: list[str] = field(default_factory=list)   # must NOT appear (anti-fabrication)
    note: str = ""


SCENARIOS = [
    Scenario("S1", "What is 25 * 17?", "happy", ["425"], ["multiply"]),
    Scenario("S2", "Calculate 120 + 45 and then multiply the result by 3.", "happy",
             ["495"], ["add", "multiply"], note="sequential: add then multiply(165, 3)"),
    Scenario("S3", "Calculate 10 * 5 and 20 + 7.", "happy", ["50", "27"], ["multiply", "add"],
             note="independent; call order / parallel vs sequential may vary"),
    Scenario("S4", "What is 10 divided by 0?", "failure", [], ["divide"],
             forbid_answer=["infinity", "inf "], note="tool raises ZeroDivisionError"),
    Scenario("S5", 'Calculate the division of 10 by "hello".', "failure", [], [],
             note="invalid arguments: non-numeric operand"),
    Scenario("S6", "What is the weather in Bangalore?", "failure", [], [],
             forbid_answer=["°c", "degrees", "sunny", "cloudy", "rain"],
             note="no tool exists; must not fabricate"),
]
