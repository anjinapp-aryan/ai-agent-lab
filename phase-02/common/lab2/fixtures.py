"""Simulated tool fixtures (test data, not a framework). EXEC records every real execution."""
import enum
from typing import Literal

EXEC: list = []   # (tool_name, args...) appended when the tool body actually runs


def calculate(a: int, b: int) -> int:
    """Add two integers and return a + b."""
    EXEC.append(("calculate", a, b))
    return a + b


def calc_kwargs(a: int, b: int, **kwargs) -> str:
    """Add two integers; reports any extra keyword arguments it received."""
    EXEC.append(("calc_kwargs", a, b, dict(kwargs)))
    return f"{a + b} extras={kwargs}"


def convert(amount: float, currency: Literal["USD", "EUR", "INR"]) -> str:
    """Convert an amount into the given currency (simulated)."""
    EXEC.append(("convert", amount, currency))
    return f"{amount} {currency}"


class Currency(enum.Enum):
    USD = "USD"
    EUR = "EUR"
    INR = "INR"


def convert_enum(amount: float, currency: Currency) -> str:
    """Convert an amount into the given currency (simulated, Enum-typed)."""
    EXEC.append(("convert_enum", amount, currency))
    return f"{amount} {currency.value}"


# ---- result-validation fixtures (Experiment 11): annotations are deliberately sometimes false ----
def r_valid() -> str:
    """Return a normal string result."""
    EXEC.append(("r_valid",))
    return "ok"


def r_dict() -> dict:
    """Return a dict result."""
    EXEC.append(("r_dict",))
    return {"status": "ok", "n": 1}


def r_none() -> int:
    """Annotated int but returns None."""
    EXEC.append(("r_none",))
    return None  # type: ignore[return-value]


def r_wrong_type() -> int:
    """Annotated int but returns a string."""
    EXEC.append(("r_wrong_type",))
    return "abc"  # type: ignore[return-value]


class _Opaque:
    def __repr__(self):
        return "<Opaque object, not JSON serializable>"


def r_object() -> str:
    """Annotated str but returns an arbitrary object."""
    EXEC.append(("r_object",))
    return _Opaque()  # type: ignore[return-value]


def r_nan() -> float:
    """Return NaN."""
    EXEC.append(("r_nan",))
    return float("nan")


HUGE_CHARS = 1_000_000


def r_huge() -> str:
    """Return a 1,000,000 character string."""
    EXEC.append(("r_huge",))
    return "x" * HUGE_CHARS
