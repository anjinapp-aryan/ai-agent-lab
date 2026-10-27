"""The single shared tool set. Plain typed functions; each framework wraps them natively.

Docstrings are the tool descriptions and are identical for both frameworks.
"""


def add(a: float, b: float) -> float:
    """Add two numbers and return a + b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and return a - b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return a * b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b and return a / b. Fails if b is zero."""
    return a / b


TOOLS = [add, subtract, multiply, divide]
TOOL_NAMES = [t.__name__ for t in TOOLS]
