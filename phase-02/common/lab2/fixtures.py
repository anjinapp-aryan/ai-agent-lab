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


# =========================== Batch 2 fixtures ===========================
import time as _time  # noqa: E402

from pydantic import BaseModel  # noqa: E402


class ConflictError(Exception):
    """Backend: the idempotency key was reused with different arguments."""


class _Strict(BaseModel):
    x: int


def failing_tool(kind: str) -> str:
    """Raise a controlled exception of the requested kind (value, runtime, timeout, validation, tool_exception, model_retry)."""
    EXEC.append(("failing_tool", kind))
    if kind == "value":
        raise ValueError("controlled ValueError")
    if kind == "runtime":
        raise RuntimeError("controlled RuntimeError")
    if kind == "timeout":
        raise TimeoutError("controlled TimeoutError")
    if kind == "validation":
        _Strict(x="not-an-int")          # raises pydantic ValidationError from inside the tool body
    if kind == "tool_exception":
        from langchain_core.tools import ToolException
        raise ToolException("controlled ToolException")
    if kind == "model_retry":
        from pydantic_ai import ModelRetry
        raise ModelRetry("controlled ModelRetry: please try again")
    return "no failure"


def make_flaky(fail_times: int, how: str = "model_retry"):
    """Tool that fails `fail_times` times then succeeds. how: model_retry | runtime."""
    state = {"n": 0}

    def flaky() -> str:
        """Fails a fixed number of times, then succeeds."""
        state["n"] += 1
        EXEC.append(("flaky", state["n"]))
        if state["n"] <= fail_times:
            if how == "model_retry":
                from pydantic_ai import ModelRetry
                raise ModelRetry(f"attempt {state['n']} failed, retry")
            raise RuntimeError(f"attempt {state['n']} failed")
        return f"succeeded on attempt {state['n']}"
    return flaky


def make_flaky_internal_retry(fail_times: int, attempts: int = 3):
    """(T) tool-internal retry: the tool loops itself; the framework sees only the final outcome."""
    state = {"n": 0}

    def flaky_internal() -> str:
        """Retries itself internally before reporting."""
        last = None
        for _ in range(attempts):
            state["n"] += 1
            EXEC.append(("flaky_internal", state["n"]))
            if state["n"] > fail_times:
                return f"succeeded on internal attempt {state['n']}"
            last = RuntimeError(f"internal attempt {state['n']} failed")
        raise last
    return flaky_internal


class Orders:
    """Simulated order backend (no real effects) + the approved tiny idempotency guard.

    ledger = real side effects. fail_first = number of EXECUTIONS whose response is 'lost' AFTER the side effect.
    The guard (when a key is given) checks, then commits side effect + key together; `delay` widens the window
    between check and commit so the non-atomic race is observable. Not production-grade by design.
    """

    def __init__(self, fail_first: int = 0, exc=RuntimeError, delay: float = 0.0, atomic: bool = False):
        import threading
        self._lock, self.atomic = threading.Lock(), atomic     # atomic=True: check+effect+commit under one lock (Batch 4)
        self.ledger, self.seen = [], {}
        self.executions = self.replays = 0
        self.fail_first, self.exc, self.delay = fail_first, exc, delay

    def _execute(self, item: str, qty: int, key=None) -> str:
        if self.atomic:
            with self._lock:
                return self._execute_inner(item, qty, key)
        return self._execute_inner(item, qty, key)

    def _execute_inner(self, item: str, qty: int, key=None) -> str:
        sig = (item, qty)
        if key is not None and key in self.seen:
            prev_sig, result = self.seen[key]
            if prev_sig != sig:
                raise ConflictError(f"idempotency key {key!r} reused with different arguments")
            self.replays += 1
            return result                                   # replay: no new side effect
        self.executions += 1
        n = self.executions
        _time.sleep(self.delay)                             # window between check and commit
        self.ledger.append(sig)                             # THE SIDE EFFECT
        result = f"order #{len(self.ledger)} created: {qty} x {item}"
        if key is not None:
            self.seen[key] = (sig, result)                  # committed together with the side effect
        if n <= self.fail_first:
            raise self.exc("response lost after the order was created")
        return result

    def unguarded(self):
        def create_order(item: str, qty: int) -> str:
            """Create an order for qty units of item (simulated)."""
            return self._execute(item, qty)
        return create_order

    def model_keyed(self):
        def create_order(item: str, qty: int, idempotency_key: str) -> str:
            """Create an order for qty units of item (simulated). idempotency_key must be the same for retries of the same order."""
            return self._execute(item, qty, idempotency_key)
        return create_order

    def context_keyed(self, key: str):
        """Key supplied by the surrounding system (closure here), not by the model."""
        def create_order(item: str, qty: int) -> str:
            """Create an order for qty units of item (simulated)."""
            return self._execute(item, qty, key)
        return create_order


def record_action_factory():
    """Duplicate-call recorder: every execution is appended with a timestamp."""
    log = []

    def record_action(action_id: str, payload: str) -> str:
        """Record an action (simulated side effect)."""
        log.append({"action_id": action_id, "payload": payload, "t": _time.monotonic(), "n": len(log) + 1})
        return f"recorded #{len(log)}"
    return record_action, log
