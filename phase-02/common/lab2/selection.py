"""Batch 5 fixtures: order-domain tools whose name / description / schema / order can be varied ONE at a time, an
execution log that separates MODEL REQUESTED from TOOL EXECUTED, a first-turn selection probe, and a small classifier.

Test data and measurement code, not a tool framework and not an evaluation framework:
- tools are plain Python functions (both frameworks derive the schema from the same function);
- the probe and the tool_choice switch are thin subclasses of the existing provider clients;
- `classify` maps one RunResult to one label, in the style of lab_common/evaluate.py.
"""
import asyncio
import itertools
import json
import re
import threading
import time
from typing import Annotated, Any

from pydantic import Field

PROBE_END = "SELECTION_PROBE_END"

# ------------------------------------------------------------------ default metadata (the controlled domain)
DESC = {
    "get_order": "Get the details of an order: items, total and status.",
    "track_order": "Track the shipment of an order: carrier, current location and estimated delivery.",
    "cancel_order": "Cancel an order.",
    "refund_order": "Refund an order.",
    "update_order_address": "Change the delivery address of an order.",
    "delete_order": "Delete an order record.",
    # collision vocabulary (5.8 / 5.12)
    "get_order_status": "Get the status of an order.",
    "find_order": "Find an order.",
    "lookup_order": "Look up an order.",
}
PARAMS = {"update_order_address": ("order_id", "address")}

RESULT = {
    "get_order": "item=wireless headphones, total=$59.00, status=SHIPPED",
    "track_order": "carrier=UPS, last scan=Memphis TN hub, estimated delivery=2026-10-30",
    "get_order_status": "status=SHIPPED",
    "find_order": "found: item=wireless headphones, status=SHIPPED",
    "lookup_order": "item=wireless headphones, status=SHIPPED",
    "cancel_order": "cancellation recorded (simulated; no real effect)",
    "refund_order": "refund recorded (simulated; no real effect)",
    "update_order_address": "address change recorded (simulated; no real effect)",
    "delete_order": "deletion recorded (simulated; no real effect)",
}
EFFECTFUL = {"cancel_order", "refund_order", "update_order_address", "delete_order"}

# 19 unrelated store tools, used only to vary the tool COUNT (5.7). One string argument each.
DISTRACTORS = [
    ("get_customer_profile", "customer_id", "Get a customer's profile: name, email and membership tier."),
    ("list_products", "category", "List the products in a catalog category."),
    ("get_product_stock", "product_id", "Get the warehouse stock level of a product."),
    ("apply_coupon", "code", "Check whether a coupon code is valid."),
    ("get_store_hours", "city", "Get the opening hours of the store in a city."),
    ("get_return_policy", "country", "Get the return policy text for a country."),
    ("check_gift_card_balance", "card_number", "Check the balance of a gift card."),
    ("subscribe_newsletter", "email", "Subscribe an email address to the newsletter."),
    ("get_shipping_rates", "postcode", "Get shipping rates to a postcode."),
    ("update_payment_method", "customer_id", "Update a customer's saved payment method."),
    ("get_loyalty_points", "customer_id", "Get a customer's loyalty points balance."),
    ("list_promotions", "category", "List current promotions in a category."),
    ("get_tax_rate", "postcode", "Get the sales tax rate for a postcode."),
    ("rate_product", "product_id", "Leave a star rating for a product."),
    ("get_faq", "topic", "Get FAQ answers for a topic."),
    ("get_exchange_rate", "currency", "Get the exchange rate for a currency."),
    ("reset_password", "email", "Send a password reset email."),
    ("get_weather", "city", "Get the weather forecast for a city."),
    ("create_support_ticket", "subject", "Open a support ticket with a subject line."),
]

# 5.6: the only difference between the schemas is the annotation of order_id
OrderIdRich = Annotated[str, Field(description="The order number as digits only, without '#' or the word 'order', e.g. '123'.",
                                   pattern=r"^\d+$")]

# ------------------------------------------------------------------ execution log
EXEC: list = []           # every tool body that started: the ground truth for "executed"
EFFECTS: list = []        # simulated side effects that were committed
_seq = itertools.count(1)
_lock = threading.Lock()


def reset():
    global _seq
    with _lock:
        EXEC.clear()
        EFFECTS.clear()
        _seq = itertools.count(1)


def _task_name():
    try:
        t = asyncio.current_task()
        return t.get_name() if t else None
    except RuntimeError:
        return None


def _execute(name, kind, args, output, check):
    with _lock:
        seq = next(_seq)
    ev = {"seq": seq, "tool": name, "kind": kind, "args": dict(args), "start": time.perf_counter(), "end": None,
          "thread": threading.get_ident(), "task": _task_name(), "status": "ok"}
    EXEC.append(ev)
    try:
        if check is not None:
            check(name, args)                      # application policy; may raise (e.g. PermissionDenied)
        if kind in EFFECTFUL:
            EFFECTS.append((name, dict(args)))
        body = output if output is not None else RESULT.get(kind, "ok")
        if not isinstance(body, str):
            return body
        return f"TOOL_EXECUTED:{name}#{seq} order {args.get('order_id', '?')}: {body}"
    except BaseException:
        ev["status"] = "failed"
        raise
    finally:
        ev["end"] = time.perf_counter()


def make_tool(name, description=None, *, kind=None, params=None, schema="plain", output=None, check=None):
    """A plain function `name(order_id: str, ...) -> str` with the given docstring.

    `kind` picks the behaviour/result (defaults to `name`), so a renamed tool keeps the same body. `schema="rich"` changes
    only the order_id annotation. Built with exec so both frameworks introspect an ordinary function signature.
    """
    kind = kind or name
    params = params or PARAMS.get(kind, ("order_id",))
    ann = {"plain": "str", "rich": "OrderIdRich"}[schema]
    sig = ", ".join(f"{p}: {ann if p == 'order_id' else 'str'}" for p in params)
    ns = {"OrderIdRich": OrderIdRich, "_impl": lambda a: _execute(name, kind, a, output, check)}
    exec(f"def {name}({sig}) -> str:\n    return _impl(dict(locals()))\n", ns)  # noqa: S102 - fixed template, lab-controlled names
    fn = ns[name]
    fn.__doc__ = DESC.get(kind, "") if description is None else description
    return fn


def tools(*names, **overrides):
    """Default-metadata tools in the given order. overrides: name -> dict(make_tool kwargs)."""
    return [make_tool(n, **overrides.get(n, {})) for n in names]


def distractors(k):
    out = []
    for name, param, desc in DISTRACTORS[:k]:
        DESC.setdefault(name, desc)
        RESULT.setdefault(name, f"{param} result")
        out.append(make_tool(name, desc, params=(param,)))
    return out


def tool_defs_lg(fns):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    return [convert_to_openai_tool(f) for f in fns]


# ------------------------------------------------------------------ classification (one label per run)
ALLOWED = ("CORRECT", "WRONG_TOOL", "NO_TOOL", "INVALID_TOOL", "INVALID_ARGUMENTS", "AMBIGUOUS", "VALID_ALTERNATIVE", "REFUSAL")
_REFUSAL = re.compile(r"(can(?:no|['’])t|unable to|not able to|(?:don['’]t|do not) have (?:a |the |any )?(?:tool|way|abilit|capabilit|access)|"
                      r"no (?:available )?tool|not (?:possible|supported|available)|isn['’]t (?:possible|supported|available)|"
                      r"(?:doesn['’]t|does not) (?:support|allow|include|have)|outside (?:of )?(?:my|the) )", re.I)


def classify(r, *, expect=(), alternatives=(), available=(), ambiguous=False, order_id="123"):
    """Label the FIRST selection of a run.

    expect: correct tool(s); alternatives: also-valid tools; ambiguous: the request legitimately supports several tools
    (then any expect/alternative tool is AMBIGUOUS, not wrong); order_id=None means the user gave no id, so any
    order_id the model sends was invented.
    """
    calls = r.tool_calls
    if not calls:
        return "REFUSAL" if _REFUSAL.search(r.final_text or "") else "NO_TOOL"
    c = calls[0]
    if c.name not in available:
        return "INVALID_TOOL"
    plausible = set(expect) | set(alternatives)
    if ambiguous and c.name in plausible:
        return "AMBIGUOUS"
    if c.name in expect:
        got = c.args.get("order_id") if isinstance(c.args, dict) else None
        if order_id is None:
            return "INVALID_ARGUMENTS" if got not in (None, "") else "CORRECT"
        return "CORRECT" if str(got).strip() == order_id else "INVALID_ARGUMENTS"
    if c.name in alternatives:
        return "VALID_ALTERNATIVE"
    return "WRONG_TOOL"


def asked_clarification(text):
    """Heuristic: a no-tool reply that asks the user a question."""
    return "?" in (text or "")


# ------------------------------------------------------------------ provider clients: probe, tool_choice, raw capture
def _capture_hooks(log, is_async):
    def on_request(request):
        try:
            log.append({"request": json.loads(request.content)})
        except Exception:  # noqa: BLE001 - non-JSON bodies are not chat completions
            pass

    def on_response_sync(response):
        response.read()
        if log:
            log[-1]["response"] = response.json() if response.status_code == 200 else {"status": response.status_code}

    async def on_request_async(request):
        on_request(request)

    async def on_response_async(response):
        await response.aread()
        if log:
            log[-1]["response"] = response.json() if response.status_code == 200 else {"status": response.status_code}

    if is_async:
        return {"request": [on_request_async], "response": [on_response_async]}
    return {"request": [on_request], "response": [on_response_sync]}


def lg_model(probe=False, tool_choice=None, capture=None, transport=None, name=None):
    """ChatOpenAI subclass (LangGraph side). probe: answer locally once a tool result exists (first-turn selection
    only, saves quota); tool_choice: forwarded to bind_tools; capture: list that receives raw request/response JSON."""
    import httpx
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI

    class LabChatOpenAI(ChatOpenAI):
        probe: bool = False
        forced_tool_choice: Any = None

        def bind_tools(self, tools, **kw):
            if self.forced_tool_choice is not None:
                kw["tool_choice"] = self.forced_tool_choice
            return super().bind_tools(tools, **kw)

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if self.probe and any(isinstance(m, ToolMessage) for m in messages):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(PROBE_END))])
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kw)

    if transport is not None:
        model_name, key, base = name or "openai/gpt-oss-120b", "k", "http://mock.invalid/v1"
    else:
        from lab_common.config import load_model_config
        cfg = load_model_config()
        model_name, key, base = cfg.model_name, cfg.api_key, cfg.base_url
    kw = {}
    if transport is not None or capture is not None:
        kw["http_client"] = httpx.Client(transport=transport, timeout=60,
                                         event_hooks=_capture_hooks(capture, False) if capture is not None else None)
    return LabChatOpenAI(model=model_name, api_key=key, base_url=base, temperature=0, probe=probe,
                         forced_tool_choice=tool_choice, **kw)


def pa_model(probe=False, capture=None, transport=None, name=None, temperature=0):
    """OpenAIChatModel (Pydantic AI side) at temperature 0 (same as the LangGraph side), optionally wrapped by the probe."""
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolReturnPart
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings

    if transport is not None:
        model_name, key, base = name or "openai/gpt-oss-120b", "k", "http://mock.invalid/v1"
    else:
        from lab_common.config import load_model_config
        cfg = load_model_config()
        model_name, key, base = cfg.model_name, cfg.api_key, cfg.base_url
    hc = None
    if transport is not None or capture is not None:
        hc = httpx.AsyncClient(transport=transport, timeout=60,
                               event_hooks=_capture_hooks(capture, True) if capture is not None else None)
    client = AsyncOpenAI(api_key=key, base_url=base, http_client=hc)
    settings = ModelSettings(temperature=temperature) if temperature is not None else None
    inner = OpenAIChatModel(model_name, provider=OpenAIProvider(openai_client=client), settings=settings)
    if not probe:
        return inner

    class Probe(WrapperModel):
        async def request(self, messages, model_settings, params):
            if any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(PROBE_END)], model_name=self.wrapped.model_name)
            return await self.wrapped.request(messages, model_settings, params)

    return Probe(inner)


def pa_tool_choice(choice):
    """Pydantic AI capability that sets ModelSettings.tool_choice on every step (the documented per-step route)."""
    from pydantic_ai.capabilities import AbstractCapability

    class ToolChoiceCap(AbstractCapability):
        def get_model_settings(self):
            return lambda ctx: {"tool_choice": choice}

    return ToolChoiceCap()
