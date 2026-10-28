"""Phase 3 measurement fixtures: a trajectory recorder for ONE agent run. Not a loop, not a runtime.

The frameworks run their own loops (LangGraph `graph.stream`, Pydantic AI `agent.iter`, both of which their
`invoke`/`run` use internally). This module only records what happens, from three independent vantage points:

- WIRE: every HTTP request/response between the provider client and the model (httpx event hooks on the
  existing provider clients; offline the transport is `lab2.wire`'s mock, live it is the real provider);
- TOOL: every tool body that actually executed (a functools.wraps wrapper around the shared Phase 1 tools);
- FRAMEWORK: every step the framework's own iteration API yields (LangGraph super-steps, Pydantic AI nodes).

Each event carries a perf_counter timestamp (ms since the run started) so the three views can be interleaved.
"""
import functools
import json
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    events: list = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, kind, actor, **detail):
        with self._lock:
            ev = {"t_ms": round((time.perf_counter() - self.t0) * 1000, 1), "kind": kind, "actor": actor, **detail}
            self.events.append(ev)
            return ev

    def of(self, kind):
        return [e for e in self.events if e["kind"] == kind]

    def accounting(self):
        """Counts from each vantage point. Deliberately NOT merged into one 'iteration' number."""
        responses = self.of("model_response")
        return {
            "model_requests": len(self.of("model_request")),
            "model_responses_with_tool_calls": sum(1 for e in responses if e["tool_calls"]),
            "tool_calls_requested": sum(len(e["tool_calls"]) for e in responses),
            "tool_executions": len(self.of("tool_end")),
            "framework_steps": len(self.of("framework_step")),
            "final_responses": sum(1 for e in responses if not e["tool_calls"]),
            "tokens_total": sum((e.get("usage") or {}).get("total_tokens", 0) for e in responses),
        }

    def ascii(self):
        lines = []
        for e in self.events:
            k = e["kind"]
            if k == "model_request":
                lines.append(f"{e['t_ms']:>8} ms  WIRE  -> request #{e['n']}: {e['n_messages']} messages {e['roles']}")
            elif k == "model_response":
                what = ", ".join(f"{c['name']}({c['args']})" for c in e["tool_calls"]) or f"text {e['text'][:50]!r}"
                lines.append(f"{e['t_ms']:>8} ms  WIRE  <- response #{e['n']}: {what} [finish={e['finish_reason']}]")
            elif k == "tool_start":
                lines.append(f"{e['t_ms']:>8} ms  TOOL  start {e['tool']}({e['args']})")
            elif k == "tool_end":
                lines.append(f"{e['t_ms']:>8} ms  TOOL  end   {e['tool']} -> {e.get('result', e.get('error'))!r}")
            elif k == "framework_step":
                lines.append(f"{e['t_ms']:>8} ms  FW    step {e['step']}: {e['label']}  (state: {e['state']})")
        return "\n".join(lines)


def traced(fn, traj):
    """Same function for the frameworks (functools.wraps keeps name, docstring and signature), plus two events."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        traj.add("tool_start", "tool", tool=fn.__name__, args=kwargs or list(args), thread=threading.get_ident())
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            traj.add("tool_end", "tool", tool=fn.__name__, error=f"{type(e).__name__}: {e}")
            raise
        traj.add("tool_end", "tool", tool=fn.__name__, result=out)
        return out
    return wrapper


def _hooks(traj, is_async):
    counter = {"n": 0}

    def on_request(request):
        try:
            body = json.loads(request.content)
        except Exception:  # noqa: BLE001
            return
        counter["n"] += 1
        msgs = body.get("messages", [])
        traj.add("model_request", "framework", n=counter["n"], n_messages=len(msgs), roles=[m["role"] for m in msgs],
                 messages=msgs, tools=[t["function"]["name"] for t in body.get("tools", [])],
                 temperature=body.get("temperature", "(absent)"), tool_choice=body.get("tool_choice", "(absent)"))

    def on_response(response):
        body = response.json() if response.status_code == 200 else {}
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        calls = [{"id": c["id"], "name": c["function"]["name"], "args": c["function"]["arguments"]} for c in msg.get("tool_calls") or []]
        traj.add("model_response", "model", n=counter["n"], status=response.status_code, tool_calls=calls,
                 text=msg.get("content") or "", finish_reason=choice.get("finish_reason"), usage=body.get("usage"))

    if not is_async:
        def resp(r):
            r.read()
            on_response(r)
        return {"request": [on_request], "response": [resp]}

    async def areq(r):
        on_request(r)

    async def aresp(r):
        await r.aread()
        on_response(r)
    return {"request": [areq], "response": [aresp]}


def _target(transport):
    if transport is not None:
        return "mock", "k", "http://mock.invalid/v1"
    from lab_common.config import load_model_config
    cfg = load_model_config()
    return cfg.model_name, cfg.api_key, cfg.base_url


def lg_model(traj, transport=None, temperature=0):
    """The provider client LangGraph uses in Phase 1 (`ChatOpenAI`), with wire hooks. temperature 0 = Phase 1 default."""
    import httpx
    from langchain_openai import ChatOpenAI
    name, key, base = _target(transport)
    return ChatOpenAI(model=name, api_key=key, base_url=base, temperature=temperature,
                      http_client=httpx.Client(transport=transport, timeout=60, event_hooks=_hooks(traj, False)))


def pa_model(traj, transport=None, temperature=0):
    """The provider client Pydantic AI uses in Phase 1 (`OpenAIChatModel`), with wire hooks.

    temperature 0 equalizes with LangGraph (the Phase 1 Pydantic AI model sends none; Phase 2 Batch 5 finding)."""
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings
    name, key, base = _target(transport)
    client = AsyncOpenAI(api_key=key, base_url=base,
                         http_client=httpx.AsyncClient(transport=transport, timeout=60, event_hooks=_hooks(traj, True)))
    return OpenAIChatModel(name, provider=OpenAIProvider(openai_client=client),
                           settings=ModelSettings(temperature=temperature) if temperature is not None else None)
