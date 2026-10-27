"""Wire-level capture: run the REAL provider clients (ChatOpenAI / OpenAIChatModel) against a mock HTTP transport.

Replays a script of canned OpenAI-style chat completions and records every request body the framework sends,
so we see exactly what would go on the wire (tool schemas, serialized tool results) without any API call.
Reuses httpx.MockTransport and the existing provider clients; no new dependency.
"""
import json

import httpx


def _completion(step, n):
    kind, payload = step
    if kind == "text":
        msg, finish = {"role": "assistant", "content": payload}, "stop"
    else:
        calls = [{"id": f"call_{n}_{i}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(payload)]
        msg, finish = {"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls"
    return {"id": f"cmpl-{n}", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def transport(script):
    """Returns (MockTransport, requests_log). Script steps: ("text", s) / ("call", [(name,args)])."""
    log = []

    def handler(request: httpx.Request):
        log.append(json.loads(request.content))
        step = script[min(len(log) - 1, len(script) - 1)]
        return httpx.Response(200, json=_completion(step, len(log)))

    return httpx.MockTransport(handler), log


def lg_wire_model(script):
    from langchain_openai import ChatOpenAI
    t, log = transport(script)
    m = ChatOpenAI(model="mock", api_key="k", base_url="http://mock.invalid/v1",
                   http_client=httpx.Client(transport=t), http_async_client=httpx.AsyncClient(transport=t))
    return m, log


def pa_wire_model(script):
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    t, log = transport(script)
    client = AsyncOpenAI(api_key="k", base_url="http://mock.invalid/v1", http_client=httpx.AsyncClient(transport=t))
    return OpenAIChatModel("mock", provider=OpenAIProvider(openai_client=client)), log
