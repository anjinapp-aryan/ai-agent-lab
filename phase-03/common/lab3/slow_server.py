"""A local, scripted, OpenAI-compatible chat server for timeout and cancellation experiments (standard library only).

Why it exists: `httpx.MockTransport` never enforces a timeout and has no socket to close, so it cannot show what a real
client timeout or cancellation does to an in-flight request. This server uses real sockets on 127.0.0.1.

It models a provider that keeps working after the client has gone: each request "works" for its scripted delay regardless of
the client, and the log records when the work finished and whether the peer had already closed. Measurement fixture only:
it serves canned completions, decides nothing, and is not an agent component.
"""
import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def completion_body(step, n, usage=None):
    """One chat.completion payload from a scripted step {content, calls, finish} (same shape lab2.wire / exp3_batch2 use)."""
    msg = {"role": "assistant", "content": step.get("content")}
    if step.get("calls"):
        msg["tool_calls"] = [{"id": f"call_{n}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                             for i, (name, args) in enumerate(step["calls"])]
    prompt, completion = usage or (1, 1)
    return {"id": f"cmpl-{n}", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{"index": 0, "message": msg, "finish_reason": step.get("finish") or ("tool_calls" if step.get("calls") else "stop")}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


def _peer_closed(conn):
    """True once the client has closed its end (FIN or RST). Does not consume data."""
    try:
        r, _, _ = select.select([conn], [], [], 0)
        if not r:
            return False
        return conn.recv(1, socket.MSG_PEEK) == b""
    except (ConnectionError, OSError):
        return True


class SlowServer:
    def __init__(self, script, delays=None, usages=None, statuses=None):
        """script: scripted steps (the last one repeats). delays[n-1]: seconds the n-th request 'works' before answering.
        usages: (prompt, completion) tokens per SUCCESSFUL response. statuses[n-1]: optional non-200 status for the n-th request
        (the body is a provider-style error). script and usages advance only on successful responses, so a client retry after an
        error status receives the next scripted completion, not the one after it."""
        self.script, self.delays, self.usages, self.statuses = script, delays or [], usages or [], statuses or []
        self.log, self._lock, self._n, self._ok = [], threading.Lock(), 0, 0
        outer = self

        class Handler(BaseHTTPRequestHandler):                      # HTTP/1.0: one connection per request, simplest disconnect semantics
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with outer._lock:
                    outer._n += 1
                    n = outer._n
                rec = {"n": n, "arrived": time.perf_counter(), "messages": len(body.get("messages", [])), "peer_closed_at": None,
                       "work_done_at": None, "responded": False, "delay": outer._at(outer.delays, n, 0.0), "max_tokens": body.get("max_tokens"), "max_completion_tokens": body.get("max_completion_tokens")}
                outer.log.append(rec)
                end = rec["arrived"] + rec["delay"]
                while time.perf_counter() < end and not outer._stop.is_set():
                    if rec["peer_closed_at"] is None and _peer_closed(self.connection):
                        rec["peer_closed_at"] = time.perf_counter()
                    time.sleep(0.01)
                rec["work_done_at"] = time.perf_counter()                       # the provider-side work ended, whoever is listening
                if rec["peer_closed_at"] is None and _peer_closed(self.connection):
                    rec["peer_closed_at"] = rec["work_done_at"]
                status = outer._at(outer.statuses, n, 200)
                rec["status"] = status
                if status != 200:
                    code = "context_length_exceeded" if status == 400 else "server_error"
                    payload = json.dumps({"error": {"message": f"simulated provider error {status}", "type": "invalid_request_error" if status == 400 else "server_error", "code": code}}).encode()
                else:
                    with outer._lock:
                        outer._ok += 1
                        k = outer._ok
                    step = outer.script[min(k, len(outer.script)) - 1]
                    payload = json.dumps(completion_body(step, k, outer._at(outer.usages, k, None))).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    self.wfile.flush()
                    rec["responded"] = rec["peer_closed_at"] is None
                except (ConnectionError, OSError):
                    pass

        self._stop = threading.Event()
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._srv.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self._srv.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @staticmethod
    def _at(seq, n, default):
        return seq[n - 1] if n - 1 < len(seq) and seq[n - 1] is not None else default

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._srv.shutdown()
        self._srv.server_close()
        return False

    def settle(self, timeout=3.0):
        """Wait until every logged request has finished its provider-side work (so the log is complete)."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end and any(r["work_done_at"] is None for r in self.log):
            time.sleep(0.02)
        return self.log
