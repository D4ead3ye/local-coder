"""Streaming chat-completions client for llama-server, stdlib only.

One call streams one assistant message: visible text, reasoning and tool
calls arrive as deltas and are folded back into a single message dict in the
OpenAI shape, ready to append to the history.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

# Qwen's recommended sampling for thinking mode on coding work
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


class Cancelled(Exception):
    pass


def stream_chat(base: str, messages, tools, on_delta, cancel, model="local",
                max_tokens=16384, sampling=None):
    """`on_delta(kind, text)` with kind in {"text", "think", "tool", "tool_name"}.
    `cancel` is a threading.Event; setting it aborts the read.

    `sampling` overrides the default coding sampling when a caller wants more
    variety. Returns (message, stats) where stats has prompt/predicted token
    counts and generation speed when the server reports them."""
    body = {"model": model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, **(sampling or SAMPLING)}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")

    content, reasoning = [], []
    calls: dict[int, dict] = {}
    stats = {}
    finish = None
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"servidor recusou ({e.code}): {e.read()[:600].decode(errors='replace')}")
    with resp:
        for raw in resp:
            if cancel.is_set():
                raise Cancelled()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if "timings" in chunk:
                tm = chunk["timings"]
                stats = {"prompt_n": tm.get("prompt_n", 0),
                         "cache_n": tm.get("cache_n", 0),
                         "predicted_n": tm.get("predicted_n", 0),
                         "tps": tm.get("predicted_per_second", 0.0),
                         "pp_tps": tm.get("prompt_per_second", 0.0)}
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                    on_delta("think", d["reasoning_content"])
                if d.get("content"):
                    content.append(d["content"])
                    on_delta("text", d["content"])
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0),
                                            {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                        on_delta("tool_name", fn["name"])
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                        on_delta("tool", fn["arguments"])
                finish = ch.get("finish_reason") or finish

    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [
            {"id": c["id"] or f"call_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
            for i, c in sorted(calls.items())]
    stats["finish"] = finish
    return msg, stats
