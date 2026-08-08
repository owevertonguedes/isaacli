"""Minimal agent loop against the local Ollama.

Usage:
    python3 agent.py "write a file hello.txt containing 'hi'" [--model qwen2.5-coder:3b]

What it does, in a cycle:
  1. sends history + tool schema to the model
  2. if the model asked for a tool, actually runs it on disk (inside the sandbox)
  3. returns the result to the model and repeats, until it answers in text or
     hits the step limit
"""
import argparse
import json
import os
import re
import urllib.request
import urllib.error

import tools

# APIs behind Cloudflare (e.g. Groq) block urllib's default User-Agent
# ("Python-urllib/x.y") with HTTP 403 (error code 1010), treating it as a bot.
USER_AGENT = "isaacli/0.1"


# Honours OLLAMA_HOST, the same variable the ollama CLI reads, so pointing the
# harness at another host does not require editing code. Accepts it with or
# without a scheme, because the CLI accepts both.
def _ollama_url():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/") + "/api/chat"


URL = _ollama_url()

# --- The "snap-on piece": knowledge injected at boot, without touching weights.
TOOLS_KNOWLEDGE = """You are an assistant that operates on files through tools.

RULES:
- To read, write or list files you MUST call the matching tool.
- NEVER invent the contents of a file: read it with read_file before stating what is inside.
- write_file ERASES all previous content. To only add something at the end, use append_file.
- Before using write_file on a file that already exists, read it first with read_file.
- Call ONE tool at a time and wait for the result before the next one.
- When the task is done, reply in short text saying what you did.
- All paths are relative to the working directory. Do not use absolute paths.
"""


def _normalize_msg(msg):
    """Convert Ollama's native response into the shape the loop already uses."""
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        if not tc.get("id"):
            tc["id"] = f"call_{f.get('name', 'tool')}"
        if not isinstance(f.get("arguments"), str):
            f["arguments"] = json.dumps(f.get("arguments") or {})
    return msg


def _usage(data):
    return {
        "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
        "eval_count": int(data.get("eval_count") or 0),
        "total_duration": int(data.get("total_duration") or 0),
        "eval_duration": int(data.get("eval_duration") or 0),
    }


def _add_usage(total, item):
    for key in ("prompt_eval_count", "eval_count", "total_duration", "eval_duration"):
        total[key] = total.get(key, 0) + int((item or {}).get(key) or 0)


def _messages_for_ollama(messages):
    """Native Ollama expects tool_calls.function.arguments as an object, not a string."""
    out = json.loads(json.dumps(messages))
    for msg in out:
        for key in list(msg):
            if key.startswith("_"):
                del msg[key]
        for tc in msg.get("tool_calls") or []:
            f = tc.get("function") or {}
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    f["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    f["arguments"] = {}
    return out


def _messages_for_openai(messages):
    out = json.loads(json.dumps(messages))
    for msg in out:
        for key in list(msg):
            if key.startswith("_"):
                del msg[key]
    return out


def _api_request(payload, api_key, base_url):
    return urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                 "User-Agent": USER_AGENT},
    )


def _api_error(e):
    try:
        detail = e.read().decode("utf-8", errors="replace")
        data = json.loads(detail)
        message = (data.get("error") or {}).get("message") or detail[:500]
        return re.sub(r"(?i)(api[_ -]?key\s*[=:]?\s*)\S+", r"\1[hidden]", message)
    except Exception:
        return str(e)


def _reasoning_effort_rejected(error_text):
    # Each OpenAI-compatible provider accepts a different set of values for
    # reasoning_effort (e.g. Groq accepts low/medium/high for some models and
    # only none/default for others), and /models does not declare that in a way
    # that is standard across providers. Instead of maintaining a per-model
    # table, we treat the provider's own HTTP 400 rejection as the source of
    # truth and retry without the parameter.
    return "reasoning_effort" in error_text.lower()


def call_api(model, messages, use_tools=True, temperature=0.0,
             tools_schema=None, thinking=None, api_key=None, base_url=None):
    if not api_key:
        raise RuntimeError("API key missing; use /setup")
    if not base_url:
        raise RuntimeError("API endpoint missing; use /setup")
    payload = {"model": model, "messages": _messages_for_openai(messages),
               "temperature": temperature, "stream": False}
    if use_tools:
        payload.update({"tools": tools_schema or tools.SCHEMA,
                        "tool_choice": "auto", "parallel_tool_calls": False})
    if thinking in ("low", "medium", "high"):
        payload["reasoning_effort"] = thinking
    thinking_rejected = False
    try:
        with urllib.request.urlopen(_api_request(payload, api_key, base_url), timeout=600) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = _api_error(e)
        if payload.get("reasoning_effort") and _reasoning_effort_rejected(detail):
            payload.pop("reasoning_effort")
            thinking_rejected = True
            try:
                with urllib.request.urlopen(_api_request(payload, api_key, base_url), timeout=600) as r:
                    data = json.load(r)
            except urllib.error.HTTPError as e2:
                raise RuntimeError(f"API: {_api_error(e2)}") from e2
        else:
            raise RuntimeError(f"API: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API: could not connect to the endpoint: {e.reason}") from e
    msg = _normalize_msg(data["choices"][0]["message"])
    usage = data.get("usage") or {}
    msg["_usage"] = {"prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                     "eval_count": int(usage.get("completion_tokens") or 0),
                     "total_duration": 0}
    msg["_thinking_rejected"] = thinking_rejected
    return msg


def call_stream_api(model, messages, use_tools=True, temperature=0.0,
                    on_token=None, tools_schema=None, thinking=None,
                    api_key=None, base_url=None):
    if not api_key:
        raise RuntimeError("API key missing; use /setup")
    if not base_url:
        raise RuntimeError("API endpoint missing; use /setup")
    payload = {"model": model, "messages": _messages_for_openai(messages),
               "temperature": temperature, "stream": True,
               "stream_options": {"include_usage": True}}
    if use_tools:
        payload.update({"tools": tools_schema or tools.SCHEMA,
                        "tool_choice": "auto", "parallel_tool_calls": False})
    if thinking in ("low", "medium", "high"):
        payload["reasoning_effort"] = thinking
    content, tc_acc = [], {}
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
    thinking_rejected = False
    try:
        response = urllib.request.urlopen(_api_request(payload, api_key, base_url), timeout=600)
    except urllib.error.HTTPError as e:
        detail = _api_error(e)
        if payload.get("reasoning_effort") and _reasoning_effort_rejected(detail):
            payload.pop("reasoning_effort")
            thinking_rejected = True
            try:
                response = urllib.request.urlopen(_api_request(payload, api_key, base_url), timeout=600)
            except urllib.error.HTTPError as e2:
                raise RuntimeError(f"API: {_api_error(e2)}") from e2
        else:
            raise RuntimeError(f"API: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API: could not connect to the endpoint: {e.reason}") from e
    with response as r:
        for line in r:
            text = line.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"):
                continue
            text = text[5:].strip()
            if text == "[DONE]":
                break
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            reported = data.get("usage") or {}
            if reported:
                usage = {"prompt_eval_count": int(reported.get("prompt_tokens") or 0),
                         "eval_count": int(reported.get("completion_tokens") or 0),
                         "total_duration": 0}
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            chunk = delta.get("content")
            if chunk:
                content.append(chunk)
                if on_token:
                    on_token(chunk)
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                acc = tc_acc.setdefault(i, {"id": tc.get("id") or f"tc{i}", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                f = tc.get("function") or {}
                acc["function"]["name"] += f.get("name") or ""
                acc["function"]["arguments"] += f.get("arguments") or ""
    msg = {"role": "assistant", "content": "".join(content)}
    if tc_acc:
        msg["tool_calls"] = [tc_acc[i] for i in sorted(tc_acc)]
    msg["_usage"] = usage
    msg["_thinking_rejected"] = thinking_rejected
    return _normalize_msg(msg)


def call(model, messages, use_tools=True, temperature=0.0, tools_schema=None,
         thinking=None, num_ctx=None):
    payload = {
        "model": model,
        "messages": _messages_for_ollama(messages),
        "temperature": temperature,
        "stream": False,
    }
    if use_tools:
        payload["tools"] = tools_schema or tools.SCHEMA
    if thinking is not None:
        payload["think"] = thinking
    if num_ctx:
        payload["options"] = {"num_ctx": int(num_ctx)}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
        msg = _normalize_msg(data["message"])
        msg["_usage"] = _usage(data)
        return msg


def call_stream(model, messages, use_tools=True, temperature=0.0, on_token=None,
                tools_schema=None, thinking=None, on_thinking=None, num_ctx=None):
    """Like call(), but streaming: on_token(chunk) is called for every token.

    Returns the same assembled message call() would return, so the caller does
    not need to know whether it arrived as a stream or not.
    """
    payload = {"model": model, "messages": _messages_for_ollama(messages),
               "temperature": temperature, "stream": True}
    if use_tools:
        payload["tools"] = tools_schema or tools.SCHEMA
    if thinking is not None:
        payload["think"] = thinking
    if num_ctx:
        payload["options"] = {"num_ctx": int(num_ctx)}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    content = []
    thoughts = []
    tc_acc = {}  # index -> accumulated tool_call (whole name, arguments in chunks)
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                delta = data.get("message", {})
            except json.JSONDecodeError:
                continue
            if data.get("done"):
                usage = _usage(data)
            thought = delta.get("thinking")
            if thought:
                thoughts.append(thought)
                if on_thinking:
                    # Normally the reasoning stays hidden and only serves to
                    # update the indicator while there is no visible text yet.
                    on_thinking(thought)
            chunk = delta.get("content")
            if chunk:
                content.append(chunk)
                if on_token:
                    on_token(chunk)
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                acc = tc_acc.setdefault(i, {"id": tc.get("id") or f"tc{i}", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                f = tc.get("function") or {}
                if f.get("name"):
                    acc["function"]["name"] = f["name"]
                if f.get("arguments"):
                    args = f["arguments"]
                    acc["function"]["arguments"] += (
                        args if isinstance(args, str) else json.dumps(args))
    text = "".join(content)
    # Some Qwen models via Ollama occasionally put the whole answer in
    # message.thinking and finish with message.content empty. Without this
    # fallback the turn looks like it generated tokens, but the user sees no
    # answer at all. Do not mix reasoning and answer when content exists, and do
    # not turn it into text if the model asked for a tool.
    if not text and not tc_acc:
        text = "".join(thoughts).strip()
        if text and on_token:
            on_token(text)
    msg = {"role": "assistant", "content": text}
    if tc_acc:
        msg["tool_calls"] = [tc_acc[i] for i in sorted(tc_acc)]
    msg["_usage"] = usage
    return _normalize_msg(msg)


# REMOVED: the "regex rescue", which fished the tool call out of loose text when
# the model did not emit a real tool_call.
#
# It existed because qwen2.5-coder:3b cannot call tools. The current model calls
# NATIVELY, measured, correct on the first try. That turned the rescue into
# dead code that MASKED failure: if one day the model stops emitting tool_call,
# now we find out instead of the patch hiding it. If this is ever needed again,
# the problem is the model, not the parser.


def run(request, model, max_steps=8, use_tools=True, verbose=True,
        on_token=None, on_tool=None, on_tool_before=None, history=None,
        tools_schema=None, thinking=None, on_working=None, provider=None,
        on_thinking=None, num_ctx=None):
    """on_token(chunk): text streaming.
    on_tool_before(name, args): BEFORE running. If it returns a string, that
      string replaces the tool execution (used by the CLI to approve/deny
      commands before they reach the executor).
    on_tool(name, args, result, via): afterwards, with the result.
    All optional."""
    if history is not None:
        msgs = history
        if not msgs:
            msgs.append({"role": "system", "content": TOOLS_KNOWLEDGE})
        msgs.append({"role": "user", "content": request})
    else:
        msgs = [
            {"role": "system", "content": TOOLS_KNOWLEDGE},
            {"role": "user", "content": request},
        ]
    calls = []
    total_usage = {"prompt_eval_count": 0, "eval_count": 0,
                   "total_duration": 0, "eval_duration": 0}
    thinking_adjusted = False
    for step in range(max_steps):
        if on_working:
            on_working()
        provider_kind = (provider or {}).get("provider", "ollama")
        active_schema = tools_schema or tools.SCHEMA

        def query(schema):
            if provider_kind == "openai_compatible" and on_token:
                return call_stream_api(
                    model, msgs, use_tools=use_tools, on_token=on_token,
                    tools_schema=schema, thinking=thinking,
                    api_key=(provider or {}).get("api_key"),
                    base_url=(provider or {}).get("base_url"))
            if provider_kind == "openai_compatible":
                return call_api(
                    model, msgs, use_tools=use_tools, tools_schema=schema,
                    thinking=thinking, api_key=(provider or {}).get("api_key"),
                    base_url=(provider or {}).get("base_url"))
            if on_token:
                return call_stream(
                    model, msgs, use_tools=use_tools, on_token=on_token,
                    tools_schema=schema, thinking=thinking,
                    on_thinking=on_thinking, num_ctx=num_ctx)
            return call(model, msgs, use_tools=use_tools,
                        tools_schema=schema, thinking=thinking, num_ctx=num_ctx)

        msg = query(active_schema)
        tc = msg.get("tool_calls")
        _add_usage(total_usage, msg.get("_usage"))
        if msg.pop("_thinking_rejected", False):
            thinking = None
            thinking_adjusted = True

        msgs.append(msg)

        if not tc:
            if verbose:
                print(f"[step {step}] FINAL ANSWER:\n{msg.get('content')}")
            return {"final": msg.get("content"), "calls": calls, "steps": step,
                    "usage": total_usage, "thinking_adjusted": thinking_adjusted}

        for c in tc:
            name = c["function"]["name"]
            args = c["function"]["arguments"]
            if verbose:
                print(f"[step {step}] NATIVE TOOL_CALL -> {name}({args})")
            early_result = on_tool_before(name, args) if on_tool_before else None
            result = (early_result if isinstance(early_result, str)
                      else tools.execute(name, args))
            if verbose:
                print(f"           <- {result[:200]}")
            calls.append((name, args, result, "native"))
            if on_tool:
                on_tool(name, args, result, "native")
            msgs.append({"role": "tool", "tool_call_id": c.get("id", name), "content": result})

    return {"final": "(step limit reached)", "calls": calls,
            "steps": max_steps, "usage": total_usage,
            "thinking_adjusted": thinking_adjusted}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--no-tools", action="store_true",
                    help="do not send the schema (tests the prompt alone)")
    a = ap.parse_args()
    r = run(a.request, a.model, use_tools=not a.no_tools)
    print("\n=== SUMMARY ===")
    print(f"steps: {r['steps']}  calls: {len(r['calls'])}")
    for n, ar, res, via in r["calls"]:
        print(f"  [{via}] {n} {ar} -> {res[:80]}")
