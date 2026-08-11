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
import math
import os
import re
import time
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


def _api_body(e):
    """Read the error body once: an HTTPError only yields it on the first read."""
    body = getattr(e, "_isaac_body", None)
    if body is None:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        try:
            e._isaac_body = body
        except AttributeError:
            pass
    return body


def _api_error(e):
    detail = _api_body(e)
    if not detail:
        return str(e)
    try:
        data = json.loads(detail)
        message = (data.get("error") or {}).get("message") or detail[:500]
    except Exception:
        message = detail[:500]
    return re.sub(r"(?i)(api[_ -]?key\s*[=:]?\s*)\S+", r"\1[hidden]", message)


# A compatible provider may answer 429 to a turn that is merely large and say
# how long the wait is. Losing the turn over a wait the provider itself measured
# in seconds is worse than waiting it out.
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_MAX_WAIT = 90.0
# Set by the CLI so the wait shows up on screen instead of looking like a freeze.
RATE_LIMIT_NOTICE = None
RATE_LIMIT_PREEMPTIVE_NOTICE = None
_RATE_LIMITS = {}


def _duration_seconds(value):
    """Parse provider reset durations such as 7.66s or 2m59.56s."""
    text = str(value or "").strip().lower()
    if not text:
        return None
    matches = list(re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)", text))
    if not matches or re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)", "", text).strip():
        return None
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(match.group(1)) * units[match.group(2)] for match in matches)


def _rate_limit_scope(payload, base_url):
    return (base_url.rstrip("/"), str(payload.get("model") or ""))


def _payload_chars(payload):
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _header_number(headers, name, integer=False):
    if not headers:
        return None
    value = headers.get(name)
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None


def _remember_rate_limit_headers(response, payload, base_url):
    """Remember generic quota headers from a successful compatible API call."""
    headers = getattr(response, "headers", None)
    remaining_tokens = _header_number(headers, "x-ratelimit-remaining-tokens", integer=True)
    remaining_requests = _header_number(headers, "x-ratelimit-remaining-requests", integer=True)
    reset_tokens = _duration_seconds(headers.get("x-ratelimit-reset-tokens")) if headers else None
    reset_requests = _duration_seconds(headers.get("x-ratelimit-reset-requests")) if headers else None
    if all(value is None for value in (
        remaining_tokens, remaining_requests, reset_tokens, reset_requests,
    )):
        return
    state = _RATE_LIMITS.setdefault(_rate_limit_scope(payload, base_url), {})
    now = time.monotonic()
    state.update({
        "remaining_tokens": remaining_tokens,
        "remaining_requests": remaining_requests,
        "reset_tokens_at": now + reset_tokens if reset_tokens is not None else None,
        "reset_requests_at": now + reset_requests if reset_requests is not None else None,
        "payload_chars": _payload_chars(payload),
    })


def _remember_rate_limit_usage(payload, base_url, usage):
    """Calibrate the next request estimate from usage reported by the provider."""
    prompt = int((usage or {}).get("prompt_tokens") or 0)
    completion = int((usage or {}).get("completion_tokens") or 0)
    if prompt <= 0:
        return
    state = _RATE_LIMITS.get(_rate_limit_scope(payload, base_url))
    if not state:
        return
    chars = _payload_chars(payload)
    state["chars_per_prompt_token"] = chars / prompt
    state["last_completion_tokens"] = completion


def _wait_for_rate_limit_capacity(payload, base_url):
    """Wait before crossing a quota advertised by the provider itself."""
    state = _RATE_LIMITS.get(_rate_limit_scope(payload, base_url))
    if not state:
        return
    delays = []
    now = time.monotonic()
    ratio = state.get("chars_per_prompt_token")
    remaining_tokens = state.get("remaining_tokens")
    if ratio and remaining_tokens is not None:
        estimated = math.ceil(_payload_chars(payload) / ratio)
        estimated += int(state.get("last_completion_tokens") or 0)
        if estimated >= remaining_tokens and state.get("reset_tokens_at") is not None:
            delays.append(state["reset_tokens_at"] - now)
    if (state.get("remaining_requests") is not None
            and state["remaining_requests"] <= 0
            and state.get("reset_requests_at") is not None):
        delays.append(state["reset_requests_at"] - now)
    delay = max(delays, default=0)
    if delay <= 0:
        return
    if delay > RATE_LIMIT_MAX_WAIT:
        return
    if RATE_LIMIT_PREEMPTIVE_NOTICE:
        RATE_LIMIT_PREEMPTIVE_NOTICE(delay)
    time.sleep(delay + 1)
    # The remembered counters describe the old window. The next response will
    # replace them with the provider's current view.
    _RATE_LIMITS.pop(_rate_limit_scope(payload, base_url), None)


def _rate_limit_wait(e):
    header = (e.headers.get("retry-after") if e.headers else None) or ""
    try:
        return float(header.strip())
    except ValueError:
        pass
    body = _api_body(e)
    seconds = re.search(r"try again in\s*([0-9.]+)\s*(ms|s)\b", body, re.I)
    if not seconds:
        return None
    value = float(seconds.group(1))
    return value / 1000 if seconds.group(2).lower() == "ms" else value


def _urlopen_api(payload, api_key, base_url, timeout=600):
    _wait_for_rate_limit_capacity(payload, base_url)
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            response = urllib.request.urlopen(
                _api_request(payload, api_key, base_url), timeout=timeout)
            _remember_rate_limit_headers(response, payload, base_url)
            return response
        except urllib.error.HTTPError as e:
            delay = _rate_limit_wait(e) if e.code == 429 else None
            if (delay is None or delay > RATE_LIMIT_MAX_WAIT
                    or attempt == RATE_LIMIT_RETRIES):
                raise
            if RATE_LIMIT_NOTICE:
                RATE_LIMIT_NOTICE(delay, attempt)
            # A whole second of margin: the announced window is the provider's
            # estimate, and coming back early spends another 429.
            time.sleep(delay + 1)


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
        with _urlopen_api(payload, api_key, base_url) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = _api_error(e)
        if payload.get("reasoning_effort") and _reasoning_effort_rejected(detail):
            payload.pop("reasoning_effort")
            thinking_rejected = True
            try:
                with _urlopen_api(payload, api_key, base_url) as r:
                    data = json.load(r)
            except urllib.error.HTTPError as e2:
                raise RuntimeError(f"API: {_api_error(e2)}") from e2
        else:
            raise RuntimeError(f"API: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API: could not connect to the endpoint: {e.reason}") from e
    msg = _normalize_msg(data["choices"][0]["message"])
    usage = data.get("usage") or {}
    _remember_rate_limit_usage(payload, base_url, usage)
    msg["_usage"] = {"prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                     "eval_count": int(usage.get("completion_tokens") or 0),
                     "total_duration": 0}
    msg["_thinking_rejected"] = thinking_rejected
    return msg


def call_stream_api(model, messages, use_tools=True, temperature=0.0,
                    on_token=None, tools_schema=None, thinking=None,
                    api_key=None, base_url=None, on_progress=None):
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
        response = _urlopen_api(payload, api_key, base_url)
    except urllib.error.HTTPError as e:
        detail = _api_error(e)
        if payload.get("reasoning_effort") and _reasoning_effort_rejected(detail):
            payload.pop("reasoning_effort")
            thinking_rejected = True
            try:
                response = _urlopen_api(payload, api_key, base_url)
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
                if on_progress:
                    on_progress(chunk)
                if on_token:
                    on_token(chunk)
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                acc = tc_acc.setdefault(i, {"id": tc.get("id") or f"tc{i}", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                f = tc.get("function") or {}
                raw_args = f.get("arguments")
                progress = (f.get("name") or "") + (
                    raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {})
                )
                if progress and on_progress:
                    on_progress(progress)
                acc["function"]["name"] += f.get("name") or ""
                acc["function"]["arguments"] += f.get("arguments") or ""
    _remember_rate_limit_usage(payload, base_url, {
        "prompt_tokens": usage["prompt_eval_count"],
        "completion_tokens": usage["eval_count"],
    })
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
                tools_schema=None, thinking=None, on_thinking=None, num_ctx=None,
                on_progress=None):
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
                if on_progress:
                    on_progress(thought)
                if on_thinking:
                    # Normally the reasoning stays hidden and only serves to
                    # update the indicator while there is no visible text yet.
                    on_thinking(thought)
            chunk = delta.get("content")
            if chunk:
                content.append(chunk)
                if on_progress:
                    on_progress(chunk)
                if on_token:
                    on_token(chunk)
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                acc = tc_acc.setdefault(i, {"id": tc.get("id") or f"tc{i}", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                f = tc.get("function") or {}
                args = f.get("arguments")
                progress = (f.get("name") or "") + (
                    args if isinstance(args, str) else json.dumps(args or {})
                )
                if progress and on_progress:
                    on_progress(progress)
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


MUTATION_RETRY = """The user asked for a change, but no changing tool has succeeded, so no change is confirmed. If the task has enough information, call exactly one appropriate tool now and continue from its result. Do not present file contents as saved unless a tool saved them. If essential information is missing, ask one concise clarification question instead."""
READ_ONLY_RESULT_NOTE = """NOTE: This tool only inspected state and changed nothing. Re-read the user's request before answering. If the requested outcome requires any persistent change, call an appropriate changing tool first; never describe an unsaved draft or a read result as a completed change."""
FAILED_CHANGE_NOTE = """NOTE: This changing tool did not succeed, so no change from this call is confirmed. Do not claim completion. Correct the call or explain the failure."""
CHANGING_TOOLS = {"write_file", "append_file", "replace_between", "replace_text"}


def run(request, model, max_steps=8, use_tools=True, verbose=True,
        on_token=None, on_tool=None, on_tool_before=None, history=None,
        tools_schema=None, thinking=None, on_working=None, provider=None,
        on_thinking=None, num_ctx=None, require_change=False,
        is_changing_tool=None, changing_tool_succeeded=None, on_progress=None):
    """on_token(chunk): text streaming.
    on_tool_before(name, args): BEFORE running. If it returns a string, that
      string replaces the tool execution (used by the CLI to approve/deny
      commands before they reach the executor).
    on_tool(name, args, result, via): afterwards, with the result.
    changing_tool_succeeded(name, args, result): confirms that a changing call
      completed successfully instead of merely being attempted.
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
    changing_calls = 0
    successful_changes = 0
    correction_pending = False
    correction_sent = False
    for step in range(max_steps):
        if on_working:
            on_working()
        provider_kind = (provider or {}).get("provider", "ollama")
        active_schema = tools_schema or tools.SCHEMA

        visible_stream = not require_change or successful_changes or correction_sent
        stream_response = bool(on_progress or (on_token and visible_stream))
        visible_token = on_token if visible_stream else None

        def query(schema):
            if provider_kind == "openai_compatible" and stream_response:
                return call_stream_api(
                    model, msgs, use_tools=use_tools, on_token=visible_token,
                    tools_schema=schema, thinking=thinking,
                    api_key=(provider or {}).get("api_key"),
                    base_url=(provider or {}).get("base_url"),
                    on_progress=on_progress)
            if provider_kind == "openai_compatible":
                return call_api(
                    model, msgs, use_tools=use_tools, tools_schema=schema,
                    thinking=thinking, api_key=(provider or {}).get("api_key"),
                    base_url=(provider or {}).get("base_url"))
            if stream_response:
                return call_stream(
                    model, msgs, use_tools=use_tools, on_token=visible_token,
                    tools_schema=schema, thinking=thinking,
                    on_thinking=on_thinking, num_ctx=num_ctx,
                    on_progress=on_progress)
            return call(model, msgs, use_tools=use_tools,
                        tools_schema=schema, thinking=thinking, num_ctx=num_ctx)

        if correction_pending:
            msgs.append({"role": "system", "content": MUTATION_RETRY})
        msg = query(active_schema)
        if correction_pending:
            msgs.pop()
            correction_pending = False
            correction_sent = True
        tc = msg.get("tool_calls")
        _add_usage(total_usage, msg.get("_usage"))
        if msg.pop("_thinking_rejected", False):
            thinking = None
            thinking_adjusted = True

        if not tc:
            if require_change and not successful_changes and not correction_sent:
                correction_pending = True
                continue
            msgs.append(msg)
            if on_token and not stream_response and msg.get("content"):
                on_token(msg["content"])
            if verbose:
                print(f"[step {step}] FINAL ANSWER:\n{msg.get('content')}")
            return {"final": msg.get("content"), "calls": calls, "steps": step,
                    "usage": total_usage, "thinking_adjusted": thinking_adjusted,
                    "changing_calls": changing_calls,
                    "successful_changes": successful_changes}

        msgs.append(msg)

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
            changing = (is_changing_tool(name, args) if is_changing_tool
                        else name in CHANGING_TOOLS)
            if changing:
                changing_calls += 1
            if on_tool:
                on_tool(name, args, result, "native")
            succeeded = False
            if changing:
                succeeded = (
                    changing_tool_succeeded(name, args, result)
                    if changing_tool_succeeded
                    else name in CHANGING_TOOLS and result.startswith("OK:")
                )
                if succeeded:
                    successful_changes += 1
            note = ("" if changing and succeeded else
                    FAILED_CHANGE_NOTE if changing else READ_ONLY_RESULT_NOTE)
            model_result = result if not note else f"{result}\n\n{note}"
            msgs.append({"role": "tool", "tool_call_id": c.get("id", name),
                         "content": model_result})

    return {"final": "(step limit reached)", "calls": calls,
            "steps": max_steps, "usage": total_usage,
            "thinking_adjusted": thinking_adjusted,
            "changing_calls": changing_calls,
            "successful_changes": successful_changes}


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
