#!/usr/bin/env python3
"""Checks that the reasoning configuration reaches the Ollama payload."""
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import agent


captured = []
original = agent.urllib.request.urlopen


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def fake_urlopen(req, timeout=0):
    captured.append(json.loads(req.data.decode()))
    return Response(json.dumps({
        "message": {"role": "assistant", "content": "ok"},
        "prompt_eval_count": 10,
        "eval_count": 1,
    }).encode())


try:
    agent.urllib.request.urlopen = fake_urlopen
    messages = [{"role": "user", "content": "hi"}]
    agent.call("gpt-oss", messages, use_tools=False, thinking="high", num_ctx=32768)
    agent.call("qwen", messages, use_tools=False, thinking=False)
    agent.call("raw-model", messages, use_tools=False)
finally:
    agent.urllib.request.urlopen = original


assert captured[0]["think"] == "high", "GPT-OSS has to receive the reasoning level"
assert captured[0]["options"]["num_ctx"] == 32768, "the chosen context has to reach Ollama"
assert captured[1]["think"] is False, "Qwen Instruct has to receive thinking disabled"
assert "think" not in captured[2], "a raw model must preserve Ollama's default"
assert agent._usage({"eval_duration": 500_000_000})["eval_duration"] == 500_000_000
print("AGENT CONFIG OK: thinking and context are separate and reach Ollama")

# The before-hook may return a denial/approval without running the tool again;
# this is the wiring used by the CLI's interactive prompt.
tool_calls = []
original_call = agent.call
original_execute = agent.tools.execute
try:
    agent.call = lambda *_a, **_kw: (
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "run_command", "arguments": {"cmd": "rm x"}},
        }]} if not tool_calls else {"role": "assistant", "content": "done"}
    )
    agent.tools.execute = lambda *_a: (_ for _ in ()).throw(
        AssertionError("the tool must not run twice"))
    result = agent.run(
        "test", "model", verbose=False,
        on_tool_before=lambda *_a: tool_calls.append(True) or "DENIED",
    )
    assert result["calls"][0][2] == "DENIED"
finally:
    agent.call = original_call
    agent.tools.execute = original_execute
print("AGENT APPROVAL OK: the callback can replace the tool execution")

# OpenAI-compatible adapter: endpoint/model are data, not fixed providers.
api_capture = {}


def urlopen_sse(req, timeout=0):
    api_capture["url"] = req.full_url
    api_capture["auth"] = req.headers.get("Authorization")
    api_capture["payload"] = json.loads(req.data.decode())
    return Response(
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
        b'data: [DONE]\n\n'
    )


original = agent.urllib.request.urlopen
try:
    agent.urllib.request.urlopen = urlopen_sse
    tokens = []
    msg = agent.call_stream_api(
        "free-model", [{"role": "user", "content": "hi"}],
        on_token=tokens.append, thinking="medium", api_key="test-key",
        base_url="https://api.example.test/v1",
    )
finally:
    agent.urllib.request.urlopen = original
assert api_capture["url"] == "https://api.example.test/v1/chat/completions"
assert api_capture["auth"] == "Bearer test-key"
assert api_capture["payload"]["model"] == "free-model"
assert api_capture["payload"]["reasoning_effort"] == "medium"
assert msg["content"] == "Hello" and tokens == ["Hello"]
assert msg["_usage"]["prompt_eval_count"] == 12
print("AGENT API OK: OpenAI-compatible endpoint, streaming and tools are configurable")

# Each OpenAI-compatible provider accepts a different set of reasoning_effort
# values (e.g. Groq accepts low/medium/high for some models and only
# none/default for others) and /models does not declare that in a standard way.
# call_api has to find out from the provider's own HTTP 400 rejection, with no
# hardcoded per-model table, and retry without the parameter.
reasoning_attempts = []


def urlopen_reasoning_rejected(req, timeout=0):
    payload = json.loads(req.data.decode())
    reasoning_attempts.append(payload)
    if "reasoning_effort" in payload:
        raise agent.urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(json.dumps(
                {"error": {"message": "`reasoning_effort` must be one of `none` or `default`"}}
            ).encode()),
        )
    return Response(json.dumps({
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }).encode())


original = agent.urllib.request.urlopen
try:
    agent.urllib.request.urlopen = urlopen_reasoning_rejected
    msg = agent.call_api(
        "qwen/qwen3.6-27b", [{"role": "user", "content": "hi"}], use_tools=False,
        thinking="medium", api_key="test-key", base_url="https://api.example.test/v1",
    )
finally:
    agent.urllib.request.urlopen = original
assert len(reasoning_attempts) == 2, "it has to retry without reasoning_effort"
assert "reasoning_effort" not in reasoning_attempts[1]
assert msg["content"] == "ok" and msg["_thinking_rejected"] is True
print("AGENT REASONING FALLBACK OK: the provider refuses reasoning_effort and call_api "
      "retries without the parameter")

# A tokens-per-minute limit answers 429 to a turn that is merely large and says
# how long the wait is. Turning that into an aborted turn loses work over a
# handful of seconds, so the call waits the announced window out and retries.
rate_limit_attempts = []


def urlopen_rate_limited(req, timeout=0):
    rate_limit_attempts.append(json.loads(req.data.decode()))
    if len(rate_limit_attempts) == 1:
        raise agent.urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", {},
            io.BytesIO(json.dumps({"error": {"message": (
                "Rate limit reached for model `qwen/qwen3.6-27b` on tokens per "
                "minute (TPM): Limit 8000, Used 7941, Requested 3738. Please try "
                "again in 2.5s.")}}).encode()),
        )
    return Response(json.dumps({
        "choices": [{"message": {"role": "assistant", "content": "after the wait"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }).encode())


notices = []
slept = []
original = agent.urllib.request.urlopen
original_sleep = agent.time.sleep
original_notice = agent.RATE_LIMIT_NOTICE
try:
    agent.urllib.request.urlopen = urlopen_rate_limited
    agent.time.sleep = slept.append
    agent.RATE_LIMIT_NOTICE = lambda seconds, attempt: notices.append((seconds, attempt))
    msg = agent.call_api(
        "qwen/qwen3.6-27b", [{"role": "user", "content": "hi"}], use_tools=False,
        api_key="test-key", base_url="https://api.example.test/v1",
    )
finally:
    agent.urllib.request.urlopen = original
    agent.time.sleep = original_sleep
    agent.RATE_LIMIT_NOTICE = original_notice
assert len(rate_limit_attempts) == 2, "the 429 has to be retried, not raised"
assert msg["content"] == "after the wait"
assert slept and 2.5 <= slept[0] <= 4, f"it waits the announced window: {slept}"
assert notices == [(2.5, 1)], f"the wait has to be announced on screen: {notices}"


def urlopen_rate_limited_forever(req, timeout=0):
    rate_limit_attempts.append(True)
    raise agent.urllib.error.HTTPError(
        req.full_url, 429, "Too Many Requests", {"retry-after": "1"},
        io.BytesIO(b'{"error": {"message": "Rate limit reached"}}'))


rate_limit_attempts.clear()
original = agent.urllib.request.urlopen
original_sleep = agent.time.sleep
try:
    agent.urllib.request.urlopen = urlopen_rate_limited_forever
    agent.time.sleep = lambda _s: None
    agent.call_api(
        "qwen/qwen3.6-27b", [{"role": "user", "content": "hi"}], use_tools=False,
        api_key="test-key", base_url="https://api.example.test/v1",
    )
    raise AssertionError("a limit that never lifts has to surface as an error")
except RuntimeError as e:
    assert "Rate limit reached" in str(e), f"the provider's reason must survive: {e}"
finally:
    agent.urllib.request.urlopen = original
    agent.time.sleep = original_sleep
assert len(rate_limit_attempts) == agent.RATE_LIMIT_RETRIES
print("AGENT RATE LIMIT OK: a 429 with an announced wait is retried, and a "
      "persistent limit still reports the provider's reason")

# run() has to stop sending reasoning_effort for the rest of the turn as soon as
# the provider rejects it, and signal that to the caller so it can persist the
# correction in the profile (cli._persist_adjusted_thinking).
thinking_calls = []


def fake_call_api(model, messages, use_tools=True, tools_schema=None,
                  thinking=None, api_key=None, base_url=None, temperature=0.0):
    thinking_calls.append(thinking)
    if len(thinking_calls) == 1:
        return {"role": "assistant", "content": "", "_thinking_rejected": True,
                "tool_calls": [{"id": "t1", "type": "function",
                                "function": {"name": "list_dir", "arguments": "{}"}}]}
    return {"role": "assistant", "content": "done"}


original_call_api = agent.call_api
original_execute2 = agent.tools.execute
try:
    agent.call_api = fake_call_api
    agent.tools.execute = lambda *_a: "ok"
    thinking_result = agent.run(
        "test", "qwen/qwen3.6-27b", verbose=False,
        provider={"provider": "openai_compatible", "api_key": "k", "base_url": "https://x"},
        thinking="medium",
    )
finally:
    agent.call_api = original_call_api
    agent.tools.execute = original_execute2
assert thinking_calls == ["medium", None], (
    "after the rejection, later calls in the same turn must not repeat reasoning_effort")
assert thinking_result["thinking_adjusted"] is True
print("AGENT REASONING TURN OFF OK: after a rejection the rest of the turn stops "
      "sending reasoning_effort")

# Ollama sends the reasoning in message.thinking. It only feeds the progress
# indicator: it must not leak into the visible answer or into the history.
def urlopen_ollama_stream(req, timeout=0):
    return Response(
        b'{"message":{"role":"assistant","thinking":"hidden step"}}\n'
        b'{"message":{"role":"assistant","content":"answer"}}\n'
        b'{"done":true,"eval_count":2,"eval_duration":100000000}\n'
    )


original = agent.urllib.request.urlopen
try:
    agent.urllib.request.urlopen = urlopen_ollama_stream
    thoughts, visible = [], []
    msg = agent.call_stream(
        "model", [{"role": "user", "content": "hi"}], use_tools=False,
        on_token=visible.append, on_thinking=thoughts.append,
    )
finally:
    agent.urllib.request.urlopen = original
assert thoughts == ["hidden step"]
assert visible == ["answer"] and msg["content"] == "answer"
assert "hidden step" not in json.dumps(msg, ensure_ascii=False)
print("AGENT THINKING OK: hidden progress kept separate from the answer")


# Some Qwen models end the turn with the whole answer in thinking and content
# empty. In that specific case the text has to become the visible answer.
def urlopen_ollama_thinking_only(req, timeout=0):
    return Response(
        b'{"message":{"role":"assistant","thinking":"recovered answer"}}\n'
        b'{"done":true,"eval_count":2,"eval_duration":100000000}\n'
    )


original = agent.urllib.request.urlopen
try:
    agent.urllib.request.urlopen = urlopen_ollama_thinking_only
    thoughts, visible = [], []
    msg = agent.call_stream(
        "model", [{"role": "user", "content": "hi"}], use_tools=False,
        on_token=visible.append, on_thinking=thoughts.append,
    )
finally:
    agent.urllib.request.urlopen = original
assert thoughts == ["recovered answer"]
assert visible == ["recovered answer"]
assert msg["content"] == "recovered answer"
print("AGENT THINKING FALLBACK OK: an empty answer recovers the generated text")

# An empty answer with no tool_call is a real model/provider error: it has to
# surface as-is, with no hidden attempt using another tool and no automatic
# retry. That was removed on purpose: it masked the cause and risked a loop.
schemas_seen = []
original_call = agent.call
try:
    def empty_call(*_args, tools_schema=None, **_kwargs):
        schemas_seen.append([
            item["function"]["name"] for item in (tools_schema or agent.tools.SCHEMA)
        ])
        return {"role": "assistant", "content": "", "_usage": {"eval_count": 80}}

    agent.call = empty_call
    result = agent.run("create a markdown DESIGN.md", "any-model", verbose=False)
finally:
    agent.call = original_call
assert len(schemas_seen) == 1, "an empty answer must not trigger a hidden second attempt"
assert not result["final"], "an empty answer surfaces as-is, with no invented text"
assert result["usage"]["eval_count"] == 80
print("AGENT NO HIDDEN FALLBACK OK: an empty answer surfaces as a real error, "
      "with no hidden retry")
