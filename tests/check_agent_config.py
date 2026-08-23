#!/usr/bin/env python3
"""Checks that the reasoning configuration reaches the Ollama payload."""
import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import agent


captured = []
original = agent.urllib.request.urlopen


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}

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
        base_url="https://api.example.test/v1", temperature=0.7, seed=21001,
    )
finally:
    agent.urllib.request.urlopen = original
assert api_capture["url"] == "https://api.example.test/v1/chat/completions"
assert api_capture["auth"] == "Bearer test-key"
assert api_capture["payload"]["model"] == "free-model"
assert api_capture["payload"]["reasoning_effort"] == "medium"
assert api_capture["payload"]["temperature"] == 0.7
assert api_capture["payload"]["seed"] == 21001
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

# Successful responses advertise the remaining quota and reset window. The
# adapter should use those provider-supplied values to pause BEFORE another
# request crosses the boundary; no provider/model limit belongs in isaacli.
preventive_attempts = []


def urlopen_quota_headers(req, timeout=0):
    preventive_attempts.append(json.loads(req.data.decode()))
    headers = ({
        "x-ratelimit-remaining-tokens": "1",
        "x-ratelimit-reset-tokens": "2.5s",
    } if len(preventive_attempts) == 1 else {})
    return Response(json.dumps({
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }).encode(), headers=headers)


preemptive_notices = []
slept = []
original = agent.urllib.request.urlopen
original_sleep = agent.time.sleep
original_preemptive_notice = agent.RATE_LIMIT_PREEMPTIVE_NOTICE
agent._RATE_LIMITS.clear()
try:
    agent.urllib.request.urlopen = urlopen_quota_headers
    agent.time.sleep = slept.append
    agent.RATE_LIMIT_PREEMPTIVE_NOTICE = preemptive_notices.append
    for _ in range(2):
        agent.call_api(
            "header-driven-model", [{"role": "user", "content": "hi"}],
            use_tools=False, api_key="test-key",
            base_url="https://headers.example.test/v1",
        )
finally:
    agent.urllib.request.urlopen = original
    agent.time.sleep = original_sleep
    agent.RATE_LIMIT_PREEMPTIVE_NOTICE = original_preemptive_notice
    agent._RATE_LIMITS.clear()
assert len(preventive_attempts) == 2, "both successful requests should complete"
assert len(slept) == 1 and 3 <= slept[0] <= 4, f"reset header drives the wait: {slept}"
assert len(preemptive_notices) == 1 and 2 <= preemptive_notices[0] <= 3
assert agent._duration_seconds("2m59.56s") == 179.56
print("AGENT PREEMPTIVE RATE LIMIT OK: successful response headers pace the next call")

# run() has to stop sending reasoning_effort for the rest of the turn as soon as
# the provider rejects it, and signal that to the caller so it can persist the
# correction in the profile (cli._persist_adjusted_thinking).
thinking_calls = []


def fake_call_api(model, messages, use_tools=True, tools_schema=None,
                  thinking=None, api_key=None, base_url=None, temperature=0.0,
                  seed=None):
    thinking_calls.append((thinking, temperature, seed))
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
        thinking="medium", temperature=0.7, seed=21001,
    )
finally:
    agent.call_api = original_call_api
    agent.tools.execute = original_execute2
assert thinking_calls == [("medium", 0.7, 21001), (None, 0.7, 21001)], (
    "after the rejection, later calls in the same turn must not repeat reasoning_effort")
assert thinking_result["thinking_adjusted"] is True
print("AGENT REASONING TURN OFF OK: after a rejection the rest of the turn stops "
      "sending reasoning_effort")

# A weak model may print a complete-looking file instead of calling write_file.
# For a mutation request, discard that uncommitted draft and give it exactly one
# explicit chance to correct itself, even after harmless read tools ran first.
mutation_messages = []
mutation_responses = [
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "read1", "type": "function",
        "function": {"name": "list_dir", "arguments": "{}"},
    }]},
    {"role": "assistant", "content": "# Design shown but not saved"},
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "write1", "type": "function",
        "function": {"name": "write_file", "arguments": json.dumps({
            "path": "design.md", "content": "# Design\n",
        })},
    }]},
    {"role": "assistant", "content": "Saved design.md"},
]


def mutation_call(_model, messages, **_kwargs):
    mutation_messages.append(json.loads(json.dumps(messages)))
    return mutation_responses.pop(0)


mutation_history = []
original_call = agent.call
original_execute = agent.tools.execute
try:
    agent.call = mutation_call
    agent.tools.execute = lambda name, _args: (
        "pages/" if name == "list_dir" else "OK: wrote design.md"
    )
    mutation_result = agent.run(
        "create design.md", "weak-local-model", verbose=False,
        history=mutation_history, require_change=True,
        is_changing_tool=lambda name, _args: name == "write_file",
    )
finally:
    agent.call = original_call
    agent.tools.execute = original_execute
assert mutation_result["final"] == "Saved design.md"
assert mutation_result["changing_calls"] == 1
assert mutation_result["successful_changes"] == 1
assert [call[0] for call in mutation_result["calls"]] == ["list_dir", "write_file"]
assert any(message.get("content") == agent.MUTATION_RETRY
           for message in mutation_messages[2])
assert all(message.get("content") != "# Design shown but not saved"
           for message in mutation_history)
assert all(message.get("content") != agent.MUTATION_RETRY
           for message in mutation_history)
print("AGENT MUTATION RETRY OK: an unsaved draft gets one explicit tool-call correction")

clarification_responses = [
    {"role": "assistant", "content": "unsaved draft"},
    {"role": "assistant", "content": "Which filename should I use?"},
]
visible_correction = []
original_call = agent.call
try:
    agent.call = lambda *_args, **_kwargs: clarification_responses.pop(0)
    clarification_result = agent.run(
        "create the file", "weak-local-model", verbose=False,
        require_change=True, on_token=visible_correction.append,
    )
finally:
    agent.call = original_call
assert clarification_result["final"] == "Which filename should I use?"
assert visible_correction == ["Which filename should I use?"]
assert not clarification_responses, "the corrective attempt must happen exactly once"
print("AGENT MUTATION CLARIFICATION OK: one failed correction stays visible and does not loop")

# Read-only results carry a language-independent reminder inside the model
# conversation. This catches uncommon phrasing without growing parallel verb
# lists in every supported UI language.
read_note_messages = []
read_note_responses = [
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "read2", "type": "function",
        "function": {"name": "list_dir", "arguments": "{}"},
    }]},
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "write3", "type": "function",
        "function": {"name": "write_file", "arguments": json.dumps({
            "path": "design.md", "content": "# Updated design\n",
        })},
    }]},
    {"role": "assistant", "content": "Saved the requested change."},
]


def read_note_call(_model, messages, **_kwargs):
    read_note_messages.append(json.loads(json.dumps(messages)))
    return read_note_responses.pop(0)


original_call = agent.call
original_execute = agent.tools.execute
try:
    agent.call = read_note_call
    agent.tools.execute = lambda name, _args: (
        "design.md" if name == "list_dir" else "OK: wrote design.md"
    )
    read_note_result = agent.run(
        "uncommon wording that names design.md", "weak-local-model", verbose=False,
    )
finally:
    agent.call = original_call
    agent.tools.execute = original_execute
assert any(agent.READ_ONLY_RESULT_NOTE in message.get("content", "")
           for message in read_note_messages[1])
assert [call[0] for call in read_note_result["calls"]] == ["list_dir", "write_file"]
assert read_note_result["successful_changes"] == 1
print("AGENT READ-ONLY REMINDER OK: uncommon mutation wording gets a generic safeguard")

# A changing call counts as a confirmed mutation only after its result says it
# succeeded; a failed write receives an explicit correction instead.
failed_change_responses = [
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "write2", "type": "function",
        "function": {"name": "write_file", "arguments": "{}"},
    }]},
    {"role": "assistant", "content": "Saved despite the error."},
    {"role": "assistant", "content": "The write failed."},
]
original_call = agent.call
original_execute = agent.tools.execute
try:
    agent.call = lambda *_args, **_kwargs: failed_change_responses.pop(0)
    agent.tools.execute = lambda *_args: "ERROR: disk full"
    failed_change_result = agent.run(
        "change the file", "weak-local-model", verbose=False,
        require_change=True,
    )
finally:
    agent.call = original_call
    agent.tools.execute = original_execute
assert failed_change_result["changing_calls"] == 1
assert failed_change_result["successful_changes"] == 0
assert not failed_change_responses
print("AGENT FAILED MUTATION OK: an attempted write is not mistaken for a saved change")

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

# Streaming progress includes hidden reasoning, visible text and tool arguments,
# while only visible answer text reaches on_token.
def urlopen_ollama_tool_progress(req, timeout=0):
    return Response(
        b'{"message":{"role":"assistant","thinking":"plan"}}\n'
        b'{"message":{"role":"assistant","tool_calls":[{"index":0,"function":{"name":"write_file","arguments":{"path":"x","content":"body"}}}]}}\n'
        b'{"done":true,"eval_count":4,"eval_duration":100000000}\n'
    )


original = agent.urllib.request.urlopen
try:
    agent.urllib.request.urlopen = urlopen_ollama_tool_progress
    progress, visible = [], []
    msg = agent.call_stream(
        "model", [{"role": "user", "content": "write"}],
        on_token=visible.append, on_progress=progress.append,
    )
finally:
    agent.urllib.request.urlopen = original
assert visible == []
assert progress[0] == "plan" and "write_file" in progress[1] and '"path": "x"' in progress[1]
assert msg["tool_calls"][0]["function"]["arguments"] == json.dumps({"path": "x", "content": "body"})
print("AGENT LIVE PROGRESS OK: tool generation updates progress without leaking content")


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
# A server the user runs has no key to demand. Requiring one here made the whole
# local-first path fail at request time, after setup had already accepted it.
# Tested by effect: the local call must fail for a different reason (nothing is
# listening on port 9), never for a missing key.
try:
    agent.call_api("m", [{"role": "user", "content": "x"}],
                   api_key="", base_url="http://127.0.0.1:9/v1")
    local_key_error = False
except RuntimeError as e:
    local_key_error = "API key missing" in str(e)
try:
    agent.call_api("m", [{"role": "user", "content": "x"}],
                   api_key="", base_url="https://api.example.com/v1")
    remote_key_error = False
except RuntimeError as e:
    remote_key_error = "API key missing" in str(e)
assert not local_key_error, "a keyless loopback endpoint must not be refused for the key"
assert remote_key_error, "a keyless remote endpoint must still be refused"

# An empty Authorization header is worse than none: some servers reject the
# malformed value outright.
assert not agent._api_request({}, "", "http://127.0.0.1:8080/v1").has_header("Authorization")
assert agent._api_request({}, "k", "https://api.example.com/v1").has_header("Authorization")
print("AGENT LOCAL KEY OK: loopback needs no API key, remote still does, no empty bearer")


print("AGENT NO HIDDEN FALLBACK OK: an empty answer surfaces as a real error, "
      "with no hidden retry")


# --- Constrained correction (task 020) ---------------------------------------
# Measured on 2026-08-20: Qwen2.5-Coder-3B emits the right call with the wrong
# wrapper (markdown fence instead of its own <tool_call> tags), so the harness
# discards it and nothing changes on disk. Constraining the decoding on the
# correction turn turned 0/6 into 6/6. These tests check the effect (a tool
# really ran, with the arguments the constrained answer carried), never the
# wording of any message.

def openai_provider():
    return {"provider": "openai_compatible", "api_key": "k",
            "base_url": "https://endpoint.example/v1"}


def openai_response(message, tokens=7):
    return Response(json.dumps({
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 20, "completion_tokens": tokens},
    }).encode())


constrained_payloads = []


def urlopen_constrained(req, timeout=0):
    payload = json.loads(req.data.decode())
    constrained_payloads.append(payload)
    if "response_format" in payload:
        constrained_number = sum(
            "response_format" in item for item in constrained_payloads)
        if constrained_number == 1:
            outcome = {"name": "write_file", "arguments": {
                "path": "index.html", "content": "<h1>hi</h1>"}}
        elif constrained_number == 2:
            outcome = {"name": "write_file", "arguments": {
                "path": "style.css", "content": "h1 { color: navy; }"}}
        return openai_response({
            "role": "assistant",
            "content": json.dumps(outcome),
        })
    if len(constrained_payloads) in (1, 3):
        # What the weak model actually does after each tool: the right call as
        # ordinary JSON content instead of a native tool_call.
        path = "index.html" if len(constrained_payloads) == 1 else "style.css"
        return openai_response({
            "role": "assistant",
            "content": json.dumps({"name": "write_file", "arguments": {
                "path": path, "content": "draft"}}),
        })
    return openai_response({"role": "assistant", "content": "Created both files."})


original_execute3 = agent.tools.execute
original_sandbox_root = agent.tools.SANDBOX_ROOT
with tempfile.TemporaryDirectory() as constrained_dir:
    try:
        agent.urllib.request.urlopen = urlopen_constrained
        agent.tools.SANDBOX_ROOT = Path(constrained_dir)
        constrained_result = agent.run(
            "create index.html and style.css", "weak-local-model", verbose=False,
            provider=openai_provider(), require_change=True,
            is_changing_tool=lambda name, _args: name == "write_file",
            temperature=0.7, seed=21001,
        )
        assert Path(constrained_dir, "index.html").read_text() == "<h1>hi</h1>"
        assert Path(constrained_dir, "style.css").read_text() == "h1 { color: navy; }"
    finally:
        agent.urllib.request.urlopen = original
        agent.tools.SANDBOX_ROOT = original_sandbox_root

assert [json.loads(call[1])["path"] for call in constrained_result["calls"]] == [
    "index.html", "style.css"], (
    "two consecutive constrained steps have to execute their distinct arguments")
assert constrained_result["successful_changes"] == 2
assert constrained_result["final"] == "Created both files."
assert [call[3] for call in constrained_result["calls"]] == [
    "constrained", "constrained"], (
    "every call obtained through the constraint must be recorded as constrained")
assert "response_format" in constrained_payloads[1], (
    "the correction turn has to carry the schema constraint")
assert "tools" not in constrained_payloads[1], (
    "offering tools and a response_format at once asks for two output shapes")
branches = constrained_payloads[1]["response_format"]["json_schema"]["schema"]["anyOf"]
assert {branch["properties"]["name"]["const"] for branch in branches} == {
    entry["function"]["name"] for entry in agent.tools.SCHEMA}, (
    "every declared tool has to be reachable through the constrained schema")
assert "tools" in constrained_payloads[0], "the first turn stays a normal tool call"
assert all("response_format" in payload and "tools" not in payload
           for payload in constrained_payloads[1:4:2])
assert all("response_format" not in payload and "tools" in payload
           for payload in constrained_payloads[2::2])
assert all(payload["temperature"] == 0.7 and payload["seed"] == 21001
           for payload in constrained_payloads), (
    "sampling parameters must survive native and constrained steps")
print("AGENT CONSTRAINED LOOP OK: two consecutive constrained calls create two "
      "files, preserve via, and stop when the model returns legitimate prose")


# A normal prose-only request never enters constrained mode and still finishes.
prose_payloads = []


def urlopen_prose(req, timeout=0):
    prose_payloads.append(json.loads(req.data.decode()))
    return openai_response({"role": "assistant", "content": "A concise explanation."})


try:
    agent.urllib.request.urlopen = urlopen_prose
    prose_result = agent.run(
        "explain what CSS does", "weak-local-model", verbose=False,
        provider=openai_provider(), require_change=False,
    )
finally:
    agent.urllib.request.urlopen = original

assert prose_result["final"] == "A concise explanation."
assert prose_result["calls"] == []
assert len(prose_payloads) == 1 and "response_format" not in prose_payloads[0]
print("AGENT LEGITIMATE PROSE OK: a non-mutation answer still ends immediately")


# Talking about JSON is not smuggling a tool call. A finished answer that
# mentions an object inline has to reach the user untouched, and must not push
# the session into another tool call it never asked for. The guard is about the
# whole answer being the object, which is what was actually measured.
inline_json_answer = 'Done. I created config.json with {"theme": "dark"} in it.'
inline_payloads = []
inline_executed = []


def urlopen_inline_json(req, timeout=0):
    inline_payloads.append(json.loads(req.data.decode()))
    if len(inline_payloads) == 1:
        return openai_response({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps(
                    {"path": "config.json", "content": "{}"})},
            }],
        })
    return openai_response({"role": "assistant", "content": inline_json_answer})


try:
    agent.urllib.request.urlopen = urlopen_inline_json
    agent.tools.execute = lambda name, args: (
        inline_executed.append(name) or "OK: wrote config.json")
    inline_result = agent.run(
        "create config.json", "weak-local-model", verbose=False,
        provider=openai_provider(), require_change=True,
        is_changing_tool=lambda name, _args: name == "write_file",
    )
finally:
    agent.urllib.request.urlopen = original
    agent.tools.execute = original_execute3

assert inline_result["final"] == inline_json_answer, (
    "an answer that merely mentions an object is a legitimate final answer")
assert inline_executed == ["write_file"], (
    "no extra tool may run because the answer talked about JSON")
assert len(inline_payloads) == 2, "no extra correction turn may be spent"
print("AGENT INLINE JSON OK: prose that mentions an object still ends the run")


# The correction used to depend on the request reading as a mutation, which was
# a list of verbs per language: "que tal demonstrar criando algo nessa pasta"
# missed it, and the call the model had already written was thrown away. What
# arms it now is the answer itself, so the same run works with require_change
# off, and it works the same in a language the verb list never had.
unasked_payloads = []
unasked_executed = []


def urlopen_unasked(req, timeout=0):
    payload = json.loads(req.data.decode())
    unasked_payloads.append(payload)
    if "response_format" in payload:
        return openai_response({
            "role": "assistant",
            "content": json.dumps({"name": "write_file", "arguments": {
                "path": "hello.py", "content": "print('Olá, mundo!')"}}),
        })
    if len(unasked_payloads) == 1:
        # Exactly what Qwen2.5-Coder-3B answered on 2026-08-23: the right call
        # in a Markdown fence instead of its own <tool_call> tags.
        return openai_response({
            "role": "assistant",
            "content": "```json\n" + json.dumps({
                "name": "write_file",
                "arguments": {"path": "hello.py",
                              "content": "print('Olá, mundo!')"},
            }) + "\n```",
        })
    return openai_response({"role": "assistant", "content": "Pronto."})


with tempfile.TemporaryDirectory() as unasked_dir:
    try:
        agent.urllib.request.urlopen = urlopen_unasked
        agent.tools.SANDBOX_ROOT = Path(unasked_dir)
        unasked_result = agent.run(
            "que tal demonstrar criando algo nessa pasta local?",
            "weak-local-model", verbose=False, provider=openai_provider(),
            require_change=False,
        )
        unasked_written = Path(unasked_dir, "hello.py").read_text()
    finally:
        agent.urllib.request.urlopen = original
        agent.tools.SANDBOX_ROOT = original_sandbox_root

assert unasked_written == "print('Olá, mundo!')", (
    "a call the model wrote in the wrong wrapper has to reach the disk without "
    "the request having to match a verb list")
assert [call[3] for call in unasked_result["calls"]] == ["constrained"], (
    "the call must come from the constrained correction, never from parsing "
    "the loose object")
assert "response_format" in unasked_payloads[1], (
    "the correction turn has to carry the schema constraint")
print("AGENT WRONG WRAPPER OK: the answer's shape arms the correction, in any "
      "language and with no mutation verb in the request")


# The other half of the same decision: an object that is not a call to a tool
# that was offered is a legitimate answer. Asking for a JSON example must not
# be turned into a tool call, which is exactly the failure a verb list in the
# request produced.
example_payloads = []


def urlopen_example(req, timeout=0):
    example_payloads.append(json.loads(req.data.decode()))
    return openai_response({
        "role": "assistant",
        "content": '{"theme": "dark", "font_size": 14}',
    })


try:
    agent.urllib.request.urlopen = urlopen_example
    example_result = agent.run(
        "me mostre um exemplo de arquivo de configuração em JSON",
        "weak-local-model", verbose=False, provider=openai_provider(),
        require_change=False,
    )
finally:
    agent.urllib.request.urlopen = original

assert example_result["final"] == '{"theme": "dark", "font_size": 14}'
assert example_result["calls"] == []
assert len(example_payloads) == 1, (
    "an answer that is an object but not a call must not spend a correction")
print("AGENT JSON EXAMPLE OK: an object that names no offered tool stays the "
      "final answer")


# Measured on 2026-08-20 against llama-server: after creating both files,
# Qwen2.5-Coder-3B answered with four call objects in a row, separated by
# newlines and wrapped in nothing. That is not one object, and it reached the
# screen as the final answer. The guard is about objects accounting for the
# whole answer, so this shape has to be caught too.
sequence_answer = "\n".join(json.dumps(
    {"name": "run_command", "arguments": {"cmd": command}})
    for command in ("git add index.html", "git add style.css", "git commit"))
sequence_payloads = []
sequence_tokens = []


def urlopen_sequence(req, timeout=0):
    payload = json.loads(req.data.decode())
    sequence_payloads.append(payload)
    if "response_format" in payload:
        return openai_response({
            "role": "assistant",
            "content": json.dumps({"name": "write_file", "arguments": {
                "path": "second.txt", "content": "second"}}),
        })
    if len(sequence_payloads) == 1:
        return openai_response({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps(
                    {"path": "first.txt", "content": "first"})},
            }],
        })
    if len(sequence_payloads) == 2:
        return openai_response({"role": "assistant", "content": sequence_answer})
    return openai_response({"role": "assistant", "content": "Both files are saved."})


sequence_executed = []
try:
    agent.urllib.request.urlopen = urlopen_sequence
    agent.tools.execute = lambda name, args: (
        sequence_executed.append((name, json.loads(args)["path"])) or "OK: wrote it")
    sequence_result = agent.run(
        "create first.txt and second.txt", "weak-local-model", verbose=False,
        provider=openai_provider(), require_change=True,
        is_changing_tool=lambda name, _args: name == "write_file",
        on_token=sequence_tokens.append,
    )
finally:
    agent.urllib.request.urlopen = original
    agent.tools.execute = original_execute3

assert sequence_result["final"] == "Both files are saved.", (
    "a run of call objects must never be the answer the user reads")
assert sequence_answer not in "".join(sequence_tokens), (
    "the object sequence must not reach the terminal at all")
assert sequence_executed == [("write_file", "first.txt"),
                             ("write_file", "second.txt")], (
    "the withheld sequence has to become one real constrained call")
print("AGENT JSON SEQUENCE OK: several objects in a row are withheld and "
      "re-decoded under the schema")


# A provider that rejects response_format must not silently lose the correction:
# the retry still goes out the way it always did, and the failure stays visible.
unsupported_payloads = []
unsupported_executed = []


def urlopen_unsupported(req, timeout=0):
    payload = json.loads(req.data.decode())
    unsupported_payloads.append(payload)
    if "response_format" in payload:
        raise agent.urllib.error.HTTPError(
            "https://endpoint.example/v1/chat/completions", 400, "Bad Request", {},
            io.BytesIO(json.dumps(
                {"error": {"message": "response_format is not supported"}}).encode()))
    return openai_response({"role": "assistant", "content": "still just prose"})


try:
    agent.urllib.request.urlopen = urlopen_unsupported
    agent.tools.execute = lambda name, args: unsupported_executed.append(name)
    unsupported_result = agent.run(
        "create index.html", "weak-local-model", verbose=False,
        provider=openai_provider(), require_change=True,
        is_changing_tool=lambda name, _args: name == "write_file",
    )
finally:
    agent.urllib.request.urlopen = original
    agent.tools.execute = original_execute3

assert unsupported_executed == [], "no tool ran, so none may be reported as run"
assert unsupported_result["successful_changes"] == 0
assert unsupported_result["final"] == "still just prose", (
    "the model's own failed answer has to reach the user, not a swallowed error")
assert "tools" in unsupported_payloads[-1] and "response_format" not in unsupported_payloads[-1], (
    "after the rejection the correction has to go out unconstrained, as before")
print("AGENT CONSTRAINED RETRY UNSUPPORTED OK: a provider without json_schema keeps "
      "today's behaviour and today's visible failure")


# If that old unconstrained path returns a JSON object, it must not become a
# final answer and must not be streamed before validation. It is not parsed as
# a tool dialect and no tool is executed.
json_text_payloads = []
json_text_tokens = []
json_text_executed = []


def urlopen_json_text(req, timeout=0):
    payload = json.loads(req.data.decode())
    json_text_payloads.append(payload)
    if "response_format" in payload:
        raise agent.urllib.error.HTTPError(
            "https://endpoint.example/v1/chat/completions", 400, "Bad Request", {},
            io.BytesIO(json.dumps(
                {"error": {"message": "response_format is not supported"}}).encode()))
    if len(json_text_payloads) == 1:
        return openai_response({"role": "assistant", "content": "draft"})
    leaked = [
        {"name": "write_file", "arguments": {"path": "leak.txt", "content": "x"}},
        {"name": "write_file", "arguments": {"path": "leak2.txt", "content": "y"}},
    ]
    return openai_response({"role": "assistant", "content": "```json\n" + "\n".join(
        json.dumps(item) for item in leaked) + "\n```"})


try:
    agent.urllib.request.urlopen = urlopen_json_text
    agent.tools.execute = lambda name, args: json_text_executed.append((name, args))
    try:
        agent.run(
            "create leak.txt", "weak-local-model", verbose=False,
            provider=openai_provider(), require_change=True,
            is_changing_tool=lambda name, _args: name == "write_file",
            on_token=json_text_tokens.append,
        )
        json_text_error = None
    except agent.ConstrainedOutputError as error:
        json_text_error = error
finally:
    agent.urllib.request.urlopen = original
    agent.tools.execute = original_execute3

assert json_text_error is not None and json_text_error.reason == "json_as_text"
assert json_text_executed == []
assert json_text_tokens == [], "a JSON object must be rejected before reaching the UI"
print("AGENT JSON TEXT GUARD OK: JSON cannot become a final answer or execute as a call")


# An endpoint may accept response_format and still violate it. That is a hard,
# visible failure, not a reason to send another unconstrained request.
invalid_constrained_payloads = []


def urlopen_invalid_constrained(req, timeout=0):
    payload = json.loads(req.data.decode())
    invalid_constrained_payloads.append(payload)
    if "response_format" in payload:
        return openai_response({"role": "assistant", "content": "{}"})
    return openai_response({"role": "assistant", "content": "draft"})


try:
    agent.urllib.request.urlopen = urlopen_invalid_constrained
    try:
        agent.run(
            "create invalid.txt", "weak-local-model", verbose=False,
            provider=openai_provider(), require_change=True,
        )
        invalid_constrained_error = None
    except agent.ConstrainedOutputError as error:
        invalid_constrained_error = error
finally:
    agent.urllib.request.urlopen = original

assert invalid_constrained_error is not None
assert invalid_constrained_error.reason == "invalid_response"
assert len(invalid_constrained_payloads) == 2
print("AGENT INVALID CONSTRAINED OUTPUT OK: accepted but unusable schema output raises")


# The conversion refuses anything that is not a call to a declared tool. The
# constraint is supposed to make this impossible; if it ever happens, the
# harness must not invent a call out of it.
assert agent.message_from_constrained("not json", agent.tools.SCHEMA) is None
assert agent.message_from_constrained(
    json.dumps({"name": "rm_rf", "arguments": {}}), agent.tools.SCHEMA) is None
assert agent.message_from_constrained(
    json.dumps({"name": "write_file", "arguments": "a string"}),
    agent.tools.SCHEMA) is None
accepted = agent.message_from_constrained(
    json.dumps({"name": "read_file", "arguments": {"path": "a.py"}}),
    agent.tools.SCHEMA)
assert accepted["tool_calls"][0]["function"]["name"] == "read_file"
assert json.loads(accepted["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
print("AGENT CONSTRAINED PARSE OK: only a call to a declared tool is accepted")


# --- debug.note --------------------------------------------------------------
# Diagnostic detail belongs to --debug, never to the screen. Tested by effect on
# stderr, not by reading the message text.
import contextlib  # noqa: E402

import debug  # noqa: E402

quiet = io.StringIO()
debug.enable(False)
with contextlib.redirect_stderr(quiet):
    debug.note("test.site", "should stay invisible")
assert quiet.getvalue() == "", "with debug off a note must print nothing at all"

loud = io.StringIO()
debug.enable(True)
try:
    with contextlib.redirect_stderr(loud):
        debug.note("test.site", "first")
        debug.note("test.site", "second from the same site")
        debug.note("other.site", "from another site")
finally:
    debug.enable(False)
printed = loud.getvalue()
assert printed.count("test.site") == 1, (
    "a site reports once per run, so a note inside a loop cannot bury the terminal")
assert "second from the same site" not in printed
assert "other.site" in printed, "a different site still reports"
print("DEBUG NOTE OK: silent by default, once per site when enabled")
