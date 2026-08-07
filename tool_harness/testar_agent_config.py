#!/usr/bin/env python3
"""Confere que a configuracao de raciocinio chega ao payload Ollama."""
import io
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import agent


capturados = []
original = agent.urllib.request.urlopen


class Resposta(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def urlopen_fake(req, timeout=0):
    capturados.append(json.loads(req.data.decode()))
    return Resposta(json.dumps({
        "message": {"role": "assistant", "content": "ok"},
        "prompt_eval_count": 10,
        "eval_count": 1,
    }).encode())


try:
    agent.urllib.request.urlopen = urlopen_fake
    mensagens = [{"role": "user", "content": "oi"}]
    agent.chamar("gpt-oss", mensagens, usar_tools=False, thinking="high")
    agent.chamar("qwen", mensagens, usar_tools=False, thinking=False)
    agent.chamar("modelo-cru", mensagens, usar_tools=False)
finally:
    agent.urllib.request.urlopen = original


assert capturados[0]["think"] == "high", "GPT-OSS precisa receber nivel de raciocinio"
assert capturados[1]["think"] is False, "Qwen Instruct precisa receber thinking desativado"
assert "think" not in capturados[2], "modelo cru deve preservar o padrao do Ollama"
assert agent._uso({"eval_duration": 500_000_000})["eval_duration"] == 500_000_000
print("AGENT CONFIG OK — thinking separado e enviado ao Ollama")

# O gancho anterior pode entregar uma recusa/aprovação sem executar a tool de
# novo; esta é a ligação usada pelo prompt interativo da CLI.
chamadas_tool = []
chamar_original = agent.chamar
executar_original = agent.tools.executar
try:
    agent.chamar = lambda *_a, **_kw: (
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "t1", "type": "function",
            "function": {"name": "run_command", "arguments": {"cmd": "rm x"}},
        }]} if not chamadas_tool else {"role": "assistant", "content": "fim"}
    )
    agent.tools.executar = lambda *_a: (_ for _ in ()).throw(
        AssertionError("a ferramenta não deve executar duas vezes"))
    resultado = agent.rodar(
        "teste", "modelo", verbose=False,
        on_tool_antes=lambda *_a: chamadas_tool.append(True) or "RECUSADO",
    )
    assert resultado["chamadas"][0][2] == "RECUSADO"
finally:
    agent.chamar = chamar_original
    agent.tools.executar = executar_original
print("AGENT APPROVAL OK — callback pode substituir execução da ferramenta")

# Adaptador OpenAI-compatible: endpoint/modelo são dados, não provedores fixos.
captura_api = {}
def urlopen_sse(req, timeout=0):
    captura_api["url"] = req.full_url
    captura_api["auth"] = req.headers.get("Authorization")
    captura_api["payload"] = json.loads(req.data.decode())
    return Resposta(
        b'data: {"choices":[{"delta":{"content":"Ola"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n'
        b'data: [DONE]\n\n'
    )
original = agent.urllib.request.urlopen
try:
    agent.urllib.request.urlopen = urlopen_sse
    tokens = []
    msg = agent.chamar_stream_api(
        "modelo-livre", [{"role": "user", "content": "oi"}],
        on_token=tokens.append, thinking="medium", api_key="chave-teste",
        base_url="https://api.exemplo.test/v1",
    )
finally:
    agent.urllib.request.urlopen = original
assert captura_api["url"] == "https://api.exemplo.test/v1/chat/completions"
assert captura_api["auth"] == "Bearer chave-teste"
assert captura_api["payload"]["model"] == "modelo-livre"
assert captura_api["payload"]["reasoning_effort"] == "medium"
assert msg["content"] == "Ola" and tokens == ["Ola"]
assert msg["_usage"]["prompt_eval_count"] == 12
print("AGENT API OK — endpoint OpenAI-compatible, streaming e tools configuráveis")
