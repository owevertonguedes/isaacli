"""Loop de agente minimo contra o Ollama local.

Uso:
    python3 agent.py "escreva um arquivo hello.txt com 'oi'" [--modelo qwen2.5-coder:3b]

O que ele faz, em ciclo:
  1. manda historico + schema das ferramentas pro modelo
  2. se o modelo pediu ferramenta, executa de verdade no disco (dentro da sandbox)
  3. devolve o resultado pro modelo e repete, ate ele responder em texto ou estourar o limite
"""
import argparse
import json
import os
import re
import sys
import urllib.request
import urllib.error

import tools

# Honours OLLAMA_HOST, the same variable the ollama CLI reads, so pointing the
# harness at another host does not require editing code. Accepts it with or
# without a scheme, because the CLI accepts both.
def _url_do_ollama():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/") + "/api/chat"


URL = _url_do_ollama()

# --- A "pecinha de montar": o conhecimento injetado no boot, sem tocar nos pesos.
CONHECIMENTO_FERRAMENTAS = """You are an assistant that operates on files through tools.

RULES:
- To read, write or list files you MUST call the matching tool.
- NEVER invent the contents of a file: read it with read_file before stating what is inside.
- write_file ERASES all previous content. To only add something at the end, use append_file.
- Before using write_file on a file that already exists, read it first with read_file.
- Call ONE tool at a time and wait for the result before the next one.
- When the task is done, reply in short text saying what you did.
- All paths are relative to the working directory. Do not use absolute paths.
"""


def _normalizar_msg(msg):
    """Converte a resposta nativa do Ollama para o formato que o loop ja usa."""
    for tc in msg.get("tool_calls") or []:
        f = tc.get("function") or {}
        if not tc.get("id"):
            tc["id"] = f"call_{f.get('name', 'tool')}"
        if not isinstance(f.get("arguments"), str):
            f["arguments"] = json.dumps(f.get("arguments") or {})
    return msg


def _uso(dado):
    return {
        "prompt_eval_count": int(dado.get("prompt_eval_count") or 0),
        "eval_count": int(dado.get("eval_count") or 0),
        "total_duration": int(dado.get("total_duration") or 0),
        "eval_duration": int(dado.get("eval_duration") or 0),
    }


def _somar_uso(total, item):
    for chave in ("prompt_eval_count", "eval_count", "total_duration", "eval_duration"):
        total[chave] = total.get(chave, 0) + int((item or {}).get(chave) or 0)


def _mensagens_para_ollama(mensagens):
    """Ollama nativo espera tool_calls.function.arguments como objeto, nao string."""
    saida = json.loads(json.dumps(mensagens))
    for msg in saida:
        for chave in list(msg):
            if chave.startswith("_"):
                del msg[chave]
        for tc in msg.get("tool_calls") or []:
            f = tc.get("function") or {}
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    f["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    f["arguments"] = {}
    return saida


def _mensagens_para_openai(mensagens):
    saida = json.loads(json.dumps(mensagens))
    for msg in saida:
        for chave in list(msg):
            if chave.startswith("_"):
                del msg[chave]
    return saida


def _api_request(payload, api_key, base_url):
    return urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )


def _api_erro(e):
    try:
        detalhe = e.read().decode("utf-8", errors="replace")
        dado = json.loads(detalhe)
        mensagem = (dado.get("error") or {}).get("message") or detalhe[:500]
        return re.sub(r"(?i)(api[_ -]?key\s*[=:]?\s*)\S+", r"\1[oculta]", mensagem)
    except Exception:
        return str(e)


def chamar_api(modelo, mensagens, usar_tools=True, temperatura=0.0,
               tools_schema=None, thinking=None, api_key=None, base_url=None):
    if not api_key:
        raise RuntimeError("chave da API ausente; use /setup")
    if not base_url:
        raise RuntimeError("endpoint da API ausente; use /setup")
    payload = {"model": modelo, "messages": _mensagens_para_openai(mensagens),
               "temperature": temperatura, "stream": False}
    if usar_tools:
        payload.update({"tools": tools_schema or tools.SCHEMA,
                        "tool_choice": "auto", "parallel_tool_calls": False})
    if thinking in ("low", "medium", "high"):
        payload["reasoning_effort"] = thinking
    try:
        with urllib.request.urlopen(_api_request(payload, api_key, base_url), timeout=600) as r:
            dado = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API: {_api_erro(e)}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API: não foi possível conectar ao endpoint — {e.reason}") from e
    msg = _normalizar_msg(dado["choices"][0]["message"])
    uso = dado.get("usage") or {}
    msg["_usage"] = {"prompt_eval_count": int(uso.get("prompt_tokens") or 0),
                     "eval_count": int(uso.get("completion_tokens") or 0),
                     "total_duration": 0}
    return msg


def chamar_stream_api(modelo, mensagens, usar_tools=True, temperatura=0.0,
                      on_token=None, tools_schema=None, thinking=None,
                      api_key=None, base_url=None):
    if not api_key:
        raise RuntimeError("chave da API ausente; use /setup")
    if not base_url:
        raise RuntimeError("endpoint da API ausente; use /setup")
    payload = {"model": modelo, "messages": _mensagens_para_openai(mensagens),
               "temperature": temperatura, "stream": True,
               "stream_options": {"include_usage": True}}
    if usar_tools:
        payload.update({"tools": tools_schema or tools.SCHEMA,
                        "tool_choice": "auto", "parallel_tool_calls": False})
    if thinking in ("low", "medium", "high"):
        payload["reasoning_effort"] = thinking
    conteudo, tc_acc = [], {}
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
    try:
        resposta = urllib.request.urlopen(_api_request(payload, api_key, base_url), timeout=600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API: {_api_erro(e)}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API: não foi possível conectar ao endpoint — {e.reason}") from e
    with resposta as r:
        for linha in r:
            texto = linha.decode("utf-8", errors="replace").strip()
            if not texto.startswith("data:"):
                continue
            texto = texto[5:].strip()
            if texto == "[DONE]":
                break
            try:
                dado = json.loads(texto)
            except json.JSONDecodeError:
                continue
            uso = dado.get("usage") or {}
            if uso:
                usage = {"prompt_eval_count": int(uso.get("prompt_tokens") or 0),
                         "eval_count": int(uso.get("completion_tokens") or 0),
                         "total_duration": 0}
            choices = dado.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            pedaco = delta.get("content")
            if pedaco:
                conteudo.append(pedaco)
                if on_token:
                    on_token(pedaco)
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                acc = tc_acc.setdefault(i, {"id": tc.get("id") or f"tc{i}", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                f = tc.get("function") or {}
                acc["function"]["name"] += f.get("name") or ""
                acc["function"]["arguments"] += f.get("arguments") or ""
    msg = {"role": "assistant", "content": "".join(conteudo)}
    if tc_acc:
        msg["tool_calls"] = [tc_acc[i] for i in sorted(tc_acc)]
    msg["_usage"] = usage
    return _normalizar_msg(msg)


def chamar(modelo, mensagens, usar_tools=True, temperatura=0.0, tools_schema=None,
           thinking=None):
    payload = {
        "model": modelo,
        "messages": _mensagens_para_ollama(mensagens),
        "temperature": temperatura,
        "stream": False,
    }
    if usar_tools:
        payload["tools"] = tools_schema or tools.SCHEMA
    if thinking is not None:
        payload["think"] = thinking
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        dado = json.load(r)
        msg = _normalizar_msg(dado["message"])
        msg["_usage"] = _uso(dado)
        return msg


def chamar_stream(modelo, mensagens, usar_tools=True, temperatura=0.0, on_token=None,
                  tools_schema=None, thinking=None):
    """Como chamar(), mas em streaming: on_token(pedaco) e chamado a cada token.

    Devolve a mesma mensagem montada que chamar() devolveria — quem chama nao
    precisa saber se veio em streaming ou nao.
    """
    payload = {"model": modelo, "messages": _mensagens_para_ollama(mensagens), "temperature": temperatura,
               "stream": True}
    if usar_tools:
        payload["tools"] = tools_schema or tools.SCHEMA
    if thinking is not None:
        payload["think"] = thinking
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    conteudo = []
    tc_acc = {}  # index -> tool_call acumulado (name inteiro, arguments em pedacos)
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
    with urllib.request.urlopen(req, timeout=600) as r:
        for linha in r:
            linha = linha.decode("utf-8", errors="replace").strip()
            if not linha:
                continue
            try:
                dado = json.loads(linha)
                delta = dado.get("message", {})
            except json.JSONDecodeError:
                continue
            if dado.get("done"):
                usage = _uso(dado)
            pedaco = delta.get("content")
            if pedaco:
                conteudo.append(pedaco)
                if on_token:
                    on_token(pedaco)
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
    msg = {"role": "assistant", "content": "".join(conteudo) or ""}
    if tc_acc:
        msg["tool_calls"] = [tc_acc[i] for i in sorted(tc_acc)]
    msg["_usage"] = usage
    return _normalizar_msg(msg)


# REMOVIDO em 2026-07-19: o "resgate por regex", que pescava a chamada de
# ferramenta do texto solto quando o modelo nao emitia tool_call de verdade.
#
# Existia porque o qwen2.5-coder:3b nao sabe chamar ferramenta. O modelo atual
# (isaac, sobre granite4:micro) chama NATIVO — medido, acertou de primeira.
# Com isso o resgate virou codigo morto que MASCARAVA falha: se um dia o modelo
# parar de emitir tool_call, agora a gente fica sabendo em vez de o remendo
# esconder. Se voltar a precisar disso, o problema e o modelo, nao o parser.
#
# Historico completo em PROGRESS.md, secao de 2026-07-19.


def rodar(pedido, modelo, max_passos=8, usar_tools=True, verbose=True,
          on_token=None, on_tool=None, on_tool_antes=None, historico=None,
          tools_schema=None, thinking=None, on_working=None, provider=None):
    """on_token(pedaco): streaming do texto.
    on_tool_antes(nome, args): ANTES de executar. Se devolver uma string, ela
      substitui a execucao da ferramenta (usado pela CLI para aprovar/recusar
      comandos antes que cheguem ao executor).
    on_tool(nome, args, resultado, via): depois, com o resultado.
    Todos opcionais."""
    if historico is not None:
        msgs = historico
        if not msgs:
            msgs.append({"role": "system", "content": CONHECIMENTO_FERRAMENTAS})
        msgs.append({"role": "user", "content": pedido})
    else:
        msgs = [
            {"role": "system", "content": CONHECIMENTO_FERRAMENTAS},
            {"role": "user", "content": pedido},
        ]
    chamadas = []
    uso_total = {"prompt_eval_count": 0, "eval_count": 0,
                 "total_duration": 0, "eval_duration": 0}
    for passo in range(max_passos):
        if on_working:
            on_working()
        provedor = (provider or {}).get("provider", "ollama")
        if provedor == "openai_compatible" and on_token:
            msg = chamar_stream_api(
                modelo, msgs, usar_tools=usar_tools, on_token=on_token,
                tools_schema=tools_schema, thinking=thinking,
                api_key=(provider or {}).get("api_key"),
                base_url=(provider or {}).get("base_url"))
        elif provedor == "openai_compatible":
            msg = chamar_api(
                modelo, msgs, usar_tools=usar_tools, tools_schema=tools_schema,
                thinking=thinking, api_key=(provider or {}).get("api_key"),
                base_url=(provider or {}).get("base_url"))
        elif on_token:
            msg = chamar_stream(
                modelo, msgs, usar_tools=usar_tools, on_token=on_token,
                tools_schema=tools_schema, thinking=thinking)
        else:
            msg = chamar(modelo, msgs, usar_tools=usar_tools,
                          tools_schema=tools_schema, thinking=thinking)
        tc = msg.get("tool_calls")
        _somar_uso(uso_total, msg.get("_usage"))

        msgs.append(msg)

        if not tc:
            if verbose:
                print(f"[passo {passo}] RESPOSTA FINAL:\n{msg.get('content')}")
            return {"final": msg.get("content"), "chamadas": chamadas, "passos": passo,
                    "uso": uso_total}

        for c in tc:
            nome = c["function"]["name"]
            args = c["function"]["arguments"]
            if verbose:
                print(f"[passo {passo}] TOOL_CALL NATIVO -> {nome}({args})")
            resultado_antecipado = on_tool_antes(nome, args) if on_tool_antes else None
            resultado = (resultado_antecipado if isinstance(resultado_antecipado, str)
                         else tools.executar(nome, args))
            if verbose:
                print(f"           <- {resultado[:200]}")
            chamadas.append((nome, args, resultado, "nativo"))
            if on_tool:
                on_tool(nome, args, resultado, "nativo")
            msgs.append({"role": "tool", "tool_call_id": c.get("id", nome), "content": resultado})

    return {"final": "(limite de passos atingido)", "chamadas": chamadas,
            "passos": max_passos, "uso": uso_total}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pedido")
    ap.add_argument("--modelo", default="qwen2.5-coder:3b")
    ap.add_argument("--sem-tools", action="store_true", help="nao manda o schema (testa so o prompt)")
    a = ap.parse_args()
    r = rodar(a.pedido, a.modelo, usar_tools=not a.sem_tools)
    print("\n=== RESUMO ===")
    print(f"passos: {r['passos']}  chamadas: {len(r['chamadas'])}")
    for n, ar, res, via in r["chamadas"]:
        print(f"  [{via}] {n} {ar} -> {res[:80]}")
