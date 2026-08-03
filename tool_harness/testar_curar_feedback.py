#!/usr/bin/env python3
"""Testes baratos da curadoria de feedback, sem Ollama/Gemini."""
import json
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import curar_feedback

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


def escrever_jsonl(path, eventos):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for evento in eventos:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")


raiz = Path(tempfile.mkdtemp())
sessoes = raiz / "cli_sessoes"
feedback = raiz / "feedback"
out = raiz / "curadoria"

sessao_ruim = sessoes / "sessao-ruim.jsonl"
escrever_jsonl(sessao_ruim, [
    {"tipo": "meta", "session_id": "sessao-ruim", "modelo": "isaac-granite", "workspace": "/repo"},
    {"tipo": "user", "content": "refaca a mensagem do commit com assinatura e pushe"},
    {"tipo": "tool_result", "nome": "run_command", "cmd": "git commit -m \"nova\"",
     "codigo": 1, "resultado": "no changes added to commit\n(codigo de saida: 1)"},
    {"tipo": "assistant_final", "content": "O commit foi realizado com sucesso."},
])

sessao_amend = sessoes / "sessao-amend.jsonl"
escrever_jsonl(sessao_amend, [
    {"tipo": "meta", "session_id": "sessao-amend", "modelo": "isaac-granite", "workspace": "/repo"},
    {"tipo": "user", "content": "corrija o ultimo commit para ficar assinado, sem push"},
    {"tipo": "tool_result", "nome": "run_command", "cmd": "git commit --amend -m \"Mensagem\"",
     "codigo": 0, "resultado": "[main abc] Mensagem\n(codigo de saida: 0)"},
    {"tipo": "assistant_final", "content": "O ultimo commit foi corrigido e assinado."},
])

sessao_boa = sessoes / "sessao-boa.jsonl"
escrever_jsonl(sessao_boa, [
    {"tipo": "meta", "session_id": "sessao-boa", "modelo": "isaac-granite", "workspace": "/repo"},
    {"tipo": "user", "content": "rode git status"},
    {"tipo": "tool_result", "nome": "run_command", "cmd": "git status",
     "codigo": 0, "resultado": "ok\n(codigo de saida: 0)"},
    {"tipo": "assistant_final", "content": "Rodei git status e nao ha pendencias."},
])
escrever_jsonl(feedback / "sessao-boa.jsonl", [
    {"feedback_tipo": "bom", "nota": 10, "comentario": "ficou util",
     "session_id": "sessao-boa", "session_path": str(sessao_boa)}
])

episodios = curar_feedback.montar_episodios(
    feedback_dir=feedback,
    session_dir=sessoes,
    sessoes_manuais=[str(sessao_ruim), str(sessao_amend)],
)
por_id = {e["session_id"]: e for e in episodios}

ruim = por_id["sessao-ruim"]
checar(ruim["sinais"]["comando_falhou"], "detecta comando falho")
checar(ruim["sinais"]["resposta_declara_sucesso"], "detecta sucesso falso apos falha")
checar(ruim["sinais"]["push_pedido"], "detecta pedido de push")
checar(not ruim["sinais"]["push_executado"], "detecta push nao executado")
checar(ruim["classe"] == "precisa_juiz", "git com falha fica precisa_juiz")

amend = por_id["sessao-amend"]
checar(amend["sinais"]["amend_necessario"], "detecta amend necessario")
checar(amend["sinais"]["amend_executado"], "detecta amend executado")
checar(not amend["sinais"]["assinatura_verificada"], "nao aceita assinatura sem trailer verificado")
checar(not amend["sinais"]["push_pedido"], "nao trata 'sem push' como pedido de push")
checar(not amend["sinais"]["push_nao_executado"], "nao marca push ausente quando usuario proibiu push")

boa = por_id["sessao-boa"]
checar(boa["classe"] == "precisa_juiz", "episodio com git continua precisa_juiz")

curar_feedback.escrever_saida(episodios, out)
checar((out / "episodios.jsonl").exists(), "gera episodios.jsonl")
checar((out / "precisa_juiz.jsonl").exists(), "gera precisa_juiz.jsonl")
checar("sessao-ruim" in (out / "episodios.jsonl").read_text(), "saida contem sessao manual")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("CURADORIA OK")
