#!/usr/bin/env python3
"""Valida o dataset curado de commit assinado do Isaac."""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATASET = RAIZ / "datasets" / "commit-assinatura-isaac-2026-07-20.jsonl"
TRAILER_RE = re.compile(
    r"^(Co-Authored-By|Signed-off-by): Isaac <[^<>@\s]+@[^<>@\s]+>$"
)

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


linhas = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]

checar(len(linhas) >= 4, "dataset tem pelo menos 4 exemplos")
ids = [l["id"] for l in linhas]
checar(len(ids) == len(set(ids)), "ids sao unicos")

positivos = [l for l in linhas if l.get("tipo") == "positivo"]
negativos = [l for l in linhas if l.get("tipo") == "negativo"]
checar(bool(positivos), "ha exemplos positivos")
checar(bool(negativos), "ha exemplos negativos")

for item in positivos:
    esperado = item.get("mensagem_esperada", {})
    comandos = "\n".join(item.get("comandos_esperados", []))
    checar(TRAILER_RE.match(esperado.get("trailer", "")),
           f"{item['id']} tem trailer Git verificavel")
    checar(esperado.get("trailer", "") not in esperado.get("titulo", ""),
           f"{item['id']} nao coloca trailer no titulo")
    checar("git log -1 --format=%B" in comandos,
           f"{item['id']} verifica mensagem depois do commit")
    checar("git push" not in comandos,
           f"{item['id']} nao inclui push em pedido sem push")

for item in negativos:
    ruim = "\n".join(item.get("comportamento_ruim", []))
    bom = "\n".join(item.get("comportamento_esperado", []))
    checar(bool(ruim and bom), f"{item['id']} tem ruim e esperado")
    checar("Co-Authored-By:" in bom or "Signed-off-by:" in bom,
           f"{item['id']} aponta algum formato de trailer valido")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("DATASET COMMIT ASSINATURA OK")
