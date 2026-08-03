#!/usr/bin/env python3
"""Valida os N4 novos ANTES de eles entrarem na bancada (task 03).

Por que este arquivo existe: a regra de ouro do projeto e que assert errado
REPROVA MODELO CERTO. Ja aconteceu — o gabarito de `match_wildcard` estava
invertido e a bancada acusava o modelo de errar o que ele acertou. Entao todo
problema novo passa por aqui primeiro: escreve-se a solucao correta e confirma-se
que ela passa nos proprios testes.

Aqui tambem se confere o oposto, que e mais facil de esquecer: uma solucao
INGENUA (a que um modelo escreveria sem pensar direito) precisa FALHAR. Se ela
passa, o problema nao mede o que se queria — e um N2 com cara de N4, que e
exatamente o defeito que esta task veio consertar.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from n4_novos import PROBLEMAS_N4, GABARITOS, INGENUAS  # noqa: E402

falhas = []


def roda(codigo, testes):
    ns = {}
    try:
        exec(codigo, ns)
        exec(testes, ns)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


for p in PROBLEMAS_N4:
    nome = p["nome"]
    print(f"\n=== {nome} ===")

    ok, erro = roda(GABARITOS[nome], p["testes"])
    print(f"[{'ok    ' if ok else 'FALHOU'}] o gabarito passa nos proprios testes"
          + ("" if ok else f"  -> {erro}"))
    if not ok:
        falhas.append(f"{nome}: gabarito nao passa ({erro})")

    ingenua = INGENUAS.get(nome)
    if ingenua:
        ok_ing, erro_ing = roda(ingenua, p["testes"])
        print(f"[{'ok    ' if not ok_ing else 'FALHOU'}] a solucao INGENUA e reprovada"
              + (f"  -> barrada em: {erro_ing}" if not ok_ing else
                 "  -> PASSOU, o problema e facil demais"))
        if ok_ing:
            falhas.append(f"{nome}: solucao ingenua passa — problema fraco demais pra N4")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("N4 NOVOS VALIDADOS — gabarito passa, ingenua reprova")
