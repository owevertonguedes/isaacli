#!/usr/bin/env python3
"""Prova o que a task 01 exige do gerenciador de lote.

O teste que importa e o do ORFAO: matar o `python3` da ponta e facil, mas o
`aprender.py` pode ter neto (o proprio Popen dele). Se o neto sobreviver, a
janela fecha e a chamada de rede continua gastando credito. Aqui a gente cria um
neto de proposito e confere que ele morreu junto.
"""
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import processos

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


def rodar(script, esperar=True, timeout=15):
    """Roda um script python temporario pelo gerenciador. Devolve (linhas, proc, fim)."""
    tmp = Path(tempfile.mkdtemp()) / "alvo.py"
    tmp.write_text(script)
    linhas = []
    fim = {}
    acabou = threading.Event()

    def on_linha(t):
        linhas.append(t)

    def on_fim(p, codigo):
        fim["codigo"] = codigo
        acabou.set()

    p = processos.Processo("teste", [str(tmp)], tmp.parent, on_linha, on_fim)
    p.iniciar()
    if esperar:
        acabou.wait(timeout=timeout)
    return linhas, p, fim, acabou


# --- 1) saida crua chega, e chega ANTES do processo terminar -----------------
linhas, p, fim, acabou = rodar(
    "import time\n"
    "for i in range(3):\n"
    "    print('linha', i)\n"
    "    time.sleep(0.2)\n"
)
checar(linhas == ["linha 0", "linha 1", "linha 2"], f"saida crua em ordem: {linhas}")
checar(fim.get("codigo") == 0, f"codigo de saida 0 (veio {fim.get('codigo')})")

# --- 2) stderr sai junto com stdout, na mesma corrente ----------------------
linhas, p, fim, acabou = rodar(
    "import sys\n"
    "print('saida normal')\n"
    "print('erro cru', file=sys.stderr)\n"
)
checar("erro cru" in linhas, f"stderr aparece na tela: {linhas}")

# --- 3) erro nao tratado sai INTEIRO (traceback), nao escondido -------------
linhas, p, fim, acabou = rodar("raise RuntimeError('estourou de proposito')\n")
juntas = "\n".join(linhas)
checar("estourou de proposito" in juntas, "mensagem do erro aparece")
checar("Traceback" in juntas, "traceback inteiro aparece, nao so a ultima linha")
checar(fim.get("codigo") not in (0, None), f"codigo de saida != 0 (veio {fim.get('codigo')})")

# --- 4) O TESTE QUE IMPORTA: parar mata o NETO tambem -----------------------
marcador = Path(tempfile.mkdtemp()) / "neto_vivo.txt"
linhas, p, fim, acabou = rodar(
    # pai dorme; neto fica escrevendo num arquivo. Se o neto sobreviver ao
    # parar(), o arquivo continua crescendo depois que o pai ja morreu.
    "import subprocess, sys, time\n"
    f"neto = subprocess.Popen([sys.executable, '-u', '-c',\n"
    f"    \"import time\\n\"\n"
    f"    \"while True:\\n\"\n"
    f"    \"    open(r'{marcador}','a').write('x')\\n\"\n"
    f"    \"    time.sleep(0.1)\\n\"])\n"
    "print('neto no ar', neto.pid)\n"
    "time.sleep(60)\n",
    esperar=False,
)
time.sleep(1.5)
checar(marcador.exists() and marcador.stat().st_size > 0, "o neto estava mesmo rodando")

p.parar()
acabou.wait(timeout=10)
checar(not p.vivo(), "o pai morreu depois de parar()")

tamanho_ao_parar = marcador.stat().st_size
time.sleep(1.0)
checar(marcador.stat().st_size == tamanho_ao_parar,
       f"o NETO parou junto (arquivo nao cresceu: {tamanho_ao_parar} -> {marcador.stat().st_size})")

# --- 5) matar_todos() limpa o registro global -------------------------------
linhas, p2, fim2, acabou2 = rodar("import time; time.sleep(60)\n", esperar=False)
time.sleep(0.5)
checar(p2.vivo(), "processo de longa duracao no ar")
processos.matar_todos()
acabou2.wait(timeout=10)
checar(not p2.vivo(), "matar_todos() derrubou o processo")
checar(processos._VIVOS == [], f"registro global vazio depois (_VIVOS={processos._VIVOS})")

# --- 6) contadores lidos da saida real dos scripts ---------------------------
c = {"gerados": 0, "aprovados": 0, "rejeitados": 0, "erro_api": 0}
for l in ["  [1/4] aprovado  (gemini-3.5-flash)",
          "  [2/4] REJEITADO no portao: AssertionError",
          "  [3/4] REJEITADO: nao vieram os 2 blocos",
          "  [4/4] ERRO API: HTTP Error 429",
          "dataset: /caminho/qualquer.jsonl"]:
    processos.contar_aprendizado(l, c)
checar(c == {"gerados": 4, "aprovados": 1, "rejeitados": 2, "erro_api": 1},
       f"contadores do aprendizado: {c}")

cj = {"ciclos": 0, "cumpridos": 0, "total": 0}
processos.contar_juiz("  Requisitos Cumpridos: 4/6", cj)
checar(cj == {"ciclos": 1, "cumpridos": 4, "total": 6}, f"contador do juiz: {cj}")
checar(processos.contar_juiz("linha qualquer sem numero", cj) is False,
       "linha sem contador nao mexe no placar")

cl = {"task05_save": "?", "commit_workflow": "?", "commit_literal_signature": "?",
      "intent_question": "?", "graphify_navigation": "?", "concluido": 0}
processos.contar_lora("task05_save: 1/5 -> 5/5", cl)
processos.contar_lora("commit_workflow: 3/8 -> 7/8", cl)
processos.contar_lora("commit_literal_signature: 3/7 -> 6/7", cl)
processos.contar_lora("intent_question: 3/5 -> 4/5", cl)
processos.contar_lora("graphify_navigation: 2/5 -> 5/5", cl)
processos.contar_lora("ciclo concluido. Adapter separado.", cl)
checar(cl == {
    "task05_save": "1/5 -> 5/5",
    "commit_workflow": "3/8 -> 7/8",
    "commit_literal_signature": "3/7 -> 6/7",
    "intent_question": "3/5 -> 4/5",
    "graphify_navigation": "2/5 -> 5/5",
    "concluido": 1,
}, f"contador LoRA: {cl}")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("LOTE CONTROLADO — sem orfao, saida crua, contadores certos")
