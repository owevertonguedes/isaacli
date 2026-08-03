#!/usr/bin/env python3
"""Prova a classificacao novo/alterado do diario de arquivos (task 01).

Nao abre janela: chama os metodos da Janela com um objeto solto no lugar do
`self`. A logica que interessa aqui e de dados, nao de widget — e o widget
precisa de tela, que num teste nao tem.

Por que este teste existe: a distincao novo/alterado depende de uma FOTO tirada
no comeco da sessao. Se a foto nao for tirada (ou for tirada depois), tudo vira
"novo" e o diario passa a mentir de um jeito silencioso — do tipo que so aparece
quando ja atrapalhou.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import app
import tools

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


# Pasta de trabalho de mentira, com um arquivo que JA EXISTIA antes da sessao.
raiz = Path(tempfile.mkdtemp())
(raiz / "ja_existia.txt").write_text("velho")
(raiz / "sub").mkdir()
(raiz / "sub" / "fundo.txt").write_text("velho tambem")
(raiz / ".escondido").write_text("nao conta")
(raiz / "__pycache__").mkdir()
(raiz / "__pycache__" / "lixo.pyc").write_text("nao conta")
tools.SANDBOX_ROOT = raiz

# Um "self" de mentira com so o que estes metodos usam.
j = SimpleNamespace(tocados={}, existiam=set())
j._varrer = lambda p, d: app.Janela._varrer(j, p, d)
app.Janela.fotografar_pasta(j)

checar((raiz / "ja_existia.txt").resolve() in j.existiam, "foto pegou arquivo da raiz")
checar((raiz / "sub" / "fundo.txt").resolve() in j.existiam, "foto desce em subpasta")
checar((raiz / ".escondido").resolve() not in j.existiam, "foto ignora arquivo oculto")
checar((raiz / "__pycache__" / "lixo.pyc").resolve() not in j.existiam,
       "foto ignora pasta da lista IGNORAR")

# Agora o isaac trabalha: mexe no que existia e cria um do zero.
app.Janela.anotar_toque(j, raiz / "ja_existia.txt")
(raiz / "criado_agora.txt").write_text("novo em folha")
app.Janela.anotar_toque(j, raiz / "criado_agora.txt")

checar(j.tocados.get((raiz / "ja_existia.txt").resolve()) == "alterado",
       "arquivo que ja existia sai como ALTERADO")
checar(j.tocados.get((raiz / "criado_agora.txt").resolve()) == "novo",
       "arquivo criado na sessao sai como NOVO")
checar(len(j.tocados) == 2, f"so os tocados entram no diario (tem {len(j.tocados)})")

# Tocar de novo nao pode reclassificar: um arquivo criado agora e alterado em
# seguida continua sendo "novo" pra quem le a lista.
app.Janela.anotar_toque(j, raiz / "criado_agora.txt")
checar(j.tocados.get((raiz / "criado_agora.txt").resolve()) == "novo",
       "segundo toque nao rebaixa NOVO pra alterado")
checar(len(j.tocados) == 2, "tocar duas vezes nao duplica a linha")

# Caminho relativo e absoluto tem que cair na MESMA entrada, senao o mesmo
# arquivo apareceria duas vezes na lista.
import os
os.chdir(raiz)
app.Janela.anotar_toque(j, Path("criado_agora.txt"))
checar(len(j.tocados) == 2, f"caminho relativo nao vira linha nova (tem {len(j.tocados)})")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("DIARIO DE ARQUIVOS CORRETO — novo e alterado nao se confundem")
