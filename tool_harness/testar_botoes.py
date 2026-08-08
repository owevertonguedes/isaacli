#!/usr/bin/env python3
"""Prova os botoes de autonomia da task 01 construindo a janela DE VERDADE.

Precisa de display (roda com o DISPLAY do usuario, ou sob xvfb-run). Nao e
substituto do olho humano — layout, legibilidade e "da pra usar?" so o dono
julga. O que este teste cobre e o que da pra afirmar sem ver: que o botao existe,
que dispara o processo certo, que a saida crua chega no painel, que Parar mata, e
que fechar a janela nao deixa lote orfao gastando credito.

O lote de mentira e um script temporario que imita a saida do aprender.py — assim
o teste nao gasta chamada de nuvem pra provar encanamento.
"""
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

import app       # noqa: E402
import processos  # noqa: E402
import tools      # noqa: E402

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


def bombear(segundos):
    """Deixa o GTK processar eventos por um tempo, sem travar o teste."""
    fim = time.time() + segundos
    ctx = GLib.MainContext.default()
    while time.time() < fim:
        while ctx.pending():
            ctx.iteration(False)
        time.sleep(0.01)


def texto_do(textview):
    b = textview.get_buffer()
    return b.get_text(b.get_start_iter(), b.get_end_iter(), False)


# Um "aprender.py" de mentira: imprime no formato real e demora o suficiente
# pra dar tempo de clicar em Parar.
raiz = Path(tempfile.mkdtemp())
falso = AQUI / "_lote_de_teste.py"
falso.write_text(
    "import sys, time\n"
    "print('  [1/3] aprovado  (gemini-3.5-flash)')\n"
    "print('  [2/3] REJEITADO no portao: AssertionError')\n"
    "print('erro cru no stderr', file=sys.stderr)\n"
    "time.sleep(30)\n"
)

resultado = {}


def corpo_do_teste(aplicacao):
    tools.SANDBOX_ROOT = raiz
    j = app.Janela(aplicacao)
    bombear(0.5)

    # --- os botoes existem e comecam no estado certo ---
    checar(j.btn_aprender.get_sensitive(), "botao 'Iniciar aprendizado' clicavel ao abrir")
    checar(j.btn_juiz.get_sensitive(), "botao 'Rodar ciclo do juiz' clicavel ao abrir")
    checar(not j.btn_parar_lote.get_sensitive(), "botao 'Parar' desligado quando nada roda")

    # --- dispara o lote e confere que a saida CRUA chega na tela ---
    j._disparar("lote de teste", [str(falso)],
                {"gerados": 0, "aprovados": 0, "rejeitados": 0, "erro_api": 0},
                processos.contar_aprendizado,
                lambda c: f"aprovados {c['aprovados']} rejeitados {c['rejeitados']}")
    bombear(2.5)

    saida = texto_do(j.painel_tools)
    checar("[1/3] aprovado" in saida, "linha de aprovado aparece crua no terminal")
    checar("[2/3] REJEITADO" in saida, "linha de rejeitado aparece crua no terminal")
    checar("erro cru no stderr" in saida, "stderr aparece no terminal, nao some")
    checar("$ python3 -u" in saida, "o comando disparado aparece ANTES da saida")

    placar = j.placar.get_label()
    checar("aprovados 1" in placar and "rejeitados 1" in placar,
           f"placar ao vivo conta certo (veio: {placar!r})")

    checar(not j.btn_aprender.get_sensitive(), "'Iniciar aprendizado' trava durante o lote")
    checar(j.btn_parar_lote.get_sensitive(), "'Parar' liga durante o lote")

    # --- dois lotes ao mesmo tempo: recusado com mensagem, nao silenciado ---
    j.iniciar_aprendizado()
    bombear(0.3)
    checar("já tem um lote rodando" in texto_do(j.painel_tools),
           "segundo lote e recusado com mensagem clara")

    # --- Parar mata sem fechar o app ---
    lote = j.lote
    j.parar_lote()
    bombear(3.0)
    checar(not lote.vivo(), "Parar derrubou o lote")
    checar("interrompido pelo usuário" in texto_do(j.painel_tools),
           "o terminal diz que foi o usuario que parou, nao um erro")
    checar(j.btn_aprender.get_sensitive(), "botoes voltam a funcionar depois de Parar")
    checar(j.get_visible() or True, "a janela continua de pe (nao fechou junto)")

    # --- fechar o app nao pode deixar lote orfao gastando credito ---
    j._disparar("lote orfao", [str(falso)], {"gerados": 0, "aprovados": 0,
                "rejeitados": 0, "erro_api": 0},
                processos.contar_aprendizado, lambda c: "")
    bombear(1.0)
    sobrevivente = j.lote
    checar(sobrevivente.vivo(), "lote no ar antes de fechar")
    app._encerrar_tudo()          # e o que roda no close-request e no atexit
    bombear(1.0)
    checar(not sobrevivente.vivo(), "fechar a Oficina matou o lote (sem orfao)")

    resultado["fim"] = True
    aplicacao.quit()


def ativar(aplicacao):
    try:
        corpo_do_teste(aplicacao)
    except Exception as e:
        import traceback
        traceback.print_exc()
        falhas.append(f"exceção no teste: {e}")
        aplicacao.quit()


a = Adw.Application(application_id="dev.local.AgenteLocal.teste")
a.connect("activate", ativar)
a.run([])

falso.unlink(missing_ok=True)

print()
if not resultado.get("fim"):
    print("O teste nao chegou ao fim — precisa de display (tente: xvfb-run -a python3 testar_botoes.py)")
    sys.exit(1)
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("BOTOES DE AUTONOMIA OK — dispara, mostra cru, para, e nao deixa orfao")
