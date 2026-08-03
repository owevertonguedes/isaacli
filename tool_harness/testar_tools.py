#!/usr/bin/env python3
"""Testes baratos das ferramentas locais."""
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import tools

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


raiz = Path(tempfile.mkdtemp())
tools.SANDBOX_ROOT = raiz

html = """<section>
  <!-- ISAAC_FORM_START -->
  <!-- ISAAC_FORM_END -->
</section>
<script>
  function render() {
    /* ISAAC_RENDER_START */
    /* ISAAC_RENDER_END */
  }
  function apagar() {
    /* ISAAC_DELETE_START */
    /* ISAAC_DELETE_END */
  }
  function salvar() {
    /* ISAAC_SAVE_START */
    /* ISAAC_SAVE_END */
  }
</script>
"""
(raiz / "a.html").write_text(html)

saida = tools.replace_between(
    "a.html",
    "ISAAC_FORM_START",
    "ISAAC_FORM_END",
    "<form id=\"f\"></form>",
)
texto = (raiz / "a.html").read_text()
checar(saida.startswith("OK:"), "replace_between retorna OK")
checar("<!-- ISAAC_FORM_START -->\n<form id=\"f\"></form>\n  <!-- ISAAC_FORM_END -->" in texto,
       "HTML entra fora do comentario dos marcadores")

tools.replace_between(
    "a.html",
    "ISAAC_RENDER_START",
    "ISAAC_RENDER_END",
    "document.body.dataset.ok = '1';",
)
texto = (raiz / "a.html").read_text()
checar("/* ISAAC_RENDER_START */\ndocument.body.dataset.ok = '1';\n    /* ISAAC_RENDER_END */" in texto,
       "JS entra fora do comentario dos marcadores")

saida = tools.replace_between(
    "a.html",
    "ISAAC_FORM_START",
    "ISAAC_RENDER_END",
    "<p>nao pode atravessar secoes</p>",
)
checar(saida.startswith("ERRO: marcadores incompat"), "replace_between recusa marcador cruzado")
checar("nao pode atravessar secoes" not in (raiz / "a.html").read_text(),
       "marcador cruzado nao altera o arquivo")

saida = tools.replace_between(
    "a.html",
    "ISAAC_DELETE_START",
    "ISAAC_DELETE_END",
    "<button>nao pode</button>",
)
checar(saida.startswith("ERRO: este marcador fica dentro de <script>"),
       "replace_between recusa HTML dentro de marcador JS de acao")

saida = tools.replace_between(
    "a.html",
    "ISAAC_DELETE_START",
    "ISAAC_DELETE_END",
    "if (index < 0 || index >= itens.length) return;",
)
checar(saida.startswith("OK:"), "replace_between permite comparacao menor-que em JS")

saida = tools.replace_between(
    "a.html",
    "ISAAC_DELETE_START",
    "ISAAC_DELETE_END",
    "function deleteTransaction(index) { return index; }",
)
checar(saida.startswith("ERRO: escreva apenas o miolo"),
       "replace_between recusa redeclarar funcao que envolve marcador")

saida = tools.replace_between(
    "a.html",
    "ISAAC_SAVE_START",
    "ISAAC_SAVE_END",
    "<p>nao pode</p>",
)
checar(saida.startswith("ERRO: este marcador fica dentro de <script>"),
       "replace_between recusa HTML dentro de marcador JS de persistencia")

saida = tools.replace_between(
    "a.html",
    "/* ISAAC_RENDER_START */",
    "/* ISAAC_RENDER_END */",
    "document.body.dataset.commentMarker = '1';",
)
checar(saida.startswith("OK:"), "replace_between aceita marcador embrulhado em comentario")

saida = tools.replace_between(
    "a.html",
    "ISAAC_RENDER_START",
    "ISAAC_RENDER_END",
    "// JavaScript aqui",
)
checar(saida.startswith("ERRO: o trecho ainda contem placeholder"),
       "replace_between recusa placeholder antes de gravar")

saida = tools.replace_between(
    "a.html",
    "ISAAC_RENDER_START",
    "ISAAC_RENDER_END",
    "function render() { return true; }",
)
checar(saida.startswith("ERRO: este marcador fica dentro de render"),
       "replace_between recusa funcao dentro de marcador de render")

saida = tools.replace_between(
    "a.html",
    "ISAAC_FORM_START",
    "ISAAC_FORM_END",
    "<!-- ISAAC_FORM_START -->\n<input id=\"limpo\">\n/* ISAAC_RENDER_END */\n<!-- ISAAC_FORM_END -->",
)
texto = (raiz / "a.html").read_text()
checar(texto.count("ISAAC_FORM_START") == 1 and texto.count("ISAAC_FORM_END") == 1,
       "replace_between remove marcadores repetidos emitidos pelo modelo")
checar(texto.count("ISAAC_RENDER_END") == 1,
       "replace_between remove marcador de outra secao emitido no conteudo")

saida = tools.replace_between(
    "a.html",
    "ISAAC_RENDER_START",
    "ISAAC_RENDER_END",
    "document.body.innerHTML = '<main>ok</main>';",
)
checar(saida.startswith("OK:"), "replace_between permite HTML em string no render")

saida = tools.replace_between(
    "a.html",
    "ISAAC_RENDER_START",
    "ISAAC_RENDER_END",
    "</script><p>quebra</p>",
)
checar(saida.startswith("ERRO: o trecho tentou alterar a estrutura principal"),
       "replace_between recusa fechamento de script")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("TOOLS OK")
