#!/usr/bin/env python3
"""Testes baratos do CLI do Isaac, sem chamar Ollama."""
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import isaac_cli
import tools

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


raiz = Path(tempfile.mkdtemp())
sub = raiz / "projeto"
sub.mkdir()

cli = isaac_cli.IsaacCLI("isaac", sub, 4, autostart_ollama=False)
checar(tools.SANDBOX_ROOT == sub.resolve(), "workspace inicial vira SANDBOX_ROOT")
checar(str(sub.resolve()) in cli.historico[0]["content"], "system prompt informa workspace")
checar(cli.session_path.exists(), "CLI cria log JSONL da sessao")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/workspace")
checar(str(sub.resolve()) in out.getvalue(), "/workspace sem argumento mostra pasta atual")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno(f"/workspace {raiz}")
checar(tools.SANDBOX_ROOT == raiz.resolve(), "/workspace troca SANDBOX_ROOT")
checar(str(raiz.resolve()) in out.getvalue(), "/workspace ecoa nova pasta")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/model outro")
checar(cli.modelo == "outro", "/model troca modelo")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/status")
status = out.getvalue()
checar(cli.session_id in status and "tokens Ollama:" in status and "feedbacks:" in status,
       "/status mostra sessao, consumo e feedback")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/tools")
tools_out = out.getvalue()
checar("executar_comando" in tools_out and "git:" in tools_out, "/tools lista ferramentas e git")

cli._tool_depois("executar_comando", {"cmd": "git status"}, "$ git status\nok\n(código de saída: 0)", "teste")
out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/show 1")
checar("$ git status" in out.getvalue(), "/show expande comando salvo")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/")
checar("/status" in out.getvalue() and "/bom" in out.getvalue(), "barra sozinha mostra ajuda")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/feedback")
checar(str(cli.feedback_path) in out.getvalue(), "/feedback mostra destino da avaliacao")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/bom ficou util")
checar(cli.feedback_path.exists() and cli.feedbacks == 1, "/bom salva feedback")
checar('"nota": 10' in cli.feedback_path.read_text(), "/bom registra nota 10")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/nota 7 faltou testar")
checar(cli.feedbacks == 2 and '"nota": 7' in cli.feedback_path.read_text(),
       "/nota salva nota numerica")

out = io.StringIO()
with redirect_stdout(out):
    try:
        isaac_cli.main(["--help"])
    except SystemExit:
        pass
checar("--max-passos" not in out.getvalue(), "--max-passos fica oculto no help normal")

cli.historico.append({"role": "user", "content": "lixo"})
cli.comando_interno("/clear")
checar(len(cli.historico) == 1, "/clear limpa historico")
checar(str(raiz.resolve()) in cli.historico[0]["content"], "/clear mantem workspace atual")

try:
    cli.comando_interno("/exit")
    saiu = False
except EOFError:
    saiu = True
checar(saiu, "/exit encerra REPL")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("CLI DO ISAAC OK — workspace, modelo e saida basica sem Ollama")
