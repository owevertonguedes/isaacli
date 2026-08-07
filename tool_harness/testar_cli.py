#!/usr/bin/env python3
"""Testes baratos do CLI do Isaac, sem chamar Ollama."""
import io
import builtins
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import isaac_cli
import config
import setup_ollama
import tools
import execucao

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


checar(not hasattr(isaac_cli, "MODELO_FALLBACK"),
       "CLI não possui modelo fallback hardcoded")


raiz = Path(tempfile.mkdtemp())
sub = raiz / "projeto"
sub.mkdir()

cli = isaac_cli.IsaacCLI("isaac-granite", sub, 4, autostart_ollama=False)
checar(tools.SANDBOX_ROOT == sub.resolve(), "workspace inicial vira SANDBOX_ROOT")
checar(str(sub.resolve()) in cli.historico[0]["content"], "system prompt informa workspace")
checar("same language" in cli.historico[0]["content"],
       "system prompt obriga resposta no idioma do usuário")
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

config_setup = raiz / "config-setup.json"
cli_setup = isaac_cli.IsaacCLI(
    "modelo-antigo", sub, 4, autostart_ollama=False, config_file=config_setup,
)
setup_original = setup_ollama.executar_setup
def setup_fake(config_file=None):
    dado = config.config_vazia()
    dado["profiles"]["novo"] = {
        "provider": "ollama", "model": "modelo-novo", "num_ctx": 16384,
        "thinking": "medium",
    }
    dado["default_profile"] = "novo"
    config.salvar(dado, config_file)
    return 0
setup_ollama.executar_setup = setup_fake
try:
    with redirect_stdout(io.StringIO()):
        cli_setup.comando_interno("/setup")
finally:
    setup_ollama.executar_setup = setup_original
checar(cli_setup.modelo == "modelo-novo" and cli_setup.thinking == "medium",
       "/setup recarrega motor na sessao sem fechar o CLI")

cli_api = isaac_cli.IsaacCLI(
    "modelo-api", sub, 4, autostart_ollama=False,
    config_file=raiz / "config-provider.json",
)
# O helper procura secrets.json ao lado do config; use o caminho padrão esperado.
config.salvar_segredo("api:teste", "chave", (raiz / "config-provider.json").with_name("secrets.json"))
provider_api = cli_api._provider_do_perfil({
    "provider": "openai_compatible", "provider_name": "Servidor livre",
    "base_url": "https://api.exemplo.test/v1", "credential": "api:teste",
})
checar(provider_api["provider"] == "openai_compatible"
       and provider_api["api_key"] == "chave",
       "CLI carrega perfil e segredo de API genérica")

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
checar("run_command" in tools_out and "git:" in tools_out, "/tools lista ferramentas e git")

cli._tool_depois("run_command", {"cmd": "git status"}, "$ git status\nok\n(código de saída: 0)", "teste")
out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/show 1")
checar("$ git status" in out.getvalue(), "/show expande comando salvo")

falhas_antes_recusa = cli.falhas
out = io.StringIO()
with redirect_stdout(out):
    cli._tool_depois(
        "run_command", {"cmd": "rm x"},
        "$ rm x\nRECUSADO PELO USUÁRIO: o comando não foi autorizado.\n(código de saída: 126)",
        "teste",
    )
checar("recusado pelo usuário" in out.getvalue() and cli.falhas == falhas_antes_recusa,
       "recusa humana não é classificada nem contabilizada como falha")

out = io.StringIO()
with redirect_stdout(out):
    cli.comando_interno("/")
checar("/setup" in out.getvalue() and "/status" in out.getvalue() and "/bom" in out.getvalue(),
       "barra sozinha mostra ajuda e reparo do motor")

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

setup_original_main = setup_ollama.executar_setup
setup_ollama.executar_setup = lambda: 1
try:
    with redirect_stdout(io.StringIO()):
        codigo_setup_falho = isaac_cli.main(["setup"])
finally:
    setup_ollama.executar_setup = setup_original_main
checar(codigo_setup_falho == 1,
       "setup explícito com falha não abre silenciosamente o fallback granite")

cli._rotulo_assistente_pendente = True
cli._working_visivel = False
out = io.StringIO()
with redirect_stdout(out):
    cli._token("Olá")
checar(out.getvalue() == "isaac: Olá",
       "primeiro texto não adiciona uma linha vazia antes de isaac")

# Política de comandos: leitura automática, mutação só após decisão humana.
exec_original = execucao.run_command
input_original_permissao = builtins.input
chamadas_exec = []
def exec_fake(cmd, autorizado=False):
    chamadas_exec.append((cmd, autorizado))
    return f"$ {cmd}\n(código de saída: 0)"
execucao.run_command = exec_fake
try:
    cli.config_file = raiz / "config-permissoes.json"
    cli._aprovar_e_executar("ls")
    checar(chamadas_exec[-1] == ("ls", False),
           "modo seguro executa leitura sem interromper")
    builtins.input = lambda _prompt="": "w"
    cli._aprovar_e_executar("rm arquivo.txt")
    dado_permissoes = config.carregar(cli.config_file)
    checar("rm" in config.regras_permissao(dado_permissoes, cli.workspace),
           "aprovação pode persistir regra somente no workspace")
    builtins.input = lambda _prompt="": (_ for _ in ()).throw(AssertionError("não deve perguntar"))
    cli._aprovar_e_executar("rm outro.txt")
    checar(chamadas_exec[-1] == ("rm outro.txt", True),
           "regra persistida autoriza nova chamada equivalente")
    cli.modo_permissao = "somente_autorizados"
    builtins.input = lambda _prompt="": "n"
    recusado = cli._aprovar_e_executar("git status")
    checar("RECUSADO PELO USUÁRIO" in recusado,
           "modo somente autorizados pergunta até para leitura")
finally:
    execucao.run_command = exec_original
    builtins.input = input_original_permissao

agent_original = isaac_cli.agent.rodar
garantir_original = cli.garantir_ollama
try:
    cli.garantir_ollama = lambda avisar=False: "teste"
    isaac_cli.agent.rodar = lambda *_a, **_kw: {
        "final": "Apaguei com sucesso.", "chamadas": [], "uso": {},
    }
    out = io.StringIO()
    with redirect_stdout(out):
        cli.perguntar("apague o arquivo")
finally:
    isaac_cli.agent.rodar = agent_original
    cli.garantir_ollama = garantir_original
checar("nenhuma ferramenta de alteração foi executada" in out.getvalue(),
       "CLI desmente sucesso alucinado quando nenhuma ferramenta alterou nada")

def agente_recusado(*_a, on_tool=None, **_kw):
    resultado = ("$ rm x\nRECUSADO PELO USUÁRIO: o comando não foi autorizado.\n"
                 "(código de saída: 126)")
    on_tool("run_command", {"cmd": "rm x"}, resultado, "teste")
    return {"final": "A exclusão foi recusada.",
            "chamadas": [("run_command", {"cmd": "rm x"}, resultado, "teste")],
            "uso": {}}
try:
    cli.garantir_ollama = lambda avisar=False: "teste"
    isaac_cli.agent.rodar = agente_recusado
    out = io.StringIO()
    with redirect_stdout(out):
        cli.perguntar("apague x")
finally:
    isaac_cli.agent.rodar = agent_original
    cli.garantir_ollama = garantir_original
checar("recusado pelo usuário" in out.getvalue() and "Nota do Isaac CLI" not in out.getvalue(),
       "recusa humana não produz nota de falha no fim do turno")

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

input_original = builtins.input
ollama_original = cli.garantir_ollama
builtins.input = lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
cli.garantir_ollama = lambda avisar=False: "teste"
try:
    with redirect_stdout(io.StringIO()):
        codigo_ctrl_c = cli._repl_tela()
finally:
    builtins.input = input_original
    cli.garantir_ollama = ollama_original
checar(codigo_ctrl_c == 130, "Ctrl+C no prompt encerra a interface sem traceback")

cli.transcript_retomada = [("user", "mensagem antiga"), ("assistant", "resposta antiga")]
builtins.input = lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
cli.garantir_ollama = lambda avisar=False: "teste"
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli._repl_tela()
finally:
    builtins.input = input_original
    cli.garantir_ollama = ollama_original
checar("mensagem antiga" in out.getvalue() and "resposta antiga" in out.getvalue(),
       "REPL redesenha conversa recente ao retomar")
cli.transcript_retomada = []

tela_original = isaac_cli.terminal_ui.tela_com_scrollback
repl_tela_original = cli._repl_tela
isaac_cli.terminal_ui.tela_com_scrollback = __import__("contextlib").nullcontext
cli._repl_tela = lambda: 0
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli.repl()
finally:
    isaac_cli.terminal_ui.tela_com_scrollback = tela_original
    cli._repl_tela = repl_tela_original
linhas_saida = out.getvalue().splitlines()
checar(f"isaacli --resume {cli.session_id}" in linhas_saida,
       "comando de retomada fica sozinho em uma linha copiável")

resume_id = "2026-08-07-123456-abcdef"
resume_path = isaac_cli.SESSOES_DIR / f"{resume_id}.jsonl"
eventos_resume = [
    {"tipo": "meta", "workspace": str(sub), "modelo": "modelo-resume"},
    {"tipo": "user", "workspace": str(sub), "modelo": "modelo-resume",
     "content": "leia o arquivo"},
    {"tipo": "tool_start", "workspace": str(sub), "modelo": "modelo-resume",
     "nome": "read_file", "args": {"path": "a.txt"}},
    {"tipo": "tool_result", "workspace": str(sub), "modelo": "modelo-resume",
     "nome": "read_file", "resultado": "conteúdo"},
    {"tipo": "assistant_final", "workspace": str(sub), "modelo": "modelo-resume",
     "content": "O arquivo contém conteúdo."},
]
resume_path.write_text("\n".join(__import__("json").dumps(e, ensure_ascii=False)
                                  for e in eventos_resume) + "\n")
retomada = isaac_cli._carregar_sessao(resume_id)
checar(retomada["model"] == "modelo-resume" and retomada["workspace"] == sub,
       "--resume recupera modelo e workspace")
checar(any(m.get("role") == "tool" and m.get("content") == "conteúdo"
           for m in retomada["history"]), "--resume reconstrói mensagens e ferramentas")
checar(retomada["transcript"] == [
    ("user", "leia o arquivo"), ("assistant", "O arquivo contém conteúdo.")
], "--resume prepara mensagens visíveis sem despejar resultados de ferramentas")
try:
    isaac_cli._carregar_sessao("../../etc/passwd")
    resume_seguro = False
except ValueError:
    resume_seguro = True
checar(resume_seguro, "--resume recusa IDs usados como caminho")

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("CLI DO ISAAC OK — workspace, modelo e saida basica sem Ollama")
