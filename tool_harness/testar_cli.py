#!/usr/bin/env python3
"""Testes baratos do CLI do Isaac, sem chamar Ollama."""
import io
import builtins
import os
import pty
import select
import sys
import tempfile
import termios
import time
from contextlib import redirect_stdout
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import isaac_cli
import config
import setup_ollama
import terminal_ui
import tools
import execucao

falhas = []


def checar(cond, desc):
    print(f"[{'ok    ' if cond else 'FALHOU'}] {desc}")
    if not cond:
        falhas.append(desc)


checar(not hasattr(isaac_cli, "MODELO_FALLBACK"),
       "CLI não possui modelo fallback hardcoded")

interativo_original_tela = terminal_ui.interativo
terminal_ui.interativo = lambda _input_fn=input: True
try:
    sequencias_tela = io.StringIO()
    with redirect_stdout(sequencias_tela), terminal_ui.tela_alternativa():
        pass
finally:
    terminal_ui.interativo = interativo_original_tela
checar("\033[?1049h" in sequencias_tela.getvalue()
       and "\033[?1049l" in sequencias_tela.getvalue()
       and "\033[?1007h" not in sequencias_tela.getvalue()
       and "\033[?1000h" not in sequencias_tela.getvalue(),
       "tela não converte roda em seta nem bloqueia seleção nativa")

which_launcher_original = isaac_cli.shutil.which
try:
    isaac_cli.shutil.which = lambda nome: str(AQUI.parent / "isaacli") if nome == "isaacli" else None
    checar(isaac_cli._comando_retomada("sessao") == "isaacli --resume sessao",
           "retomada usa comando curto quando instalação global aponta para este app")
    isaac_cli.shutil.which = lambda _nome: None
    checar(str(AQUI.parent / "isaacli") in isaac_cli._comando_retomada("sessao"),
           "retomada usa path completo quando isaacli ainda não está no PATH")
finally:
    isaac_cli.shutil.which = which_launcher_original

class FechamentoInterrompidoFake:
    def __init__(self):
        self.tentativas = 0
    def fechar(self):
        self.tentativas += 1
        if self.tentativas == 1:
            raise KeyboardInterrupt

fechamento_fake = FechamentoInterrompidoFake()
isaac_cli._fechar_sem_interrupcao(fechamento_fake)
checar(fechamento_fake.tentativas == 2,
       "Ctrl+C repetido não interrompe nem exibe traceback durante o cleanup")

comandos_filtrados = isaac_cli._filtrar_comandos("/sta")
checar(comandos_filtrados and comandos_filtrados[0][0] == "/status",
       "busca incremental prioriza prefixo do comando")
checar(len(isaac_cli._filtrar_comandos("/")) == len(isaac_cli.COMANDOS_BARRA),
       "barra sozinha oferece todos os comandos")
checar(any(comando == "/sessions"
           for comando, _descricao in isaac_cli._filtrar_comandos("sessões")),
       "busca de comandos também encontra texto da descrição")

if isaac_cli.PromptSession is not None:
    from prompt_toolkit.document import Document
    completador = isaac_cli._CompletadorComandos()
    conclusoes = list(completador.get_completions(Document("/sta"), None))
    checar(conclusoes and conclusoes[0].text == "/status"
           and conclusoes[0].start_position == -4,
           "menu substitui a consulta pelo comando selecionado ao completar")


raiz = Path(tempfile.mkdtemp())
sub = raiz / "projeto"
sub.mkdir()

cli = isaac_cli.IsaacCLI("isaac-granite", sub, 4, autostart_ollama=False)
checar(bool(isaac_cli.SESSION_ID_UUID.fullmatch(cli.session_id)),
       "novas sessões usam UUIDv4 completo")
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

config_picker = raiz / "config-picker.json"
dado_picker = config.config_vazia()
dado_picker["profiles"]["qwen"] = {
    "provider": "ollama", "model": "modelo-qwen", "num_ctx": 16384,
    "context_limit": 32768, "thinking": "medium",
}
dado_picker["default_profile"] = "qwen"
config.salvar(dado_picker, config_picker)
cli_picker = isaac_cli.IsaacCLI(
    "modelo-qwen", sub, 4, autostart_ollama=False, thinking="medium",
    num_ctx=16384, config_file=config_picker,
)
redesenhos_modelo = []
cli_picker.redesenhar_sessao = lambda mensagem=None: redesenhos_modelo.append(mensagem)
seletor_modelo_original = setup_ollama.executar_seletor_modelo
def seletor_modelo_fake(config_file=None):
    dado = config.carregar(config_file)
    item = dado["profiles"]["qwen"]
    item["thinking"] = "high"
    item["num_ctx"] = 32768
    dado["default_profile"] = "qwen"
    config.salvar(dado, config_file)
    return 0
setup_ollama.executar_seletor_modelo = seletor_modelo_fake
try:
    with redirect_stdout(io.StringIO()):
        cli_picker.comando_interno("/model")
finally:
    setup_ollama.executar_seletor_modelo = seletor_modelo_original
dado_picker = config.carregar(config_picker)
checar(cli_picker.modelo == "modelo-qwen" and cli_picker.thinking == "high"
       and cli_picker.num_ctx == 32768,
       "/model seleciona perfil, esforço e contexto por menus")
checar(dado_picker["default_profile"] == "qwen"
       and dado_picker["profiles"]["qwen"]["num_ctx"] == 32768,
       "/model persiste a seleção rápida sem repetir /setup")
checar(redesenhos_modelo and "modelo: qwen" in redesenhos_modelo[-1],
       "/model redesenha a sessão depois de fechar o menu de tela inteira")

cli_redraw = isaac_cli.IsaacCLI(
    "modelo-redraw", sub, 4, autostart_ollama=False,
    config_file=raiz / "config-redraw.json",
)
cli_redraw._log("user", content="pergunta anterior")
cli_redraw._log("assistant_final", content="resposta anterior\n" * 30)
cli_redraw.garantir_ollama = lambda avisar=False: "teste"
interativo_original_redraw = terminal_ui.interativo
limpar_original_redraw = terminal_ui.limpar
terminal_ui.interativo = lambda _input_fn=input: True
terminal_ui.limpar = lambda _input_fn=input: None
saida_redraw = io.StringIO()
try:
    with redirect_stdout(saida_redraw):
        cli_redraw.redesenhar_sessao("modelo alterado")
finally:
    terminal_ui.interativo = interativo_original_redraw
    terminal_ui.limpar = limpar_original_redraw
checar("pergunta anterior" in saida_redraw.getvalue()
       and "resposta anterior" in saida_redraw.getvalue()
       and "modelo alterado" in saida_redraw.getvalue(),
       "redesenho restaura conversa atual como uma retomada")

# /history não abre mais uma camada de tela cheia: é print simples, então
# fica no scrollback nativo do terminal, com markdown formatado e copiável.
saida_history = io.StringIO()
with redirect_stdout(saida_history):
    cli_redraw.mostrar_historico()
checar("pergunta anterior" in saida_history.getvalue()
       and "resposta anterior" in saida_history.getvalue(),
       "/history imprime a conversa completa sem sequestrar a tela")

cli_new = isaac_cli.IsaacCLI(
    "modelo-novo", sub, 4, autostart_ollama=False, config_file=raiz / "config-new.json",
)
sessao_anterior = cli_new.session_id
caminho_anterior = cli_new.session_path
cli_new.historico.append({"role": "user", "content": "contexto antigo"})
cli_new.turnos = 3
cli_new.comandos.append({"id": 1})
with redirect_stdout(io.StringIO()):
    cli_new.comando_interno("/new")
checar(cli_new.session_id != sessao_anterior and cli_new.session_path != caminho_anterior,
       "/new cria outro ID e outro arquivo de sessão")
checar(len(cli_new.historico) == 1 and cli_new.turnos == 0 and not cli_new.comandos,
       "/new zera contexto e contadores sem fechar o CLI")
checar(f'"proxima_sessao": "{cli_new.session_id}"' in caminho_anterior.read_text()
       and cli_new.session_path.exists(),
       "/new fecha o log anterior e inicia o novo com rastreabilidade")

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

cli._log("user", content="mensagem para histórico interno")
cli._log("assistant_final", content="resposta para histórico interno")
historico_interno = cli._texto_historico()
checar("mensagem para histórico interno" in historico_interno
       and "resposta para histórico interno" in historico_interno,
       "/history reconstrói mensagens da sessão sem usar scrollback do shell")

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
checar("fetch_url" in tools_out, "/tools mostra a leitura web dedicada")
checar(isaac_cli._comando_leitura_segura(
    "gh issue view 246 --repo aws-cloudformation/cloudformation-validate"),
    "consultas gh somente leitura não pedem aprovação desnecessária")
checar(not isaac_cli._comando_leitura_segura("gh issue close 246"),
       "operações mutáveis do gh nunca são classificadas como leitura")

issue_url = "https://github.com/aws-cloudformation/cloudformation-validate/issues/246"
checar(tools._normalizar_url_web(issue_url) ==
       "https://api.github.com/repos/aws-cloudformation/cloudformation-validate/issues/246",
       "fetch_url converte link de issue do GitHub para a API pública")
getaddrinfo_original = tools.socket.getaddrinfo
try:
    tools.socket.getaddrinfo = lambda *_a, **_kw: [
        (tools.socket.AF_INET, tools.socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80)),
    ]
    web_privada = tools.executar("fetch_url", {"url": "http://localhost/segredo"})
finally:
    tools.socket.getaddrinfo = getaddrinfo_original
checar("não acessa localhost" in web_privada,
       "fetch_url bloqueia localhost e redes privadas antes da conexão")

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

out = io.StringIO()
with redirect_stdout(out):
    try:
        isaac_cli.main(["--version"])
    except SystemExit as e:
        codigo_versao = e.code
checar(codigo_versao == 0 and f"Isaac CLI v{isaac_cli.APP_VERSION}" in out.getvalue(),
       "--version informa a versão do aplicativo")

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
    cli._mostrar_working()
    cli._primeiro_token_em = time.monotonic() - 1
    cli._geracao_status_em = float("inf")
    cli._token("Olá")
    cli._limpar_working()
checar(out.getvalue().startswith("\nTrabalhando…")
       and "\r\033[2Kisaac: Olá" in out.getvalue(),
       "Trabalhando é transitório e deixa uma separação antes da resposta")

painel = isaac_cli._linhas_boas_vindas(
    "modelo-longo", "Ollama 0.30.10", sub, largura=100, usuario="Weverton",
)
checar(all(isaac_cli._largura_visual(linha) == 100 for linha in painel)
       and f"Isaac CLI v{isaac_cli.APP_VERSION}" in painel[0]
       and "Bem-vindo de volta, Weverton!" in "\n".join(painel)
       and "┬" in painel[0] and painel[1].count("│") == 3
       and all(linha in "\n".join(painel) for linha in isaac_cli.WORDMARK_ISAAC)
       and "Shift+Tab alterna permissões" in "\n".join(painel)
       and "🐏" not in "\n".join(painel) and "Gênesis" not in "\n".join(painel),
       "boas-vindas tem versão, identidade e alinhamento estável")
painel_compacto = isaac_cli._linhas_boas_vindas(
    "modelo-com-nome-muito-longo", "motor", sub, largura=40,
    usuario="Nome de usuário muito comprido",
)
checar(all(isaac_cli._largura_visual(linha) == 40 for linha in painel_compacto),
       "boas-vindas também se ajusta a terminal estreito")

markdown = isaac_cli._formatar_markdown_terminal(
    "# Título\n**forte** e `código`\n- [x] feito\n```python\nprint(1)\n```\n"
    "[site](https://example.test)\x1b[2J",
    cores=True,
)
checar("**" not in markdown and "```" not in markdown
       and "\033[1mforte\033[0m" in markdown
       and "•" in markdown and "☑" in markdown
       and "\x1b[2J" not in markdown and "https://example.test" in markdown,
       "Markdown comum ganha estilo e controles do modelo são removidos")

out = io.StringIO()
with redirect_stdout(out):
    cli._mostrar_working()
    cli._primeiro_token_em = time.monotonic() - 1
    cli._thinking_token("não exibir este raciocínio")
checar("Trabalhando… · ≈" in out.getvalue()
       and "não exibir este raciocínio" not in out.getvalue(),
       "thinking atualiza tok/s sem revelar o raciocínio")

# Enquanto o agente trabalha não há prompt lendo stdin. Setas e rolagem não
# podem ser ecoadas, mas Ctrl+C precisa continuar sendo um sinal.
master_fd, slave_fd = pty.openpty()
try:
    antes = termios.tcgetattr(slave_fd)
    with terminal_ui.entrada_ocupada(fd=slave_fd):
        durante = termios.tcgetattr(slave_fd)
        checar(not (durante[3] & termios.ECHO), "entrada ocupada desativa echo")
        checar(not (durante[3] & termios.ICANON), "entrada ocupada desativa modo de linha")
        checar(bool(durante[3] & termios.ISIG), "entrada ocupada preserva Ctrl+C")
        os.write(master_fd, b"\x1b[B\x1b[A")
    depois = termios.tcgetattr(slave_fd)
    checar(depois == antes, "entrada ocupada restaura o terminal")
    pronto, _, _ = select.select([slave_fd], [], [], 0)
    checar(not pronto, "teclas durante a inicialização não vazam para o primeiro prompt")
finally:
    os.close(master_fd)
    os.close(slave_fd)

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

try:
    cli.garantir_ollama = lambda avisar=False: "teste"
    isaac_cli.agent.rodar = lambda *_a, **_kw: {
        "final": "", "chamadas": [], "uso": {"eval_count": 3},
    }
    out = io.StringIO()
    with redirect_stdout(out):
        codigo_resposta_vazia = cli.perguntar("teste vazio")
finally:
    isaac_cli.agent.rodar = agent_original
    cli.garantir_ollama = garantir_original
checar(codigo_resposta_vazia == 1 and "sem resposta visível" in out.getvalue(),
       "CLI denuncia resposta realmente vazia em vez de mostrar só métricas")

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

tela_original = isaac_cli.terminal_ui.tela_alternativa
repl_tela_original = cli._repl_tela
isaac_cli.terminal_ui.tela_alternativa = __import__("contextlib").nullcontext
cli._repl_tela = lambda: 0
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli.repl()
finally:
    isaac_cli.terminal_ui.tela_alternativa = tela_original
    cli._repl_tela = repl_tela_original
linhas_saida = out.getvalue().splitlines()
checar(isaac_cli._comando_retomada(cli.session_id) in linhas_saida,
       "comando de retomada fica sozinho em uma linha copiável")

resume_id = "2026-08-07-123456-abcdef"
resume_path = isaac_cli.SESSOES_DIR / f"{resume_id}.jsonl"
eventos_resume = [
    {"tipo": "meta", "workspace": str(sub), "modelo": "modelo-resume"},
    {"tipo": "user", "workspace": str(sub), "modelo": "modelo-resume",
     "content": "leia o arquivo"},
    {"tipo": "tool_start", "workspace": str(sub), "modelo": "modelo-resume",
     "nome": "read_file", "args": {"path": "a.txt"}},
    {"tipo": "permission", "workspace": str(sub), "modelo": "modelo-resume",
     "cmd": "cat a.txt", "decisao": "uma_vez"},
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
checar([papel for papel, _ in retomada["transcript"]] == [
    "user", "tool_start", "permission", "tool_result", "assistant"
], "--resume prepara conversa visível incluindo ferramentas e permissões")
cli.transcript_retomada = retomada["transcript"]
builtins.input = lambda _prompt="": (_ for _ in ()).throw(KeyboardInterrupt())
cli.garantir_ollama = lambda avisar=False: "teste"
try:
    out = io.StringIO()
    with redirect_stdout(out):
        cli._repl_tela()
finally:
    builtins.input = input_original
    cli.garantir_ollama = ollama_original
checar("[read_file] →" in out.getvalue() and "[read_file] ← conteúdo" in out.getvalue()
       and "Permissão: Permitir uma vez" in out.getvalue(),
       "REPL redesenha ações executadas no histórico retomado")
cli.transcript_retomada = []
try:
    isaac_cli._carregar_sessao("../../etc/passwd")
    resume_seguro = False
except ValueError:
    resume_seguro = True
checar(resume_seguro, "--resume recusa IDs usados como caminho")

# Duas sessões devem compartilhar o servidor iniciado pelo Isaac. A primeira
# a sair não pode derrubá-lo; somente a última encerra o processo gerenciado.
runtime_original = os.environ.get("ISAACLI_RUNTIME_DIR")
ok_original = isaac_cli._ollama_ok
which_original = isaac_cli.shutil.which
popen_original = isaac_cli.subprocess.Popen
identidade_original = isaac_cli._identidade_pid
igual_original = isaac_cli._processo_igual
kill_original = isaac_cli.os.kill
servidor = {"ativo": False}
identidades = {101: "cliente-a", 202: "cliente-b"}
mortes = []
class ProcessoOllamaFake:
    pid = 999
    returncode = None
    def __init__(self, *_a, **kwargs):
        checar(kwargs.get("start_new_session") is True,
               "Ollama gerenciado nasce fora do grupo do terminal")
        servidor["ativo"] = True
        identidades[self.pid] = "servidor"
    def poll(self):
        return None if servidor["ativo"] else 0
    def terminate(self):
        servidor["ativo"] = False
    def wait(self, timeout=None):
        return 0
try:
    os.environ["ISAACLI_RUNTIME_DIR"] = str(raiz / "runtime")
    isaac_cli._ollama_ok = lambda timeout=2: "0.30-teste" if servidor["ativo"] else None
    isaac_cli.shutil.which = lambda _nome: "/usr/bin/ollama"
    isaac_cli.subprocess.Popen = ProcessoOllamaFake
    isaac_cli._identidade_pid = lambda pid: identidades.get(int(pid))
    isaac_cli._processo_igual = lambda pid, inicio: identidades.get(int(pid or -1)) == inicio
    def kill_fake(pid, sinal):
        mortes.append((pid, sinal))
        if pid == 999:
            servidor["ativo"] = False
            identidades.pop(999, None)
    isaac_cli.os.kill = kill_fake
    cli_a = isaac_cli.IsaacCLI("modelo", sub, 2, config_file=raiz / "cfg-a.json")
    cli_b = isaac_cli.IsaacCLI("modelo", sub, 2, config_file=raiz / "cfg-b.json")
    cli_a._runtime_pid, cli_a._runtime_start = 101, "cliente-a"
    cli_b._runtime_pid, cli_b._runtime_start = 202, "cliente-b"
    checar(cli_a.garantir_ollama() == "0.30-teste", "primeira sessão inicia Ollama")
    checar(cli_b.garantir_ollama() == "0.30-teste", "segunda sessão compartilha Ollama")
    cli_a.fechar()
    checar(servidor["ativo"] and not mortes,
           "fechar uma sessão preserva Ollama usado por outra")
    cli_b.fechar()
    checar(not servidor["ativo"] and mortes[-1][0] == 999,
           "última sessão encerra Ollama gerenciado")
finally:
    if runtime_original is None:
        os.environ.pop("ISAACLI_RUNTIME_DIR", None)
    else:
        os.environ["ISAACLI_RUNTIME_DIR"] = runtime_original
    isaac_cli._ollama_ok = ok_original
    isaac_cli.shutil.which = which_original
    isaac_cli.subprocess.Popen = popen_original
    isaac_cli._identidade_pid = identidade_original
    isaac_cli._processo_igual = igual_original
    isaac_cli.os.kill = kill_original

print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
print("CLI DO ISAAC OK — workspace, modelo e saida basica sem Ollama")
