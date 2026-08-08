#!/usr/bin/env python3
"""CLI bruto do Isaac.

Uso:
    isaac
    isaac "rode git status e me diga o que esta pendente"
    isaac --workspace /caminho/do/projeto

O processo roda em foreground. Fechar o terminal derruba este Python; comandos
executados pelo Isaac nascem em grupo proprio e com bwrap --die-with-parent.
"""
import argparse
import datetime as dt
import getpass
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import shlex
from contextlib import contextmanager
from pathlib import Path

try:
    import readline
except ImportError:  # pragma: no cover - Windows/ambiente minimo
    readline = None

try:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
except ImportError:  # pragma: no cover - dependencia opcional
    PromptSession = None
    Completer = object
    Completion = None
    CompleteStyle = None
    FormattedText = None
    KeyBindings = None

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import agent
import config
import terminal_ui
import tools

SESSOES_DIR = AQUI / "cli_sessoes"
FEEDBACK_DIR = AQUI / "feedback"
COMANDOS_INFO = (
    ("/help", "comandos disponíveis"),
    ("/setup", "configurar modelos e motores"),
    ("/status", "sessão, workspace e consumo"),
    ("/tools", "ferramentas disponíveis"),
    ("/sessions", "sessões salvas"),
    ("/history", "mostra a conversa completa desta sessão"),
    ("/show", "expandir saída de um comando"),
    ("/log", "arquivo JSONL desta sessão"),
    ("/feedback", "como avaliar a tarefa"),
    ("/bom", "marcar a tarefa como boa"),
    ("/ruim", "marcar a tarefa como ruim"),
    ("/nota", "dar uma nota de 0 a 10"),
    ("/workspace", "mostrar ou trocar a pasta"),
    ("/model", "selecionar modelo, esforço e contexto"),
    ("/permissions", "autorizações persistentes"),
    ("/mode", "alternar modo de permissões"),
    ("/language", "mudar idioma da interface"),
    ("/clear", "limpar contexto da conversa"),
    ("/new", "iniciar uma nova sessão"),
    ("/exit", "sair do Isaac"),
)
COMANDOS_BARRA = [comando for comando, _descricao in COMANDOS_INFO]
MAX_PREVIEW_CHARS = 1800
MAX_PREVIEW_LINHAS = 28
APP_VERSION = "0.1.0-dev"
SESSION_ID_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
SESSION_ID_LEGADO = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6}-[a-f0-9]{6}"
)
WORDMARK_ISAAC = tuple(linha.ljust(23) for linha in (
    "╻  ┏━╸  ┏━┓  ┏━┓  ┏━╸",
    "┃  ┗━┓  ┣━┫  ┣━┫  ┃",
    "╹  ╺━┛  ╹ ╹  ╹ ╹  ┗━╸",
))
ANSI = {
    "prompt": "\033[1;36m",
    "assistant": "\033[1;32m",
    "tool": "\033[1;34m",
    "warn": "\033[1;33m",
    "bad": "\033[1;31m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}

COMANDOS_SOMENTE_LEITURA = {"ls", "cat", "head", "tail", "wc", "grep", "find"}
GIT_SOMENTE_LEITURA = {"status", "diff", "log", "show"}
GH_SOMENTE_LEITURA = {
    ("issue", "view"), ("pr", "view"), ("repo", "view"),
    ("release", "view"), ("run", "view"), ("auth", "status"),
    ("search", "issues"), ("search", "prs"), ("search", "repos"),
    ("search", "commits"),
}


def _runtime_ollama_dir():
    base = os.environ.get("ISAACLI_RUNTIME_DIR") or os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "isaacli"
    return Path("/tmp") / f"isaacli-{os.getuid()}"


def _identidade_pid(pid):
    """Identidade estável para não sinalizar um PID que tenha sido reutilizado."""
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split()[21]
    except (OSError, ValueError, IndexError):
        return None


def _processo_igual(pid, identidade):
    atual = _identidade_pid(pid)
    return bool(atual and identidade and atual == str(identidade))


@contextmanager
def _estado_ollama_compartilhado():
    """Serializa o autostart/autostop entre várias sessões do Isaac."""
    import fcntl

    pasta = _runtime_ollama_dir()
    pasta.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = pasta / "ollama.lock"
    state_path = pasta / "ollama.json"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                estado = json.loads(state_path.read_text()) if state_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                estado = {}
            yield estado
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(estado))
            os.replace(tmp, state_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

CONHECIMENTO_CLI = """You are isaacli running as a local CLI in the user's terminal.

OPERATING CONTEXT:
- Always answer in the same language as the user's latest message. If the user
  writes in Portuguese, answer in Brazilian Portuguese; do not switch to English.
- The current working directory is: {workspace}
- File and terminal tools are confined to that directory.
- To read public web content (pages, documentation, links or HTTP APIs), use
  fetch_url. It is a general web-reading tool, not a GitHub-only workaround.
- For structured, read-only GitHub queries, you may use `gh issue view`,
  `gh pr view`, `gh repo view`, `gh release view`, `gh run view` or `gh search`.
  Prefer fetch_url for public links; use gh when its GitHub-specific structure or
  authenticated access is useful. If gh reports missing/invalid authentication
  for a public link, use fetch_url immediately; do not inspect tokens, environment
  variables or credential files, and do not clone a repository just to read it.
  Never use curl through run_command.
- Before asking the user to clarify a local file, directory or project target,
  try to resolve it with list_dir, find, grep or read_file. If the user says
  "the txt file", "the config" or similar and the workspace can identify it,
  inspect the workspace instead of asking for an exact name.
- To inspect the project, use run_command with short commands: git status,
  git diff, ls, find, wc, pytest, python3.
- run_command executes exactly one program without a shell. Never use pipes,
  redirections, `&&`, `||`, `;`, `cd`, `$VARIABLE` or `2>/dev/null`; make separate
  tool calls instead.
- If `graphify-out/graph.json` exists and the user asks where a flow, resource,
  module, test or architectural relation lives, look it up first with
  `graphify query "question" --graph graphify-out/graph.json --budget 700`.
  Graphify is for locating context; after that read the files and verify before
  declaring success. If there is no graph, fall back to local search with
  find/rg, and do not edit before locating.
- To change files, use read_file first and write_file/append_file after.
- To delete a file or perform another operation not covered by a specialized
  file tool, call run_command with the exact terminal command (for example,
  `rm hello-world.txt`). The CLI, not you, handles user approval.
- Never claim that you created, edited, deleted, committed, tested or otherwise
  changed something unless at least one tool actually performed that action in
  this turn and its result confirms success.
- For git: run git status and git diff before proposing a commit.
- You may use git add and git commit when the user asks for it.
- If the user asks you to sign a commit, read "signature" as a valid Git trailer
  in the message body, not loose text in the subject line. Pick an appropriate
  trailer, such as `Co-Authored-By: Name <email>` or
  `Signed-off-by: Name <email>`, always with name and email inside `<...>`.
  Use a message with a clear subject, a blank line, a short explanation of why
  the commit exists, and the trailer on its own line. Since there is no shell,
  pass each part in quotes with `-m`, for example:
  git commit -m "Clear subject" -m "Short reason for the commit." -m "Signed-off-by: Name <name@localhost>"
  Writing "Co-Authored-By" in the subject is not enough. If the command has only
  one `-m`, it probably did not create a body or a trailer. The trailer has to
  appear on its own line in the body shown by `git log -1 --format=%B`.
  Check with git log -1 --format=%B afterwards. Do not reply that the commit is
  signed before that check shows a valid trailer. Do not use `git commit -S` for
  this; `-S` is the user's GPG signature, not a textual trailer.
- NEVER run git push to a real remote without explicit confirmation from the
  user in the current conversation. Normal push exists, push --force is blocked.
- If any tool returns a non-zero exit code, that is a failure. NEVER say a
  commit, push or test worked when the output showed an error.
- If git commit fails, stop and explain the error before trying to push.
- There is no shell: no pipes, no redirection, no &&, no ; and no $().
- Keep decisions and results short. The terminal shows a summary of the commands
  and keeps the full output in the session log.
"""


def _instalar_sinais():
    def sair(_signum, _frame):
        raise SystemExit(130)

    # SIGINT precisa virar KeyboardInterrupt para o REPL restaurar a tela e
    # imprimir o resumo da sessão. HUP/TERM continuam encerrando imediatamente.
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (AttributeError, ValueError):
        pass
    for sig in (signal.SIGHUP, signal.SIGTERM):
        try:
            signal.signal(sig, sair)
        except (AttributeError, ValueError):
            pass


def _fechar_sem_interrupcao(cli):
    """Conclui o cleanup mesmo quando o usuário repete Ctrl+C ao sair."""
    anteriores = {}
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        try:
            anteriores[sig] = signal.getsignal(sig)
            signal.signal(sig, signal.SIG_IGN)
        except (AttributeError, ValueError):
            pass
    try:
        while True:
            try:
                cli.fechar()
                return
            except KeyboardInterrupt:
                # Pode haver um SIGINT já entregue no instante em que o finally
                # começou. A partir daqui novos SIGINTs estão ignorados.
                continue
    finally:
        for sig, anterior in anteriores.items():
            try:
                signal.signal(sig, anterior)
            except (AttributeError, ValueError):
                pass


def _ollama_ok(timeout=2):
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=timeout) as r:
            return json.load(r).get("version") or "ok"
    except Exception:
        return None


def _agora():
    return dt.datetime.now().isoformat(timespec="seconds")


def _novo_session_id():
    return str(uuid.uuid4())


def _session_id_valido(session_id):
    return bool(SESSION_ID_UUID.fullmatch(session_id) or SESSION_ID_LEGADO.fullmatch(session_id))


def _comando_retomada(session_id):
    launcher = AQUI.parent / "isaacli"
    global_no_path = shutil.which("isaacli")
    if global_no_path:
        try:
            if Path(global_no_path).resolve() == launcher.resolve():
                return f"isaacli --resume {session_id}"
        except OSError:
            pass
    return f"{shlex.quote(str(launcher))} --resume {session_id}"


def _usa_cor():
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _cor(texto, nome):
    if not _usa_cor():
        return texto
    return f"{ANSI[nome]}{texto}{ANSI['reset']}"


def _prompt_colorido(texto, nome="prompt"):
    """Marca ANSI como não imprimível para o readline calcular quebras corretamente."""
    if not _usa_cor():
        return texto
    return f"\001{ANSI[nome]}\002{texto}\001{ANSI['reset']}\002"


def _texto_terminal_seguro(texto):
    """Remove controles que uma resposta do modelo não pode enviar ao terminal."""
    texto = re.sub(
        r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)?|.)",
        "", str(texto),
    )
    return "".join(
        caractere for caractere in texto
        if caractere in "\n\t" or ord(caractere) >= 32 and ord(caractere) != 127
    )


def _markdown_inline(texto, cores=True):
    """Renderiza o subconjunto inline mais comum do Markdown de LLMs."""
    if not cores:
        return texto
    saida = []
    i = 0
    estilos = {"bold": False, "italic": False, "strike": False, "code": False}

    def alternar(nome, ansi):
        estilos[nome] = not estilos[nome]
        return ansi if estilos[nome] else ANSI["reset"]

    while i < len(texto):
        if texto.startswith("**", i) or texto.startswith("__", i):
            saida.append(alternar("bold", "\033[1m"))
            i += 2
            continue
        if texto.startswith("~~", i):
            saida.append(alternar("strike", "\033[9m"))
            i += 2
            continue
        if texto[i] == "`":
            saida.append(alternar("code", "\033[38;5;81m"))
            i += 1
            continue
        if texto[i] == "*":
            saida.append(alternar("italic", "\033[3m"))
            i += 1
            continue
        if texto[i] == "[":
            fim_rotulo = texto.find("](https://", i)
            if fim_rotulo < 0:
                fim_rotulo = texto.find("](http://", i)
            if fim_rotulo >= 0:
                fim_url = texto.find(")", fim_rotulo + 2)
                if fim_url >= 0:
                    rotulo = texto[i + 1:fim_rotulo]
                    url = texto[fim_rotulo + 2:fim_url]
                    saida.append(f"\033[4m{rotulo}{ANSI['reset']} {ANSI['dim']}<{url}>{ANSI['reset']}")
                    i = fim_url + 1
                    continue
        saida.append(texto[i])
        i += 1
    if any(estilos.values()):
        saida.append(ANSI["reset"])
    return "".join(saida)


def _formatar_markdown_terminal(texto, cores=None):
    """Converte Markdown de chat em uma apresentação ANSI simples e segura."""
    texto = _texto_terminal_seguro(texto)
    cores = _usa_cor() if cores is None else cores
    if not cores:
        return texto
    linhas = []
    em_codigo = False
    for linha in texto.splitlines(keepends=True):
        fim = "\n" if linha.endswith("\n") else ""
        corpo = linha[:-1] if fim else linha
        cerca = re.match(r"^\s*```\s*([^`]*)$", corpo)
        if cerca:
            if em_codigo:
                linhas.append(ANSI["reset"] + fim)
                em_codigo = False
            else:
                linguagem = cerca.group(1).strip()
                rotulo = f" código · {linguagem} " if linguagem else " código "
                linhas.append(f"{ANSI['dim']}──{rotulo}────────────────{ANSI['reset']}" + fim)
                em_codigo = True
            continue
        if em_codigo:
            linhas.append(f"\033[38;5;81m  {corpo}{ANSI['reset']}" + fim)
            continue
        titulo = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", corpo)
        if titulo:
            linhas.append(
                f"{ANSI['assistant']}▌{ANSI['reset']} \033[1m"
                f"{_markdown_inline(titulo.group(1), cores=True)}{ANSI['reset']}" + fim
            )
            continue
        citacao = re.match(r"^\s*>\s?(.*)$", corpo)
        if citacao:
            linhas.append(
                f"{ANSI['dim']}│{ANSI['reset']} "
                f"{_markdown_inline(citacao.group(1), cores=True)}" + fim
            )
            continue
        item = re.match(r"^(\s*)[-+*]\s+(.*)$", corpo)
        if item:
            conteudo = item.group(2)
            conteudo = re.sub(r"^\[ \]\s*", "☐ ", conteudo)
            conteudo = re.sub(r"^\[[xX]\]\s*", "☑ ", conteudo)
            linhas.append(
                f"{item.group(1)}{ANSI['assistant']}•{ANSI['reset']} "
                f"{_markdown_inline(conteudo, cores=True)}" + fim
            )
            continue
        numerado = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", corpo)
        if numerado:
            linhas.append(
                f"{numerado.group(1)}{ANSI['assistant']}{numerado.group(2)}.{ANSI['reset']} "
                f"{_markdown_inline(numerado.group(3), cores=True)}" + fim
            )
            continue
        if re.match(r"^\s*(?:---+|___+|\*\*\*+)\s*$", corpo):
            linhas.append(f"{ANSI['dim']}{'─' * 32}{ANSI['reset']}" + fim)
            continue
        linhas.append(_markdown_inline(corpo, cores=True) + fim)
    if em_codigo:
        linhas.append(ANSI["reset"])
    return "".join(linhas)


def _encurtar(texto, limite):
    texto = str(texto)
    if len(texto) <= limite:
        return texto
    return texto[:max(1, limite - 1)] + "…"


def _largura_visual(texto):
    return len(str(texto))


def _preencher_visual(texto, largura, alinhamento="left"):
    faltam = max(0, largura - _largura_visual(texto))
    if alinhamento == "center":
        esquerda = faltam // 2
        return " " * esquerda + texto + " " * (faltam - esquerda)
    return texto + " " * faltam


def _caminho_amigavel(caminho):
    texto = str(caminho)
    home = str(Path.home())
    return "~" + texto[len(home):] if texto == home or texto.startswith(home + os.sep) else texto


def _linhas_boas_vindas(modelo, motor, workspace, largura=None, usuario=None):
    """Monta o painel inicial sem ANSI para manter alinhamento previsível."""
    colunas = largura if largura is not None else shutil.get_terminal_size((100, 24)).columns - 2
    largura = max(36, min(colunas, 112))
    interior = largura - 2
    usuario = usuario or getpass.getuser().replace("_", " ").title()
    linhas = []

    def corpo(texto=""):
        texto = _encurtar(texto, interior - 2)
        linhas.append("│ " + _preencher_visual(texto, interior - 2) + " │")

    if largura >= 88:
        esquerda = (interior - 5) // 2
        direita = interior - esquerda - 5
        titulo = f"─── Isaac CLI v{APP_VERSION} "
        linhas.append(
            "╭" + titulo + "─" * max(0, esquerda + 2 - len(titulo))
            + "┬" + "─" * (direita + 2) + "╮"
        )
        lado_a = [
            "",
            "",
            *WORDMARK_ISAAC,
            "",
            "",
            "",
        ]
        lado_b = [
            f"Bem-vindo de volta, {usuario}!",
            "",
            "Começando",
            "/help     comandos disponíveis",
            "/setup    modelos e motores",
            "/status   sessão e consumo",
            "/history  ver conversa completa",
            "Shift+Tab alterna permissões",
        ]
        for a, b in zip(lado_a, lado_b):
            linhas.append(
                "│ " + _preencher_visual(_encurtar(a, esquerda), esquerda, "center") + " │ "
                + _preencher_visual(_encurtar(b, direita), direita) + " │"
            )
        linhas.append("├" + "─" * (esquerda + 2) + "┴" + "─" * (direita + 2) + "┤")
    else:
        titulo = f"╭─── Isaac CLI v{APP_VERSION} "
        linhas.append(titulo + "─" * max(0, largura - len(titulo) - 1) + "╮")
        corpo(f"Bem-vindo de volta, {usuario}!")
        for linha_wordmark in WORDMARK_ISAAC:
            corpo(_preencher_visual(linha_wordmark, interior - 2, "center"))
        corpo("/help comandos · /setup modelos · /status sessão")
        linhas.append("├" + "─" * interior + "┤")

    valor = interior - 13
    for rotulo, conteudo in (
        ("modelo", modelo), ("motor", motor),
        ("workspace", _caminho_amigavel(workspace)),
    ):
        corpo(f"{rotulo:<10} {_encurtar(conteudo, valor)}")
    linhas.append("╰" + "─" * interior + "╯")
    return linhas


def _imprimir_boas_vindas(modelo, motor, workspace):
    for indice, linha in enumerate(_linhas_boas_vindas(modelo, motor, workspace)):
        if indice == 0 or linha.startswith(("├", "╰")):
            print(_cor(linha, "assistant"))
            continue
        # A divisória e a moldura usam o verde do Isaac.
        decorada = linha
        for glifo in "│╭╮╯╰─╱╲":
            decorada = decorada.replace(glifo, _cor(glifo, "assistant"))
        # O wordmark é uma unidade visual: todas as letras recebem a mesma cor.
        for trecho in WORDMARK_ISAAC:
            decorada = decorada.replace(trecho, _cor(trecho, "assistant"))
        print(decorada)


def _codigo_saida(resultado):
    m = re.search(r"\(c[oó]digo de sa[ií]da: (-?\d+)\)\s*$", resultado.strip())
    return int(m.group(1)) if m else None


def _partes_comando(cmd):
    try:
        return shlex.split(cmd)
    except ValueError:
        return []


def _regra_comando(cmd):
    partes = _partes_comando(cmd)
    if not partes:
        return ""
    if partes[0] == "git" and len(partes) > 1:
        sub = next((p for p in partes[1:] if not p.startswith("-")), "*")
        return f"git {sub}"
    if partes[0] == "gh" and len(partes) > 2:
        rota = [p for p in partes[1:] if not p.startswith("-")][:2]
        return "gh " + " ".join(rota)
    return partes[0]


def _comando_leitura_segura(cmd):
    partes = _partes_comando(cmd)
    if not partes:
        return False
    if partes[0] in COMANDOS_SOMENTE_LEITURA:
        return True
    if partes[0] == "gh":
        rota = tuple(p for p in partes[1:] if not p.startswith("-"))[:2]
        return rota in GH_SOMENTE_LEITURA
    return (partes[0] == "git" and len(partes) > 1
            and next((p for p in partes[1:] if not p.startswith("-")), None)
            in GIT_SOMENTE_LEITURA)


def _contexto_curto(valor):
    if isinstance(valor, int) and valor % 1024 == 0:
        return f"{valor // 1024}K"
    return str(valor)


def _preview(texto):
    linhas = texto.splitlines()
    cortou_linhas = len(linhas) > MAX_PREVIEW_LINHAS
    if cortou_linhas:
        linhas = linhas[:MAX_PREVIEW_LINHAS]
    saida = "\n".join(linhas)
    cortou_chars = len(saida) > MAX_PREVIEW_CHARS
    if cortou_chars:
        saida = saida[:MAX_PREVIEW_CHARS].rstrip()
    return saida, cortou_linhas or cortou_chars


def _instalar_autocomplete():
    if readline is None:
        return

    def completar(texto, estado):
        opcoes = [c for c in COMANDOS_BARRA if c.startswith(texto)]
        try:
            return opcoes[estado] + " "
        except IndexError:
            return None

    readline.set_completer(completar)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    # Shift+Tab envia /mode como se o usuário o tivesse digitado. Funciona no
    # GNU readline sem substituir o editor de linha (e suas quebras corretas).
    readline.parse_and_bind('"\\e[Z": "/mode\\n"')


def _pontuar_comando(consulta, comando, descricao):
    """Ordena prefixos primeiro e aceita busca aproximada no restante."""
    consulta = consulta.casefold().strip()
    comando_normal = comando.casefold()
    descricao_normal = descricao.casefold()
    if not consulta or consulta == "/":
        return (0, COMANDOS_BARRA.index(comando))
    if comando_normal.startswith(consulta):
        return (0, len(comando))
    termo = consulta.removeprefix("/")
    nome = comando_normal.removeprefix("/")
    if termo in nome:
        return (1, nome.index(termo), len(nome))
    if termo in descricao_normal:
        return (2, descricao_normal.index(termo), len(descricao_normal))
    posicao = 0
    distancia = 0
    for caractere in termo:
        achou = nome.find(caractere, posicao)
        if achou < 0:
            return None
        distancia += achou - posicao
        posicao = achou + 1
    return (3, distancia, len(nome))


def _filtrar_comandos(consulta):
    encontrados = []
    for comando, descricao in COMANDOS_INFO:
        pontos = _pontuar_comando(consulta, comando, descricao)
        if pontos is not None:
            encontrados.append((pontos, comando, descricao))
    encontrados.sort(key=lambda item: item[0])
    return [(comando, descricao) for _pontos, comando, descricao in encontrados]


class _CompletadorComandos(Completer):
    def get_completions(self, documento, _evento):
        texto = documento.text_before_cursor
        if not texto.startswith("/") or any(c.isspace() for c in texto):
            return
        for comando, descricao in _filtrar_comandos(texto):
            yield Completion(
                comando,
                start_position=-len(texto),
                display=comando,
                display_meta=descricao,
            )


_SESSAO_PROMPT = None


def _sessao_prompt():
    global _SESSAO_PROMPT
    if _SESSAO_PROMPT is None:
        teclas = KeyBindings()

        @teclas.add("s-tab")
        def _alternar_permissoes(evento):
            evento.app.exit(result="/mode")

        @teclas.add("escape", "enter")
        def _quebra_de_linha(evento):
            evento.current_buffer.insert_text("\n")

        _SESSAO_PROMPT = PromptSession(
            completer=_CompletadorComandos(),
            complete_while_typing=False,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=12,
            key_bindings=teclas,
            enable_history_search=True,
        )

        def _atualizar_paleta(buffer):
            texto = buffer.document.text_before_cursor
            if texto.startswith("/") and not any(c.isspace() for c in texto):
                buffer.start_completion(select_first=False)
            elif buffer.complete_state is not None:
                buffer.cancel_completion()

        _SESSAO_PROMPT.default_buffer.on_text_changed += _atualizar_paleta
    return _SESSAO_PROMPT


def _ler_entrada():
    if PromptSession is not None and terminal_ui.interativo():
        prompt = FormattedText([("ansibrightcyan bold", "❯ ")])
        return _sessao_prompt().prompt(prompt)
    return input(_prompt_colorido("❯ "))


def _montar_historico(workspace):
    return [{"role": "system", "content": (
        agent.CONHECIMENTO_FERRAMENTAS + "\n\n" +
        CONHECIMENTO_CLI.format(workspace=str(workspace))
    )}]


def _carregar_sessao(session_id):
    """Reconstrói conversa e tool calls de um JSONL local pelo ID exato."""
    if not _session_id_valido(session_id):
        raise ValueError("ID de sessão inválido")
    caminho = SESSOES_DIR / f"{session_id}.jsonl"
    if not caminho.is_file():
        raise ValueError(f"sessão não encontrada: {session_id}")
    if caminho.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("sessão grande demais para retomada automática")

    eventos = []
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), 1):
        try:
            evento = json.loads(linha)
        except json.JSONDecodeError as e:
            raise ValueError(f"log de sessão inválido na linha {numero}") from e
        if isinstance(evento, dict):
            eventos.append(evento)
    if not eventos:
        raise ValueError("sessão vazia")

    workspace = Path(eventos[-1].get("workspace") or os.getcwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace da sessão não existe mais: {workspace}")
    modelo = next((e.get("modelo") for e in reversed(eventos) if e.get("modelo")), None)
    historico = _montar_historico(workspace)
    transcript = []
    tool_pendente = None
    tool_numero = 0
    for evento in eventos:
        tipo = evento.get("tipo")
        if tipo == "meta" and evento.get("evento") == "clear":
            historico = _montar_historico(workspace)
            transcript = []
        elif tipo == "user" and isinstance(evento.get("content"), str):
            historico.append({"role": "user", "content": evento["content"]})
            transcript.append(("user", evento["content"]))
        elif tipo == "tool_start":
            tool_numero += 1
            nome = evento.get("nome") or "unknown"
            args = evento.get("args")
            if args is None and nome == "run_command":
                args = {"cmd": evento.get("cmd", "")}
            tool_pendente = f"resume-tool-{tool_numero}"
            historico.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": tool_pendente, "type": "function",
                                "function": {"name": nome, "arguments": args or {}}}],
            })
            transcript.append(("tool_start", {
                "nome": nome, "args": args or {}, "cmd": evento.get("cmd"),
            }))
        elif tipo == "permission":
            transcript.append(("permission", {
                "cmd": evento.get("cmd"), "decisao": evento.get("decisao"),
            }))
        elif tipo == "tool_result" and isinstance(evento.get("resultado"), str):
            historico.append({"role": "tool", "tool_call_id": tool_pendente or "resume-tool",
                              "content": evento["resultado"]})
            transcript.append(("tool_result", {
                "nome": evento.get("nome") or "unknown",
                "codigo": evento.get("codigo"),
                "resultado": evento["resultado"],
            }))
            tool_pendente = None
        elif tipo == "assistant_final" and isinstance(evento.get("content"), str):
            historico.append({"role": "assistant", "content": evento["content"]})
            if evento["content"]:
                transcript.append(("assistant", evento["content"]))
    return {"id": session_id, "path": caminho, "workspace": workspace,
            "model": modelo, "history": historico, "transcript": transcript}


class IsaacCLI:
    def __init__(self, modelo, workspace, max_passos, autostart_ollama=True,
                 thinking=None, num_ctx=None, config_file=None, provider=None):
        self.modelo = modelo
        self.thinking = thinking
        self.num_ctx = num_ctx
        self.config_file = config_file
        self.provider = provider or {"provider": "ollama"}
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_passos = max_passos
        self.autostart_ollama = autostart_ollama
        self.ollama_proc = None
        self._ollama_registrado = False
        self._runtime_pid = os.getpid()
        self._runtime_start = _identidade_pid(self._runtime_pid)
        self.historico = []
        self.session_id = _novo_session_id()
        self.session_path = SESSOES_DIR / f"{self.session_id}.jsonl"
        self.feedback_path = FEEDBACK_DIR / f"{self.session_id}.jsonl"
        self.turnos = 0
        self.falhas = 0
        self.comandos = []
        self.uso_total = {"prompt_eval_count": 0, "eval_count": 0,
                          "total_duration": 0, "eval_duration": 0}
        self.ultima_resposta = ""
        self.feedbacks = 0
        self.modo_permissao = "seguro"
        self._working_visivel = False
        self._rotulo_assistente_pendente = True
        self._geracao_inicio = None
        self._turno_inicio = None
        self._geracao_pedacos = 0
        self._geracao_status_em = 0.0
        self._token_buffer = []
        self._primeiro_token_em = None
        self._stream_iniciado = False
        self._bloco_saida = False
        self.transcript_retomada = []
        self.definir_workspace(self.workspace, reset=True)
        self._log("meta", evento="inicio", pid=os.getpid(), modelo=self.modelo,
                  workspace=str(self.workspace))

    def _provider_do_perfil(self, item):
        if not item or item.get("provider", "ollama") == "ollama":
            return {"provider": "ollama"}
        segredo_path = (Path(self.config_file).with_name("secrets.json")
                        if self.config_file else None)
        return {
            "provider": "openai_compatible",
            "provider_name": item.get("provider_name") or "API",
            "base_url": item.get("base_url"),
            "api_key": config.carregar_segredo(item.get("credential"), segredo_path),
        }

    def garantir_ollama(self, avisar=False):
        if self.provider.get("provider") != "ollama":
            return ((self.provider.get("provider_name") or "API")
                    if self.provider.get("api_key") and self.provider.get("base_url") else None)
        with _estado_ollama_compartilhado() as estado:
            versao = _ollama_ok()
            servidor_valido = (
                estado.get("managed")
                and _processo_igual(estado.get("server_pid"), estado.get("server_start"))
            )
            clientes = [
                item for item in estado.get("clients", [])
                if _processo_igual(item.get("pid"), item.get("start"))
            ]
            if not servidor_valido:
                estado.clear()
                clientes = []
            if versao:
                if servidor_valido:
                    atual = {"pid": self._runtime_pid, "start": self._runtime_start}
                    clientes = [c for c in clientes if c.get("pid") != self._runtime_pid]
                    clientes.append(atual)
                    estado["clients"] = clientes
                    self._ollama_registrado = True
                # Sem estado válido, o servidor pertence ao usuário/sistema.
                return versao
            if not self.autostart_ollama:
                return None
            exe = shutil.which("ollama")
            if not exe:
                if avisar:
                    print(_cor("Ollama nao encontrado no PATH.", "bad"))
                return None

            if avisar:
                print(_cor("Ollama nao estava rodando; iniciando ollama serve...", "warn"))
            try:
                self.ollama_proc = subprocess.Popen(
                    [exe, "serve"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                self._log("error", erro=f"ollama_autostart: {e}")
                if avisar:
                    print(_cor(f"Nao consegui iniciar ollama serve: {e}", "bad"))
                return None

            self._log("meta", evento="ollama_autostart", pid=self.ollama_proc.pid)
            for _ in range(40):
                time.sleep(0.25)
                versao = _ollama_ok(timeout=1)
                if versao:
                    estado.update({
                        "managed": True,
                        "server_pid": self.ollama_proc.pid,
                        "server_start": _identidade_pid(self.ollama_proc.pid),
                        "clients": [{"pid": self._runtime_pid,
                                     "start": self._runtime_start}],
                    })
                    self._ollama_registrado = True
                    return versao
                if self.ollama_proc.poll() is not None:
                    self._log("error", erro=(
                        f"ollama serve saiu com codigo {self.ollama_proc.returncode}"
                    ))
                    return None
            versao = _ollama_ok(timeout=1)
            if versao:
                estado.update({
                    "managed": True,
                    "server_pid": self.ollama_proc.pid,
                    "server_start": _identidade_pid(self.ollama_proc.pid),
                    "clients": [{"pid": self._runtime_pid,
                                 "start": self._runtime_start}],
                })
                self._ollama_registrado = True
                return versao
            if self.ollama_proc.poll() is None:
                self.ollama_proc.terminate()
                self.ollama_proc.wait(timeout=3)
            return None

    def fechar(self):
        if not self._ollama_registrado:
            return
        with _estado_ollama_compartilhado() as estado:
            clientes = [
                item for item in estado.get("clients", [])
                if item.get("pid") != self._runtime_pid
                and _processo_igual(item.get("pid"), item.get("start"))
            ]
            estado["clients"] = clientes
            servidor_pid = estado.get("server_pid")
            servidor_valido = (
                estado.get("managed")
                and _processo_igual(servidor_pid, estado.get("server_start"))
            )
            if clientes or not servidor_valido:
                if not servidor_valido:
                    estado.clear()
                self._ollama_registrado = False
                return

            # A trava continua ativa até o processo sair: uma nova sessão não
            # pode enxergar o servidor e se registrar no intervalo do shutdown.
            try:
                os.kill(int(servidor_pid), signal.SIGTERM)
                limite = time.monotonic() + 3
                while _processo_igual(servidor_pid, estado.get("server_start")):
                    if time.monotonic() >= limite:
                        os.kill(int(servidor_pid), signal.SIGKILL)
                        break
                    time.sleep(0.05)
            except ProcessLookupError:
                pass
            estado.clear()
            self._ollama_registrado = False
            self._log("meta", evento="ollama_autostop", pid=servidor_pid)

    def _log(self, tipo, **dados):
        SESSOES_DIR.mkdir(parents=True, exist_ok=True)
        evento = {
            "ts": _agora(),
            "tipo": tipo,
            "session_id": self.session_id,
            "modelo": self.modelo,
            "workspace": str(self.workspace),
            **dados,
        }
        with self.session_path.open("a") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")

    def definir_workspace(self, caminho, reset=False):
        novo = Path(caminho).expanduser().resolve()
        if not novo.exists():
            raise FileNotFoundError(f"workspace nao existe: {novo}")
        if not novo.is_dir():
            raise NotADirectoryError(f"workspace nao e diretorio: {novo}")
        self.workspace = novo
        tools.SANDBOX_ROOT = novo
        if reset or not self.historico:
            self.historico = _montar_historico(novo)
        else:
            self.historico.append({
                "role": "system",
                "content": f"A pasta de trabalho agora e: {novo}",
            })
            self._log("meta", evento="workspace", workspace=str(novo))

    def ajuda(self):
        print("""comandos:
  /help                 mostra isto
  /setup                configura ou repara o motor sem fechar o Isaac
  /status               mostra sessao, workspace, modelo e consumo Ollama
  /tools                lista ferramentas e comandos de terminal permitidos
  /sessions             lista sessoes CLI salvas
  /history              mostra a conversa completa desta sessao
  /show [n|last]        expande a saida completa de um comando recolhido
  /log                  mostra o arquivo JSONL desta sessao
  /feedback             mostra como avaliar a sessao/tarefa
  /bom [comentario]     marca a ultima tarefa como boa
  /ruim [comentario]    marca a ultima tarefa como ruim
  /nota 0-10 [texto]    da uma nota numerica para a ultima tarefa
  /workspace [caminho]  mostra ou troca a pasta de trabalho
  /model [perfil|nome]  seleciona modelo, esforço e contexto
  /permissions          mostra autorizações persistentes aplicáveis
  /mode                 alterna entre automático seguro e somente autorizados
  /language             muda o idioma da interface (fica salvo)
  /clear                limpa o contexto da conversa
  /new                  encerra a sessão atual e inicia outra
  /exit                 sai

atalho:
  digite / para buscar comandos; setas selecionam e Tab completa
  Alt+Enter insere uma quebra de linha sem enviar a mensagem
  Shift+Tab alterna o modo de aprovação de comandos

exemplos:
  rode git status
  veja o diff e diga o que falta testar
  commite as mudancas desta task com uma mensagem curta
""")

    def comando_interno(self, texto):
        if not texto.startswith("/"):
            return False
        if texto == "/":
            self.ajuda()
            return True
        partes = texto.split(maxsplit=1)
        cmd = partes[0]
        arg = partes[1].strip() if len(partes) > 1 else ""

        if cmd in ("/exit", "/quit"):
            raise EOFError
        if cmd in ("/help", "/?"):
            self.ajuda()
            return True
        if cmd == "/setup":
            import setup_ollama
            codigo = setup_ollama.executar_setup(config_file=self.config_file)
            if codigo == 0:
                try:
                    dado = config.carregar(self.config_file)
                    nome, item = config.perfil(dado)
                except ValueError as e:
                    self.redesenhar_sessao(
                        f"setup concluiu, mas não consegui reler a configuração: {e}"
                    )
                    return True
                if item:
                    self.modelo = item["model"]
                    self.thinking = item.get("thinking")
                    self.num_ctx = item.get("num_ctx")
                    self.provider = self._provider_do_perfil(item)
                    self._log("meta", evento="setup", perfil=nome,
                              modelo=self.modelo, thinking=self.thinking)
                    self.redesenhar_sessao(
                        f"Perfil carregado nesta sessão: {nome} ({self.modelo})"
                    )
            else:
                self.redesenhar_sessao(
                    "setup cancelado; o modelo anterior continua ativo"
                    if codigo == 130 else
                    "setup não concluído; corrija o motor e tente novamente"
                )
            return True
        if cmd == "/status":
            self.status()
            return True
        if cmd == "/tools":
            self.listar_tools()
            return True
        if cmd == "/sessions":
            self.listar_sessoes()
            return True
        if cmd == "/history":
            self.mostrar_historico(arg)
            return True
        if cmd == "/show":
            self.mostrar_comando(arg or "last")
            return True
        if cmd == "/log":
            print(self.session_path)
            return True
        if cmd == "/feedback":
            self.ajuda_feedback()
            return True
        if cmd == "/bom":
            self.salvar_feedback("bom", 10, arg)
            return True
        if cmd == "/ruim":
            self.salvar_feedback("ruim", 0, arg)
            return True
        if cmd == "/nota":
            self.comando_nota(arg)
            return True
        if cmd == "/workspace":
            if not arg:
                print(self.workspace)
            else:
                self.definir_workspace(arg)
                print(f"workspace: {self.workspace}")
            return True
        if cmd == "/model":
            if not arg:
                self.selecionar_modelo()
            else:
                try:
                    dado = config.carregar(self.config_file)
                except ValueError:
                    dado = config.config_vazia()
                item = (dado.get("profiles") or {}).get(arg)
                if item:
                    self.modelo = item["model"]
                    self.thinking = item.get("thinking")
                    self.num_ctx = item.get("num_ctx")
                    self.provider = self._provider_do_perfil(item)
                    origem = f"perfil {arg}"
                else:
                    self.modelo = arg
                    self.thinking = None
                    self.num_ctx = None
                    self.provider = {"provider": "ollama"}
                    origem = "nome Ollama direto"
                self._log("meta", evento="model", modelo=self.modelo,
                          thinking=self.thinking)
                print(f"modelo: {self.modelo} ({origem})")
            return True
        if cmd == "/mode":
            self.modo_permissao = (
                "somente_autorizados" if self.modo_permissao == "seguro" else "seguro"
            )
            print("modo de comandos: " + (
                "somente autorizações salvas" if self.modo_permissao == "somente_autorizados"
                else "automático seguro (somente leitura)"
            ))
            self._log("meta", evento="permission_mode", modo=self.modo_permissao)
            return True
        if cmd == "/permissions":
            try:
                dado = config.carregar(self.config_file)
            except ValueError as e:
                print(f"configuração: {e}")
                return True
            permissoes = dado.get("permissions") or {}
            if arg in ("clear workspace", "clear global"):
                if arg == "clear global":
                    permissoes["global"] = []
                else:
                    (permissoes.get("workspaces") or {}).pop(str(self.workspace), None)
                config.salvar(dado, self.config_file)
                print("autorizações removidas: " + arg.removeprefix("clear "))
                return True
            globais = permissoes.get("global") or []
            locais = (permissoes.get("workspaces") or {}).get(str(self.workspace), [])
            print(f"modo: {self.modo_permissao}")
            print("globais: " + (", ".join(globais) if globais else "(nenhuma)"))
            print("neste workspace: " + (", ".join(locais) if locais else "(nenhuma)"))
            print("remover: /permissions clear workspace | /permissions clear global")
            return True
        if cmd == "/language":
            from i18n import SUPPORTED_LANGUAGES
            try:
                dado = config.carregar(self.config_file)
            except ValueError:
                dado = config.config_vazia()
            codigos = list(SUPPORTED_LANGUAGES)
            atual = dado.get("language") or "en"
            inicial = codigos.index(atual) if atual in codigos else 0
            indice = terminal_ui.selecionar(
                "Isaac CLI · Language / Idioma",
                [SUPPORTED_LANGUAGES[codigo] for codigo in codigos],
                inicial=inicial,
            )
            dado["language"] = codigos[indice]
            config.salvar(dado, self.config_file)
            print(f"language: {SUPPORTED_LANGUAGES[codigos[indice]]}")
            self._log("meta", evento="language", idioma=codigos[indice])
            return True
        if cmd == "/clear":
            self.historico = _montar_historico(self.workspace)
            self._log("meta", evento="clear")
            print("contexto limpo")
            return True
        if cmd == "/new":
            self.nova_sessao()
            return True

        print(f"comando desconhecido: {cmd}  (use /help)")
        return True

    def selecionar_modelo(self):
        import setup_ollama

        codigo = setup_ollama.executar_seletor_modelo(config_file=self.config_file)
        if codigo != 0:
            mensagem = ("seleção de modelo cancelada" if codigo == 130
                        else "modelo não alterado")
            self.redesenhar_sessao(mensagem)
            return
        try:
            dado = config.carregar(self.config_file)
        except ValueError as e:
            print(f"configuração: {e}")
            return
        nome, item = config.perfil(dado)
        if not item:
            print("modelo não alterado: perfil selecionado ausente")
            return
        self.modelo = item["model"]
        self.thinking = item.get("thinking")
        self.num_ctx = item.get("num_ctx")
        self.provider = self._provider_do_perfil(item)
        self._log("meta", evento="model", perfil=nome, modelo=self.modelo,
                  thinking=self.thinking, num_ctx=self.num_ctx)
        contexto = f" · contexto {_contexto_curto(self.num_ctx)}" if self.num_ctx else ""
        esforco = (f" · esforço {self.thinking}"
                   if self.thinking not in (None, False) else " · sem raciocínio")
        self.redesenhar_sessao(f"modelo: {nome}{contexto}{esforco}")

    def nova_sessao(self):
        anterior_id = self.session_id
        anterior_path = self.session_path
        novo_id = _novo_session_id()
        self._log("meta", evento="nova_sessao", proxima_sessao=novo_id)

        self.session_id = novo_id
        self.session_path = SESSOES_DIR / f"{novo_id}.jsonl"
        self.feedback_path = FEEDBACK_DIR / f"{novo_id}.jsonl"
        self.turnos = 0
        self.falhas = 0
        self.comandos = []
        self.uso_total = {"prompt_eval_count": 0, "eval_count": 0,
                          "total_duration": 0, "eval_duration": 0}
        self.ultima_resposta = ""
        self.feedbacks = 0
        self.transcript_retomada = []
        self._working_visivel = False
        self._rotulo_assistente_pendente = True
        self._token_buffer = []
        self._bloco_saida = False
        self.definir_workspace(self.workspace, reset=True)
        self._log("meta", evento="inicio", pid=os.getpid(), modelo=self.modelo,
                  workspace=str(self.workspace), sessao_anterior=anterior_id)

        terminal_ui.limpar()
        print(_cor(f"Nova sessão · {novo_id}", "assistant"))
        print(_cor(f"Anterior · {anterior_path}", "dim"))
        print(_cor(f"Retomar · {_comando_retomada(anterior_id)}", "dim"))

    def status(self):
        versao = (_ollama_ok() or "sem resposta") if self.provider.get("provider") == "ollama" else (
            self.provider.get("provider_name") or "API compatível com OpenAI")
        duracao_s = self.uso_total.get("total_duration", 0) / 1_000_000_000
        print(f"sessao: {self.session_id}")
        print(f"log: {self.session_path}")
        print(f"pid: {os.getpid()}")
        print(f"modelo: {self.modelo}")
        print(f"raciocinio: {self.thinking if self.thinking is not None else 'padrao do modelo'}")
        print(f"contexto: {_contexto_curto(self.num_ctx) if self.num_ctx else 'padrao do modelo'}")
        print(f"workspace: {self.workspace}")
        print(f"motor: {versao}")
        print(f"turnos: {self.turnos}")
        print(f"comandos: {len(self.comandos)}  falhas: {self.falhas}")
        print(f"permissões: {self.modo_permissao}")
        print(f"feedbacks: {self.feedbacks}  arquivo: {self.feedback_path}")
        print(
            "tokens Ollama: "
            f"prompt={self.uso_total.get('prompt_eval_count', 0)} "
            f"resposta={self.uso_total.get('eval_count', 0)} "
            f"tempo={duracao_s:.2f}s"
        )
        print("slash: " + " ".join(COMANDOS_BARRA))
        self._log("status", turnos=self.turnos, comandos=len(self.comandos),
                  falhas=self.falhas, uso=self.uso_total)

    def listar_tools(self):
        import execucao

        nomes = [s["function"]["name"] for s in tools.SCHEMA]
        print("ferramentas: " + ", ".join(nomes))
        print("terminal: " + ", ".join(sorted(execucao.PERMITIDOS)))
        print("git: " + ", ".join(sorted(execucao.GIT_PERMITIDOS)) + " (sem --force)")

    def listar_sessoes(self):
        SESSOES_DIR.mkdir(parents=True, exist_ok=True)
        arquivos = sorted(SESSOES_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not arquivos:
            print("(nenhuma sessao CLI salva)")
            return
        for p in arquivos[:12]:
            stat = p.stat()
            mod = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            atual = "  atual" if p == self.session_path else ""
            print(f"{p.stem}  {mod}  {stat.st_size} bytes{atual}")

    def _texto_historico(self):
        eventos = list(self.transcript_retomada)
        try:
            eventos.extend(_carregar_sessao(self.session_id)["transcript"])
        except ValueError:
            pass
        linhas = []
        for papel, conteudo in eventos:
            if papel == "user":
                linhas.extend(["", f"❯ {conteudo}"])
            elif papel == "assistant":
                linhas.extend(["", f"isaac: {conteudo}"])
            elif papel == "tool_start":
                nome = conteudo.get("nome") or "unknown"
                if nome == "run_command":
                    cmd = conteudo.get("cmd") or (conteudo.get("args") or {}).get("cmd", "")
                    linhas.extend(["", f"$ {cmd}"])
                else:
                    args = json.dumps(conteudo.get("args") or {}, ensure_ascii=False)
                    linhas.extend(["", f"[{nome}] → {args}"])
            elif papel == "permission":
                linhas.append(f"permissão: {conteudo.get('decisao') or 'desconhecida'}")
            elif papel == "tool_result":
                resultado = conteudo.get("resultado") or ""
                linhas.append(resultado)
        return "\n".join(linhas).strip() or "(a sessão ainda não possui mensagens)"

    def redesenhar_sessao(self, mensagem=None):
        """Restaura a conversa depois que um menu de tela inteira foi fechado."""
        if not terminal_ui.interativo():
            if mensagem:
                print(mensagem)
            return
        terminal_ui.limpar()
        if self.provider.get("provider") == "ollama":
            versao = self.garantir_ollama(avisar=False)
            motor = f"Ollama {versao}" if versao else "MOTOR INDISPONÍVEL · use /setup"
        else:
            motor = self.provider.get("provider_name") or "API compatível com OpenAI"
        _imprimir_boas_vindas(self.modelo, motor, self.workspace)
        texto = self._texto_historico()
        if not texto.startswith("(a sessão ainda não possui mensagens)"):
            altura = max(5, shutil.get_terminal_size((100, 30)).lines - 17)
            linhas = texto.splitlines()
            usuarios = [i for i, linha in enumerate(linhas) if linha.startswith("❯ ")]
            inicio_turno = usuarios[-1] if usuarios else max(0, len(linhas) - altura)
            turno = linhas[inicio_turno:]
            if len(turno) <= altura:
                inicio = max(0, len(linhas) - altura)
                visiveis = linhas[inicio:]
                omitidas = inicio
            else:
                # A pergunta recente não pode sumir atrás de uma resposta longa.
                cabeca = turno[:2]
                espaco_cauda = max(1, altura - len(cabeca) - 1)
                visiveis = [*cabeca, "… trecho intermediário omitido …", *turno[-espaco_cauda:]]
                omitidas = len(linhas) - len(visiveis)
            if omitidas:
                print(_cor(f"… {omitidas} linha(s) acima · /history mostra tudo", "dim"))
            print(_formatar_markdown_terminal("\n".join(visiveis)))
        if mensagem:
            print("\n" + _cor(mensagem, "dim"))
        print()

    def mostrar_historico(self, movimento=""):
        # Impresso normal, sem tela cheia nem captura do mouse: fica no
        # scrollback nativo do terminal, com markdown formatado e copiável.
        print(_formatar_markdown_terminal(self._texto_historico()))

    def mostrar_comando(self, qual):
        if not self.comandos:
            print("nenhum comando executado nesta sessao")
            return
        if qual == "last":
            item = self.comandos[-1]
        else:
            try:
                numero = int(qual)
            except ValueError:
                print("use /show last ou /show <numero>")
                return
            item = next((c for c in self.comandos if c["id"] == numero), None)
            if item is None:
                print(f"comando #{numero} nao existe nesta sessao")
                return
        print(_cor(f"comando #{item['id']} completo: {item['cmd']}", "tool"))
        print(item["resultado"])

    def ajuda_feedback(self):
        print("""avaliacao:
  /bom [comentario]       salva que a ultima tarefa foi boa
  /ruim [comentario]      salva que a ultima tarefa foi ruim
  /nota 0-10 [comentario] salva uma nota numerica

destino:
  feedback bruto: {feedback}
  sessao completa: {sessao}

uso posterior:
  isto NAO entra direto no conhecimento do Isaac. Uma task posterior deve passar
  esses registros por juiz/curadoria antes de destilar ou fixar aprendizado.
""".format(feedback=self.feedback_path, sessao=self.session_path))

    def comando_nota(self, arg):
        if not arg:
            print("use: /nota 0-10 [comentario]")
            return
        partes = arg.split(maxsplit=1)
        try:
            nota = int(partes[0])
        except ValueError:
            print("nota precisa ser numero inteiro de 0 a 10")
            return
        if nota < 0 or nota > 10:
            print("nota precisa ficar entre 0 e 10")
            return
        comentario = partes[1] if len(partes) > 1 else ""
        self.salvar_feedback("nota", nota, comentario)

    def salvar_feedback(self, tipo, nota, comentario):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        evento = {
            "ts": _agora(),
            "feedback_tipo": tipo,
            "nota": nota,
            "comentario": comentario,
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "modelo": self.modelo,
            "workspace": str(self.workspace),
            "turnos": self.turnos,
            "comandos": len(self.comandos),
            "falhas": self.falhas,
            "uso": self.uso_total,
            "ultima_resposta": self.ultima_resposta,
        }
        with self.feedback_path.open("a") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")
        self.feedbacks += 1
        self._log("feedback", **evento)
        print(f"feedback salvo: nota={nota} arquivo={self.feedback_path}")

    def lembrar_feedback(self, houve_comando):
        if self.turnos == 0:
            return
        if not houve_comando and self.turnos % 3 != 0:
            return
        print()
        print(_cor("Avaliação opcional: /bom [comentário], /ruim [comentário] ou /nota 0-10 [comentário]", "dim"))

    def _mostrar_working(self):
        self._rotulo_assistente_pendente = True
        self._geracao_inicio = time.monotonic()
        self._geracao_pedacos = 0
        self._geracao_status_em = 0.0
        self._token_buffer = []
        self._primeiro_token_em = None
        self._stream_iniciado = False
        print()
        self._bloco_saida = False
        print(_cor("Trabalhando… · aguardando tokens", "dim"), end="", flush=True)
        self._working_visivel = True

    def _despejar_tokens(self):
        if not self._token_buffer:
            return False
        texto = "".join(self._token_buffer)
        self._token_buffer = []
        if self._working_visivel:
            print("\r\033[2K", end="", flush=True)
            self._working_visivel = False
        if self._rotulo_assistente_pendente:
            print(_cor("isaac:", "assistant"), end=" ", flush=True)
            self._rotulo_assistente_pendente = False
        print(_formatar_markdown_terminal(texto), end="", flush=True)
        self._stream_iniciado = True
        self._bloco_saida = True
        return True

    def _limpar_working(self):
        if self._despejar_tokens():
            return
        if self._working_visivel:
            print("\r\033[2K", end="", flush=True)
            self._working_visivel = False

    def _token(self, pedaco):
        if not pedaco:
            return
        agora = time.monotonic()
        if self._primeiro_token_em is None:
            self._primeiro_token_em = agora
        self._token_buffer.append(pedaco)
        self._geracao_pedacos += 1
        decorrido = agora - self._primeiro_token_em
        if decorrido >= 0.05 and agora - self._geracao_status_em >= 0.05:
            print(
                "\r\033[2K" + _cor(
                    f"Trabalhando… · ≈ {max(self._geracao_pedacos - 1, 1) / decorrido:.1f} tok/s",
                    "dim",
                ),
                end="", flush=True,
            )
            self._geracao_status_em = agora
        # Mantemos a etapa inteira no buffer para não quebrar marcadores Markdown
        # (por exemplo ** e ```) que podem chegar divididos entre vários tokens.

    def _thinking_token(self, pedaco):
        """Conta o stream de raciocínio sem revelar seu conteúdo no terminal."""
        if not pedaco or not self._working_visivel:
            return
        agora = time.monotonic()
        if self._primeiro_token_em is None:
            self._primeiro_token_em = agora
        self._geracao_pedacos += 1
        decorrido = agora - self._primeiro_token_em
        if decorrido >= 0.05 and agora - self._geracao_status_em >= 0.05:
            print(
                "\r\033[2K" + _cor(
                    f"Trabalhando… · ≈ {max(self._geracao_pedacos - 1, 1) / decorrido:.1f} tok/s",
                    "dim",
                ),
                end="", flush=True,
            )
            self._geracao_status_em = agora

    def _tool_antes(self, nome, args):
        self._limpar_working()
        if self._bloco_saida:
            print()
        try:
            dados = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            dados = {}
        if nome == "run_command":
            cmd = dados.get("cmd", args)
            print(_cor(f"$ {cmd}", "tool"), flush=True)
            self._bloco_saida = True
            self._log("tool_start", nome=nome, cmd=cmd)
            return self._aprovar_e_executar(cmd)
        else:
            resumo = json.dumps(dados, ensure_ascii=False) if dados else str(args)
            print(_cor(f"[{nome}] → {resumo[:180]}", "tool"), flush=True)
            self._bloco_saida = True
            self._log("tool_start", nome=nome, args=dados or args)

    def _aprovar_e_executar(self, cmd):
        """Aplica política humana antes de entregar o comando ao bwrap."""
        import execucao

        regra = _regra_comando(cmd)
        # Valida antes de oferecer uma escolha: aprovar algo que continuará
        # proibido (shell, find destrutivo, force-push) seria enganoso.
        try:
            execucao.revisar(cmd, autorizado=True)
        except execucao.Recusado as e:
            return f"$ {cmd}\nRECUSADO: {e}\n(código de saída: 126)"
        try:
            dado = config.carregar(self.config_file)
        except ValueError:
            dado = config.config_vazia()
        salvo = regra and regra in config.regras_permissao(dado, self.workspace)
        automatico = self.modo_permissao == "seguro" and _comando_leitura_segura(cmd)
        if salvo or automatico:
            return execucao.run_command(cmd, autorizado=salvo)

        print(_cor("Permissão necessária · use ↑/↓ e Enter ou w/g/n", "warn"))
        print("O comando poderá alterar este workspace; o sandbox não expõe o restante do computador.")
        try:
            indice = terminal_ui.selecionar_inline(
                [
                    "Permitir uma vez",
                    f"Sempre permitir “{regra}” neste workspace  [w]",
                    f"Sempre permitir “{regra}” globalmente  [g]",
                    "Recusar  [n]",
                ],
                atalhos={"w": 1, "g": 2, "n": 3}, input_fn=input, inicial=0,
            )
        except (EOFError, KeyboardInterrupt):
            indice = 3
            print()
        if indice == 3:
            self._log("permission", cmd=cmd, regra=regra, decisao="recusado")
            return (f"$ {cmd}\nRECUSADO PELO USUÁRIO: o comando não foi autorizado.\n"
                    "(código de saída: 126)")
        if indice in (1, 2) and regra:
            config.adicionar_permissao(
                dado, regra, workspace=self.workspace if indice == 1 else None,
            )
            config.salvar(dado, self.config_file)
        decisao = {0: "uma_vez", 1: "workspace", 2: "global"}[indice]
        self._log("permission", cmd=cmd, regra=regra, decisao=decisao)
        return execucao.run_command(cmd, autorizado=True)

    def _tool_depois(self, nome, args, resultado, _via):
        if nome == "run_command":
            try:
                dados = json.loads(args) if isinstance(args, str) else (args or {})
            except json.JSONDecodeError:
                dados = {}
            cmd = dados.get("cmd", args)
            codigo = _codigo_saida(resultado)
            item = {
                "id": len(self.comandos) + 1,
                "cmd": cmd,
                "codigo": codigo,
                "resultado": resultado,
                "recusado": "RECUSADO PELO USUÁRIO:" in resultado,
            }
            self.comandos.append(item)
            if codigo is not None and codigo != 0 and not item["recusado"]:
                self.falhas += 1
            if item["recusado"]:
                status, cor = "recusado pelo usuário", "warn"
            elif codigo == 0:
                status, cor = "ok", "tool"
            else:
                status = f"falhou · código {codigo}" if codigo is not None else "sem código"
                cor = "bad"
            print(_cor(f"comando #{item['id']} · {status}", cor), flush=True)
            texto, cortado = _preview(resultado)
            print(texto, flush=True)
            if cortado:
                print(_cor(f"[saida recolhida: use /show {item['id']} para ver tudo]", "dim"), flush=True)
            self._log("tool_result", nome=nome, cmd=cmd, codigo=codigo, resultado=resultado)
        else:
            texto, cortado = _preview(resultado)
            print(_cor(f"[{nome}] ← {texto}", "tool"), flush=True)
            if cortado:
                print(_cor("[saida recolhida no log da sessao]", "dim"), flush=True)
            self._log("tool_result", nome=nome, resultado=resultado)
        self._bloco_saida = True
        self._rotulo_assistente_pendente = True

    def perguntar(self, pedido):
        if not self.garantir_ollama(avisar=True):
            if self.provider.get("provider") == "ollama":
                print("ERRO: Ollama não respondeu e não consegui iniciar automaticamente.")
                erro = "ollama_indisponivel"
            else:
                print("ERRO: credencial ou endpoint da API ausente. Use /setup para reparar.")
                erro = "api_indisponivel"
            self._log("error", erro=erro)
            return 1
        self._log("user", content=pedido)
        comandos_antes = len(self.comandos)
        self._turno_inicio = time.monotonic()
        try:
            with terminal_ui.entrada_ocupada():
                r = agent.rodar(
                    pedido,
                    self.modelo,
                    max_passos=self.max_passos,
                    verbose=False,
                    on_token=self._token,
                    on_thinking=self._thinking_token,
                    on_tool_antes=self._tool_antes,
                    on_tool=self._tool_depois,
                    on_working=self._mostrar_working,
                    historico=self.historico,
                    thinking=self.thinking,
                    num_ctx=self.num_ctx,
                    provider=self.provider,
                )
        except RuntimeError as e:
            self._limpar_working()
            print(f"ERRO: {e}")
            self._log("error", erro=str(e))
            return 1
        except urllib.error.URLError as e:
            self._limpar_working()
            if self.provider.get("provider") != "ollama":
                print(f"ERRO: a API não respondeu ({e}).")
            elif self.garantir_ollama(avisar=True):
                print("\nOllama iniciou agora; tente o pedido de novo.")
            else:
                print(f"\nERRO: Ollama nao respondeu ({e}) e o inicio automatico falhou.")
            self._log("error", erro=str(e))
            return 1
        except KeyboardInterrupt:
            self._limpar_working()
            print("\ninterrompido")
            self._log("error", erro="interrompido")
            return 130

        self._limpar_working()
        final = (r or {}).get("final") or ""
        chamadas = (r or {}).get("chamadas") or []
        resposta_vazia = not final and not chamadas
        self.ultima_resposta = final
        uso = (r or {}).get("uso") or {}
        for chave in self.uso_total:
            self.uso_total[chave] += int(uso.get(chave) or 0)
        self.turnos += 1
        if final and not final.endswith("\n"):
            print()
        if resposta_vazia:
            print(_cor(
                "ERRO: o modelo terminou sem resposta visível nem chamada de ferramenta.",
                "bad",
            ))
        eval_count = int(uso.get("eval_count") or 0)
        eval_duration = int(uso.get("eval_duration") or 0) / 1_000_000_000
        tempo_medicao = eval_duration or max(
            time.monotonic() - (self._turno_inicio or time.monotonic()), 0.001,
        )
        if eval_count:
            aproximado = "" if eval_duration else "≈ "
            print()
            print(_cor(
                f"{aproximado}{eval_count / tempo_medicao:.1f} tok/s · "
                f"{eval_count} tokens gerados", "dim",
            ))
        pediu_mutacao = bool(re.search(
            r"\b(apag(?:ue|ar)|delet(?:e|ar)|remov(?:a|er)|exclu(?:a|ir)|"
            r"cri(?:e|ar)|edit(?:e|ar)|alter(?:e|ar)|modifi(?:que|car)|"
            r"delete|remove|create|edit|modify)\b", pedido, re.IGNORECASE,
        ))
        if pediu_mutacao and not chamadas:
            print(_cor(
                "\nAviso do Isaac CLI: nenhuma ferramenta de alteração foi executada; "
                "portanto nenhuma mudança foi confirmada.", "warn",
            ))
        novos = self.comandos[comandos_antes:]
        if (novos and not novos[-1].get("recusado")
                and novos[-1].get("codigo") not in (None, 0)):
            print(_cor(
                f"\nNota do Isaac CLI: o último comando falhou com código {novos[-1]['codigo']}; "
                "trate a resposta acima como não concluída se ela disser sucesso.",
                "warn",
            ))
        self._log("assistant_final", content=final, uso=uso,
                  chamadas=len((r or {}).get("chamadas") or []))
        self.lembrar_feedback(bool(novos))
        print()
        return 1 if resposta_vazia else 0

    def repl(self):
        with terminal_ui.tela_alternativa():
            codigo = self._repl_tela()
        print()
        print(_cor("Retome esta sessão com:", "dim"))
        print(_comando_retomada(self.session_id))
        print()
        return codigo

    def _inicializar_repl(self):
        _instalar_autocomplete()
        versao = self.garantir_ollama(avisar=False)
        if self.provider.get("provider") == "ollama":
            motor = f"Ollama {versao}" if versao else "MOTOR INDISPONÍVEL · use /setup"
        else:
            motor = self.provider.get("provider_name") or "API compatível com OpenAI"
        _imprimir_boas_vindas(self.modelo, motor, self.workspace)
        print()

        if self.transcript_retomada:
            limite = 20
            itens = self.transcript_retomada[-limite:]
            omitidas = len(self.transcript_retomada) - len(itens)
            print(_cor("conversa retomada", "dim"))
            if omitidas:
                print(_cor(f"… {omitidas} evento(s) anterior(es) omitido(s)", "dim"))
            for papel, conteudo in itens:
                if papel in ("user", "assistant"):
                    texto = conteudo if len(conteudo) <= 2000 else conteudo[:2000] + "…"
                    rotulo = "❯" if papel == "user" else "isaac:"
                    cor = "prompt" if papel == "user" else "assistant"
                    visivel = (_formatar_markdown_terminal(texto)
                               if papel == "assistant" else _texto_terminal_seguro(texto))
                    print("\n" + _cor(rotulo, cor), visivel)
                elif papel == "tool_start":
                    nome = conteudo.get("nome") or "unknown"
                    if nome == "run_command":
                        cmd = conteudo.get("cmd") or (conteudo.get("args") or {}).get("cmd", "")
                        print("\n" + _cor(f"$ {cmd}", "tool"))
                    else:
                        args = json.dumps(conteudo.get("args") or {}, ensure_ascii=False)
                        print("\n" + _cor(f"[{nome}] → {args[:500]}", "tool"))
                elif papel == "permission":
                    decisoes = {
                        "uma_vez": "Permitir uma vez",
                        "workspace": "Sempre permitido neste workspace",
                        "global": "Sempre permitido globalmente",
                        "recusado": "Recusado pelo usuário",
                    }
                    decisao = decisoes.get(conteudo.get("decisao"), conteudo.get("decisao") or "desconhecida")
                    print(_cor(f"Permissão: {decisao}", "dim"))
                elif papel == "tool_result":
                    nome = conteudo.get("nome") or "unknown"
                    resultado = conteudo.get("resultado") or ""
                    texto = resultado if len(resultado) <= 2000 else resultado[:2000] + "…"
                    if nome == "run_command":
                        codigo = conteudo.get("codigo")
                        status = "ok" if codigo == 0 else f"código {codigo}"
                        print(_cor(f"comando · {status}", "tool"))
                        print(texto)
                    else:
                        print(_cor(f"[{nome}] ← {texto}", "tool"))
            print("\n" + _cor("fim do histórico retomado", "dim") + "\n")

    def _repl_tela(self):
        # Desde antes do autostart do Ollama até o desenho final, stdin fica sem
        # eco e é esvaziado ao abrir o primeiro prompt. Isso fecha a corrida de
        # teclas digitadas enquanto o aplicativo ainda está carregando.
        with terminal_ui.entrada_ocupada():
            self._inicializar_repl()
        while True:
            try:
                texto = _ler_entrada().strip()
            except EOFError:
                print()
                self._log("meta", evento="eof")
                return 0
            except KeyboardInterrupt:
                print()
                self._log("meta", evento="ctrl_c_exit")
                return 130
            if not texto:
                continue
            try:
                if self.comando_interno(texto):
                    continue
            except EOFError:
                self._log("meta", evento="exit")
                return 0
            self.perguntar(texto)


def main(argv=None):
    _instalar_sinais()
    argumentos = list(sys.argv[1:] if argv is None else argv)
    setup_solicitado = bool(argumentos and argumentos[0] == "setup")
    if argumentos and argumentos[0] == "setup":
        if len(argumentos) > 1:
            print("use: isaacli setup")
            return 2
        argumentos = []

    ap = argparse.ArgumentParser(
        prog="isaac",
        epilog="primeiro uso: isaacli setup",
    )
    ap.add_argument("--version", action="version", version=f"Isaac CLI v{APP_VERSION}")
    ap.add_argument("pedido", nargs="*", help="pedido unico; sem isto abre o REPL")
    ap.add_argument("--model", "--modelo", dest="modelo", default=None)
    ap.add_argument("--resume", metavar="ID", help="retoma uma sessão salva")
    ap.add_argument("--workspace", "--dir", default=os.getcwd())
    ap.add_argument("--max-passos", type=int, default=12, help=argparse.SUPPRESS)
    args = ap.parse_args(argumentos)

    retomada = None
    if args.resume:
        if args.pedido:
            print("use --resume sem um pedido na mesma linha")
            return 2
        try:
            retomada = _carregar_sessao(args.resume)
        except ValueError as e:
            print(f"ERRO: {e}")
            return 2

    try:
        dado_config = config.carregar()
    except ValueError as e:
        print(f"AVISO: {e}; usando configuracao padrao.")
        dado_config = config.config_vazia()
    _perfil_nome, perfil_padrao = config.perfil(dado_config)
    precisa_setup = (
        setup_solicitado
        or (
        args.modelo is None
        and not os.environ.get("AGENTE_MODELO")
        and perfil_padrao is None
        and retomada is None
        and sys.stdin.isatty()
        )
    )
    if precisa_setup:
        import setup_ollama
        codigo = setup_ollama.executar_setup()
        if codigo == 0:
            dado_config = config.carregar()
            _perfil_nome, perfil_padrao = config.perfil(dado_config)
        elif codigo == 130:
            return 130
        elif setup_solicitado:
            # Nunca esconder falha/cancelamento abrindo outro modelo.
            return codigo
        elif perfil_padrao is None:
            # Sem perfil não há motor seguro para abrir; a mensagem abaixo
            # orienta uma nova tentativa sem escolher um modelo pelo usuário.
            dado_config = config.config_vazia()
            perfil_padrao = None
    modelo = (
        args.modelo
        or os.environ.get("AGENTE_MODELO")
        or (retomada or {}).get("model")
        or (perfil_padrao or {}).get("model")
    )
    if not modelo:
        print("Nenhum modelo foi configurado. Execute: isaacli setup")
        return 2
    if (args.modelo is None and not os.environ.get("AGENTE_MODELO")
            and retomada is None and perfil_padrao
            and perfil_padrao.get("model") == modelo):
        perfil_modelo = perfil_padrao
    else:
        _nome_modelo, perfil_modelo = config.perfil_do_modelo(dado_config, modelo)
    thinking = (perfil_modelo or {}).get("thinking")

    workspace = retomada["workspace"] if retomada else args.workspace
    cli = IsaacCLI(
        modelo, workspace, args.max_passos, thinking=thinking,
        num_ctx=(perfil_modelo or {}).get("num_ctx"),
    )
    cli.provider = cli._provider_do_perfil(perfil_modelo)
    if retomada:
        cli.historico = retomada["history"]
        cli.transcript_retomada = retomada["transcript"]
        cli._log("meta", evento="resume", origem=retomada["id"],
                 origem_log=str(retomada["path"]))
    try:
        if args.pedido:
            return cli.perguntar(" ".join(args.pedido))
        return cli.repl()
    finally:
        _fechar_sem_interrupcao(cli)


if __name__ == "__main__":
    raise SystemExit(main())
