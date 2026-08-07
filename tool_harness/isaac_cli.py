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
from pathlib import Path

try:
    import readline
except ImportError:  # pragma: no cover - Windows/ambiente minimo
    readline = None

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import agent
import config
import terminal_ui
import tools

SESSOES_DIR = AQUI / "cli_sessoes"
FEEDBACK_DIR = AQUI / "feedback"
COMANDOS_BARRA = [
    "/help", "/setup", "/status", "/tools", "/sessions", "/show", "/log", "/feedback",
    "/bom", "/ruim", "/nota",
    "/workspace", "/model", "/permissions", "/mode", "/clear", "/exit",
]
MAX_PREVIEW_CHARS = 1800
MAX_PREVIEW_LINHAS = 28
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

CONHECIMENTO_CLI = """You are isaacli running as a local CLI in the user's terminal.

OPERATING CONTEXT:
- Always answer in the same language as the user's latest message. If the user
  writes in Portuguese, answer in Brazilian Portuguese; do not switch to English.
- The current working directory is: {workspace}
- Everything you do through tools is confined to that directory.
- Before asking the user to clarify a local file, directory or project target,
  try to resolve it with list_dir, find, grep or read_file. If the user says
  "the txt file", "the config" or similar and the workspace can identify it,
  inspect the workspace instead of asking for an exact name.
- To inspect the project, use run_command with short commands: git status,
  git diff, ls, find, wc, pytest, python3.
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


def _ollama_ok(timeout=2):
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=timeout) as r:
            return json.load(r).get("version") or "ok"
    except Exception:
        return None


def _agora():
    return dt.datetime.now().isoformat(timespec="seconds")


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
    return partes[0]


def _comando_leitura_segura(cmd):
    partes = _partes_comando(cmd)
    if not partes:
        return False
    if partes[0] in COMANDOS_SOMENTE_LEITURA:
        return True
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


def _montar_historico(workspace):
    return [{"role": "system", "content": (
        agent.CONHECIMENTO_FERRAMENTAS + "\n\n" +
        CONHECIMENTO_CLI.format(workspace=str(workspace))
    )}]


def _carregar_sessao(session_id):
    """Reconstrói conversa e tool calls de um JSONL local pelo ID exato."""
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6}-[a-f0-9]{6}", session_id):
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
        elif tipo == "tool_result" and isinstance(evento.get("resultado"), str):
            historico.append({"role": "tool", "tool_call_id": tool_pendente or "resume-tool",
                              "content": evento["resultado"]})
            tool_pendente = None
        elif tipo == "assistant_final" and isinstance(evento.get("content"), str):
            historico.append({"role": "assistant", "content": evento["content"]})
            if evento["content"]:
                transcript.append(("assistant", evento["content"]))
    return {"id": session_id, "path": caminho, "workspace": workspace,
            "model": modelo, "history": historico, "transcript": transcript}


class IsaacCLI:
    def __init__(self, modelo, workspace, max_passos, autostart_ollama=True,
                 thinking=None, config_file=None, provider=None):
        self.modelo = modelo
        self.thinking = thinking
        self.config_file = config_file
        self.provider = provider or {"provider": "ollama"}
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_passos = max_passos
        self.autostart_ollama = autostart_ollama
        self.ollama_proc = None
        self.historico = []
        self.session_id = dt.datetime.now().strftime("%Y-%m-%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
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
        versao = _ollama_ok()
        if versao:
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
                return versao
            if self.ollama_proc.poll() is not None:
                self._log("error", erro=f"ollama serve saiu com codigo {self.ollama_proc.returncode}")
                return None
        return _ollama_ok(timeout=1)

    def fechar(self):
        if self.ollama_proc and self.ollama_proc.poll() is None:
            self.ollama_proc.terminate()
            try:
                self.ollama_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.ollama_proc.kill()
                self.ollama_proc.wait(timeout=3)
            self._log("meta", evento="ollama_autostop", pid=self.ollama_proc.pid)

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
  /show [n|last]        expande a saida completa de um comando recolhido
  /log                  mostra o arquivo JSONL desta sessao
  /feedback             mostra como avaliar a sessao/tarefa
  /bom [comentario]     marca a ultima tarefa como boa
  /ruim [comentario]    marca a ultima tarefa como ruim
  /nota 0-10 [texto]    da uma nota numerica para a ultima tarefa
  /workspace [caminho]  mostra ou troca a pasta de trabalho
  /model [perfil|nome]  lista perfis ou troca o modelo Ollama
  /permissions          mostra autorizações persistentes aplicáveis
  /mode                 alterna entre automático seguro e somente autorizados
  /clear                limpa o contexto da conversa
  /exit                 sai

atalho:
  digite / e pressione Tab para completar comandos
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
                    print(f"setup concluiu, mas nao consegui reler a configuracao: {e}")
                    return True
                if item:
                    self.modelo = item["model"]
                    self.thinking = item.get("thinking")
                    self.provider = self._provider_do_perfil(item)
                    self._log("meta", evento="setup", perfil=nome,
                              modelo=self.modelo, thinking=self.thinking)
                    print(f"perfil carregado nesta sessao: {nome} ({self.modelo})")
            else:
                print("O Isaac continua aberto. Corrija o motor e use /setup novamente.")
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
                print(f"modelo atual: {self.modelo}")
                try:
                    dado = config.carregar(self.config_file)
                except ValueError as e:
                    print(f"configuracao: {e}")
                    return True
                perfis = dado.get("profiles") or {}
                if perfis:
                    print("perfis configurados:")
                    for nome, item in perfis.items():
                        marca = " *" if item.get("model") == self.modelo else ""
                        print(
                            f"  {nome}: {item.get('model')} "
                            f"(contexto={_contexto_curto(item.get('num_ctx'))}, "
                            f"raciocinio={item.get('thinking')}){marca}"
                        )
                    print("use: /model <perfil>")
                    print("use /setup para adicionar modelo ou mudar contexto/raciocinio")
            else:
                try:
                    dado = config.carregar(self.config_file)
                except ValueError:
                    dado = config.config_vazia()
                item = (dado.get("profiles") or {}).get(arg)
                if item:
                    self.modelo = item["model"]
                    self.thinking = item.get("thinking")
                    self.provider = self._provider_do_perfil(item)
                    origem = f"perfil {arg}"
                else:
                    self.modelo = arg
                    self.thinking = None
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
        if cmd == "/clear":
            self.historico = _montar_historico(self.workspace)
            self._log("meta", evento="clear")
            print("contexto limpo")
            return True

        print(f"comando desconhecido: {cmd}  (use /help)")
        return True

    def status(self):
        versao = (_ollama_ok() or "sem resposta") if self.provider.get("provider") == "ollama" else (
            self.provider.get("provider_name") or "API compatível com OpenAI")
        duracao_s = self.uso_total.get("total_duration", 0) / 1_000_000_000
        print(f"sessao: {self.session_id}")
        print(f"log: {self.session_path}")
        print(f"pid: {os.getpid()}")
        print(f"modelo: {self.modelo}")
        print(f"raciocinio: {self.thinking if self.thinking is not None else 'padrao do modelo'}")
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
        print(_cor("avaliação opcional: /bom [comentário], /ruim [comentário] ou /nota 0-10 [comentário]", "dim"))

    def _mostrar_working(self):
        self._rotulo_assistente_pendente = True
        self._geracao_inicio = time.monotonic()
        self._geracao_pedacos = 0
        self._geracao_status_em = 0.0
        terminal_ui.status("Working…")
        if _usa_cor() and not self._working_visivel:
            print(_cor("Working…", "dim"), end="", flush=True)
            self._working_visivel = True

    def _limpar_working(self, limpar_status=True):
        if limpar_status:
            terminal_ui.status(None)
        if self._working_visivel:
            print("\r\033[2K", end="", flush=True)
            self._working_visivel = False

    def _token(self, pedaco):
        self._limpar_working(limpar_status=False)
        self._geracao_pedacos += 1
        agora = time.monotonic()
        decorrido = agora - (self._geracao_inicio or agora)
        if decorrido > 0 and agora - self._geracao_status_em >= 0.2:
            terminal_ui.status(
                f"Gerando · ≈ {self._geracao_pedacos / decorrido:.1f} tok/s"
            )
            self._geracao_status_em = agora
        if self._rotulo_assistente_pendente and pedaco:
            print(_cor("isaac:", "assistant"), end=" ", flush=True)
            self._rotulo_assistente_pendente = False
        print(pedaco, end="", flush=True)

    def _tool_antes(self, nome, args):
        self._limpar_working()
        try:
            dados = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            dados = {}
        if nome == "run_command":
            cmd = dados.get("cmd", args)
            print(_cor(f"$ {cmd}", "tool"), flush=True)
            self._log("tool_start", nome=nome, cmd=cmd)
            return self._aprovar_e_executar(cmd)
        else:
            resumo = json.dumps(dados, ensure_ascii=False) if dados else str(args)
            print(_cor(f"[{nome}] {resumo[:180]}", "tool"), flush=True)
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
            print(_cor(f"[{nome}] {texto}", "tool"), flush=True)
            if cortado:
                print(_cor("[saida recolhida no log da sessao]", "dim"), flush=True)
            self._log("tool_result", nome=nome, resultado=resultado)
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
            r = agent.rodar(
                pedido,
                self.modelo,
                max_passos=self.max_passos,
                verbose=False,
                on_token=self._token,
                on_tool_antes=self._tool_antes,
                on_tool=self._tool_depois,
                on_working=self._mostrar_working,
                historico=self.historico,
                thinking=self.thinking,
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
        self.ultima_resposta = final
        uso = (r or {}).get("uso") or {}
        for chave in self.uso_total:
            self.uso_total[chave] += int(uso.get(chave) or 0)
        self.turnos += 1
        if final and not final.endswith("\n"):
            print()
        eval_count = int(uso.get("eval_count") or 0)
        eval_duration = int(uso.get("eval_duration") or 0) / 1_000_000_000
        tempo_medicao = eval_duration or max(
            time.monotonic() - (self._turno_inicio or time.monotonic()), 0.001,
        )
        if eval_count:
            aproximado = "" if eval_duration else "≈ "
            print(_cor(
                f"{aproximado}{eval_count / tempo_medicao:.1f} tok/s · "
                f"{eval_count} tokens gerados", "dim",
            ))
        chamadas = (r or {}).get("chamadas") or []
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
        return 0

    def repl(self):
        with terminal_ui.tela_alternativa():
            codigo = self._repl_tela()
        print()
        print(_cor("sessão Isaac encerrada · para retomar:", "dim"))
        print(f"isaacli --resume {self.session_id}")
        print()
        return codigo

    def _repl_tela(self):
        _instalar_autocomplete()
        versao = self.garantir_ollama(avisar=False)
        if self.provider.get("provider") == "ollama":
            motor = f"Ollama {versao}" if versao else "MOTOR INDISPONÍVEL · use /setup"
        else:
            motor = self.provider.get("provider_name") or "API compatível com OpenAI"
        linhas = [
            "Isaac CLI",
            f"modelo     {self.modelo}",
            f"motor      {motor}",
            f"workspace  {self.workspace}",
            "Shift+Tab muda permissões · / para comandos · /exit sai",
        ]
        largura = max(len(linha) for linha in linhas)
        print(_cor("┌" + "─" * (largura + 2) + "┐", "assistant"))
        for linha in linhas:
            print(_cor("│", "assistant") + f" {linha:<{largura}} "
                  + _cor("│", "assistant"))
        print(_cor("└" + "─" * (largura + 2) + "┘", "assistant") + "\n")

        if self.transcript_retomada:
            limite = 20
            itens = self.transcript_retomada[-limite:]
            omitidas = len(self.transcript_retomada) - len(itens)
            print(_cor("conversa retomada", "dim"))
            if omitidas:
                print(_cor(f"… {omitidas} mensagem(ns) anterior(es) omitida(s)", "dim"))
            for papel, conteudo in itens:
                texto = conteudo if len(conteudo) <= 2000 else conteudo[:2000] + "…"
                rotulo = "❯" if papel == "user" else "isaac:"
                cor = "prompt" if papel == "user" else "assistant"
                print("\n" + _cor(rotulo, cor), texto)
            print("\n" + _cor("fim do histórico retomado", "dim") + "\n")

        while True:
            try:
                texto = input(_prompt_colorido("❯ ")).strip()
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
    cli = IsaacCLI(modelo, workspace, args.max_passos, thinking=thinking)
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
        cli.fechar()


if __name__ == "__main__":
    raise SystemExit(main())
