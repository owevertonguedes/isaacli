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
from pathlib import Path

try:
    import readline
except ImportError:  # pragma: no cover - Windows/ambiente minimo
    readline = None

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import agent
import tools

MODELO_PADRAO = os.environ.get("AGENTE_MODELO", "isaac-granite")
SESSOES_DIR = AQUI / "cli_sessoes"
FEEDBACK_DIR = AQUI / "feedback"
COMANDOS_BARRA = [
    "/help", "/status", "/tools", "/sessions", "/show", "/log", "/feedback",
    "/bom", "/ruim", "/nota",
    "/workspace", "/model", "/clear", "/exit",
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

CONHECIMENTO_CLI = """You are isaacli running as a local CLI in the user's terminal.

OPERATING CONTEXT:
- The current working directory is: {workspace}
- Everything you do through tools is confined to that directory.
- To inspect the project, use run_command with short commands: git status,
  git diff, ls, find, wc, pytest, python3.
- If `graphify-out/graph.json` exists and the user asks where a flow, resource,
  module, test or architectural relation lives, look it up first with
  `graphify query "question" --graph graphify-out/graph.json --budget 700`.
  Graphify is for locating context; after that read the files and verify before
  declaring success. If there is no graph, fall back to local search with
  find/rg, and do not edit before locating.
- To change files, use read_file first and write_file/append_file after.
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

    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
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


def _codigo_saida(resultado):
    m = re.search(r"\(c[oó]digo de sa[ií]da: (-?\d+)\)\s*$", resultado.strip())
    return int(m.group(1)) if m else None


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


def _montar_historico(workspace):
    return [{"role": "system", "content": (
        agent.CONHECIMENTO_FERRAMENTAS + "\n\n" +
        CONHECIMENTO_CLI.format(workspace=str(workspace))
    )}]


class IsaacCLI:
    def __init__(self, modelo, workspace, max_passos, autostart_ollama=True):
        self.modelo = modelo
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
        self.uso_total = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
        self.ultima_resposta = ""
        self.feedbacks = 0
        self.definir_workspace(self.workspace, reset=True)
        self._log("meta", evento="inicio", pid=os.getpid(), modelo=self.modelo,
                  workspace=str(self.workspace))

    def garantir_ollama(self, avisar=False):
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
  /model [nome]         mostra ou troca o modelo Ollama
  /clear                limpa o contexto da conversa
  /exit                 sai

atalho:
  digite / e pressione Tab para completar comandos

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
                print(self.modelo)
            else:
                self.modelo = arg
                self._log("meta", evento="model", modelo=self.modelo)
                print(f"modelo: {self.modelo}")
            return True
        if cmd == "/clear":
            self.historico = _montar_historico(self.workspace)
            self._log("meta", evento="clear")
            print("contexto limpo")
            return True

        print(f"comando desconhecido: {cmd}  (use /help)")
        return True

    def status(self):
        versao = _ollama_ok() or "sem resposta"
        duracao_s = self.uso_total.get("total_duration", 0) / 1_000_000_000
        print(f"sessao: {self.session_id}")
        print(f"log: {self.session_path}")
        print(f"pid: {os.getpid()}")
        print(f"modelo: {self.modelo}")
        print(f"workspace: {self.workspace}")
        print(f"Ollama: {versao}")
        print(f"turnos: {self.turnos}")
        print(f"comandos: {len(self.comandos)}  falhas: {self.falhas}")
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
        print(_cor("avaliacao opcional: /bom [comentario], /ruim [comentario] ou /nota 0-10 [comentario]", "dim"))

    def _token(self, pedaco):
        print(pedaco, end="", flush=True)

    def _tool_antes(self, nome, args):
        try:
            dados = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            dados = {}
        if nome == "run_command":
            cmd = dados.get("cmd", args)
            print(_cor(f"\nrodando: {cmd}", "tool"), flush=True)
            self._log("tool_start", nome=nome, cmd=cmd)
        else:
            resumo = json.dumps(dados, ensure_ascii=False) if dados else str(args)
            print(_cor(f"\n[{nome}] {resumo[:180]}", "tool"), flush=True)
            self._log("tool_start", nome=nome, args=dados or args)

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
            }
            self.comandos.append(item)
            if codigo is not None and codigo != 0:
                self.falhas += 1
            status = "ok" if codigo == 0 else (f"falhou codigo {codigo}" if codigo is not None else "sem codigo")
            print(_cor(f"comando #{item['id']} · {status}", "bad" if codigo else "tool"), flush=True)
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

    def perguntar(self, pedido):
        if not self.garantir_ollama(avisar=True):
            print("ERRO: Ollama nao respondeu e nao consegui iniciar automaticamente.")
            self._log("error", erro="ollama_indisponivel")
            return 1
        self._log("user", content=pedido)
        comandos_antes = len(self.comandos)
        print(_cor("isaac:", "assistant"), end=" ", flush=True)
        try:
            r = agent.rodar(
                pedido,
                self.modelo,
                max_passos=self.max_passos,
                verbose=False,
                on_token=self._token,
                on_tool_antes=self._tool_antes,
                on_tool=self._tool_depois,
                historico=self.historico,
            )
        except urllib.error.URLError as e:
            if self.garantir_ollama(avisar=True):
                print("\nOllama iniciou agora; tente o pedido de novo.")
            else:
                print(f"\nERRO: Ollama nao respondeu ({e}) e o inicio automatico falhou.")
            self._log("error", erro=str(e))
            return 1
        except KeyboardInterrupt:
            print("\ninterrompido")
            self._log("error", erro="interrompido")
            return 130

        final = (r or {}).get("final") or ""
        self.ultima_resposta = final
        uso = (r or {}).get("uso") or {}
        for chave in self.uso_total:
            self.uso_total[chave] += int(uso.get(chave) or 0)
        self.turnos += 1
        if final and not final.endswith("\n"):
            print()
        novos = self.comandos[comandos_antes:]
        if novos and novos[-1].get("codigo") not in (None, 0):
            print(_cor(
                f"nota: o ultimo comando falhou com codigo {novos[-1]['codigo']}; "
                "trate a resposta acima como nao concluida se ela disser sucesso.",
                "warn",
            ))
        self._log("assistant_final", content=final, uso=uso,
                  chamadas=len((r or {}).get("chamadas") or []))
        self.lembrar_feedback(bool(novos))
        return 0

    def repl(self):
        _instalar_autocomplete()
        versao = self.garantir_ollama(avisar=True)
        print(f"Isaac CLI · modelo={self.modelo} · workspace={self.workspace}")
        if versao:
            print(f"Ollama OK · {versao}")
        else:
            print("Ollama nao respondeu em 127.0.0.1:11434 e o inicio automatico falhou.")
        print(f"sessao={self.session_id} · log={self.session_path}")
        print("Digite / e Enter para comandos; Tab completa; /exit sai.\n")

        while True:
            try:
                texto = input(_cor("isaac> ", "prompt")).strip()
            except EOFError:
                print()
                self._log("meta", evento="eof")
                return 0
            except KeyboardInterrupt:
                print("\nuse /exit para sair")
                continue
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
    ap = argparse.ArgumentParser(prog="isaac")
    ap.add_argument("pedido", nargs="*", help="pedido unico; sem isto abre o REPL")
    ap.add_argument("--model", "--modelo", dest="modelo", default=MODELO_PADRAO)
    ap.add_argument("--workspace", "--dir", default=os.getcwd())
    ap.add_argument("--max-passos", type=int, default=12, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    cli = IsaacCLI(args.modelo, args.workspace, args.max_passos)
    try:
        if args.pedido:
            return cli.perguntar(" ".join(args.pedido))
        return cli.repl()
    finally:
        cli.fechar()


if __name__ == "__main__":
    raise SystemExit(main())
