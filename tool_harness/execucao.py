"""Execucao de comando pelo isaac, confinada (task 02).

O QUE ISTO PROTEGE, E POR QUE PRECISOU DE MAIS DE UMA CAMADA
------------------------------------------------------------
`tools._safe()` confina CAMINHO DE ARQUIVO numa raiz. Comando de shell nao e
caminho: `rm -rf ~`, `curl | sh` e `git push` passam por fora daquela protecao
inteira. Entao aqui sao tres camadas, e cada uma sozinha tem furo conhecido:

1. SEM SHELL. O comando e quebrado com shlex e entregue direto ao execve. Sem
   shell nao existe pipe, redirecionamento, `&&`, `$(...)`, `~` nem glob. Isto
   mata a familia inteira do `curl | sh` de uma vez — nao por lista, por
   construcao. Sozinha nao basta: `python3 -c "..."` ainda faria qualquer coisa.

2. LISTA DE PERMITIDOS, curta de proposito. Amplia com uso, nao por antecipacao.
   Sozinha nao basta: allowlist e jogo de adivinhar argumento perigoso, e quem
   escreve a lista sempre esquece um.

3. BWRAP — a unica que vale de verdade, porque e o kernel dizendo nao, nao um
   `if` nosso. Sem rede (`--unshare-net`), sistema de arquivos inteiro em
   somente-leitura, e a pasta de trabalho como UNICA coisa gravavel. Aqui o
   `rm -rf /home/usuario` nao alcanca a home real do usuario.

Se o bwrap nao existir na maquina, este modulo RECUSA executar. Cair de volta
pro host "so pra funcionar" transformaria a contencao em teatro — e teatro de
seguranca e pior que nenhuma, porque a gente para de olhar.

EXCECAO DELIBERADA (2026-07-19): `git push` agora e permitido e reabre rede SO
para aquela chamada (`--share-net`, ver `_precisa_de_rede`) -- todo o resto
(python3, pytest, git status/diff/...) continua sem rede nenhuma. HOME
continua sendo a pasta de trabalho, nunca a home real: nenhuma chave privada
ou credencial HTTPS do usuario e montada dentro do sandbox. A autenticacao
passa pelo SOCKET do ssh-agent (`SSH_AUTH_SOCK`), que so assina desafios --
nao existe operacao que deixe `cat`/`python3` lerem a chave privada por ele.
`--force` continua bloqueado (unica acao daqui de fato irreversivel).

EXCECAO DELIBERADA (2026-07-20): `graphify query/path/explain/diagnose` agora
e permitido para consultar mapas `graphify-out/graph.json`. Para o binario
funcionar, a jaula monta em modo somente-leitura apenas o `uv tool` do Graphify
e o Python embutido que ele usa. Isso cria caminhos-pai como `/home/usuario`,
mas nao monta a home real, `.ssh`, DevTools, historicos ou credenciais.

NOTA DE PROCESSO: a task 02 diz, e esta certa, que quem escreve a jaula nao pode
ser quem ja provou nao respeitar a jaula. Por isso os testes de
`testar_execucao.py` tentam ESCAPAR de verdade (escrever fora, abrir rede,
apagar a home) em vez de so conferir que a mensagem de recusa apareceu.
"""
import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

TETO_SEGUNDOS = 60          # teto por comando
TETO_SAIDA = 20_000         # corta saida gigante ANTES de voltar pro modelo

# Curta de proposito (task 02). Cada entrada precisa de motivo pra estar aqui.
PERMITIDOS = {
    "ls", "cat", "head", "tail", "wc", "grep", "find",
    "python3", "pytest", "git", "graphify",
}

# git de leitura + add/commit/push (2026-07-19, pedido explicito do usuario pra
# testar o fluxo "isaac ve o diff, commita e da push" neste repo). commit e
# local, sem risco. push e o unico subcomando que abre rede (ver
# `_precisa_de_rede`) -- e mesmo assim so pro proprio push, nunca pros outros
# comandos da lista (python3, pytest etc. continuam 100% sem rede).
GIT_PERMITIDOS = {"status", "diff", "log", "show", "branch", "add", "commit", "push"}

# Graphify entra como consulta local de mapa estrutural. Subcomandos que podem
# baixar, extrair, vigiar ou reescrever grafo ficam fora.
GRAPHIFY_PERMITIDOS = {"query", "path", "explain", "diagnose"}

# push --force reescreve historico remoto: e a unica acao daqui que e
# irreversivel de verdade (apaga trabalho de outras pessoas se o remoto ja
# tiver avancado). Fica fora mesmo com push liberado.
PUSH_FLAGS_PROIBIDAS = {"--force", "-f", "--force-with-lease", "--force-if-includes"}

# Caminho publico do known_hosts do usuario (so chaves publicas de servidor,
# nada secreto) -- deixa o ssh validar o host sem precisar de TTY interativo
# pra "yes/no" na primeira conexao.
KNOWN_HOSTS_HOST = Path.home() / ".ssh" / "known_hosts"
GRAPHIFY_TOOL_ROOT = Path.home() / ".local" / "share" / "uv" / "tools" / "graphifyy"
GRAPHIFY_PYTHON_ROOT = (
    Path.home()
    / ".var"
    / "app"
    / "com.visualstudio.code"
    / "data"
    / "uv"
    / "python"
)


class Recusado(Exception):
    """Comando barrado antes de rodar. A mensagem vai crua pro modelo e pra tela."""


def _git_config_global(chave):
    try:
        r = subprocess.run(
            ["git", "config", "--global", "--get", chave],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _git_identidade():
    """Identidade de commit herdada do host, sem expor a HOME real ao sandbox."""
    nome = _git_config_global("user.name") or os.environ.get("USER") or "Isaac"
    email = _git_config_global("user.email") or f"{nome.lower().replace(' ', '.')}@localhost"
    return nome, email


def _binarios_do_sistema():
    """Onde os executaveis moram nesta distro. Fedora usa /usr + links."""
    reais, links = [], []
    for caminho in ("/usr", "/etc"):
        if os.path.isdir(caminho):
            reais.append(caminho)
    for link in ("/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(link):
            links.append((os.readlink(link), link))
        elif os.path.isdir(link):
            reais.append(link)
    return reais, links


def revisar(cmd):
    """Decide se o comando pode rodar. Devolve a lista de argumentos ja quebrada.

    Levanta Recusado com o MOTIVO. Nunca recusa em silencio: modelo pequeno que
    recebe silencio tenta de novo igual; modelo que recebe motivo corrige.
    """
    if not cmd or not cmd.strip():
        raise Recusado("comando vazio")

    # punctuation_chars=True separa os operadores de shell em tokens proprios,
    # em vez de deixar eles grudados ('ls;' virava um "programa" chamado 'ls;').
    # Isso importa pra checagem logo abaixo ser por TOKEN e nao pela string crua:
    # procurar ';' na string inteira recusaria
    # `python3 -c "import time; time.sleep(1)"`, que e legitimo e nem tem shell
    # pra interpretar aquele ';'. Recusar comando bom e tao ruim quanto aceitar
    # comando mau — o modelo so ve que "nao deu" e tenta a mesma coisa de novo.
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        partes = list(lex)
    except ValueError as e:
        raise Recusado(f"nao consegui ler o comando (aspas abertas?): {e}")
    if not partes:
        raise Recusado("comando vazio")

    # Operadores de shell como token SOLTO. Sem shell eles nunca seriam
    # interpretados — chegariam ao execve como argumento literal e o comando
    # faria em silencio outra coisa que o modelo nao pediu. Dizer o que houve e
    # melhor que executar uma versao truncada do pedido.
    OPERADORES = {"|", "||", ">", ">>", "<", "<<", "&&", "&", ";", ";;", "(", ")", "$"}
    for parte in partes:
        if parte in OPERADORES or "`" in parte:
            raise Recusado(
                f"'{parte}' nao funciona aqui: os comandos rodam sem shell, "
                f"entao nao existe pipe, redirecionamento nem encadeamento. "
                f"Rode um comando por vez.")

    programa = partes[0]
    if "/" in programa:
        raise Recusado(
            f"chame o programa pelo nome ('{Path(programa).name}'), nao pelo "
            f"caminho ('{programa}').")
    if programa not in PERMITIDOS:
        raise Recusado(
            f"'{programa}' nao esta na lista de comandos permitidos.\n"
            f"Permitidos: {', '.join(sorted(PERMITIDOS))}")

    # `find` tem flags que executam outro programa (`-exec sh -c ...` traria o
    # shell de volta pela janela) e uma que apaga arquivo. O bwrap ainda seguraria
    # o estrago dentro da pasta de trabalho, mas "dentro da pasta de trabalho" e
    # justamente onde mora o trabalho do isaac.
    if programa == "find":
        for flag in ("-exec", "-execdir", "-delete", "-ok", "-okdir",
                     "-fprintf", "-fls", "-fprint"):
            if flag in partes:
                raise Recusado(
                    f"'find {flag}' nao e permitido — o find aqui so procura e "
                    f"lista. Para agir sobre o que achou, rode outro comando.")

    if programa == "git":
        sub = next((p for p in partes[1:] if not p.startswith("-")), None)
        if sub is None:
            raise Recusado("diga o subcomando do git (ex: git status)")
        if sub not in GIT_PERMITIDOS:
            raise Recusado(
                f"'git {sub}' nao e permitido.\n"
                f"Permitidos: {', '.join(sorted(GIT_PERMITIDOS))}")
        if sub == "push" and (set(partes[1:]) & PUSH_FLAGS_PROIBIDAS):
            raise Recusado(
                "push com --force nao e permitido — reescreve historico remoto "
                "de forma irreversivel. Push normal (sem --force) esta liberado.")

    if programa == "graphify":
        sub = next((p for p in partes[1:] if not p.startswith("-")), None)
        if sub is None:
            raise Recusado("diga o subcomando do graphify (ex: graphify query)")
        if sub not in GRAPHIFY_PERMITIDOS:
            raise Recusado(
                f"'graphify {sub}' nao e permitido.\n"
                f"Permitidos: {', '.join(sorted(GRAPHIFY_PERMITIDOS))}")

    return partes


def _precisa_de_rede(partes):
    """So 'git push' abre rede. Todo o resto do sandbox continua sem rede."""
    return len(partes) >= 2 and partes[0] == "git" and partes[1] == "push"


def montar_bwrap(argv, raiz, rede=False):
    """Monta a linha do bwrap: nada gravavel a nao ser a pasta de trabalho.

    rede=True (so pra 'git push'): reabre a rede com --share-net e encaminha o
    SOCKET do ssh-agent (nao o arquivo de chave -- socket so assina, nao deixa
    ler a chave privada de dentro do sandbox nem com 'cat') mais o known_hosts
    (chave PUBLICA do servidor, nada secreto). HOME continua sendo a pasta de
    trabalho: nenhuma credencial do host fica exposta a `cat`/`python3` dentro
    do sandbox, so o suficiente pro ssh completar o push.
    """
    reais, links = _binarios_do_sistema()
    git_nome, git_email = _git_identidade()
    linha = [shutil.which("bwrap")]
    for caminho in reais:
        linha += ["--ro-bind", caminho, caminho]
    for alvo, link in links:
        linha += ["--symlink", alvo, link]
    path_env = "/usr/bin:/bin"
    if GRAPHIFY_TOOL_ROOT.exists():
        linha += ["--ro-bind", str(GRAPHIFY_TOOL_ROOT), str(GRAPHIFY_TOOL_ROOT)]
        path_env = f"{GRAPHIFY_TOOL_ROOT / 'bin'}:{path_env}"
    if GRAPHIFY_PYTHON_ROOT.exists():
        linha += ["--ro-bind", str(GRAPHIFY_PYTHON_ROOT), str(GRAPHIFY_PYTHON_ROOT)]
    linha += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # A pasta de trabalho no MESMO caminho de fora: assim o que o modelo ve
        # aqui bate com o que ele ve nas outras ferramentas.
        "--bind", str(raiz), str(raiz),
        "--chdir", str(raiz),
        "--setenv", "HOME", str(raiz),
        "--setenv", "PATH", path_env,
        "--setenv", "GIT_AUTHOR_NAME", git_nome,
        "--setenv", "GIT_AUTHOR_EMAIL", git_email,
        "--setenv", "GIT_COMMITTER_NAME", git_nome,
        "--setenv", "GIT_COMMITTER_EMAIL", git_email,
        "--unshare-all",        # sem rede, sem nada, por padrao
        "--die-with-parent",    # fechar a Oficina derruba o comando junto
        "--new-session",        # sem tty herdado (evita injecao por TIOCSTI)
    ]
    if rede:
        for diretorio in ("/run", "/run/systemd", "/run/NetworkManager"):
            linha += ["--dir", diretorio]
        for caminho in (
            "/run/systemd/resolve",
            "/run/NetworkManager/resolv.conf",
            "/run/NetworkManager/no-stub-resolv.conf",
        ):
            if Path(caminho).exists():
                linha += ["--ro-bind", caminho, caminho]
        linha += ["--share-net"]   # so desfaz o --unshare-net; resto continua isolado
        sock = os.environ.get("SSH_AUTH_SOCK")
        if sock and Path(sock).exists():
            linha += ["--bind", sock, sock, "--setenv", "SSH_AUTH_SOCK", sock]
        if KNOWN_HOSTS_HOST.exists():
            destino = "/tmp/.known_hosts_ro"
            linha += ["--ro-bind", str(KNOWN_HOSTS_HOST), destino,
                      "--setenv", "GIT_SSH_COMMAND",
                      f"ssh -o UserKnownHostsFile={destino} -o StrictHostKeyChecking=yes"]
    linha += ["--", *argv]
    return linha


def executar_comando(cmd: str) -> str:
    """Roda um comando confinado e devolve a saida CRUA.

    Crua de proposito: stdout, stderr e codigo de saida, sem resumir e sem
    "limpar". O modelo precisa do erro real pra corrigir — mensagem mastigada
    esconde justamente a linha que diz o que fazer.
    """
    import tools  # tardio: SANDBOX_ROOT muda quando o usuario troca de pasta

    if shutil.which("bwrap") is None:
        return ("RECUSADO: o bwrap nao esta instalado, e sem ele nao ha "
                "confinamento. Nao vou rodar comando direto no host.\n"
                "Instale com: sudo dnf install bubblewrap")

    try:
        argv = revisar(cmd)
    except Recusado as e:
        return f"$ {cmd}\nRECUSADO: {e}\n(código de saída: 126)"

    raiz = Path(tools.SANDBOX_ROOT).resolve()
    try:
        raiz.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"ERRO: pasta de trabalho inacessivel ({e})"

    proc = None
    try:
        proc = subprocess.Popen(
            montar_bwrap(argv, raiz, rede=_precisa_de_rede(argv)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,   # grupo proprio: da pra matar filho e neto
        )
        saida, erro = proc.communicate(timeout=TETO_SEGUNDOS)
        codigo = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate(timeout=5)
        except Exception:
            pass
        return (f"$ {cmd}\n"
                f"INTERROMPIDO: passou de {TETO_SEGUNDOS}s e foi morto.")
    except OSError as e:
        return f"ERRO: nao consegui rodar o comando ({e})"

    partes = [f"$ {cmd}"]
    if saida.strip():
        partes.append(saida.rstrip("\n"))
    if erro.strip():
        partes.append("--- stderr ---")
        partes.append(erro.rstrip("\n"))
    partes.append(f"(código de saída: {codigo})")
    texto = "\n".join(partes)

    if len(texto) > TETO_SAIDA:
        cortado = len(texto) - TETO_SAIDA
        texto = texto[:TETO_SAIDA] + f"\n… (cortei {cortado} caracteres)"
    return texto


ESQUEMA = {
    "type": "function",
    "function": {
        "name": "executar_comando",
        "description": (
            "Roda UM comando no terminal, confinado na pasta de trabalho, e "
            "devolve a saida crua (stdout, stderr e codigo de saida). "
            "Nao existe shell: nada de pipe, '>', '&&' ou ';' — um comando por "
            "vez. Permitidos: "
            + ", ".join(sorted(PERMITIDOS)) +
            f" (git: {', '.join(sorted(GIT_PERMITIDOS))} — sem --force no "
            "push; a identidade de commit vem do git global do host, entao "
            "'git config' nao e necessario nem permitido). Nao ha rede, exceto "
            "no proprio 'git push'. Use pra "
            "conferir o que voce fez: listar arquivos, contar linhas, rodar "
            "um teste, ver o diff, commitar e dar push."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string",
                        "description": "o comando, ex: 'ls -la' ou 'wc -l arquivo.py'"}
            },
            "required": ["cmd"],
        },
    },
}
