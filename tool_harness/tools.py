"""Ferramentas locais e de leitura web que o modelo pode chamar.

Arquivos são confinados a SANDBOX_ROOT. A leitura web aceita somente HTTP(S)
público, sem cookies, credenciais, proxies ou acesso à rede local.
"""
import ipaddress
import json
import os
import re
import socket
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# Pasta de trabalho do agente. Trocavel por env pra ele operar num projeto real
# em vez da sandbox de teste — continua confinado, so muda a raiz.
SANDBOX_ROOT = Path(os.environ.get("AGENTE_RAIZ", Path(__file__).parent / "sandbox"))
MAX_WEB_BYTES = 80_000


def _safe(path: str) -> Path:
    p = (SANDBOX_ROOT / path.lstrip("/")).resolve()
    root = SANDBOX_ROOT.resolve()
    if not (p == root or root in p.parents):
        raise ValueError(f"caminho fora da sandbox: {path}")
    return p


def read_file(path: str) -> str:
    p = _safe(path)
    if not p.is_file():
        return f"ERRO: arquivo nao existe: {path}"
    return p.read_text()


def write_file(path: str, content: str) -> str:
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    texto = _desescapar(content)
    p.write_text(texto)
    return f"OK: escrevi {len(texto)} bytes em {path}"


def list_dir(path: str = ".") -> str:
    p = _safe(path)
    if not p.is_dir():
        return f"ERRO: nao e diretorio: {path}"
    itens = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return "\n".join(itens) if itens else "(vazio)"


def append_file(path: str, content: str) -> str:
    """Acrescenta no fim sem o modelo precisar reproduzir o que ja existe.

    Modelo pequeno erra ao reescrever arquivo inteiro (chuta conteudo, escapa \\n
    errado). Dar uma ferramenta que nao exige reproduzir o conteudo elimina a
    classe de erro inteira, em vez de tentar consertar o modelo.
    """
    p = _safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    texto = _desescapar(content)
    if not texto.endswith("\n"):
        texto += "\n"
    with p.open("a") as f:
        f.write(texto)
    return f"OK: acrescentei {len(texto)} bytes no fim de {path}"


def replace_between(path: str, start_marker: str, end_marker: str, content: str) -> str:
    """Substitui um trecho pequeno entre dois marcadores existentes.

    Isto e andaime para modelo pequeno: em vez de pedir um HTML inteiro de novo
    a cada requisito, o agente troca so a parte que esta trabalhando agora.
    """
    start_marker = _normalizar_marcador(start_marker)
    end_marker = _normalizar_marcador(end_marker)
    p = _safe(path)
    if not p.is_file():
        return f"ERRO: arquivo nao existe: {path}"
    inicio_familia = re.fullmatch(r"(.+)_START", start_marker)
    fim_familia = re.fullmatch(r"(.+)_END", end_marker)
    familia = inicio_familia.group(1) if inicio_familia else ""
    if inicio_familia and fim_familia and familia != fim_familia.group(1):
        return (
            "ERRO: marcadores incompatíveis. "
            f"{start_marker} deve fechar com {inicio_familia.group(1)}_END, "
            f"nao com {end_marker}. Faca uma troca por vez.")
    texto = p.read_text()
    inicio = texto.find(start_marker)
    fim = texto.find(end_marker)
    if inicio < 0:
        return f"ERRO: marcador inicial nao encontrado: {start_marker}"
    if fim < 0:
        return f"ERRO: marcador final nao encontrado: {end_marker}"
    if fim <= inicio:
        return "ERRO: marcador final aparece antes do marcador inicial"

    trecho = _desescapar(content)
    baixo_trecho = trecho.lower()
    for placeholder in ("javascript aqui", "css aqui", "seu codigo aqui", "codigo aqui"):
        if placeholder in baixo_trecho:
            return (
                f'ERRO: o trecho ainda contem placeholder "{placeholder}". '
                "Substitua por codigo real antes de chamar replace_between.")
    if any(perigoso in trecho.lower() for perigoso in (
        "<script", "</script", "<body", "</body", "<html", "</html",
    )):
        return (
            "ERRO: o trecho tentou alterar a estrutura principal do HTML. "
            "Use apenas o miolo entre os marcadores, sem <script>, <body> ou <html>.")
    if familia in {"ISAAC_ADD", "ISAAC_DELETE", "ISAAC_SAVE", "ISAAC_LOAD"} and _contem_tag_html(trecho):
        return (
            "ERRO: este marcador fica dentro de <script>; nao escreva HTML aqui. "
            "Para botao, tabela ou formulario use os marcadores HTML correspondentes.")
    wrappers_js = {
        "ISAAC_ADD": "addTransaction",
        "ISAAC_DELETE": "deleteTransaction",
        "ISAAC_SAVE": "saveTransactions",
        "ISAAC_LOAD": "loadTransactions",
    }
    funcao_wrapper = wrappers_js.get(familia)
    if funcao_wrapper and re.search(rf"\bfunction\s+{re.escape(funcao_wrapper)}\s*\(", trecho):
        return (
            "ERRO: escreva apenas o miolo dentro dos marcadores, sem declarar "
            f"function {funcao_wrapper} de novo.")
    if (familia == "ISAAC_RENDER" or familia.startswith("ISAAC_RENDER_")) and re.search(r"\bfunction\s+\w+\s*\(", trecho):
        return (
            "ERRO: este marcador fica dentro de render(); escreva apenas comandos "
            "do miolo, sem declarar funcoes.")
    # Os marcadores ficam em linhas de comentario, por exemplo:
    #   <!-- ISAAC_FORM_START -->
    #   /* ISAAC_RENDER_START */
    # Substituir a partir do texto "ISAAC_*" colocava o codigo DENTRO do
    # comentario e quebrava o HTML/JS. A troca correta preserva as linhas dos
    # marcadores e altera apenas o miolo entre elas.
    inicio_conteudo = texto.find("\n", inicio)
    if inicio_conteudo < 0:
        inicio_conteudo = inicio + len(start_marker)
    else:
        inicio_conteudo += 1
    fim_conteudo = texto.rfind("\n", 0, fim)
    if fim_conteudo >= 0:
        fim_conteudo += 1
    if fim_conteudo < inicio_conteudo:
        fim_conteudo = fim
    linhas_trecho = [
        linha for linha in trecho.splitlines()
        if not re.search(r"ISAAC_[A-Z0-9_]+_(START|END)", linha)
    ]
    miolo = "\n".join(linhas_trecho).strip()
    if miolo:
        miolo = miolo + "\n"
    novo = texto[:inicio_conteudo] + miolo + texto[fim_conteudo:]
    p.write_text(novo)
    return (
        f"OK: substitui {len(trecho)} bytes entre {start_marker} e "
        f"{end_marker} em {path}")


def _normalizar_marcador(marker: str) -> str:
    """Aceita marcador puro ou embrulhado no comentario onde ele aparece."""
    m = str(marker or "").strip()
    for prefixo, sufixo in (("<!--", "-->"), ("/*", "*/")):
        if m.startswith(prefixo) and m.endswith(sufixo):
            return m[len(prefixo):-len(sufixo)].strip()
    return m


def _contem_tag_html(texto: str) -> bool:
    """Diferencia tag HTML de comparacao JS como `index < 0`."""
    return bool(re.search(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9-]*(?:\s|>|/)", texto))


def _desescapar(s: str) -> str:
    r"""Modelo pequeno emite '\n' literal (2 chars) achando que e quebra de linha.

    So converte quando NAO ha nenhuma quebra real no texto — assim nao estraga
    conteudo que legitimamente contenha uma barra invertida.
    """
    if "\n" not in s and "\\n" in s:
        return s.replace("\\n", "\n").replace("\\t", "\t")
    return s


def check_file(path: str) -> str:
    """Checagem mecanica e barata de uma pagina HTML (~1s), SEM opiniao de modelo.

    Ordem do mais barato pro mais caro; para no primeiro degrau que falhar:
      1. existe / tamanho / estrutura / placeholder / sintaxe JS (verificar_jogo)
      2. abre headless por ~1s e captura erro de runtime (checagem_rapida.js)

    E filtro de sanidade, NAO criterio de sucesso — quem diz "pronto" e o juiz
    comportamental. Isto so evita acordar o juiz pra codigo obviamente quebrado.
    """
    p = _safe(path)
    import verificar_jogo
    ok, probs = verificar_jogo.verificar(p)
    if not ok:
        return "PROBLEMAS ENCONTRADOS:\n" + "\n".join(f"- {x}" for x in probs)

    script = Path(__file__).parent / "checagem_rapida.js"
    try:
        r = subprocess.run(
            ["node", str(script), str(p)],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent),
        )
        saida = json.loads(r.stdout)
    except Exception as e:
        return f"ERRO: a checagem de runtime nao rodou ({e})"
    if not saida.get("ok"):
        return "PROBLEMAS ENCONTRADOS:\n" + "\n".join(f"- {x}" for x in saida.get("problemas", []))
    return "OK: o arquivo abre sem erro de JavaScript e mostra conteudo."


def _normalizar_url_web(url: str) -> str:
    url = str(url or "").strip()
    if len(url) > 2048:
        raise ValueError("URL longa demais")
    partes = urllib.parse.urlsplit(url)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ValueError("use uma URL completa começando com http:// ou https://")
    if partes.username or partes.password:
        raise ValueError("URL com usuário ou senha não é permitida")
    try:
        _ = partes.port
    except ValueError as e:
        raise ValueError("porta inválida na URL") from e

    issue = re.fullmatch(r"/([^/]+)/([^/]+)/issues/(\d+)/?", partes.path)
    if partes.hostname.casefold() == "github.com" and issue:
        dono, repo, numero = issue.groups()
        return f"https://api.github.com/repos/{dono}/{repo}/issues/{numero}"
    return urllib.parse.urlunsplit(partes)


def _validar_destino_web(url: str) -> None:
    partes = urllib.parse.urlsplit(url)
    if partes.scheme not in ("http", "https") or not partes.hostname:
        raise ValueError("redirecionamento para protocolo não permitido")
    porta = partes.port or (443 if partes.scheme == "https" else 80)
    try:
        destinos = socket.getaddrinfo(partes.hostname, porta, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"não foi possível resolver {partes.hostname}: {e}") from e
    ips = {ipaddress.ip_address(item[4][0]) for item in destinos}
    if not ips or any(not ip.is_global for ip in ips):
        raise ValueError("a ferramenta web não acessa localhost nem redes privadas/reservadas")


class _RedirecionamentoWebSeguro(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destino = urllib.parse.urljoin(req.full_url, newurl)
        _validar_destino_web(destino)
        return super().redirect_request(req, fp, code, msg, headers, destino)


class _ExtratorHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.partes = []
        self.oculto = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "svg", "noscript"):
            self.oculto += 1
        elif not self.oculto and tag in ("p", "div", "br", "li", "h1", "h2", "h3", "tr"):
            self.partes.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "svg", "noscript") and self.oculto:
            self.oculto -= 1
        elif not self.oculto and tag in ("p", "div", "li", "h1", "h2", "h3", "tr"):
            self.partes.append("\n")

    def handle_data(self, data):
        if not self.oculto:
            self.partes.append(data)

    def texto(self):
        linhas = (re.sub(r"[ \t]+", " ", linha).strip()
                  for linha in "".join(self.partes).splitlines())
        return "\n".join(linha for linha in linhas if linha)


def fetch_url(url: str) -> str:
    """Lê conteúdo público sem conceder rede ao terminal confinado."""
    destino = _normalizar_url_web(url)
    _validar_destino_web(destino)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RedirecionamentoWebSeguro(),
    )
    pedido = urllib.request.Request(
        destino,
        headers={
            "User-Agent": "IsaacCLI/0.1 (+read-only fetch_url)",
            "Accept": "text/html, application/json, text/plain, application/xml;q=0.9",
        },
    )
    try:
        with opener.open(pedido, timeout=20) as resposta:
            dados = resposta.read(MAX_WEB_BYTES + 1)
            final = resposta.geturl()
            tipo = resposta.headers.get_content_type()
            charset = resposta.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as e:
        return f"ERRO HTTP {e.code} ao acessar {destino}: {e.reason}"
    except urllib.error.URLError as e:
        return f"ERRO DE REDE ao acessar {destino}: {e.reason}"

    if not (tipo.startswith("text/") or tipo in (
            "application/json", "application/xml", "application/xhtml+xml")):
        return f"ERRO: conteúdo não textual recusado ({tipo})"
    cortado = len(dados) > MAX_WEB_BYTES
    texto = dados[:MAX_WEB_BYTES].decode(charset, errors="replace")
    if tipo in ("text/html", "application/xhtml+xml"):
        parser = _ExtratorHTML()
        parser.feed(texto)
        texto = parser.texto()
    cabecalho = f"URL final: {final}\nTipo: {tipo}\n"
    sufixo = "\n… conteúdo cortado pelo limite da ferramenta" if cortado else ""
    return cabecalho + texto + sufixo


def run_command(cmd: str) -> str:
    """Roda comando confinado. A contencao inteira mora em execucao.py.

    Import tardio de proposito: `tools` e importado por todo mundo aqui
    (bancada, testes, scripts de lote), e `execucao` so faz sentido quando
    alguem de fato vai rodar comando.
    """
    import execucao
    return execucao.run_command(cmd)


IMPLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "append_file": append_file,
    "replace_between": replace_between,
    "check_file": check_file,
    "fetch_url": fetch_url,
    "run_command": run_command,
}

# Schema no formato OpenAI — é isso que vai no campo `tools` da chamada da API.
SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Le o conteudo de um arquivo de texto e devolve como string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "caminho relativo do arquivo"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escreve (ou sobrescreve) um arquivo de texto com o conteudo dado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "caminho relativo do arquivo"},
                    "content": {"type": "string", "description": "conteudo completo a escrever"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": (
                "Acrescenta texto no FIM de um arquivo, preservando o que ja existe. "
                "Use esta em vez de write_file quando for so adicionar algo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "caminho relativo do arquivo"},
                    "content": {"type": "string", "description": "texto a acrescentar no fim"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_between",
            "description": (
                "Substitui apenas o conteudo entre dois marcadores textuais ja "
                "existentes no arquivo. Use para alterar uma secao pequena sem "
                "reescrever o arquivo inteiro."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "caminho relativo do arquivo"},
                    "start_marker": {"type": "string", "description": "marcador inicial literal"},
                    "end_marker": {"type": "string", "description": "marcador final literal"},
                    "content": {"type": "string", "description": "novo conteudo do trecho"},
                },
                "required": ["path", "start_marker", "end_marker", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_file",
            "description": (
                "Testa mecanicamente uma pagina HTML: sintaxe do JavaScript, erro ao abrir "
                "no navegador, placeholder esquecido. Use SEMPRE depois de escrever um jogo, "
                "antes de dizer que terminou. Devolve OK ou a lista de problemas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "caminho relativo do arquivo HTML"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Ferramenta geral para ler conteúdo textual de uma URL HTTP(S) pública: "
                "páginas, documentação, links compartilhados e APIs. Use sempre que precisar "
                "consultar a web pública; não tente curl pelo terminal. Não acessa redes privadas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL pública completa"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lista os arquivos e pastas de um diretorio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "caminho relativo do diretorio"}
                },
                "required": [],
            },
        },
    },
]


def _juntar_esquema_de_comando():
    """Anexa o schema de run_command sem duplicar a descricao dele aqui.

    A descricao mora junto das regras que ela descreve (execucao.py). Copiar pra
    ca criaria duas versoes da verdade, e a que o modelo LE seria a copia — o
    jeito classico de a lista de permitidos mudar e o modelo continuar
    acreditando na antiga.
    """
    try:
        import execucao
    except ImportError:
        return  # sem o modulo, a ferramenta simplesmente nao existe pro modelo
    SCHEMA.append(execucao.ESQUEMA)


_juntar_esquema_de_comando()


def schema_filtrado(nomes):
    """Devolve so as ferramentas necessarias para a tarefa atual."""
    permitidos = set(nomes)
    return [s for s in SCHEMA if s["function"]["name"] in permitidos]


def executar(nome: str, args_json: str) -> str:
    """Executa a ferramenta pedida pelo modelo e devolve o resultado como texto."""
    if nome not in IMPLS:
        return f"ERRO: ferramenta desconhecida '{nome}'. Disponiveis: {list(IMPLS)}"
    try:
        args = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
    except json.JSONDecodeError as e:
        return f"ERRO: argumentos nao sao JSON valido ({e}): {args_json}"
    try:
        return IMPLS[nome](**args)
    except TypeError as e:
        return f"ERRO: argumentos errados para {nome}: {e}"
    except Exception as e:
        return f"ERRO ao executar {nome}: {e}"
