"""Ferramentas de arquivo que o modelo local pode chamar.

Tudo é confinado a SANDBOX_ROOT — o modelo não consegue ler/escrever fora dela,
mesmo que peça um caminho absoluto ou com "..".
"""
import json
import os
import re
import subprocess
from pathlib import Path

# Pasta de trabalho do agente. Trocavel por env pra ele operar num projeto real
# em vez da sandbox de teste — continua confinado, so muda a raiz.
SANDBOX_ROOT = Path(os.environ.get("AGENTE_RAIZ", Path(__file__).parent / "sandbox"))


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


def checar_arquivo(path: str) -> str:
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


def executar_comando(cmd: str) -> str:
    """Roda comando confinado. A contencao inteira mora em execucao.py.

    Import tardio de proposito: `tools` e importado por todo mundo aqui
    (bancada, testes, scripts de lote), e `execucao` so faz sentido quando
    alguem de fato vai rodar comando.
    """
    import execucao
    return execucao.executar_comando(cmd)


IMPLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "append_file": append_file,
    "replace_between": replace_between,
    "checar_arquivo": checar_arquivo,
    "executar_comando": executar_comando,
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
            "name": "checar_arquivo",
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
    """Anexa o schema de executar_comando sem duplicar a descricao dele aqui.

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
