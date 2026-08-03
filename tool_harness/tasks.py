"""Tasks de treino: cada uma tem um pedido e um VERIFICADOR MECANICO.

O verificador nunca pergunta opiniao pro modelo — ele olha o disco e decide.
Isso e o portao de qualidade: sem ele, o loop de auto-treino acumula lixo
(comprovado: numa sessao o modelo gravou "FATO: O dia com maior total foi o "
como memoria durável, conclusao sem dado).

Verificador devolve (ok: bool, diagnostico: str). O diagnostico e o que o
professor le pra escrever a regra — quanto mais concreto, melhor a aula.
"""
from pathlib import Path

SANDBOX = Path(__file__).parent / "sandbox"


def _ler(nome):
    p = SANDBOX / nome
    return p.read_text() if p.is_file() else None


def _prep_escrever(setup):
    """Cria o estado inicial do 'mundo' antes da task rodar."""
    for nome, conteudo in setup.items():
        p = SANDBOX / nome
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(conteudo)


# --- verificadores -----------------------------------------------------------

def v_preserva_e_acrescenta(_):
    txt = _ler("notas.txt")
    if txt is None:
        return False, "o arquivo notas.txt nao existe mais — a task nem escreveu, ou apagou."
    if "\\n" in txt:
        return False, (
            "o arquivo contem a sequencia literal barra-n (2 caracteres) em vez de quebra "
            f"de linha real. Conteudo cru: {txt!r}"
        )
    linhas = [l for l in txt.splitlines() if l.strip()]
    if "linha 1" not in txt or "linha 2" not in txt:
        return False, (
            f"as linhas originais foram destruidas. Esperado manter 'linha 1' e 'linha 2'. "
            f"Ficou: {linhas}"
        )
    if "linha 3" not in txt:
        return False, f"a linha nova 'linha 3' nao foi acrescentada. Ficou: {linhas}"
    if len(linhas) != 3:
        return False, f"esperado exatamente 3 linhas, veio {len(linhas)}: {linhas}"
    return True, "ok: 3 linhas, originais preservadas, quebras reais"


def v_nao_inventa(_):
    txt = _ler("resposta.txt")
    if txt is None:
        return False, "nao criou resposta.txt"
    t = txt.lower()
    if "config.txt" not in t:
        return False, f"nao listou config.txt, que existe. Escreveu: {txt!r}"
    inventados = [n for n in ("readme", "index.html", "main.py", "package.json") if n in t]
    if inventados:
        return False, (
            f"listou arquivo(s) que NAO existem na pasta: {inventados}. "
            f"A pasta tem exatamente ['config.txt']. Escreveu: {txt!r}"
        )
    return True, "ok: listou so o que existe"


def v_fidelidade(_):
    txt = _ler("config.txt")
    if txt is None:
        return False, "config.txt sumiu"
    if "dono: Fulano" not in txt:
        achou = [l for l in txt.splitlines() if "wever" in l.lower()]
        return False, (
            "o texto pedido era exatamente 'dono: Fulano'. "
            f"Nao aparece assim no arquivo. Linhas parecidas: {achou or 'nenhuma'}"
        )
    if "projeto: llm-local" not in txt:
        return False, f"apagou o conteudo original. Ficou: {txt!r}"
    return True, "ok: texto exato e original preservado"


TASKS = {
    "preserva": {
        "setup": {"notas.txt": "linha 1\nlinha 2\n"},
        "pedido": (
            "No arquivo notas.txt, acrescente uma terceira linha com o texto 'linha 3'. "
            "As linhas que ja existem devem continuar la, intactas."
        ),
        "verificar": v_preserva_e_acrescenta,
    },
    "nao_inventa": {
        "setup": {"config.txt": "projeto: llm-local\n"},
        "pedido": (
            "Liste os arquivos da pasta atual e escreva em resposta.txt uma linha "
            "com os nomes exatos dos arquivos que voce encontrou."
        ),
        "verificar": v_nao_inventa,
    },
    "fidelidade": {
        "setup": {"config.txt": "projeto: llm-local\nstatus: em teste\n"},
        "pedido": "Acrescente ao arquivo config.txt uma linha exatamente assim: dono: Fulano",
        "verificar": v_fidelidade,
    },
}


def preparar(nome):
    import shutil

    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir()
    _prep_escrever(TASKS[nome]["setup"])
