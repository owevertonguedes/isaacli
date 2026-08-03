"""Ciclo de geracoes: juiz + professor ensinando o modelo local.

Uma geracao:
  1. prepara o 'mundo' (sandbox limpa com o estado inicial da task)
  2. modelo local tenta resolver, com o conhecimento acoplado ATUAL
  3. JUIZ MECANICO olha o disco e diz ok/falhou + diagnostico concreto
  4. se falhou, o PROFESSOR (modelo forte) le o diagnostico e escreve UMA regra
  5. a regra entra no conhecimento acoplado (conhecimento.md -> Modelfile -> ollama create)
  6. proxima geracao roda a MESMA task, do zero, sem dizer que ensinou

Uso:
    python3 geracoes.py preserva --geracoes 4
    python3 geracoes.py todas --geracoes 3 --professor gemini
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import agent
import tasks

AQUI = Path(__file__).parent
CONHECIMENTO = AQUI / "conhecimento.md"
MODELFILE = AQUI / "Modelfile.gerado"
MODELO = "isaac"
BASE = "granite4:micro"

BASE_CONHECIMENTO = """Voce e um assistente que opera arquivos atraves de ferramentas.

REGRAS:
- Para ler, escrever ou listar arquivos voce DEVE chamar a ferramenta correspondente.
- Chame UMA ferramenta por vez e espere o resultado antes da proxima.
- Quando terminar, responda em texto curto dizendo o que fez.
- Todos os caminhos sao relativos a pasta de trabalho.
"""


def conhecimento_atual():
    if not CONHECIMENTO.exists():
        CONHECIMENTO.write_text(BASE_CONHECIMENTO)
    return CONHECIMENTO.read_text()


def acoplar(texto):
    """Reconstroi o modelo com o conhecimento novo. Custa ~1s, nao duplica disco."""
    MODELFILE.write_text(
        f"FROM {BASE}\nPARAMETER temperature 0\nPARAMETER num_ctx 8192\n"
        f'SYSTEM """{texto}"""\n'
    )
    r = subprocess.run(
        ["ollama", "create", MODELO, "-f", str(MODELFILE)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ollama create falhou: {r.stderr[:300]}")


# --- professores -------------------------------------------------------------

PROMPT_PROFESSOR = """Voce e professor de um modelo de linguagem PEQUENO (3B) que usa ferramentas de arquivo.
Ferramentas disponiveis: read_file(path), write_file(path, content) [SOBRESCREVE TUDO],
append_file(path, content) [acrescenta no fim], list_dir(path).

O aluno recebeu esta task:
{pedido}

Ele FALHOU. O juiz mecanico (que olhou o disco de verdade) reportou:
{diagnostico}

Ferramentas que ele chamou, em ordem:
{chamadas}

Escreva UMA UNICA regra imperativa, curta (no maximo 2 linhas), que colocada no system
prompt dele evitaria exatamente esse erro no futuro. A regra deve ser GERAL o suficiente
pra valer em casos parecidos, mas CONCRETA (cite a ferramenta certa a usar).
Nao explique, nao comente, nao use markdown. Responda SO com a regra, comecando com "- ".
"""


def professor_gemini(prompt):
    # Validação obrigatória das credenciais no ambiente
    for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"]:
        if var not in __import__("os").environ:
            raise RuntimeError(
                f"Erro: A variável de ambiente '{var}' não está definida.\n"
                f"Por favor, exporte-a antes de prosseguir:\n"
                f"export {var}=<caminho_ou_id_correto>"
            )

    r = subprocess.run(
        ["gemini", "--skip-trust", "-p", prompt],
        capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ,
             "GOOGLE_CLOUD_LOCATION": "global",
             "GOOGLE_GENAI_USE_VERTEXAI": "true"},
    )
    if r.returncode != 0:
        raise RuntimeError(f"gemini falhou: {r.stderr[:200]}")
    return r.stdout


def professor_codex(prompt):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "regra.txt"
        subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "-o", str(out), prompt],
            capture_output=True, text=True, timeout=420, stdin=subprocess.DEVNULL,
        )
        return out.read_text() if out.exists() else ""


PROFESSORES = {"gemini": professor_gemini, "codex": professor_codex}


_STOP = {"a","o","as","os","de","da","do","em","no","na","para","que","e","um","uma",
         "sempre","voce","ferramenta","arquivo","use","utilize","dados","tarefa",
         "quando","com","ou","obrigatoriamente","solicitar","pedir","conteudo"}


def _parecida(regra, texto, limiar=0.6):
    """Duplicata SEMANTICA por sobreposicao de palavras significativas.

    Sem isso o professor reescreve a mesma aula com outras palavras a cada geracao
    e o conhecimento vira lixo acumulado (observado: 3 regras identicas em conteudo).
    """
    def sig(s):
        import re, unicodedata
        s = unicodedata.normalize("NFKD", s.lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return {p for p in re.findall(r"[a-z_]{3,}", s) if p not in _STOP}

    nova = sig(regra)
    if not nova:
        return True
    for linha in texto.splitlines():
        if not linha.strip().startswith("- "):
            continue
        velha = sig(linha)
        if velha and len(nova & velha) / len(nova) >= limiar:
            return True
    return False


def limpar_regra(bruto):
    """O professor as vezes tagarela. Fica so a primeira linha que parece regra."""
    for linha in (bruto or "").splitlines():
        l = linha.strip().strip("`")
        if l.startswith("- ") and len(l) > 8:
            return l
    for linha in (bruto or "").splitlines():
        l = linha.strip().strip("`*-# ")
        if len(l) > 15 and not l.lower().startswith(("regra", "aqui", "claro")):
            return "- " + l
    return None


# --- o ciclo -----------------------------------------------------------------

def uma_geracao(nome_task, professor, n):
    t = tasks.TASKS[nome_task]
    tasks.preparar(nome_task)
    agent.CONHECIMENTO_FERRAMENTAS = conhecimento_atual()

    print(f"\n{'='*70}\nGERACAO {n} — task '{nome_task}'\n{'='*70}")
    try:
        r = agent.rodar(t["pedido"], MODELO, verbose=False)
    except Exception as e:
        return False, f"o agente estourou: {e}", None

    chamadas = "\n".join(f"  {c[0]}({c[1]}) -> {c[2][:90]}" for c in r["chamadas"]) or "  (nenhuma)"
    print(chamadas)

    ok, diag = t["verificar"](None)
    print(f"\nJUIZ: {'PASSOU' if ok else 'FALHOU'} — {diag}")
    if ok:
        return True, diag, None

    prompt = PROMPT_PROFESSOR.format(pedido=t["pedido"], diagnostico=diag, chamadas=chamadas)
    print(f"\n[professor: {professor}] escrevendo a regra...")
    try:
        regra = limpar_regra(PROFESSORES[professor](prompt))
    except Exception as e:
        print(f"  professor indisponivel ({e}) — geracao sem aula")
        return False, diag, None

    if not regra:
        print("  professor nao produziu regra utilizavel")
        return False, diag, None

    atual = conhecimento_atual()
    if _parecida(regra, atual):
        print(f"  REGRA SEMANTICAMENTE REPETIDA — descartada: {regra}")
        print("  -> a aula ja estava acoplada e nao adiantou. Nao poluir o conhecimento.")
        return False, diag, "REPETIDA"

    print(f"  REGRA NOVA: {regra}")
    CONHECIMENTO.write_text(atual.rstrip() + "\n" + regra + "\n")
    acoplar(conhecimento_atual())
    return False, diag, regra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="nome da task, ou 'todas'")
    ap.add_argument("--geracoes", type=int, default=4)
    ap.add_argument("--professor", default="gemini", choices=list(PROFESSORES))
    ap.add_argument("--reset", action="store_true", help="apaga o conhecimento aprendido")
    a = ap.parse_args()

    if a.reset and CONHECIMENTO.exists():
        CONHECIMENTO.unlink()
        print("conhecimento resetado")

    acoplar(conhecimento_atual())
    alvos = list(tasks.TASKS) if a.task == "todas" else [a.task]
    placar = {}

    for nome in alvos:
        historico, diags, estagnado = [], [], False
        for n in range(1, a.geracoes + 1):
            ok, diag, regra = uma_geracao(nome, a.professor, n)
            historico.append(ok)
            diags.append(diag)
            if ok:
                print(f"\n>>> '{nome}' resolvido na geracao {n}")
                break
            # Estagnacao: mesma falha 2x seguidas E a aula ja estava acoplada.
            # Continuar aqui e desperdicio — e sinal de teto de prompt, nao de aula ruim.
            if len(diags) >= 2 and diags[-1] == diags[-2] and regra == "REPETIDA":
                print(
                    f"\n>>> ESTAGNOU em '{nome}': mesma falha 2 geracoes seguidas com a "
                    "aula ja acoplada.\n    Prompt nao resolve este caso — o proximo passo "
                    "e LoRA (nivel 2) ou mudar o desenho da ferramenta, nao mais regras."
                )
                estagnado = True
                break
        placar[nome] = (historico, estagnado)

    print(f"\n{'='*70}\nPLACAR\n{'='*70}")
    for nome, (h, est) in placar.items():
        marcas = " ".join("PASSOU" if x else "falhou" for x in h)
        print(f"  {nome:14s} {marcas}{'   [ESTAGNOU -> candidato a LoRA]' if est else ''}")
    print(f"\nconhecimento acumulado ({CONHECIMENTO}):")
    print(conhecimento_atual())


if __name__ == "__main__":
    main()
