"""Loop de auto-correcao: Isaac escreve, o juiz JOGA, o erro volta pra ele, repete.

A pergunta que este script responde: o Isaac consegue usar um erro CONCRETO pra
consertar o proprio codigo? Nao e auto-reflexao (modelo revisando a si mesmo, que
costuma trocar um bug por outro) — o sinal vem de fora, de um navegador de verdade.

    python3 corrigir.py --tentativas 4 --visivel
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import agent
import tools

AQUI = Path(__file__).parent
JUIZ = AQUI / "juiz_comportamental.js"

CONHECIMENTO = """Voce e Isaac, um programador que cria jogos de navegador em UM arquivo HTML.

REGRAS DO ARQUIVO:
- Sempre escreva o arquivo COMPLETO com write_file, de <!DOCTYPE html> ate </html>.
- NUNCA deixe comentario de placeholder tipo "// JavaScript aqui" ou "/* CSS aqui */".
  Se voce escreveu um comentario desses, o codigo real esta faltando: escreva o codigo.
- Todo o JavaScript vai DENTRO de <script>...</script>. Nada de codigo solto no HTML.
- Todo elemento interativo precisa estar ligado: se voce escreve uma funcao jogar(),
  precisa existir um <button onclick="jogar()"> que a chame.
- Nada de biblioteca externa, nada de link http. Um arquivo so.
- Depois de escrever o arquivo, chame check_file nele. Se vier problema,
  conserte e escreva de novo ANTES de dizer que terminou.
"""

PRIMEIRA = """Crie o arquivo jogos/{arquivo} — um jogo de navegador chamado "{titulo}".

{descricao}

Escreva o arquivo HTML completo de uma vez com write_file."""

CONSERTO = """O arquivo jogos/{arquivo} que voce escreveu NAO FUNCIONA.

Um navegador de verdade abriu o jogo, clicou nos botoes, preencheu os campos e
apertou teclas. Isto foi o que deu errado:

{problemas}

{evidencias}

Leia o arquivo com read_file, descubra por que isso acontece, e escreva o arquivo
CORRIGIDO INTEIRO de novo com write_file. Nao escreva so o pedaco: escreva o arquivo
completo, de <!DOCTYPE html> ate </html>, com o problema resolvido."""


def checagem_rapida(arquivo_relativo):
    """Filtro de ~1s antes do juiz: sintaxe, runtime ao abrir, placeholder.

    NAO e criterio de sucesso (quem aprova continua sendo o juiz comportamental);
    so evita gastar o juiz com codigo obviamente quebrado.
    """
    saida = tools.check_file(arquivo_relativo)
    if saida.startswith("OK"):
        return {"ok": True, "problemas": [], "evidencias": []}
    probs = [l[2:] for l in saida.splitlines() if l.startswith("- ")] or [saida]
    return {"ok": False, "problemas": probs, "evidencias": [], "rapida": True}


def julgar(caminho, visivel):
    cmd = ["node", str(JUIZ), str(caminho)] + (["--visivel"] if visivel else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "problemas": [f"o juiz nao rodou: {r.stderr[:200]}"], "evidencias": []}


def um_jogo(arquivo, titulo, descricao, modelo, tentativas, visivel):
    raiz = tools.SANDBOX_ROOT
    alvo = raiz / "jogos" / arquivo
    agent.CONHECIMENTO_FERRAMENTAS = CONHECIMENTO
    historico = []

    for t in range(1, tentativas + 1):
        if t == 1:
            pedido = PRIMEIRA.format(arquivo=arquivo, titulo=titulo, descricao=descricao)
        else:
            ult = historico[-1]
            pedido = CONSERTO.format(
                arquivo=arquivo,
                problemas="\n".join(f"- {p}" for p in ult["problemas"]),
                evidencias="\n".join(ult.get("evidencias", [])),
            )

        print(f"\n--- {arquivo} | tentativa {t}/{tentativas} ---")
        t0 = time.time()
        try:
            r = agent.rodar(pedido, modelo, max_passos=4, verbose=False)
            escreveu = [c[0] for c in r["chamadas"]]
        except Exception as e:
            print(f"  agente estourou: {e}")
            escreveu = []
        dt = time.time() - t0

        # Degrau barato primeiro: se nem abre, o erro volta sem acordar o juiz.
        v = checagem_rapida(f"jogos/{arquivo}")
        via_juiz = v["ok"]
        if via_juiz:
            v = julgar(alvo, visivel)
        tam = alvo.stat().st_size if alvo.exists() else 0
        print(f"  {dt:.0f}s | {tam} bytes | chamou: {escreveu}")
        rotulo = "JUIZ" if via_juiz else "CHECAGEM RAPIDA"
        print(f"  {rotulo}: {'PASSOU ✅' if v['ok'] else 'FALHOU'}")
        for p in v.get("problemas", []):
            print(f"    - {p}")
        historico.append(v)

        if v["ok"]:
            return True, t, historico
    return False, tentativas, historico


JOGOS = [
    ("adivinha.html", "Adivinhe o Numero",
     "O jogador digita um palpite de 1 a 100 e clica num botao. O jogo responde na "
     "propria pagina se o numero secreto e maior ou menor, e mostra o total de tentativas."),
    ("clique.html", "Teste de Reflexo",
     "Depois de um tempo aleatorio a tela muda de cor e o jogador deve clicar. O jogo "
     "mostra na pagina o tempo de reacao em milissegundos e o melhor tempo ate agora."),
    ("forca.html", "Jogo da Forca",
     "Sorteia uma palavra de uma lista fixa. Mostra a palavra com underscores. O jogador "
     "clica em botoes de letras; acerto revela a letra, erro conta ate 6 e perde."),
    ("memoria.html", "Jogo da Memoria",
     "Tabuleiro 4x4 com 8 pares de simbolos escondidos. Clicar em duas cartas iguais as "
     "mantem viradas. Mostra o numero de jogadas na pagina."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="isaac-granite")
    ap.add_argument("--tentativas", type=int, default=4)
    ap.add_argument("--visivel", action="store_true")
    ap.add_argument("--jogo", help="roda so um jogo (nome do arquivo)")
    a = ap.parse_args()

    (tools.SANDBOX_ROOT / "jogos").mkdir(parents=True, exist_ok=True)
    alvos = [j for j in JOGOS if not a.jogo or j[0] == a.jogo]
    placar = {}
    t0 = time.time()

    for arquivo, titulo, desc in alvos:
        print(f"\n{'='*72}\n{titulo}  ({arquivo})\n{'='*72}")
        ok, n, hist = um_jogo(arquivo, titulo, desc, a.modelo, a.tentativas, a.visivel)
        placar[arquivo] = (ok, n)

    print(f"\n{'='*72}\nPLACAR — modelo: {a.modelo} | {(time.time()-t0)/60:.1f} min")
    for arq, (ok, n) in placar.items():
        print(f"  {arq:16s} {'PASSOU na tentativa ' + str(n) if ok else 'falhou em ' + str(n) + ' tentativas'}")
    print(f"  TOTAL: {sum(1 for ok, _ in placar.values() if ok)}/{len(placar)} jogaveis")

    Path(AQUI / f"resultado_correcao_{a.modelo}.json").write_text(
        json.dumps({k: {"ok": v[0], "tentativas": v[1]} for k, v in placar.items()}, indent=2))


if __name__ == "__main__":
    main()
