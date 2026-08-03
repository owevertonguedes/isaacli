"""Manda o agente construir jogos de verdade e mede. Sem aula: isto e a linha de base.

Uso:
    AGENTE_RAIZ=/home/usuario/DevTools/minijogos python3 construir.py --modelo isaac
"""
import argparse
import json
import time
from pathlib import Path

import agent
import tools
import verificar_jogo

JOGOS = [
    ("adivinha.html", "Adivinhe o Numero",
     "O jogador tenta adivinhar um numero secreto de 1 a 100. A cada palpite o jogo "
     "diz 'maior' ou 'menor', e conta quantas tentativas foram usadas."),
    ("clique.html", "Teste de Reflexo",
     "Um botao aparece em posicao aleatoria na tela. Quando o jogador clica, mede o "
     "tempo de reacao em milissegundos e mostra o melhor tempo."),
    ("memoria.html", "Jogo da Memoria",
     "Um tabuleiro 4x4 com 8 pares de simbolos virados para baixo. O jogador clica em "
     "duas cartas; se forem iguais ficam viradas para cima. Conta as jogadas."),
    ("forca.html", "Jogo da Forca",
     "O jogo sorteia uma palavra de uma lista fixa em portugues. O jogador clica em "
     "letras; erros contam ate 6. Mostra a palavra com underscores nas letras ocultas."),
]

INSTRUCAO = """Crie o arquivo jogos/{arquivo} — um jogo de navegador chamado "{titulo}".

{descricao}

REQUISITOS OBRIGATORIOS:
- UM unico arquivo HTML completo, comecando com <!DOCTYPE html> e com <html>, <head> e <body>.
- Todo o CSS dentro de <style> e todo o JavaScript dentro de <script>, no mesmo arquivo.
- NAO use nenhuma biblioteca externa, nenhum link http, nenhum arquivo separado.
- O jogo tem que funcionar so abrindo o arquivo no navegador.
- Escreva o arquivo inteiro de uma vez com a ferramenta write_file.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="isaac-granite")
    ap.add_argument("--rodada", default="baseline")
    a = ap.parse_args()

    raiz = tools.SANDBOX_ROOT
    (raiz / "jogos").mkdir(parents=True, exist_ok=True)
    print(f"raiz do agente: {raiz}\nmodelo: {a.modelo}\nrodada: {a.rodada}\n")

    resultados = []
    t_total = time.time()
    for arquivo, titulo, desc in JOGOS:
        pedido = INSTRUCAO.format(arquivo=arquivo, titulo=titulo, descricao=desc)
        print(f"{'='*70}\n{arquivo} — {titulo}\n{'='*70}")
        t0 = time.time()
        try:
            r = agent.rodar(pedido, a.modelo, max_passos=4, verbose=False)
            chamadas = [(c[0], len(str(c[1]))) for c in r["chamadas"]]
        except Exception as e:
            chamadas = []
            print(f"  agente estourou: {e}")
        dt = time.time() - t0

        alvo = raiz / "jogos" / arquivo
        ok, probs = verificar_jogo.verificar(alvo)
        tam = alvo.stat().st_size if alvo.exists() else 0
        print(f"  tempo: {dt:.1f}s | escreveu: {tam} bytes | chamadas: {chamadas}")
        print(f"  JUIZ: {'PASSOU' if ok else 'FALHOU'}")
        for x in probs:
            print(f"    - {x}")
        resultados.append({"arquivo": arquivo, "ok": ok, "segundos": round(dt, 1),
                           "bytes": tam, "problemas": probs})

    total = time.time() - t_total
    passou = sum(1 for r in resultados if r["ok"])
    print(f"\n{'='*70}\nPLACAR {a.rodada}: {passou}/{len(JOGOS)} jogos validos")
    print(f"tempo total: {total/60:.1f} min | media por arquivo: {total/len(JOGOS):.0f}s")
    if total > 0:
        print(f"ritmo: {len(JOGOS)/(total/3600):.1f} arquivos/hora "
              f"({len(JOGOS)/(total/3600)*5:.0f} em 5 horas)")

    saida = Path(__file__).parent / f"resultado_{a.rodada}.json"
    saida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"detalhes em {saida}")


if __name__ == "__main__":
    main()
