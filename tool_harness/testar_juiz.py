"""Calibracao do juiz: jogos REAIS tem que passar, FRAUDES tem que falhar.

O juiz e o produto (principio 7) — entao o juiz tem teste. Rode isto depois de
qualquer mudanca no juiz_comportamental.js ou specs.js:

    python3 testar_juiz.py

As fraudes em calibracao/fraudes/ reproduzem o reward hacking medido em
2026-07-19 (conteudo muda mas o jogo nao existe). Se alguma fraude PASSAR,
o juiz esta frouxo e NADA pode ser destilado em cima dele.
"""
import json
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).parent
JUIZ = AQUI / "juiz_comportamental.js"
CASOS = [(p, True) for p in sorted((AQUI / "calibracao").glob("*.html"))] + \
        [(p, False) for p in sorted((AQUI / "calibracao" / "fraudes").glob("*.html"))]


def julgar(caminho):
    r = subprocess.run(["node", str(JUIZ), str(caminho)],
                       capture_output=True, text=True, timeout=180, cwd=str(AQUI))
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "problemas": [f"juiz nao rodou: {r.stderr[:200]}"]}


def main():
    falhas = 0
    for caminho, deve_passar in CASOS:
        v = julgar(caminho)
        rotulo = "real  " if deve_passar else "fraude"
        certo = v["ok"] == deve_passar
        status = "ok" if certo else "ERRADO"
        print(f"[{status:6s}] {rotulo} {caminho.parent.name}/{caminho.name}: "
              f"juiz disse {'PASSOU' if v['ok'] else 'FALHOU'}")
        if not certo:
            falhas += 1
            for p in v.get("problemas", []):
                print(f"           - {p}")
    print(f"\n{'JUIZ CALIBRADO' if falhas == 0 else f'{falhas} caso(s) errado(s) — juiz descalibrado'}")
    sys.exit(0 if falhas == 0 else 1)


if __name__ == "__main__":
    main()
