"""Confinamento do _safe(): com a raiz escolhivel no app, isto e a UNICA protecao.

Rode depois de QUALQUER mudanca em tools.py ou no seletor de pasta do app:

    python3 testar_sandbox.py

Testa ../, caminho absoluto e symlink — inclusive depois de trocar a raiz em
tempo de execucao (que e o que o seletor de pasta do app faz).
"""
import sys
import tempfile
from pathlib import Path

import tools


def esperar_bloqueio(descricao, fn):
    try:
        fn()
    except ValueError:
        print(f"[ok    ] {descricao}: bloqueado")
        return True
    print(f"[FUROU ] {descricao}: PASSOU POR FORA DA SANDBOX")
    return False


def testar_raiz(raiz: Path) -> int:
    tools.SANDBOX_ROOT = raiz
    raiz.mkdir(parents=True, exist_ok=True)
    furos = 0

    fora = raiz.parent / "fora_da_sandbox.txt"
    fora.write_text("segredo")

    if not esperar_bloqueio(f"{raiz.name}: ../", lambda: tools._safe("../fora_da_sandbox.txt")):
        furos += 1
    # Caminho absoluto NAO lanca erro: e neutralizado (lstrip('/') o trata como
    # relativo). O contrato e que o resultado caia DENTRO da raiz, sempre.
    for absoluto in (str(fora), "/etc/passwd"):
        p = tools._safe(absoluto)
        if tools.SANDBOX_ROOT.resolve() not in p.parents:
            print(f"[FUROU ] {raiz.name}: absoluto {absoluto} resolveu pra fora: {p}")
            furos += 1
        else:
            print(f"[ok    ] {raiz.name}: absoluto {absoluto} vira caminho interno")

    link = raiz / "atalho"
    link.unlink(missing_ok=True)
    link.symlink_to(fora.parent)
    if not esperar_bloqueio(f"{raiz.name}: symlink pra fora", lambda: tools._safe("atalho/fora_da_sandbox.txt")):
        furos += 1
    link.unlink()
    fora.unlink()
    return furos


def main():
    furos = 0
    with tempfile.TemporaryDirectory() as tmp:
        # raiz padrao e uma raiz trocada em runtime (o que o seletor do app faz)
        furos += testar_raiz(Path(tmp) / "raiz_a" / "sandbox")
        furos += testar_raiz(Path(tmp) / "raiz_b" / "projeto")
    print(f"\n{'SANDBOX CONFINADA' if furos == 0 else f'{furos} FURO(S) — NAO ligar o seletor de pasta'}")
    sys.exit(0 if furos == 0 else 1)


if __name__ == "__main__":
    main()
