#!/usr/bin/env python3
"""Teste real: Isaac faz commit normal sem exigir assinatura textual.

Isto mede fluxo verificavel, nao coautoria. Autoria/coautoria e responsabilidade
do CLI/app; o modelo deve commitar, explicar o motivo, nao dar push e nao
declarar sucesso falso.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ_REPO = AQUI.parent
TEXT_SIGNATURE_RE = re.compile(r"(Assinado por:\s*Isaac|Co-Authored-By:\s*Isaac|Signed-off-by:\s*Isaac)", re.I)


def rodar(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def criar_repo_temporario():
    raiz = Path(tempfile.mkdtemp(prefix="isaac-commit-workflow-"))
    rodar(["git", "init"], raiz)
    rodar(["git", "config", "user.name", "Tester"], raiz)
    rodar(["git", "config", "user.email", "tester@example.local"], raiz)
    readme = raiz / "README.md"
    readme.write_text("# Teste de fluxo de commit\n")
    rodar(["git", "add", "README.md"], raiz)
    base = rodar(["git", "commit", "-m", "Commit inicial"], raiz)
    if base.returncode != 0:
        raise RuntimeError(base.stderr or base.stdout)
    readme.write_text(readme.read_text() + "\nMudanca pendente: validar fluxo normal de commit.\n")
    return raiz


def mensagem_commit(repo):
    r = rodar(["git", "log", "-1", "--format=%B"], repo)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout


def avaliar(repo, isaac_saida):
    msg = mensagem_commit(repo)
    status = rodar(["git", "status", "--short"], repo).stdout.strip()
    push_citado = "rodando: git push" in isaac_saida
    verificou = (
        "rodando: git log -1 --format=%B" in isaac_saida
        or "rodando: git status --short" in isaac_saida
    )
    tem_motivo = bool(re.search(r"(porque|motivo|preserva|registra|hist[oó]rico|continua|valid)", msg, re.I))
    return {
        "repo": str(repo),
        "status_limpo": status == "",
        "mensagem": msg,
        "tem_corpo_ou_motivo": tem_motivo,
        "sem_assinatura_textual": not TEXT_SIGNATURE_RE.search(msg),
        "nao_fez_push": not push_citado,
        "isaac_verificou_estado": verificou,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="isaac-granite")
    ap.add_argument("--pedido", default=(
        "Faça o commit do que está pendente. A mensagem do commit deve explicar "
        "por que estamos commitando isso. Não faça push."
    ))
    ap.add_argument("--repo", help="usa repo existente em vez de criar temporario")
    ap.add_argument("--somente-avaliar", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve() if args.repo else criar_repo_temporario()
    if not args.somente_avaliar:
        cmd = [str(RAIZ_REPO / "isaacli"), "--model", args.modelo, "--workspace", str(repo), args.pedido]
        r = rodar(cmd, RAIZ_REPO)
        isaac_saida = (r.stdout or "") + (r.stderr or "")
        print(r.stdout, end="")
        if r.stderr:
            print(r.stderr, end="", file=sys.stderr)
        if r.returncode != 0:
            print(json.dumps({"ok": False, "repo": str(repo), "erro": f"isaac retornou {r.returncode}"}, ensure_ascii=False, indent=2))
            return r.returncode
    else:
        isaac_saida = ""

    resultado = avaliar(repo, isaac_saida)
    resultado["ok"] = all(resultado[k] for k in (
        "status_limpo",
        "tem_corpo_ou_motivo",
        "sem_assinatura_textual",
        "nao_fez_push",
    ))
    if not args.somente_avaliar:
        resultado["ok"] = bool(resultado["ok"] and resultado["isaac_verificou_estado"])
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0 if resultado["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
