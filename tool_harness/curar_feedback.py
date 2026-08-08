#!/usr/bin/env python3
"""Curadoria local de feedback e sessoes do Isaac CLI.

Nao chama Ollama, Gemini nem rede. O objetivo e transformar JSONL bruto em
episodios pequenos, com sinais mecanicos que evitem destilar mentira como
aprendizado.
"""
import argparse
import json
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SESSOES_DIR = AQUI / "cli_sessoes"
FEEDBACK_DIR = AQUI / "feedback"
CURADORIA_DIR = AQUI / "curadoria"

SUCESSO_RE = re.compile(
    r"\b("
    r"feito|feita|realizado|realizada|concluido|concluida|corrigido|corrigida|"
    r"sucesso|atualizado|atualizada|publicado|publicada|pushado|commitado"
    r")\b",
    re.I,
)
FALHA_RE = re.compile(r"\b(falh|erro|nao foi|nao consegui|não foi|não consegui)\b", re.I)
PUSH_RE = re.compile(r"\b(push|pushe|pushar|publica|publique|publicar)\b", re.I)
PUSH_NEGADO_RE = re.compile(
    r"\b(nao|não|sem)\b.{0,20}\b(push|pushe|pushar|publicar)\b|"
    r"\b(push|pushe|pushar|publicar)\b.{0,20}\b(nao|não)\b",
    re.I,
)
AMEND_RE = re.compile(
    r"\b(commit|mensagem|assinatura)\b.{0,100}\b(ultimo|último|refa\w*|corrij\w*|alter\w*|mud\w*)|"
    r"\b(refa\w*|corrij\w*|alter\w*|mud\w*)\b.{0,100}\b(commit|mensagem|assinatura)\b",
    re.I,
)
ASSINATURA_RE = re.compile(r"\b(assinatura|assinado|assinada|signed|co-authored)\b", re.I)
TRAILER_RE = re.compile(r"^(Co-Authored-By|Signed-off-by):\s+.+<[^<>@\s]+@[^<>@\s]+>\s*$", re.M)


def ler_jsonl(path):
    eventos = []
    if not path.exists():
        return eventos
    for n, linha in enumerate(path.read_text().splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            eventos.append(json.loads(linha))
        except json.JSONDecodeError as e:
            eventos.append({
                "tipo": "json_invalido",
                "path": str(path),
                "linha": n,
                "erro": str(e),
                "bruto": linha,
            })
    return eventos


def gravar_jsonl(path, itens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in itens:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def resolver_sessao(valor, base=SESSOES_DIR):
    p = Path(valor)
    if p.exists():
        return p
    if p.suffix != ".jsonl":
        p = base / f"{valor}.jsonl"
    else:
        p = base / p.name
    return p


def carregar_feedbacks(feedback_dir=FEEDBACK_DIR):
    feedbacks = []
    for path in sorted(feedback_dir.glob("*.jsonl")):
        for item in ler_jsonl(path):
            item["_feedback_file"] = str(path)
            feedbacks.append(item)
    return feedbacks


def resposta_declara_sucesso(texto):
    return bool(SUCESSO_RE.search(texto or "") and not FALHA_RE.search(texto or ""))


def _cmds(eventos):
    return [e for e in eventos if e.get("tipo") == "tool_result" and e.get("cmd")]


def _ultimo_assistant(eventos):
    finais = [e for e in eventos if e.get("tipo") == "assistant_final"]
    return finais[-1].get("content", "") if finais else ""


def _texto_usuarios(eventos):
    return "\n".join(e.get("content", "") for e in eventos if e.get("tipo") == "user")


def _assinatura_verificada(eventos):
    for e in _cmds(eventos):
        cmd = e.get("cmd", "")
        if cmd.startswith("git log") and "%B" in cmd and e.get("codigo") == 0:
            if TRAILER_RE.search(e.get("resultado", "")):
                return True
    return False


def sinais_mecanicos(eventos):
    comandos = _cmds(eventos)
    texto_user = _texto_usuarios(eventos)
    final = _ultimo_assistant(eventos)
    falhou = any(c.get("codigo") not in (None, 0) for c in comandos)
    push_mencionado = bool(PUSH_RE.search(texto_user))
    push_negado = bool(PUSH_NEGADO_RE.search(texto_user))
    push_cmds = [c for c in comandos if c.get("cmd", "").strip().startswith("git push")]
    amend_cmds = [c for c in comandos if "git commit --amend" in c.get("cmd", "")]
    assinatura_pedida = bool(ASSINATURA_RE.search(texto_user))

    return {
        "comando_falhou": falhou,
        "resposta_declara_sucesso": bool(falhou and resposta_declara_sucesso(final)),
        "push_pedido": bool(push_mencionado and not push_negado),
        "push_executado": any(c.get("codigo") == 0 for c in push_cmds),
        "push_bloqueado_por_risco": bool(push_mencionado and not push_cmds and PUSH_NEGADO_RE.search(final or "")),
        "push_nao_executado": bool(push_mencionado and not push_negado and not push_cmds),
        "amend_necessario": bool(AMEND_RE.search(texto_user)),
        "amend_executado": any(c.get("codigo") == 0 for c in amend_cmds),
        "assinatura_pedida": assinatura_pedida,
        "assinatura_verificada": bool(assinatura_pedida and _assinatura_verificada(eventos)),
    }


def classificar(feedback, sinais, comandos):
    risco_git = any((c.get("cmd") or "").startswith("git ") for c in comandos)
    risco_push = sinais["push_pedido"] or any((c.get("cmd") or "").startswith("git push") for c in comandos)
    if risco_git or risco_push:
        return "precisa_juiz", "episodio envolve git/push; exige verificacao mecanica"
    if sinais["comando_falhou"] or sinais["resposta_declara_sucesso"]:
        return "precisa_juiz", "ha comando falho ou possivel sucesso falso"
    tipo = (feedback or {}).get("feedback_tipo")
    nota = (feedback or {}).get("nota")
    if tipo == "ruim" or (isinstance(nota, int) and nota <= 3):
        return "rejeitado", "feedback negativo nao entra como comportamento desejado"
    if tipo in {"bom", "nota"} and isinstance(nota, int) and nota >= 7:
        return "aprovado", "feedback positivo sem sinal mecanico de risco"
    return "precisa_juiz", "sem feedback suficiente para aprovar"


def extrair_episodio(session_path, feedback=None):
    eventos = ler_jsonl(session_path)
    comandos = _cmds(eventos)
    users = [e.get("content", "") for e in eventos if e.get("tipo") == "user"]
    finais = [e.get("content", "") for e in eventos if e.get("tipo") == "assistant_final"]
    sinais = sinais_mecanicos(eventos)
    classe, motivo = classificar(feedback, sinais, comandos)

    return {
        "session_id": session_path.stem,
        "session_path": str(session_path),
        "feedback": feedback or {"feedback_tipo": "manual", "nota": None, "comentario": ""},
        "classe": classe,
        "motivo": motivo,
        "pedido": users[-1] if users else "",
        "pedidos": users,
        "resposta_final": finais[-1] if finais else "",
        "comandos": [
            {
                "cmd": c.get("cmd"),
                "codigo": c.get("codigo"),
                "falhou": c.get("codigo") not in (None, 0),
            }
            for c in comandos
        ],
        "erros": [
            {
                "cmd": c.get("cmd"),
                "codigo": c.get("codigo"),
                "resultado": c.get("resultado", "")[-1200:],
            }
            for c in comandos
            if c.get("codigo") not in (None, 0)
        ],
        "sinais": sinais,
    }


def montar_episodios(feedback_dir=FEEDBACK_DIR, session_dir=SESSOES_DIR, sessoes_manuais=None):
    episodios = []
    vistos = set()
    for fb in carregar_feedbacks(feedback_dir):
        session_path = Path(fb.get("session_path") or "")
        if not session_path.exists():
            sid = fb.get("session_id")
            session_path = session_dir / f"{sid}.jsonl"
        if not session_path.exists():
            episodios.append({
                "session_id": fb.get("session_id", ""),
                "session_path": str(session_path),
                "feedback": fb,
                "classe": "rejeitado",
                "motivo": "sessao apontada pelo feedback nao existe",
                "pedido": "",
                "pedidos": [],
                "resposta_final": "",
                "comandos": [],
                "erros": [],
                "sinais": {},
            })
            continue
        episodios.append(extrair_episodio(session_path, fb))
        vistos.add(str(session_path.resolve()))

    for sessao in sessoes_manuais or []:
        session_path = resolver_sessao(sessao, session_dir)
        if not session_path.exists():
            raise FileNotFoundError(f"sessao nao existe: {sessao}")
        chave = str(session_path.resolve())
        if chave not in vistos:
            episodios.append(extrair_episodio(session_path))
            vistos.add(chave)
    return episodios


def escrever_saida(episodios, out_dir=CURADORIA_DIR):
    aprovados = [e for e in episodios if e["classe"] == "aprovado"]
    rejeitados = [e for e in episodios if e["classe"] == "rejeitado"]
    pendentes = [e for e in episodios if e["classe"] == "precisa_juiz"]
    gravar_jsonl(out_dir / "episodios.jsonl", episodios)
    gravar_jsonl(out_dir / "aprovados.jsonl", aprovados)
    gravar_jsonl(out_dir / "rejeitados.jsonl", rejeitados)
    gravar_jsonl(out_dir / "precisa_juiz.jsonl", pendentes)
    (out_dir / "README.md").write_text(
        "# Curadoria do Isaac CLI\n\n"
        "Gerado por `python3 tool_harness/curar_feedback.py`.\n\n"
        "Arquivos:\n"
        "- `episodios.jsonl`: episodios extraidos de feedback/sessoes.\n"
        "- `aprovados.jsonl`: candidatos sem sinal mecanico de risco.\n"
        "- `rejeitados.jsonl`: feedback negativo ou sessao invalida.\n"
        "- `precisa_juiz.jsonl`: casos que exigem verificacao mecanica/humana.\n\n"
        "Feedback bruto nao vira aprendizado direto. Git, push, producao, dinheiro "
        "e credencial entram como `precisa_juiz` por padrao.\n"
    )


def resumo(episodios):
    contagem = {"aprovado": 0, "rejeitado": 0, "precisa_juiz": 0}
    for e in episodios:
        contagem[e["classe"]] = contagem.get(e["classe"], 0) + 1
    return contagem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback-dir", default=str(FEEDBACK_DIR))
    ap.add_argument("--session-dir", default=str(SESSOES_DIR))
    ap.add_argument("--out-dir", default=str(CURADORIA_DIR))
    ap.add_argument("--session", action="append", default=[],
                    help="id ou caminho de sessao para curar mesmo sem feedback")
    ap.add_argument("--dry-run", action="store_true",
                    help="apenas mostra contagens; nao escreve tool_harness/curadoria")
    args = ap.parse_args(argv)

    episodios = montar_episodios(
        feedback_dir=Path(args.feedback_dir),
        session_dir=Path(args.session_dir),
        sessoes_manuais=args.session,
    )
    contagem = resumo(episodios)
    print(
        "feedback/sessoes curados: "
        f"total={len(episodios)} "
        f"aprovados={contagem.get('aprovado', 0)} "
        f"rejeitados={contagem.get('rejeitado', 0)} "
        f"precisa_juiz={contagem.get('precisa_juiz', 0)}"
    )
    for e in episodios:
        print(f"- {e['session_id']}: {e['classe']} ({e['motivo']})")
    if not args.dry_run:
        escrever_saida(episodios, Path(args.out_dir))
        print(f"saida: {Path(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
