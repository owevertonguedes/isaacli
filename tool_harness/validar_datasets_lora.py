#!/usr/bin/env python3
"""Valida datasets ativos do LoRA sem carregar torch/transformers.

O objetivo e barrar dado ruim antes de qualquer ciclo T4: ferramenta inexistente,
tool-call malformado, Graphify como comando errado, push/GPG indevido, assinatura
textual ensinada como padrao estrutural ou resposta que declara sucesso sem
evidencia.
"""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "datasets"
REPORT_DIR = RAIZ / "reports" / "datasets"

ACTIVE_DATASETS = [
    "task05-andaime-marcadores-gemini-2026-07-20.jsonl",
    "intencao-pergunta-vs-execucao-isaac-2026-07-20.jsonl",
    "comportamento-agente-workflow-2026-07-20.jsonl",
]

KNOWN_TOOLS = {
    "read_file",
    "write_file",
    "append_file",
    "replace_between",
    "check_file",
    "run_command",
}
ALLOWED_GRAPHIFY = {"query", "path", "explain", "diagnose"}
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TEXT_SIGNATURE_RE = re.compile(r"(Assinado por:\s*Isaac|Co-Authored-By:\s*Isaac|Signed-off-by:\s*Isaac)")


def jdump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def tool_line(name, arguments):
    return "<tool_call>" + jdump({"name": name, "arguments": arguments}) + "</tool_call>"


def load_rows(name):
    path = DATA_DIR / name
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield path, lineno, json.loads(line)


def commit_answer(row):
    commands = row.get("comandos_esperados") or row.get("comportamento_esperado") or []
    out = []
    for cmd in commands:
        if isinstance(cmd, str) and cmd.startswith("git "):
            out.append(tool_line("run_command", {"cmd": cmd}))
    pedido = row.get("pedido", "")
    assinatura_literal = re.search(r"assin.*(t[ií]tulo|texto|corpo|mensagem)", pedido, re.I)
    if not out and assinatura_literal:
        out.extend([
            tool_line("run_command", {"cmd": "git status"}),
            tool_line("run_command", {"cmd": "git diff"}),
            tool_line("run_command", {"cmd": "git add ."}),
            tool_line("run_command", {"cmd": "git commit -m \"Registra ajuste solicitado\" -m \"A mudança foi commitada porque o usuário pediu para preservar o ajuste atual.\" -m \"Assinado por: Isaac\""}),
            tool_line("run_command", {"cmd": "git log -1 --format=%B"}),
        ])
    elif not out and re.search(r"assin|assine", pedido, re.I):
        out.append("Esse pedido e ambiguo: autoria/coautoria estrutural e responsabilidade do CLI/app. Se voce quer assinatura textual no commit, diga se deve ser no titulo, corpo ou trailer. Nao vou usar `git commit -S` por conta propria.")
    checked_log = any(isinstance(cmd, str) and cmd == "git log -1 --format=%B" for cmd in commands)
    checked_log = checked_log or any("git log -1 --format=%B" in line for line in out)
    if checked_log:
        out.append("Verifiquei a mensagem do commit antes de responder.")
    elif not out:
        out.append("Nao vou declarar commit sem comando Git real e sem evidencia de sucesso.")
    return "\n".join(out).strip()


def intent_answer(row):
    commands = row.get("comandos_esperados") or []
    out = []
    for cmd in commands:
        if isinstance(cmd, str) and cmd.startswith("write_file"):
            out.append(tool_line("write_file", {"path": "index.html", "content": "<!DOCTYPE html>\n<html lang=\"pt-BR\"></html>"}))
        elif isinstance(cmd, str) and cmd.startswith("read_file"):
            out.append(tool_line("read_file", {"path": "index.html"}))
        elif isinstance(cmd, str) and cmd.startswith("check_file"):
            out.append(tool_line("check_file", {"path": "index.html"}))
    return "\n".join(out).strip() if out else row.get("resposta_esperada", "").strip()


def messages_answer(row):
    out = []
    for msg in row["messages"][2:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                out.append(tool_line(fn["name"], fn["arguments"]))
        elif msg.get("role") == "assistant" and msg.get("content"):
            out.append(msg["content"].strip())
    return "\n".join(out).strip()


def build_answer(dataset_name, row):
    if row.get("messages"):
        return messages_answer(row)
    if dataset_name.startswith("commit"):
        return commit_answer(row)
    return intent_answer(row)


def user_text(row):
    if row.get("messages"):
        return "\n".join(msg.get("content", "") for msg in row["messages"] if msg.get("role") == "user")
    return row.get("pedido", "")


def parse_tool_calls(text):
    parsed = []
    for raw in TOOL_RE.findall(text):
        try:
            data = json.loads(raw)
        except Exception as exc:
            parsed.append({"ok": False, "error": str(exc), "name": None, "arguments": None})
            continue
        parsed.append({
            "ok": isinstance(data, dict) and isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict),
            "name": data.get("name"),
            "arguments": data.get("arguments"),
        })
    return parsed


def raw_message_issues(row):
    issues = []
    for idx, msg in enumerate(row.get("messages") or []):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
            if name not in KNOWN_TOOLS:
                issues.append(f"raw message {idx}: ferramenta desconhecida {name!r}")
            if not isinstance(args, dict):
                issues.append(f"raw message {idx}: argumentos de {name!r} nao sao objeto")
    return issues


def command_issues(cmd):
    issues = []
    if "git commit -S" in cmd:
        issues.append("usa git commit -S para assinatura textual")
    if "git push" in cmd:
        issues.append("inclui git push em exemplo de commit sem push")
    if cmd.startswith("graphify-out/graph.json"):
        issues.append("tenta executar graphify-out/graph.json como comando")
    if "graphify-out/graph.json find" in cmd:
        issues.append("usa graphify-out/graph.json find")
    if cmd.startswith("graphify "):
        parts = cmd.split()
        sub = parts[1] if len(parts) > 1 else ""
        if sub not in ALLOWED_GRAPHIFY:
            issues.append(f"subcomando graphify proibido: {sub!r}")
    return issues


def validate_row(dataset_name, path, lineno, row):
    issues = raw_message_issues(row)
    answer = build_answer(dataset_name, row)
    calls = parse_tool_calls(answer)
    command_strings = []
    for call in calls:
        if not call["ok"]:
            issues.append("tool-call gerado malformado")
            continue
        if call["name"] not in KNOWN_TOOLS:
            issues.append(f"ferramenta gerada desconhecida: {call['name']!r}")
        if call["name"] == "run_command":
            cmd = (call["arguments"] or {}).get("cmd", "")
            if not isinstance(cmd, str):
                issues.append("run_command sem cmd string")
            else:
                command_strings.append(cmd)
                issues.extend(command_issues(cmd))

    forbidden = ["ISAAC_CHOOSE", '"name":"commit"', '"name": "commit"', "graphify-out/graph.json find"]
    for marker in forbidden:
        if marker in answer:
            issues.append(f"marcador proibido no alvo: {marker}")

    is_commit = dataset_name.startswith("commit") or str(row.get("id", "")).startswith("commit-")
    commit_failed_honestly = bool(re.search(r"(nao foi criado|não foi criado|falh|nothing to commit)", answer, re.I))
    literal_signature = bool(re.search(
        r"assine seu nome|assinatura textual|no corpo|no texto|no titulo|no título",
        user_text(row),
        re.I,
    ))
    has_commit_command = any(cmd.startswith("git commit") for cmd in command_strings)
    if is_commit and has_commit_command and not commit_failed_honestly:
        if not any(cmd in {"git log -1 --format=%B", "git status --short"} for cmd in command_strings):
            issues.append("commit sem verificacao de estado/log")
        if literal_signature and not TEXT_SIGNATURE_RE.search(answer):
            issues.append("pedido literal de assinatura sem assinatura textual")
        if not literal_signature and TEXT_SIGNATURE_RE.search(answer):
            issues.append("assinatura textual ensinada em commit normal")

    return {
        "dataset": dataset_name,
        "path": str(path.relative_to(RAIZ)),
        "line": lineno,
        "id": row.get("id"),
        "tool_calls": len(calls),
        "issues": issues,
    }


def main():
    rows = []
    for dataset in ACTIVE_DATASETS:
        for path, lineno, row in load_rows(dataset):
            rows.append(validate_row(dataset, path, lineno, row))

    issues = [row for row in rows if row["issues"]]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "lora-datasets-ativos-2026-07-20.json"
    report = {
        "active_datasets": ACTIVE_DATASETS,
        "rows": len(rows),
        "issues": len(issues),
        "items": rows,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "issues": len(issues),
        "report": str(report_path.relative_to(RAIZ)),
    }, ensure_ascii=False, indent=2))
    if issues:
        for item in issues:
            print(f"{item['path']}:{item['line']} {item['id']}: {item['issues']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
