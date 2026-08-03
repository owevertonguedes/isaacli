#!/usr/bin/env python3
"""Inventaria historicos locais do Claude Code sem exportar dados.

Cria uma pasta ignorada pelo Git com manifest e links apenas para arquivos que
podem virar fonte de curadoria. Nao chama rede, nao chama modelo e nao copia
credenciais.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


HOME = Path.home()
DEFAULT_DEST = Path("claude_historico")

PRUNE_DIRS = {
    ".cache",
    ".local/share/Trash",
    ".npm",
    ".ollama",
    ".rustup",
    ".cargo",
    ".var",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "lora_runs",
    "finetune_test/hf_cache",
}

EXCLUDED_PARTS = {
    "backups",
    "cache",
    "jobs",
    "plugins",
    "shell-snapshots",
    "statsig",
}

SENSITIVE_NAMES = {
    ".credentials.json",
    "credentials.json",
    "settings.json",
    "stats-cache.json",
}


def is_under(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def should_prune_dir(path):
    text = str(path)
    parts = set(path.parts)
    if parts.intersection({"node_modules", ".cache", ".npm", ".ollama", ".rustup", ".cargo", ".var"}):
        return True
    return any(text.endswith(p) or f"/{p}/" in text for p in PRUNE_DIRS)


def discover_roots(scan_root=HOME):
    roots = []
    for current, dirs, _files in os.walk(scan_root):
        current_path = Path(current)
        dirs[:] = [d for d in dirs if not should_prune_dir(current_path / d)]
        if ".claude" in dirs:
            roots.append(current_path / ".claude")
            dirs.remove(".claude")
    return sorted(set(roots))


def classify(relpath):
    parts = relpath.parts
    name = relpath.name
    suffix = relpath.suffix.lower()

    if name in SENSITIVE_NAMES:
        return "segredo_ou_config_excluido", False, "arquivo de credencial/configuracao excluido por padrao"
    if any(part in EXCLUDED_PARTS for part in parts):
        return "cache_ou_estado_excluido", False, "cache, plugin, backup ou estado operacional"
    if parts and parts[0] == "file-history":
        return "file_history_excluido", False, "historico de arquivos pode conter codigo/segredos; curar separadamente"
    if parts and parts[0] == "paste-cache":
        return "paste_cache_excluido", False, "pastes podem conter segredos; curar separadamente"
    if name == "history.jsonl":
        return "history_jsonl", True, "historico global de prompts/respostas"
    if len(parts) >= 3 and parts[0] == "projects" and suffix == ".jsonl":
        return "session_jsonl", True, "sessao Claude Code por projeto"
    if "memory" in parts and suffix == ".md":
        return "memory_md", True, "memoria textual curavel"
    if suffix in {".md", ".jsonl"}:
        return "claude_texto_local", True, "texto local pequeno potencialmente curavel"
    return "outro_excluido", False, "formato nao entra no primeiro passe"


def file_entry(path, root):
    rel = path.relative_to(root)
    stat = path.stat()
    category, included, reason = classify(rel)
    return {
        "path": str(path),
        "root": str(root),
        "relpath": str(rel),
        "category": category,
        "included": included,
        "reason": reason,
        "size_bytes": stat.st_size,
        "mtime": int(stat.st_mtime),
    }


def iter_files(root):
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        rel_current = current_path.relative_to(root)
        dirs[:] = [
            d
            for d in dirs
            if classify(rel_current / d)[0] not in {"cache_ou_estado_excluido"}
            and d not in EXCLUDED_PARTS
        ]
        for name in files:
            path = current_path / name
            try:
                if path.is_file() or path.is_symlink():
                    yield path
            except OSError:
                continue


def safe_link_name(index, entry):
    raw = f"{index:05d}-{entry['category']}-{entry['path']}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}-{entry['category']}-{digest}{Path(entry['path']).suffix}"


def write_links(entries, dest):
    links_dir = dest / "links"
    links_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for index, entry in enumerate([e for e in entries if e["included"]], start=1):
        link = links_dir / safe_link_name(index, entry)
        target = Path(entry["path"])
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(target)
            entry["link"] = str(link)
            created += 1
        except OSError as exc:
            entry["link_error"] = str(exc)
    return created


def summarize(entries, roots, link_count):
    by_category = {}
    for entry in entries:
        data = by_category.setdefault(entry["category"], {"files": 0, "bytes": 0, "included": 0})
        data["files"] += 1
        data["bytes"] += entry["size_bytes"]
        if entry["included"]:
            data["included"] += 1
    return {
        "roots": [str(r) for r in roots],
        "root_count": len(roots),
        "file_count": len(entries),
        "included_count": sum(1 for e in entries if e["included"]),
        "excluded_count": sum(1 for e in entries if not e["included"]),
        "link_count": link_count,
        "bytes_total": sum(e["size_bytes"] for e in entries),
        "bytes_included": sum(e["size_bytes"] for e in entries if e["included"]),
        "by_category": by_category,
    }


def write_readme(dest, summary):
    lines = [
        "# Historico local do Claude Code",
        "",
        "Gerado por `python3 tool_harness/importar_historico_claude.py`.",
        "",
        "Esta pasta e local e ignorada pelo Git. Ela centraliza inventario e links",
        "para sessoes/memorias do Claude Code, sem copiar credenciais, backups,",
        "cache, plugins, paste-cache ou file-history no primeiro passe.",
        "",
        "Regra: isto nao e dataset pronto. O proximo passo e extrair padroes de",
        "trabalho e exemplos pequenos, passar por filtro de segredo e juiz, e so",
        "entao gerar JSONL curado para LoRA/adapters.",
        "",
        "## Resumo",
        "",
        f"- raizes `.claude`: {summary['root_count']}",
        f"- arquivos inventariados: {summary['file_count']}",
        f"- arquivos incluidos por link: {summary['included_count']}",
        f"- arquivos excluidos: {summary['excluded_count']}",
        f"- bytes incluidos: {summary['bytes_included']}",
        "",
        "## Categorias",
        "",
    ]
    for category, data in sorted(summary["by_category"].items()):
        lines.append(
            f"- `{category}`: {data['files']} arquivos, {data['included']} incluidos, {data['bytes']} bytes"
        )
    lines.append("")
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    dest = args.dest.resolve()
    roots = [Path(p).expanduser().resolve() for p in args.root] if args.root else discover_roots()
    entries = []
    for root in roots:
        if not root.exists():
            continue
        for path in iter_files(root):
            try:
                entries.append(file_entry(path, root))
            except OSError:
                continue
    entries.sort(key=lambda e: (e["root"], e["category"], e["relpath"]))

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
    link_count = 0
    if args.links and not args.dry_run:
        link_count = write_links(entries, dest)
    summary = summarize(entries, roots, link_count)

    if not args.dry_run:
        (dest / "manifest.json").write_text(
            json.dumps({"summary": summary, "entries": entries}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (dest / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_readme(dest, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--root", action="append", help="raiz .claude especifica; pode repetir")
    parser.add_argument("--no-links", dest="links", action="store_false", help="nao criar symlinks")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(links=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
