#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path

from importar_historico_claude import run


class Args:
    def __init__(self, root, dest):
        self.root = [str(root)]
        self.dest = dest
        self.links = True
        self.dry_run = False


def test_importa_apenas_fontes_curaveis():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        claude = base / ".claude"
        session_dir = claude / "projects" / "-tmp-projeto"
        memory_dir = session_dir / "memory"
        file_history = claude / "file-history" / "abc"
        session_dir.mkdir(parents=True)
        memory_dir.mkdir()
        file_history.mkdir(parents=True)

        (claude / "history.jsonl").write_text('{"type":"user","text":"oi"}\n')
        (claude / ".credentials.json").write_text('{"token":"secret"}\n')
        (session_dir / "sessao.jsonl").write_text('{"type":"assistant","text":"ok"}\n')
        (memory_dir / "fluxo.md").write_text("# fluxo\n")
        (file_history / "arquivo.txt").write_text("conteudo sensivel\n")

        dest = base / "saida"
        run(Args(claude, dest))

        manifest = json.loads((dest / "manifest.json").read_text())
        included = [e for e in manifest["entries"] if e["included"]]
        excluded = [e for e in manifest["entries"] if not e["included"]]

        assert {e["category"] for e in included} == {
            "history_jsonl",
            "memory_md",
            "session_jsonl",
        }
        assert any(e["category"] == "segredo_ou_config_excluido" for e in excluded)
        assert any(e["category"] == "file_history_excluido" for e in excluded)
        assert len(list((dest / "links").iterdir())) == 3


if __name__ == "__main__":
    test_importa_apenas_fontes_curaveis()
    print("ok")
