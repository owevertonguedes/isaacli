"""Juiz mecanico para um jogo de navegador de arquivo unico.

Nao pergunta nada pro modelo. Abre o arquivo, checa estrutura e roda o JS pelo
parser do Node. Devolve (ok, [problemas]) — a lista e o material de aula do professor.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_BYTES = 400


def verificar(caminho):
    p = Path(caminho)
    probs = []
    if not p.is_file():
        return False, [f"o arquivo {p.name} nao existe"]

    html = p.read_text(errors="replace")

    if len(html) < MIN_BYTES:
        probs.append(f"arquivo curto demais ({len(html)} bytes) — jogo incompleto")

    if "\\n" in html and "\n" not in html:
        probs.append("o arquivo tem barra-n literal em vez de quebras de linha reais")

    baixo = html.lower()
    for tag in ("<html", "<head", "<body"):
        if tag not in baixo:
            probs.append(f"falta a tag {tag}>")

    # Placeholder no lugar de codigo = codigo faltando. Mesma lista do juiz.
    for marcador in ("javascript aqui", "css aqui", "seu codigo aqui", "todo", "codigo aqui"):
        if marcador in baixo:
            probs.append(f'sobrou o placeholder "{marcador}" no lugar de codigo real')

    if "<script" not in baixo:
        probs.append("nao ha bloco <script> — jogo sem logica nao e jogavel")

    if "<canvas" not in baixo and not re.search(r"addeventlistener|onclick|onkey", baixo):
        probs.append("nao ha <canvas> nem nenhum listener de evento — nada responde ao jogador")

    # Autocontido: nada de CDN/rede. Regra do projeto.
    externos = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html, re.I)
    if externos:
        probs.append(f"usa recurso externo (o jogo deve ser autocontido): {externos[:3]}")

    # Sintaxe do JS: extrai os <script> inline e passa pelo parser do Node.
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S | re.I)
    codigo = "\n;\n".join(s for s in scripts if s.strip())
    if codigo.strip():
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(codigo)
            tmp = f.name
        r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
        Path(tmp).unlink(missing_ok=True)
        if r.returncode != 0:
            erro = (r.stderr or "").strip().splitlines()
            linha = next((l for l in erro if "Error" in l or "^" not in l), erro[0] if erro else "?")
            probs.append(f"o JavaScript nao compila: {linha.strip()[:160]}")

    return (not probs), probs


if __name__ == "__main__":
    ok, probs = verificar(sys.argv[1])
    print("PASSOU" if ok else "FALHOU")
    for x in probs:
        print(" -", x)
