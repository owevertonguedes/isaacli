"""Configuracao local do isaacli, fora da workspace e sem segredos."""
import json
import os
import tempfile
from pathlib import Path


CONFIG_VERSION = 1


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    raiz = Path(base).expanduser() if base else Path.home() / ".config"
    return raiz / "isaacli" / "config.json"


def secrets_path() -> Path:
    return config_path().with_name("secrets.json")


def salvar_segredo(nome, valor, path=None):
    alvo = Path(path) if path else secrets_path()
    dados = {}
    if alvo.exists():
        try:
            dados = json.loads(alvo.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dados = {}
    dados[nome] = valor
    salvar(dados, alvo)


def carregar_segredo(nome, path=None):
    alvo = Path(path) if path else secrets_path()
    if not alvo.exists():
        return None
    try:
        return json.loads(alvo.read_text(encoding="utf-8")).get(nome)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def config_vazia():
    return {
        "version": CONFIG_VERSION,
        "language": None,
        "default_profile": None,
        "profiles": {},
        "permissions": {"global": [], "workspaces": {}},
    }


def carregar(path=None):
    alvo = Path(path) if path else config_path()
    if not alvo.exists():
        return config_vazia()
    try:
        dado = json.loads(alvo.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"configuracao invalida em {alvo}: {e}") from e
    if not isinstance(dado, dict) or not isinstance(dado.get("profiles", {}), dict):
        raise ValueError(f"configuracao invalida em {alvo}: formato inesperado")
    dado.setdefault("version", CONFIG_VERSION)
    dado.setdefault("language", None)
    dado.setdefault("default_profile", None)
    dado.setdefault("profiles", {})
    permissoes = dado.setdefault("permissions", {})
    if not isinstance(permissoes, dict):
        permissoes = dado["permissions"] = {}
    permissoes.setdefault("global", [])
    permissoes.setdefault("workspaces", {})
    return dado


def regras_permissao(dado, workspace):
    permissoes = dado.get("permissions") or {}
    globais = set(permissoes.get("global") or [])
    locais = set((permissoes.get("workspaces") or {}).get(str(Path(workspace).resolve()), []))
    return globais | locais


def adicionar_permissao(dado, regra, workspace=None):
    permissoes = dado.setdefault("permissions", {"global": [], "workspaces": {}})
    if workspace is None:
        destino = permissoes.setdefault("global", [])
    else:
        chave = str(Path(workspace).resolve())
        destino = permissoes.setdefault("workspaces", {}).setdefault(chave, [])
    if regra not in destino:
        destino.append(regra)


def salvar(dado, path=None):
    alvo = Path(path) if path else config_path()
    alvo.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dado, ensure_ascii=False, indent=2) + "\n"
    fd, temporario = tempfile.mkstemp(prefix="config-", suffix=".tmp", dir=alvo.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as arquivo:
            arquivo.write(payload)
        os.replace(temporario, alvo)
    except Exception:
        try:
            os.unlink(temporario)
        except OSError:
            pass
        raise
    return alvo


def perfil(dado, nome=None):
    escolhido = nome or dado.get("default_profile")
    item = (dado.get("profiles") or {}).get(escolhido)
    return escolhido, item


def modelo_padrao(fallback=None, path=None):
    try:
        dado = carregar(path)
    except ValueError:
        return fallback
    _nome, item = perfil(dado)
    return item.get("model", fallback) if item else fallback


def pensar_do_modelo(dado, modelo):
    for item in (dado.get("profiles") or {}).values():
        if item.get("model") == modelo:
            return item.get("thinking")
    return None


def perfil_do_modelo(dado, modelo):
    for nome, item in (dado.get("profiles") or {}).items():
        if item.get("model") == modelo:
            return nome, item
    return None, None
