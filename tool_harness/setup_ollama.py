"""Configuração guiada de modelos Ollama locais."""
import json
import getpass
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import config
import terminal_ui
from i18n import SUPPORTED_LANGUAGES, Translator


MODEL_CATALOG_PATH = Path(__file__).resolve().parent / "model_catalog.json"


def _carregar_recomendados(path=MODEL_CATALOG_PATH):
    """Carrega atalhos de download; o catálogo não escolhe o modelo em uso."""
    try:
        dado = json.loads(Path(path).read_text(encoding="utf-8"))
        modelos = dado["recommended"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(f"catálogo de modelos inválido em {path}: {e}") from e
    if not isinstance(modelos, list) or not modelos or not all(
            isinstance(item, str) and item.strip() for item in modelos):
        raise RuntimeError(f"catálogo de modelos inválido em {path}: expected string list")
    return modelos


RECOMENDADOS = _carregar_recomendados()

NIVEIS_CONTEXTO = [
    ("context.compact", 8192),
    ("context.standard", 16384),
    ("context.long", 32768),
    ("context.extended", 65536),
]
CONTEXTO_MINIMO = 8192


def _base_url():
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host.rstrip("/")


class OllamaLocal:
    def __init__(self, base_url=None):
        self.base_url = (base_url or _base_url()).rstrip("/")

    def _request(self, metodo, path, payload=None, timeout=10):
        dados = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=dados, method=metodo,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resposta:
            return json.load(resposta)

    def version(self):
        return self._request("GET", "/api/version", timeout=2).get("version")

    def modelos(self):
        return self._request("GET", "/api/tags").get("models", [])

    def show(self, modelo):
        return self._request("POST", "/api/show", {"model": modelo})


def contexto_maximo(show):
    candidatos = []
    for chave, valor in (show.get("model_info") or {}).items():
        if chave.endswith(".context_length") and "original_context" not in chave:
            try:
                candidatos.append(int(valor))
            except (TypeError, ValueError):
                pass
    return max(candidatos) if candidatos else None


def formatar_contexto(valor):
    if valor % 1024 == 0:
        return f"{valor // 1024}K"
    return f"{valor:,}".replace(",", ".")


def ler_contexto(texto):
    valor = texto.strip().lower().replace(".", "")
    multiplicador = 1024 if valor.endswith("k") else 1
    if multiplicador != 1:
        valor = valor[:-1]
    try:
        return int(valor) * multiplicador
    except ValueError:
        return 0


def _titulo(tr, titulo, explicacao=None):
    partes = [titulo]
    if explicacao:
        partes.extend(("", explicacao))
    return "\n".join(partes)


def _selecionar(tr, titulo, opcoes, input_fn=input, explicacao=None, inicial=0,
                desabilitados=None):
    return terminal_ui.selecionar(
        _titulo(tr, titulo, explicacao), opcoes, input_fn=input_fn,
        prompt=tr.t("select.prompt"), invalido=tr.t("select.invalid"), inicial=inicial,
        desabilitados=desabilitados,
    )


def _escolher_idioma(input_fn):
    idiomas = list(SUPPORTED_LANGUAGES)
    indice = _selecionar(
        Translator("pt-BR"), "Isaac CLI · Language / Idioma",
        [SUPPORTED_LANGUAGES[codigo] for codigo in idiomas], input_fn,
        "Use ↑/↓ e Enter · Use ↑/↓ and Enter",
    )
    return idiomas[indice]


def _instrucoes_instalar_ollama(tr):
    terminal_ui.limpar()
    print(tr.t("ollama.missing.title"))
    print(tr.t("ollama.missing.explain"))
    chave = {"linux": "ollama.install.linux", "darwin": "ollama.install.macos",
             "windows": "ollama.install.windows"}.get(platform.system().lower(),
                                                       "ollama.install.other")
    print(tr.t(chave))
    print(tr.t("ollama.install.retry"))


def _garantir_servidor(cliente, ollama_exe, tr=None):
    tr = tr or Translator()
    try:
        return cliente.version(), None
    except Exception:
        pass
    proc = subprocess.Popen([ollama_exe, "serve"], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(0.25)
        try:
            return cliente.version(), proc
        except Exception:
            if proc.poll() is not None:
                break
    if proc.poll() is None:
        proc.terminate()
    raise RuntimeError(tr.t("ollama.server.failed"))


def _baixar_modelo(ollama_exe, modelo, tr=None):
    tr = tr or Translator()
    print(tr.t("model.download.running", model=modelo))
    resultado = subprocess.run([ollama_exe, "pull", modelo], check=False)
    if resultado.returncode != 0:
        raise RuntimeError(tr.t("model.download.failed", code=resultado.returncode))


def _item_modelo(modelo, recomendado=False):
    normalizado = modelo.removesuffix(":latest")
    slug = re.sub(r"[^a-z0-9]+", "-", normalizado.lower()).strip("-") or "local"
    return {
        "id": slug, "name": modelo, "base_model": modelo,
        "temperature": 0, "thinking_kind": "detect", "recommended": recomendado,
    }


def _nome_derivado(catalogo, num_ctx):
    sufixo = f"{num_ctx // 1024}k" if num_ctx % 1024 == 0 else str(num_ctx)
    return f"isaac-{catalogo['id']}-{sufixo}"


def _catalogo_recomendado():
    return [_item_modelo(modelo, recomendado=True) for modelo in RECOMENDADOS]


def _modelos_instalados(instalados):
    return [_item_modelo(modelo) for modelo in sorted(instalados, key=str.casefold)]


def _esta_instalado(modelo, instalados):
    procurado = modelo.removesuffix(":latest").casefold()
    return any(item.removesuffix(":latest").casefold() == procurado for item in instalados)


def _rotulo_modelo(item, instalados, tr):
    base = item["base_model"]
    estado = ("model.installed" if _esta_instalado(base, instalados)
              else "model.not_installed")
    return f"{base} [{tr.t(estado)}]"


def _criar_modelo(ollama_exe, base_model, model_name, num_ctx, temperature):
    conteudo = (f"FROM {base_model}\n\nPARAMETER temperature {temperature}\n"
                f"PARAMETER num_ctx {num_ctx}\n")
    fd, caminho = tempfile.mkstemp(prefix="isaac-model-", suffix=".Modelfile")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
        resultado = subprocess.run([ollama_exe, "create", model_name, "-f", caminho],
                                   check=False, text=True, capture_output=True)
        if resultado.returncode != 0:
            detalhe = (resultado.stderr or resultado.stdout).strip()
            raise RuntimeError(f"ollama create: {detalhe or resultado.returncode}")
    finally:
        Path(caminho).unlink(missing_ok=True)


def _escolher_contexto(limite, input_fn, tr):
    niveis = [nivel for nivel in NIVEIS_CONTEXTO if not limite or nivel[1] <= limite]
    opcoes = ([tr.t(chave) for chave, _ in niveis]
              + [tr.t("context.manual"), tr.t("navigation.back")])
    explicacao = tr.t("context.explain")
    if limite:
        explicacao += "\n" + tr.t("context.limit", limit=formatar_contexto(limite))
    indice = _selecionar(tr, tr.t("context.title"), opcoes, input_fn, explicacao)
    if indice == len(niveis) + 1:
        return None
    if indice < len(niveis):
        return niveis[indice][1]
    while True:
        print(_titulo(tr, tr.t("context.title"), explicacao))
        valor = ler_contexto(input_fn(tr.t("context.manual.prompt")))
        if valor >= CONTEXTO_MINIMO and (not limite or valor <= limite):
            return valor
        teto = formatar_contexto(limite) if limite else "∞"
        print(tr.t("context.manual.invalid", limit=teto))


def _escolher_thinking(item, input_fn, tr):
    if item["thinking_kind"] == "none":
        return False
    indice = _selecionar(
        tr, tr.t("thinking.gpt.title"),
        [tr.t("thinking.low"), tr.t("thinking.medium"), tr.t("thinking.high"),
         tr.t("navigation.back")],
        input_fn, tr.t("thinking.gpt.explain"), inicial=1,
    )
    return ["low", "medium", "high", "__context__"][indice]


def _normalizar_api_url(base_url):
    url = base_url.strip().rstrip("/")
    for sufixo in ("/chat/completions", "/models"):
        if url.endswith(sufixo):
            url = url[:-len(sufixo)]
    partes = urllib.parse.urlparse(url)
    if partes.scheme not in ("http", "https") or not partes.netloc:
        raise RuntimeError("o endpoint precisa ser uma URL http:// ou https:// válida")
    return url


def _mensagem_http_api(erro):
    detalhe = ""
    try:
        corpo = erro.read().decode("utf-8", errors="replace")
        dado = json.loads(corpo)
        item = dado.get("error", dado)
        detalhe = item.get("message", "") if isinstance(item, dict) else str(item)
    except (OSError, ValueError, AttributeError):
        pass
    detalhe = re.sub(r"(?i)(api[_ -]?key\s*[=:]?\s*)\S+", r"\1[oculta]", detalhe)
    return f"HTTP {erro.code}" + (f" — {detalhe}" if detalhe else "")


def _validar_api(base_url, api_key, model):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resposta:
            payload = json.load(resposta)
    except urllib.error.HTTPError as e:
        raise RuntimeError(_mensagem_http_api(e)) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"não foi possível conectar ao endpoint — {e.reason}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError("o endpoint /models não retornou JSON válido") from e
    modelos = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if model not in modelos:
        proximos = sorted(m for m in modelos if model.lower() in m.lower() or
                          m.lower() in model.lower())[:5]
        sugestao = f"; opções parecidas: {', '.join(proximos)}" if proximos else ""
        raise RuntimeError(f"o modelo “{model}” não está disponível para esta chave{sugestao}")


def _setup_api(idioma, input_fn, config_file, tr):
    erro_campos = None
    while True:
        terminal_ui.limpar()
        explicacao = tr.t("api.explain")
        if erro_campos:
            explicacao += "\n\n" + erro_campos
        print(_titulo(tr, tr.t("api.title"), explicacao))
        erro_campos = None
        nome = input_fn(tr.t("api.name.prompt")).strip()
        base_url = input_fn(tr.t("api.url.prompt")).strip()
        modelo = input_fn(tr.t("api.model.prompt")).strip()
        if not nome or not base_url or not modelo:
            erro_campos = tr.t("api.fields.missing")
            continue
        try:
            base_url = _normalizar_api_url(base_url)
        except RuntimeError as e:
            erro_campos = tr.t("api.validation.failed", error=e)
            continue
        chave = (getpass.getpass(tr.t("api.key.prompt")) if input_fn is input
                 else input_fn(tr.t("api.key.prompt"))).strip()
        if not chave:
            erro_campos = tr.t("api.key.missing")
            continue
        print(tr.t("api.validating"))
        try:
            _validar_api(base_url, chave, modelo)
        except RuntimeError as e:
            erro_validacao = tr.t("api.validation.failed", error=e)
            acao = _selecionar(
                tr, tr.t("api.retry.title"),
                [tr.t("api.retry.yes"), tr.t("api.save.unverified"),
                 tr.t("navigation.back")], input_fn, erro_validacao,
            )
            if acao == 0:
                continue
            if acao == 2:
                return "__engine__"
        break
    indice = _selecionar(
        tr, tr.t("thinking.api.title"),
        [tr.t("thinking.disabled"), tr.t("thinking.low"),
         tr.t("thinking.medium"), tr.t("thinking.high")], input_fn,
        tr.t("thinking.api.explain"), inicial=2,
    )
    thinking = [None, "low", "medium", "high"][indice]
    slug = re.sub(r"[^a-z0-9]+", "-", nome.lower()).strip("-") or "api"
    perfil_nome = f"{slug}-{re.sub(r'[^a-z0-9]+', '-', modelo.lower()).strip('-')}"
    credencial = f"api:{perfil_nome}"
    segredo_path = (Path(config_file).with_name("secrets.json")
                    if config_file else None)
    config.salvar_segredo(credencial, chave, segredo_path)
    dado = config.carregar(config_file)
    dado["language"] = idioma
    dado["profiles"][perfil_nome] = {
        "provider": "openai_compatible", "provider_name": nome,
        "base_url": base_url, "model": modelo, "thinking": thinking,
        "credential": credencial, "temperature": 0,
    }
    dado["default_profile"] = perfil_nome
    config.salvar(dado, config_file)
    return 0


def _executar_setup(input_fn=input, config_file=None, idioma_inicial=None):
    if idioma_inicial:
        idioma = idioma_inicial
    else:
        try:
            idioma = _escolher_idioma(input_fn)
        except KeyboardInterrupt:
            print("\n" + Translator("pt-BR").t("setup.cancelled"))
            return 130
    tr = Translator(idioma)
    print(tr.t("setup.title"), "\n")

    try:
        motor = _selecionar(
            tr, tr.t("engine.title"),
            [tr.t("engine.ollama"), tr.t("engine.api")], input_fn,
            tr.t("engine.explain"),
        )
        if motor == 1:
            resultado_api = _setup_api(idioma, input_fn, config_file, tr)
            if resultado_api == "__engine__":
                return _executar_setup(input_fn, config_file, idioma_inicial=idioma)
            return resultado_api
    except (RuntimeError, ValueError, urllib.error.URLError) as e:
        print(tr.t("setup.error", error=e))
        return 1
    except KeyboardInterrupt:
        print("\n" + tr.t("setup.cancelled"))
        return 130

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        _instrucoes_instalar_ollama(tr)
        return 2

    cliente = OllamaLocal()
    iniciado = None
    try:
        versao, iniciado = _garantir_servidor(cliente, ollama_exe, tr)
        instalados = {m.get("name", "").removesuffix(":latest") for m in cliente.modelos()}
        while True:
            recomendados = _catalogo_recomendado()
            locais = _modelos_instalados(instalados)
            entradas = [None, *recomendados, None, *locais, "__back__"]
            opcoes = [
                tr.t("model.section.recommended"),
                *[_rotulo_modelo(item, instalados, tr) for item in recomendados],
                tr.t("model.section.installed", count=len(locais)),
                *[item["base_model"] for item in locais],
                tr.t("navigation.back"),
            ]
            cabecalhos = {0, len(recomendados) + 1}
            indice_modelo = _selecionar(
                tr, tr.t("model.title"), opcoes, input_fn,
                terminal_ui.sutil(
                    tr.t("model.recommended.explain", version=versao), input_fn,
                ),
                desabilitados=cabecalhos,
            )
            escolhido = entradas[indice_modelo]
            if escolhido == "__back__":
                return _executar_setup(input_fn, config_file, idioma_inicial=idioma)
            base = escolhido["base_model"]
            if not _esta_instalado(base, instalados):
                indice = _selecionar(
                    tr, tr.t("model.download.confirm", model=base),
                    [tr.t("model.download.yes"), tr.t("navigation.back")], input_fn,
                )
                if indice:
                    continue
                _baixar_modelo(ollama_exe, base, tr)
                instalados.add(base.removesuffix(":latest"))

            info = cliente.show(base)
            if "tools" not in set(info.get("capabilities") or []):
                print(tr.t("model.tools.missing", model=base))
                return 1
            if escolhido.get("thinking_kind") == "detect":
                escolhido["thinking_kind"] = (
                    "levels" if "thinking" in set(info.get("capabilities") or []) else "none"
                )
            limite = contexto_maximo(info)
            voltar_modelo = False
            while True:
                num_ctx = _escolher_contexto(limite, input_fn, tr)
                if num_ctx is None:
                    voltar_modelo = True
                    break
                thinking = _escolher_thinking(escolhido, input_fn, tr)
                if thinking == "__context__":
                    continue
                if thinking == "__model__":
                    voltar_modelo = True
                    break
                break
            if not voltar_modelo:
                break

        nome_modelo = _nome_derivado(escolhido, num_ctx)
        print(tr.t("model.creating", model=nome_modelo))
        _criar_modelo(ollama_exe, base, nome_modelo, num_ctx, escolhido["temperature"])

        dado = config.carregar(config_file)
        dado["language"] = idioma
        perfil_nome = nome_modelo.removeprefix("isaac-")
        dado["profiles"][perfil_nome] = {
            "provider": "ollama", "model": nome_modelo, "base_model": base,
            "num_ctx": num_ctx, "context_limit": limite, "thinking": thinking,
            "temperature": escolhido["temperature"],
        }
        dado["default_profile"] = perfil_nome
        config.salvar(dado, config_file)
        return 0
    except (RuntimeError, ValueError, urllib.error.URLError) as e:
        print(tr.t("setup.error", error=e))
        return 1
    except KeyboardInterrupt:
        print("\n" + tr.t("setup.cancelled"))
        return 130
    finally:
        if iniciado and iniciado.poll() is None:
            iniciado.terminate()
            try:
                iniciado.wait(timeout=3)
            except subprocess.TimeoutExpired:
                iniciado.kill()
                iniciado.wait(timeout=3)


def executar_setup(input_fn=input, config_file=None):
    # Um único buffer alternativo evita que o terminal principal apareça entre
    # as etapas. Sem Ollama, as instruções precisam permanecer visíveis.
    if not shutil.which("ollama") or not terminal_ui.interativo(input_fn):
        return _executar_setup(input_fn, config_file)
    with terminal_ui.tela_alternativa(input_fn):
        codigo = _executar_setup(input_fn, config_file)
    if codigo == 130:
        try:
            idioma = config.carregar(config_file).get("language", "pt-BR")
        except ValueError:
            idioma = "pt-BR"
        print(Translator(idioma).t("setup.cancelled"))
    elif codigo not in (0,):
        print("Setup não concluído. Execute /setup para tentar novamente.")
    return codigo
