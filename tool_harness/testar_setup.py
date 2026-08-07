#!/usr/bin/env python3
"""Testes do setup guiado sem rede, downloads ou Ollama real."""
import io
import json
import sys
import tempfile
import stat
from contextlib import redirect_stdout
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

import config
import setup_ollama


falhas = []


def checar(condicao, descricao):
    print(f"[{'ok    ' if condicao else 'FALHOU'}] {descricao}")
    if not condicao:
        falhas.append(descricao)


class ClienteFake:
    def __init__(self, instalados, infos):
        self.instalados = instalados
        self.infos = infos

    def modelos(self):
        return [{"name": nome} for nome in self.instalados]

    def show(self, modelo):
        return self.infos[modelo]


def entrada(*respostas):
    itens = iter(respostas)
    return lambda _prompt="": next(itens)


original_which = setup_ollama.shutil.which
original_cliente = setup_ollama.OllamaLocal
original_servidor = setup_ollama._garantir_servidor
original_criar = setup_ollama._criar_modelo
original_baixar = setup_ollama._baixar_modelo
original_validar_api = setup_ollama._validar_api

try:
    raiz = Path(tempfile.mkdtemp())
    arquivo_config = raiz / "config.json"
    criados = []
    downloads = []
    qwen36 = "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ1_M"
    infos = {
        qwen36: {
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {"qwen3.context_length": 262144},
        },
        "gpt-oss:20b": {
            "capabilities": ["completion", "tools", "thinking"],
            "model_info": {
                "gptoss.context_length": 131072,
                "gptoss.rope.scaling.original_context_length": 4096,
            },
        },
    }
    cliente = ClienteFake(
        [qwen36, "gpt-oss:20b", "modelo-teste:7b"], infos,
    )
    setup_ollama.shutil.which = lambda _nome: "/usr/bin/ollama"
    setup_ollama.OllamaLocal = lambda: cliente
    setup_ollama._garantir_servidor = lambda _cliente, _exe, _tr=None: ("teste", None)
    setup_ollama._criar_modelo = (
        lambda exe, base, nome, ctx, temp: criados.append((exe, base, nome, ctx, temp))
    )
    setup_ollama._baixar_modelo = lambda exe, nome, tr=None: downloads.append((exe, nome))

    out = io.StringIO()
    with redirect_stdout(out):
        codigo = setup_ollama.executar_setup(
            entrada("1", "1", "1", "5", "12K", "1"), config_file=arquivo_config,
        )
    dado = json.loads(arquivo_config.read_text())
    perfil_qwen = dado["profiles"][dado["default_profile"]]
    checar(codigo == 0, "setup Qwen3.6 recomendado conclui")
    checar(perfil_qwen["num_ctx"] == 12288, "modo manual aceita contexto amigavel 12K")
    checar(perfil_qwen["thinking"] == "low", "thinking e detectado do manifesto, não do catálogo")
    checar(dado["language"] == "pt-BR", "setup salva idioma da interface")
    checar(criados[-1][3:] == (12288, 0),
           "perfil derivado usa o contexto selecionado")
    checar(not downloads, "modelo instalado nao e baixado novamente")
    menu_recomendado = setup_ollama._catalogo_recomendado()
    menu_local = setup_ollama._modelos_instalados(cliente.instalados)
    checar([item["base_model"] for item in menu_recomendado] == setup_ollama.RECOMENDADOS,
           "seção de recomendações preserva a curadoria e sua ordem")
    checar(any(item["base_model"] == "modelo-teste:7b" for item in menu_local),
           "seção de instalados inclui modelos consultados ao vivo no Ollama")
    checar("Recomendações do Isaac" in out.getvalue()
           and "Instalados no Ollama" in out.getvalue()
           and "modelo-teste:7b" in out.getvalue(),
           "menu exibe recomendações e todos os instalados na mesma tela")

    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(
            entrada("1", "1", "5", "3", "3"), config_file=arquivo_config,
        )
    dado = config.carregar(arquivo_config)
    perfil_gpt = dado["profiles"][dado["default_profile"]]
    checar(codigo == 0, "setup GPT-OSS conclui")
    checar(perfil_gpt["num_ctx"] == 32768, "preset longo GPT-OSS e 32K")
    checar(perfil_gpt["thinking"] == "high", "GPT-OSS salva thinking high separado")
    checar(perfil_gpt["temperature"] == 0,
           "setup não injeta ajuste específico hardcoded para GPT-OSS")
    checar(len(dado["profiles"]) == 2, "novo perfil preserva perfil anterior")

    antes_falha = arquivo_config.read_text()
    info_qwen_original = cliente.infos[qwen36]
    cliente.infos[qwen36] = {
        "capabilities": ["completion"],
        "model_info": {"qwen3.context_length": 262144},
    }
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(entrada("1", "1", "1"), config_file=arquivo_config)
    checar(codigo == 1 and arquivo_config.read_text() == antes_falha,
           "modelo sem tools e recusado sem alterar perfil anterior")
    cliente.infos[qwen36] = info_qwen_original

    criar_fake = setup_ollama._criar_modelo
    setup_ollama._criar_modelo = lambda *_args: (_ for _ in ()).throw(RuntimeError("falha criada"))
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(entrada("1", "1", "1", "1", "1"), config_file=arquivo_config)
    setup_ollama._criar_modelo = criar_fake
    checar(codigo == 1 and arquivo_config.read_text() == antes_falha,
           "falha ao criar modelo nao troca perfil padrao")

    setup_ollama.shutil.which = lambda _nome: None
    ausente = raiz / "ausente.json"
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(entrada("1", "1"), config_file=ausente)
    checar(codigo == 2 and not ausente.exists(), "Ollama ausente orienta e nao grava config parcial")

    setup_ollama._validar_api = lambda url, key, model: None
    api_config = raiz / "api-config.json"
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(
            entrada("1", "2", "Groq", "https://api.groq.com/openai/v1",
                    "openai/gpt-oss-20b", "segredo-teste", "3"),
            config_file=api_config,
        )
    dado_api = config.carregar(api_config)
    _, perfil_api = config.perfil(dado_api)
    segredo_api = config.carregar_segredo(
        perfil_api["credential"], api_config.with_name("secrets.json"))
    checar(codigo == 0 and perfil_api["provider"] == "openai_compatible",
           "setup cria perfil de API compatível sem provedor hardcoded")
    checar(perfil_api["base_url"] == "https://api.groq.com/openai/v1"
           and perfil_api["model"] == "openai/gpt-oss-20b",
           "endpoint e modelo da API são dados configuráveis")
    checar(segredo_api == "segredo-teste" and "segredo-teste" not in api_config.read_text(),
           "API key fica fora do config.json")
    checar(stat.S_IMODE(api_config.with_name("secrets.json").stat().st_mode) == 0o600,
           "arquivo de segredos usa permissão 0600")

    tentativas = []
    def validar_na_segunda(url, key, model):
        tentativas.append((url, key, model))
        if len(tentativas) == 1:
            raise RuntimeError("HTTP 401 — chave inválida")
    setup_ollama._validar_api = validar_na_segunda
    api_retry_config = raiz / "api-retry-config.json"
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(
            entrada(
                "1", "2", "Servidor", "https://api.teste/v1/chat/completions",
                "modelo-teste", "chave-errada", "1",
                "Servidor", "https://api.teste/v1", "modelo-teste", "chave-certa", "1",
            ),
            config_file=api_retry_config,
        )
    _, perfil_retry = config.perfil(config.carregar(api_retry_config))
    checar(codigo == 0 and len(tentativas) == 2,
           "falha de validação permite corrigir os dados sem reiniciar o setup")
    checar(perfil_retry["base_url"] == "https://api.teste/v1",
           "endpoint completo é normalizado antes de salvar")
    setup_ollama._validar_api = lambda url, key, model: None

    setup_ollama.shutil.which = lambda _nome: "/usr/bin/ollama"
    voltar_config = raiz / "voltar-config.json"
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(
            entrada("1", "1", "9", "2", "Servidor", "https://api.teste/v1",
                    "modelo-teste", "chave", "1"),
            config_file=voltar_config,
        )
    _, perfil_voltar = config.perfil(config.carregar(voltar_config))
    checar(codigo == 0 and perfil_voltar["provider"] == "openai_compatible",
           "voltar no menu de modelos retorna ao motor sem repetir idioma")
    checar(setup_ollama.RECOMENDADOS[0].endswith(":UD-IQ1_M"),
           "Qwen3.6-35B-A3B UD-IQ1_M é a primeira recomendação")
    checar("qwen3:4b-instruct-2507-q4_K_M" not in setup_ollama.RECOMENDADOS,
           "Qwen de teste antigo não aparece na curadoria")
    checar(
        setup_ollama._normalizar_api_url(
            "https://api.groq.com/openai/v1/chat/completions/"
        ) == "https://api.groq.com/openai/v1",
        "setup corrige endpoint colado com /chat/completions",
    )

    checar(
        setup_ollama.contexto_maximo(infos["gpt-oss:20b"]) == 131072,
        "detecta contexto nominal e ignora original_context_length",
    )
    checar(setup_ollama.ler_contexto("16K") == 16384, "entrada humana 16K vira tokens")
    respostas_contexto = entrada("5", "4K", "12K")
    with redirect_stdout(io.StringIO()):
        contexto = setup_ollama._escolher_contexto(262144, respostas_contexto,
                                                   setup_ollama.Translator("pt-BR"))
    checar(contexto == 12288, "contexto manual rejeita 4K e aceita 12K")
    with redirect_stdout(io.StringIO()):
        voltar = setup_ollama._escolher_contexto(
            262144, entrada("6"), setup_ollama.Translator("pt-BR"),
        )
        voltar_thinking = setup_ollama._escolher_thinking(
            dict(setup_ollama._item_modelo("teste"), thinking_kind="levels"),
            entrada("4"), setup_ollama.Translator("pt-BR"),
        )
    checar(voltar is None and voltar_thinking == "__context__",
           "menus de contexto e raciocinio permitem voltar")

    def interromper(_prompt=""):
        raise KeyboardInterrupt
    setup_ollama.shutil.which = lambda _nome: "/usr/bin/ollama"
    with redirect_stdout(io.StringIO()):
        codigo = setup_ollama.executar_setup(interromper, config_file=arquivo_config)
    checar(codigo == 130, "Ctrl+C cancela o setup sem traceback")
finally:
    setup_ollama.shutil.which = original_which
    setup_ollama.OllamaLocal = original_cliente
    setup_ollama._garantir_servidor = original_servidor
    setup_ollama._criar_modelo = original_criar
    setup_ollama._baixar_modelo = original_baixar
    setup_ollama._validar_api = original_validar_api


print()
if falhas:
    print(f"{len(falhas)} FALHA(S):")
    for falha in falhas:
        print(f"  - {falha}")
    raise SystemExit(1)
print("SETUP DO ISAAC OK — perfis, contexto e raciocinio separados")
