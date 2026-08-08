#!/usr/bin/env python3
"""Teste automatizado para o histórico de conversas da Oficina.

Este teste roda de forma 100% autônoma e sem necessidade de GUI (headless).
Ele valida:
  1. Geração de ID curto e ditável.
  2. Inicialização e gravação atômica da conversa em disco.
  3. Gravação incremental de trocas e chamadas de ferramenta.
  4. Sobrevivência a crash (leitura de arquivo persistido).
"""
import json
import os
import sys
import time
from pathlib import Path

# Adiciona o diretório do tool_harness ao sys.path para importações
AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))

from app import gerar_id_conversa, salvar_conversa_atomica, DIR_CONVERSAS


def testar_fluxo_historico():
    print("=== INICIANDO TESTE DE HISTÓRICO DE CONVERSAS ===")
    
    # 1. Geração de ID curto e ditável
    cid = gerar_id_conversa()
    print(f"[1] ID Gerado: {cid}")
    assert len(cid) > 10, "ID deve conter data e sufixo"
    assert "-" in cid, "ID deve ser formatado com hífens"
    
    # 2. Inicialização dos dados da conversa
    dados_conversa = {
        "id": cid,
        "modelo": "teste-modelo-isaac",
        "timestamp_criacao": time.time(),
        "trocas": [],
        "mensagens": [
            {"role": "system", "content": "Prompt do sistema de teste"}
        ]
    }
    
    caminho_arquivo = DIR_CONVERSAS / f"{cid}.json"
    print(f"[2] Salvando conversa inicial em: {caminho_arquivo}")
    salvar_conversa_atomica(caminho_arquivo, dados_conversa)
    
    assert caminho_arquivo.exists(), "O arquivo de conversa deve ser criado"
    
    # Valida conteúdo inicial
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        lido = json.load(f)
    assert lido["id"] == cid
    assert lido["modelo"] == "teste-modelo-isaac"
    assert len(lido["trocas"]) == 0
    assert len(lido["mensagens"]) == 1
    
    # 3. Troca 1 (Primeira pergunta com ferramenta)
    print("[3] Simulando Troca 1 (Pergunta -> Tool Call -> Resposta)")
    troca1 = {
        "pedido": "Qual é a pasta de trabalho?",
        "resposta": "",
        "chamadas": [],
        "timestamp": time.time()
    }
    dados_conversa["trocas"].append(troca1)
    dados_conversa["mensagens"].append({"role": "user", "content": troca1["pedido"]})
    salvar_conversa_atomica(caminho_arquivo, dados_conversa)
    
    # Simula tool call incremental
    tool_call1 = {
        "nome": "list_dir",
        "args": "{}",
        "resultado": "['app.py', 'agent.py']"
    }
    dados_conversa["trocas"][-1]["chamadas"].append(tool_call1)
    dados_conversa["mensagens"].append({"role": "assistant", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]})
    dados_conversa["mensagens"].append({"role": "tool", "tool_call_id": "tc1", "content": tool_call1["resultado"]})
    salvar_conversa_atomica(caminho_arquivo, dados_conversa)
    
    # Simula resposta final
    troca1_resposta = "A pasta de trabalho contém os arquivos app.py e agent.py."
    dados_conversa["trocas"][-1]["resposta"] = troca1_resposta
    dados_conversa["mensagens"].append({"role": "assistant", "content": troca1_resposta})
    salvar_conversa_atomica(caminho_arquivo, dados_conversa)
    
    # 4. Troca 2 (Segunda pergunta)
    print("[4] Simulando Troca 2 (Mensagem simples)")
    troca2 = {
        "pedido": "Obrigado!",
        "resposta": "De nada! Estou aqui para ajudar.",
        "chamadas": [],
        "timestamp": time.time()
    }
    dados_conversa["trocas"].append(troca2)
    dados_conversa["mensagens"].append({"role": "user", "content": troca2["pedido"]})
    dados_conversa["mensagens"].append({"role": "assistant", "content": troca2["resposta"]})
    salvar_conversa_atomica(caminho_arquivo, dados_conversa)
    
    # 5. Simulando crash e recarga (releitura fria do disco)
    print("[5] Simulando crash... Limpando dados na memória e relendo do disco...")
    del dados_conversa
    
    # Recarrega o arquivo como se o app tivesse reiniciado
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        dados_recarregados = json.load(f)
        
    print("[6] Validando integridade dos dados recarregados...")
    assert dados_recarregados["id"] == cid
    assert dados_recarregados["modelo"] == "teste-modelo-isaac"
    assert len(dados_recarregados["trocas"]) == 2, "Devem haver exatamente 2 trocas registradas"
    
    # Valida Troca 1
    t1 = dados_recarregados["trocas"][0]
    assert t1["pedido"] == "Qual é a pasta de trabalho?"
    assert len(t1["chamadas"]) == 1, "Deve haver 1 chamada de ferramenta"
    assert t1["chamadas"][0]["nome"] == "list_dir"
    assert t1["chamadas"][0]["resultado"] == "['app.py', 'agent.py']"
    assert t1["resposta"] == "A pasta de trabalho contém os arquivos app.py e agent.py."
    
    # Valida Troca 2
    t2 = dados_recarregados["trocas"][1]
    assert t2["pedido"] == "Obrigado!"
    assert t2["resposta"] == "De nada! Estou aqui para ajudar."
    assert len(t2["chamadas"]) == 0
    
    print("[7] Limpando arquivo de teste do disco...")
    caminho_arquivo.unlink()
    
    print("=== TESTE CONCLUÍDO COM SUCESSO! 100% VÁLIDO! ===")


if __name__ == "__main__":
    try:
        testar_fluxo_historico()
    except AssertionError as e:
        print(f"‼ FALHA NO TESTE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"‼ ERRO INESPERADO NO TESTE: {e}")
        sys.exit(1)
