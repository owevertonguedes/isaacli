#!/usr/bin/env python3
"""Juiz Gemini: portão mecânico + avaliação de requisitos via Gemini.

Este script executa:
1. Portão mecânico: verifica sintaxe, abre com Playwright para pegar erros de runtime.
2. Juiz Gemini: se passou no portão mecânico, envia o código para o Gemini avaliar
   cada um dos requisitos de negócio fornecidos na especificação.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
import tools
import aprender

# Validação obrigatória das credenciais no ambiente
for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"]:
    if var not in os.environ:
        raise RuntimeError(
            f"Erro: A variável de ambiente '{var}' não está definida.\n"
            f"Por favor, exporte-a antes de prosseguir:\n"
            f"export {var}=<caminho_ou_id_correto>"
        )

PROMPT_SISTEMA_JUIZ = """Você é um juiz de garantia de qualidade (QA) de software extremamente rigoroso.
Sua tarefa é analisar o código-fonte de um arquivo HTML contendo uma aplicação web de página única (Single Page Application) e julgar se ela atende a uma lista de requisitos funcionais.

Para cada requisito listado na especificação, você deve analisar o código-fonte detalhadamente e verificar se ele foi devidamente implementado (não aceitando soluções falsas ou meramente ilustrativas sem lógica real).

Você deve retornar um objeto JSON estritamente válido contendo as seguintes chaves:
{
  "ok": true/false (true apenas se TODOS os requisitos passarem),
  "requisitos": [
    {
      "requisito": "nome do requisito original",
      "passou": true/false,
      "detalhes": "explicação detalhada do porquê passou ou falhou, citando partes de tags, classes, IDs ou funções do código real"
    }
  ]
}

IMPORTANTE: Responda APENAS com o JSON. Não use blocos de código markdown (como ```json) e não escreva nenhuma introdução ou texto antes/depois do JSON.
"""

PROMPT_USUARIO_JUIZ = """Aqui está a especificação dos requisitos funcionais:
{especificacao}

Aqui está o código-fonte completo da aplicação:
```html
{codigo}
```

Analise minuciosamente e dê o seu veredito no formato JSON especificado.
"""

def carregar_spec(nome_projeto):
    """Carrega os requisitos funcionais correspondentes do projeto."""
    spec_path = AQUI / f"specs_{nome_projeto}.json"
    if not spec_path.exists():
        # Fallback para spec padrão se não achar
        return [
            "O arquivo deve possuir um formulário para entrada de dados.",
            "O arquivo deve possuir exibição visual dos resultados."
        ]
    try:
        return json.loads(spec_path.read_text())
    except Exception as e:
        print(f"Erro ao carregar a spec {nome_projeto}: {e}")
        return []

def julgar(caminho_arquivo, nome_projeto="financeiro"):
    """Executa a checagem mecânica e, se aprovada, chama o Gemini Juiz."""
    # resolve antes de comparar com a sandbox: caminho relativo ao cwd nao e
    # relativo a SANDBOX_ROOT, e sem isto o portao dizia "nao existe" pra
    # arquivo que existe — erro de regua, nao de codigo julgado.
    p = Path(caminho_arquivo).resolve()
    if not p.exists():
        return {
            "ok": False,
            "etapa": "portao_mecanico",
            "problemas": [f"Arquivo não existe: {caminho_arquivo}"],
            "evidencias": []
        }

    # --- 1) PORTÃO MECÂNICO ---
    print(f"[Portão Mecânico] Verificando sanidade de {p.name}...")
    saida_checar = tools.checar_arquivo(str(p.relative_to(tools.SANDBOX_ROOT) if p.is_relative_to(tools.SANDBOX_ROOT) else p))
    
    if not saida_checar.startswith("OK"):
        # Se falhou na checagem mecânica, retorna o erro imediatamente sem gastar Gemini
        linhas = saida_checar.splitlines()
        problemas = [l[2:] if l.startswith("- ") else l for l in linhas if l and "PROBLEMAS ENCONTRADOS" not in l]
        return {
            "ok": False,
            "etapa": "portao_mecanico",
            "problemas": problemas,
            "evidencias": ["Falha no portão mecânico de sanidade (sintaxe ou runtime)."]
        }

    print("[Portão Mecânico] OK! O arquivo abre sem erros e possui estrutura básica.")

    # --- 2) JUIZ GEMINI ---
    print(f"[Gemini Juiz] Avaliando requisitos para {nome_projeto}...")
    requisitos = carregar_spec(nome_projeto)
    if not requisitos:
        return {
            "ok": False,
            "etapa": "preparacao",
            "problemas": ["Especificação de requisitos vazia ou inexistente."],
            "evidencias": []
        }

    codigo = p.read_text(encoding="utf-8")
    
    # Monta o prompt
    espec_texto = ""
    for i, req in enumerate(requisitos, 1):
        espec_texto += f"{i}. {req['titulo']}: {req['descricao']}\n"

    prompt = PROMPT_SISTEMA_JUIZ + "\n\n" + PROMPT_USUARIO_JUIZ.format(especificacao=espec_texto, codigo=codigo)
    
    # Chama o Gemini via Agent Builder
    try:
        cab, url = aprender._sessao()
        resposta_texto, modelo_usado = aprender.gerar(cab, url, prompt)
    except Exception as e:
        return {
            "ok": False,
            "etapa": "gemini_juiz",
            "problemas": [f"Erro ao chamar o Gemini Juiz via Agent Builder: {str(e)[:300]}"],
            "evidencias": []
        }

    resposta_texto = resposta_texto.strip()
    
    # Remove eventuais blocos de marcação de código markdown
    if resposta_texto.startswith("```"):
        linhas = resposta_texto.splitlines()
        if linhas[0].startswith("```"):
            linhas = linhas[1:]
        if linhas and linhas[-1].startswith("```"):
            linhas = linhas[:-1]
        resposta_texto = "\n".join(linhas).strip()

    try:
        resultado = json.loads(resposta_texto)
        resultado["etapa"] = "gemini_juiz"
        return resultado
    except Exception as e:
        # Se a saída não puder ser parsed como JSON, tenta extrair o que puder ou logar o erro
        print(f"Erro ao analisar JSON do Gemini: {e}")
        print("Saída recebida:")
        print(resposta_texto[:1000])
        return {
            "ok": False,
            "etapa": "gemini_juiz_parse_erro",
            "problemas": [f"Resposta do Gemini não pôde ser lida como JSON: {str(e)[:150]}"],
            "evidencias": [resposta_texto]
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 juiz_gemini.py <caminho_arquivo> [nome_projeto]")
        sys.exit(1)
    
    arq = sys.argv[1]
    proj = sys.argv[2] if len(sys.argv) > 2 else "financeiro"
    res = julgar(arq, proj)
    print(json.dumps(res, indent=2, ensure_ascii=False))
