#!/usr/bin/env python3
"""Loop de Desenvolvimento do Isaac com o Juiz Gemini.

Este script orquestra o ciclo completo de:
1. Isaac (granite4 via Ollama/isaac) escrevendo/corrigindo o código do projeto real.
2. Portão Mecânico de sanidade (se quebrar, nem gasta chamada).
3. Gemini Juiz avaliando cada um dos requisitos da especificação técnica.
4. Registro de cada ciclo no diário JSON (sandbox/diario_juiz.json) e Markdown (sandbox/diario_juiz.md).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
import agent
import tools

DIARIO_JSON = tools.SANDBOX_ROOT / "diario_juiz.json"
DIARIO_MD = tools.SANDBOX_ROOT / "diario_juiz.md"
FERRAMENTAS_ANDAIME = tools.schema_filtrado([
    "read_file", "replace_between", "checar_arquivo",
])

PROJETO = "financeiro"
ARQUIVO_RELATIVO = "jogos/financeiro.html"
ALVO = tools.SANDBOX_ROOT / ARQUIVO_RELATIVO

CONHECIMENTO_JUIZ_LOOP = """Voce e Isaac, um programador experiente que cria aplicacoes web de pagina unica (SPA) em UM unico arquivo HTML completo.

REGRAS DO ARQUIVO:
- O arquivo inicial ja existe. Para evoluir a aplicacao, prefira replace_between
  nos marcadores ISAAC_* em vez de reescrever o HTML inteiro.
- Nao reescreva o arquivo inteiro. O andaime existe para trocar trechos pequenos.
- NUNCA deixe comentarios de placeholder tipo "// JavaScript aqui" ou "/* CSS aqui */".
  Se voce escreveu um comentario desses, o codigo real esta faltando: escreva o codigo completo e funcional.
- Todo o JavaScript vai DENTRO de <script>...</script> integrado na pagina.
- Quando um marcador esta dentro de uma funcao existente, escreva apenas
  comandos do miolo. NUNCA declare a funcao de novo dentro dos marcadores.
- Todo elemento interativo precisa estar conectado na interface do usuario.
- Nada de bibliotecas externas via CDN ou links HTTP que dependam de internet. Tudo em um unico arquivo.
- Apos escrever o arquivo, chame obrigatoriamente checar_arquivo nele para garantir que o portao mecanico o aprove.
"""

ESQUELETO_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gerenciador Financeiro Pessoal</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f6f7fb; color: #1f2937; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 16px; }
    section { margin: 16px 0; }
    .grid { display: grid; gap: 12px; }
    button, input, select { font: inherit; padding: 10px; }
    button { cursor: pointer; border: 0; background: #2563eb; color: white; border-radius: 6px; }
    table { width: 100%; border-collapse: collapse; background: white; }
    th, td { padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }
    .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .card { background: white; padding: 14px; border-radius: 8px; border: 1px solid #e5e7eb; }
    .bar { height: 20px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
    .bar span { display: block; height: 100%; width: 0%; background: #ef4444; }
    @media (max-width: 720px) { .cards { grid-template-columns: 1fr; } }
    /* ISAAC_CSS_START */
    /* ISAAC_CSS_END */
  </style>
</head>
<body>
  <main>
    <h1>Gerenciador Financeiro Pessoal</h1>
    <p id="status">Base pronta para evolucao por etapas.</p>
    <section id="entrada">
      <!-- ISAAC_FORM_START -->
      <!-- ISAAC_FORM_END -->
    </section>
    <section id="resumo" class="cards">
      <!-- ISAAC_SUMMARY_START -->
      <!-- ISAAC_SUMMARY_END -->
    </section>
    <section id="grafico">
      <!-- ISAAC_CHART_START -->
      <!-- ISAAC_CHART_END -->
    </section>
    <section id="lista">
      <!-- ISAAC_LIST_START -->
      <!-- ISAAC_LIST_END -->
    </section>
    <button id="health-check" type="button">Verificar base</button>
  </main>
  <script>
    const STORAGE_KEY = 'isaac-financeiro-base';
    let transactions = [];

    function money(value) {
      return Number(value || 0).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
    }

    function saveTransactions() {
      /* ISAAC_SAVE_START */
      return transactions;
      /* ISAAC_SAVE_END */
    }

    function loadTransactions() {
      /* ISAAC_LOAD_START */
      transactions = [];
      /* ISAAC_LOAD_END */
    }

    function render() {
      document.getElementById('status').textContent = 'Base carregada.';
      /* ISAAC_RENDER_LIST_START */
      /* ISAAC_RENDER_LIST_END */
      /* ISAAC_RENDER_SUMMARY_START */
      /* ISAAC_RENDER_SUMMARY_END */
      /* ISAAC_RENDER_CHART_START */
      /* ISAAC_RENDER_CHART_END */
    }

    function addTransaction(event) {
      /* ISAAC_ADD_START */
      if (event) event.preventDefault();
      /* ISAAC_ADD_END */
    }

    function deleteTransaction(index) {
      /* ISAAC_DELETE_START */
      return index;
      /* ISAAC_DELETE_END */
    }

    document.getElementById('health-check').addEventListener('click', () => {
      document.getElementById('status').textContent = 'Base operacional.';
    });
    loadTransactions();
    render();
  </script>
</body>
</html>
"""

ETAPAS = [
    {
        "nome": "formulario de transacao",
        "marcadores": "ISAAC_FORM_START/ISAAC_FORM_END e ISAAC_ADD_START/ISAAC_ADD_END",
        "requisito": "Adicionar transacoes com descricao, valor maior que zero, tipo Receita/Despesa, categoria e data.",
    },
    {
        "nome": "lista de transacoes com acao excluir",
        "marcadores": "ISAAC_LIST_START/ISAAC_LIST_END e ISAAC_RENDER_LIST_START/ISAAC_RENDER_LIST_END",
        "requisito": (
            "Exibir todas as transacoes em tabela/lista com descricao, valor, tipo, "
            "categoria, data e uma coluna Acoes. Para cada transacao, renderize um "
            "botao Excluir que chame deleteTransaction(indiceRealDaIteracao). "
            "Nao implemente deleteTransaction nesta etapa; apenas conecte o botao."
        ),
    },
    {
        "nome": "deleteTransaction em JS puro",
        "marcadores": "ISAAC_DELETE_START/ISAAC_DELETE_END",
        "requisito": (
            "Implementar somente os comandos dentro da funcao deleteTransaction "
            "que ja existe. O conteudo deve ser statements JavaScript, por exemplo "
            "if/return, transactions.splice(...), saveTransactions() e render(). "
            "Nao escreva HTML e nao escreva 'function deleteTransaction'."
        ),
    },
    {
        "nome": "resumo financeiro",
        "marcadores": "ISAAC_SUMMARY_START/ISAAC_SUMMARY_END e ISAAC_RENDER_SUMMARY_START/ISAAC_RENDER_SUMMARY_END",
        "requisito": "Mostrar Total Receitas, Total Despesas e Saldo Atual sempre atualizados.",
    },
    {
        "nome": "estrutura do grafico visual",
        "marcadores": "ISAAC_CHART_START/ISAAC_CHART_END",
        "requisito": (
            "Inserir apenas o HTML do grafico visual, sem <style> e sem <script>. "
            "Use elementos simples com IDs estaveis para receitas, despesas e saldo."
        ),
    },
    {
        "nome": "estilo do grafico visual",
        "marcadores": "ISAAC_CSS_START/ISAAC_CSS_END",
        "requisito": (
            "Inserir apenas CSS para o grafico visual nos marcadores de CSS. "
            "Nao inclua tag <style>, HTML ou JavaScript."
        ),
    },
    {
        "nome": "atualizacao do grafico no render",
        "marcadores": "ISAAC_RENDER_CHART_START/ISAAC_RENDER_CHART_END",
        "requisito": (
            "Inserir apenas JavaScript que atualize os elementos do grafico ja "
            "existentes com base em transactions. Nao inclua HTML, CSS ou tag <script>."
        ),
    },
    {
        "nome": "persistencia local",
        "marcadores": "ISAAC_SAVE_START/ISAAC_SAVE_END",
        "requisito": (
            "Implementar somente o miolo JavaScript de saveTransactions(): salvar "
            "o array transactions em localStorage usando STORAGE_KEY e devolver "
            "transactions. Nao declare function saveTransactions de novo."
        ),
    },
    {
        "nome": "carregamento local",
        "marcadores": "ISAAC_LOAD_START/ISAAC_LOAD_END",
        "requisito": (
            "Implementar somente o miolo JavaScript de loadTransactions(): ler "
            "localStorage com STORAGE_KEY, converter JSON com seguranca e preencher "
            "transactions com array valido. Nao declare function loadTransactions de novo."
        ),
    },
]

PEDIDO_ETAPA = """Evolua a aplicacao em {arquivo} trabalhando SOMENTE nesta etapa: {nome}.

Requisito desta etapa:
{requisito}

Marcadores provaveis para mexer:
{marcadores}

Feedback anterior, se houver:
{feedback}

Procedimento obrigatorio:
1. Leia o arquivo com read_file.
2. Use replace_between para trocar apenas o trecho necessario.
   Nos argumentos start_marker/end_marker, use somente o nome do marcador, por
   exemplo ISAAC_CSS_START e ISAAC_CSS_END. Nao inclua /* */, <!-- --> nem
   misture familias diferentes de marcador.
3. Chame checar_arquivo em {arquivo}.
4. Responda curto.

Nao reescreva o arquivo inteiro se os marcadores existirem."""

PEDIDO_CORRECAO = """O juiz ainda encontrou problemas em {arquivo}.

Aqui esta o veredito do Juiz Gemini detalhando o que esta pendente ou com erros:

{erros_feedback}

Corrija em passos pequenos. Leia o arquivo, use replace_between nos marcadores
mais proximos do problema e chame checar_arquivo."""


def registrar_ciclo(ciclo, total_reqs, cumpridos, tentativas, detalhes, uso=None,
                    modo_juiz="gemini"):
    """Salva os resultados no diário JSON e Markdown para persistência e desenho do gráfico."""
    # 1) Diário JSON (Usado pelo gráfico do App)
    diario_dados = []
    if DIARIO_JSON.exists():
        try:
            diario_dados = json.loads(DIARIO_JSON.read_text(encoding="utf-8"))
        except Exception:
            diario_dados = []

    novo_registro = {
        "ciclo": ciclo,
        "total_requisitos": total_reqs,
        "requisitos_cumpridos": cumpridos,
        "tentativas_no_ciclo": tentativas,
        "modo_juiz": modo_juiz,
        "uso": uso or {},
        "timestamp": time.time(),
        "data_hora": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    diario_dados.append(novo_registro)
    
    DIARIO_JSON.parent.mkdir(parents=True, exist_ok=True)
    DIARIO_JSON.write_text(json.dumps(diario_dados, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) Diário Markdown (Leitura de status de manhã)
    modo = "a" if DIARIO_MD.exists() else "w"
    with open(DIARIO_MD, modo, encoding="utf-8") as f:
        if modo == "w":
            f.write("# Diário de Evolução do Isaac (Loop do Juiz Gemini)\n\n")
        f.write(f"## Ciclo {ciclo} — {novo_registro['data_hora']}\n")
        f.write(f"- **Requisitos Cumpridos:** {cumpridos} de {total_reqs} ({(cumpridos/total_reqs)*100:.1f}%)\n")
        f.write(f"- **Tentativas gastas no ciclo:** {tentativas}\n\n")
        f.write(f"- **Modo do juiz:** {modo_juiz}\n")
        if uso:
            f.write(
                "- **Uso Ollama:** "
                f"prompt={uso.get('prompt_eval_count', 0)} · "
                f"resposta={uso.get('eval_count', 0)} · "
                f"tempo={uso.get('total_duration', 0) / 1_000_000_000:.2f}s\n\n")
        f.write("### Detalhes por Requisito:\n")
        for d in detalhes:
            status = "✅ PASSOU" if d.get("passou") else "❌ FALHOU"
            f.write(f"#### {d.get('requisito')} - {status}\n")
            f.write(f"{d.get('detalhes')}\n\n")
        f.write("---\n\n")


def garantir_esqueleto(forcar=False):
    """Cria um arquivo inicial valido e pequeno o bastante para o modelo editar."""
    if ALVO.exists() and ALVO.stat().st_size > 0 and not forcar:
        return False
    tools.write_file(ARQUIVO_RELATIVO, ESQUELETO_HTML)
    return True


def checagem_local():
    """Portao mecanico sem Gemini; usado entre etapas para nao empilhar erro."""
    saida = tools.checar_arquivo(ARQUIVO_RELATIVO)
    if saida.startswith("OK"):
        return True, saida
    return False, saida


def rodar_etapa(etapa, modelo, feedback="", tentativas=2):
    """Executa uma etapa curta com retentativa baseada no erro mecanico."""
    ultimo_feedback = feedback or "(nenhum)"
    uso_total = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
    for tentativa in range(1, tentativas + 1):
        print(f"  -> Etapa: {etapa['nome']} (tentativa {tentativa}/{tentativas})")
        pedido = PEDIDO_ETAPA.format(
            arquivo=ARQUIVO_RELATIVO,
            nome=etapa["nome"],
            requisito=etapa["requisito"],
            marcadores=etapa["marcadores"],
            feedback=ultimo_feedback,
        )
        r = agent.rodar(
            pedido, modelo, max_passos=8, verbose=False,
            tools_schema=FERRAMENTAS_ANDAIME)
        for chave in uso_total:
            uso_total[chave] += int((r.get("uso") or {}).get(chave) or 0)
        chamadas = [c[0] for c in r.get("chamadas", [])]
        ok, saida = checagem_local()
        mudou = any(
            nome == "replace_between" and str(resultado).startswith("OK:")
            for nome, _args, resultado, _via in r.get("chamadas", [])
        )
        erros_replace = [
            str(resultado).splitlines()[0]
            for nome, _args, resultado, _via in r.get("chamadas", [])
            if nome == "replace_between" and not str(resultado).startswith("OK:")
        ]
        print(
            f"     ferramentas: {chamadas} | patch: {'sim' if mudou else 'nao'} "
            f"| checagem: {'OK' if ok else 'FALHOU'}")
        if erros_replace:
            print("     replace_between recusou: " + " | ".join(erros_replace[:3]))
        if ok and mudou:
            return True, saida, chamadas, uso_total
        if not mudou:
            if erros_replace:
                ultimo_feedback = (
                    "Sua tentativa de replace_between foi recusada. Corrija exatamente "
                    "o motivo abaixo e tente uma troca menor:\n"
                    + "\n".join(f"- {erro}" for erro in erros_replace[:5]))
            else:
                ultimo_feedback = (
                    "Voce leu ou respondeu, mas nao alterou o arquivo. "
                    "Esta etapa so conta se voce chamar replace_between nos marcadores indicados.")
        else:
            ultimo_feedback = saida
    return False, ultimo_feedback, chamadas, uso_total


def feedback_por_etapa(feedback_juiz):
    if not feedback_juiz:
        return ""
    return "Feedback do juiz para considerar nesta etapa:\n" + feedback_juiz[:2500]


def _somar_uso(total, item):
    for chave in ("prompt_eval_count", "eval_count", "total_duration"):
        total[chave] = total.get(chave, 0) + int((item or {}).get(chave) or 0)


def avaliacao_sem_gemini():
    """Resultado auditavel quando o usuario ainda nao autorizou juiz externo."""
    ok_local, saida_local = checagem_local()
    spec_path = AQUI / f"specs_{PROJETO}.json"
    try:
        requisitos = json.loads(spec_path.read_text()) if spec_path.exists() else []
    except Exception:
        requisitos = []
    detalhes = []
    if not ok_local:
        return {
            "ok": False,
            "etapa": "portao_mecanico",
            "total_requisitos": 1,
            "requisitos_cumpridos": 0,
            "detalhes": [{
                "requisito": "Portao mecanico local",
                "passou": False,
                "detalhes": saida_local,
            }],
            "feedback": saida_local,
        }
    for req in requisitos:
        titulo = req.get("titulo") or req.get("descricao") or "requisito"
        detalhes.append({
            "requisito": titulo,
            "passou": False,
            "detalhes": "Nao avaliado: modo local sem Gemini. Portao mecanico passou.",
        })
    if not detalhes:
        detalhes.append({
            "requisito": "Portao mecanico local",
            "passou": True,
            "detalhes": saida_local,
        })
    return {
        "ok": False,
        "etapa": "sem_gemini",
        "total_requisitos": len(detalhes),
        "requisitos_cumpridos": 0,
        "detalhes": detalhes,
        "feedback": "Portao mecanico passou; falta juiz Gemini para requisitos de negocio.",
    }


def rodar_loop(modelo="isaac", max_ciclos=5, tentativas_por_ciclo=4,
               usar_gemini=True, reset_arquivo=False):
    """Executa o loop principal de desenvolvimento e correção autônoma."""
    print(f"\n=======================================================")
    print(f"Iniciando Loop do Juiz Gemini para {PROJETO}")
    print(f"Modelo do programador: {modelo}")
    print(f"Limites: até {max_ciclos} ciclos, {tentativas_por_ciclo} tentativas por ciclo")
    print(f"Juiz externo: {'Gemini' if usar_gemini else 'desativado (--sem-gemini)'}")
    print(f"=======================================================\n")
    
    # SOMA ao conhecimento de ferramentas, nao SUBSTITUI.
    #
    # Isto era um bug silencioso e caro (achado em 2026-07-19): trocar o system
    # prompt inteiro apagava as regras de COMO chamar ferramenta, e o isaac
    # parava de chamar qualquer uma. A API respondia 200, com
    # completion_tokens=2017 e content="" — ou seja, o modelo gerava e a saida
    # sumia. O loop rodava, registrava ciclo no diario e desenhava ponto no
    # grafico, sempre com arquivo de 0 byte. Parecia "o modelo nao consegue",
    # era "eu apaguei a instrucao".
    #
    # Medido, com o mesmo pedido e o mesmo modelo, mudando so o system:
    #   CONHECIMENTO_FERRAMENTAS  -> chamadas=['write_file'], arquivo criado
    #   CONHECIMENTO_JUIZ_LOOP    -> chamadas=[], nada criado
    agent.CONHECIMENTO_FERRAMENTAS = (
        agent.CONHECIMENTO_FERRAMENTAS + "\n\n" + CONHECIMENTO_JUIZ_LOOP)
    (tools.SANDBOX_ROOT / "jogos").mkdir(parents=True, exist_ok=True)

    ciclo = 1
    feedback_erros = ""

    while ciclo <= max_ciclos:
        print(f"\n--- INICIANDO CICLO {ciclo}/{max_ciclos} ---")

        criado = garantir_esqueleto(forcar=reset_arquivo and ciclo == 1)
        if criado:
            print(f"  Esqueleto inicial criado em {ARQUIVO_RELATIVO}")

        # Executa o agente em etapas pequenas. O parametro antigo
        # tentativas_por_ciclo agora vira tentativas por etapa, porque o gargalo
        # medido era tamanho de pedido, nao falta de ciclos no juiz.
        print("Isaac evoluindo o arquivo em etapas verificaveis...")
        t0 = time.time()
        chamadas_executadas = []
        uso_ciclo = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
        feedback_etapa = feedback_por_etapa(feedback_erros)
        for etapa in ETAPAS:
            try:
                ok_etapa, feedback_etapa, chamadas, uso_etapa = rodar_etapa(
                    etapa, modelo, feedback=feedback_etapa,
                    tentativas=max(1, min(tentativas_por_ciclo, 3)))
                chamadas_executadas.extend(chamadas)
                _somar_uso(uso_ciclo, uso_etapa)
                if not ok_etapa:
                    print("  [Aviso] etapa nao passou no portao mecanico; seguindo para o juiz registrar.")
                    break
            except Exception as e:
                print(f"  [Erro] Isaac estourou na etapa '{etapa['nome']}': {e}")
                feedback_etapa = str(e)
                break

        if feedback_erros:
            pedido = PEDIDO_CORRECAO.format(
                arquivo=ARQUIVO_RELATIVO, erros_feedback=feedback_erros)
            try:
                r = agent.rodar(
                    pedido, modelo, max_passos=8, verbose=False,
                    tools_schema=FERRAMENTAS_ANDAIME)
                chamadas_executadas.extend(c[0] for c in r.get("chamadas", []))
                _somar_uso(uso_ciclo, r.get("uso"))
            except Exception as e:
                print(f"  [Erro] Isaac estourou na correcao final: {e}")
        
        duracao = time.time() - t0
        tam_arquivo = ALVO.stat().st_size if ALVO.exists() else 0
        print(f"  Finalizado em {duracao:.1f}s | tamanho do arquivo: {tam_arquivo} bytes")
        print(f"  Ferramentas chamadas: {chamadas_executadas}")
        print(
            "  Uso Ollama: "
            f"prompt={uso_ciclo.get('prompt_eval_count', 0)} "
            f"resposta={uso_ciclo.get('eval_count', 0)} "
            f"tempo={uso_ciclo.get('total_duration', 0) / 1_000_000_000:.2f}s")

        # Avaliação pelo Juiz Gemini
        if usar_gemini:
            import juiz_gemini
            resultado_juiz = juiz_gemini.julgar(ALVO, PROJETO)
            modo_juiz = "gemini"
        else:
            resultado_juiz = avaliacao_sem_gemini()
            modo_juiz = "sem_gemini"
        
        # Analisa resultados
        ok = resultado_juiz.get("ok", False)
        etapa = resultado_juiz.get("etapa", "desconhecida")
        requisitos_lista = resultado_juiz.get("requisitos", [])
        
        total_reqs = (
            len(requisitos_lista) if requisitos_lista
            else resultado_juiz.get("total_requisitos", 6))
        cumpridos = (
            sum(1 for r in requisitos_lista if r.get("passou")) if requisitos_lista
            else resultado_juiz.get("requisitos_cumpridos", 0))
        
        if etapa == "portao_mecanico":
            print(f"❌ [CICLO {ciclo} FALHOU no Portão Mecânico]")
            problemas = "\n".join(f"- {p}" for p in resultado_juiz.get("problemas", []))
            if not problemas and resultado_juiz.get("detalhes"):
                problemas = "\n".join(d.get("detalhes", "") for d in resultado_juiz["detalhes"])
            print(problemas)
            feedback_erros = f"PORTÃO MECÂNICO (SINTAXE/RUNTIME):\n{problemas}"
            detalhes_registro = [{"requisito": "Portão Mecânico de Sanidade", "passou": False, "detalhes": problemas}]
        elif etapa == "sem_gemini":
            print(f"📊 [CICLO {ciclo} SEM GEMINI]")
            print("  Portao mecanico passou; requisitos de negocio nao foram julgados.")
            detalhes_registro = resultado_juiz.get("detalhes", [])
            feedback_erros = resultado_juiz.get("feedback", "")
        else:
            print(f"📊 [CICLO {ciclo} RESULTADOS DO JUIZ GEMINI]")
            print(f"  Aprovado Geral: {'Sim ✅' if ok else 'Não ❌'}")
            print(f"  Requisitos Cumpridos: {cumpridos}/{total_reqs}")
            
            feedback_linhas = []
            detalhes_registro = []
            for i, r in enumerate(requisitos_lista, 1):
                status = "✅" if r.get("passou") else "❌"
                print(f"    [{status}] {r.get('requisito')}")
                detalhes_registro.append({
                    "requisito": r.get("requisito"),
                    "passou": r.get("passou"),
                    "detalhes": r.get("detalhes")
                })
                if not r.get("passou"):
                    feedback_linhas.append(f"- REQUISITO FALHO: {r.get('requisito')}\n  Motivo: {r.get('detalhes')}")
            
            feedback_erros = "\n\n".join(feedback_linhas)

        # Registra no diário
        registrar_ciclo(
            ciclo, total_reqs, cumpridos, len(chamadas_executadas),
            detalhes_registro, uso=uso_ciclo, modo_juiz=modo_juiz)

        if ok:
            print(f"\n🎉 SUCESSO! Isaac construiu a aplicação atendendo a 100% dos requisitos no ciclo {ciclo}!")
            break
            
        ciclo += 1

    print("\nFim do loop de desenvolvimento do Isaac.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="isaac")
    ap.add_argument("--ciclos", type=int, default=5)
    ap.add_argument("--sem-gemini", action="store_true",
                    help="roda so Isaac + portao mecanico local, sem chamada externa")
    ap.add_argument("--reset-arquivo", action="store_true",
                    help="recria o HTML alvo antes do primeiro ciclo")
    a = ap.parse_args()
    
    rodar_loop(
        modelo=a.modelo,
        max_ciclos=a.ciclos,
        usar_gemini=not a.sem_gemini,
        reset_arquivo=a.reset_arquivo,
    )
