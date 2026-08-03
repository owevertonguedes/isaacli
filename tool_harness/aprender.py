"""Destilacao: o professor (Gemini) gera exemplos, o portao mecanico filtra,
sai um .jsonl pronto pra treinar LoRA. Task 09.

POR QUE PELO AGENT BUILDER E NAO PELA VERTEX DIRETO:
O credito "Trial credit for GenAI App Builder" (EUR 859, expira 21/06/2027) so
paga SKUs de Vertex AI Search/Conversation. Chamada normal em
aiplatform.googleapis.com/...:generateContent NAO consome ele -- cai no Free
Trial, que e menor e expira antes (12/09/2026).

O endpoint abaixo (discoveryengine .../assistants/*:streamAssist) roda o mesmo
Gemini mas pela superficie do Agent Builder. Verificado funcionando em
2026-07-19. Se um dia der 404, o metodo standalone :generateGroundedContent foi
REMOVIDO da API -- nao tente ressuscitar, o caminho e este.

REGRA QUE NAO SE PULA: exemplo que nao passa no portao e DESCARTADO, nunca
"consertado". Professor bom erra, e erro consertado por nos vira alucinacao nos
pesos -- foi exatamente o defeito do treino.jsonl antigo.
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import google.auth
import google.auth.transport.requests
import urllib.request

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATASETS = RAIZ / "datasets"
ENGINE = "<ENGINE_ID>"

# Cascata de professores: melhor primeiro, cai pro proximo se der cota/erro.
# Sondado em 2026-07-19 neste endpoint -- estes 3 respondem, o resto da 400
# ("model id invalido"): gemini-3.5-pro, gemini-3-pro, gemini-3-flash,
# gemini-3.1-pro nao existem aqui. Nao adianta tentar de novo sem sondar.
# O 2.5-flash e o piso: abaixo dele o professor nao compensa (task 09).
MODELOS = ["gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"]


def _sessao():
    creds, proj = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    cab = {
        "Authorization": "Bearer " + creds.token,
        "Content-Type": "application/json",
        "X-Goog-User-Project": proj,
    }
    url = (f"https://discoveryengine.googleapis.com/v1/projects/{proj}"
           f"/locations/global/collections/default_collection/engines/{ENGINE}"
           f"/assistants/default_assistant:streamAssist")
    return cab, url


def _uma_chamada(cab, url, pedido, modelo):
    """Devolve o texto final do professor, sem o raciocinio.

    O streamAssist devolve um ARRAY de pedacos; os de raciocinio vem marcados
    com thought=true. Concatenar tudo sem filtrar mistura o "pensando..." no
    codigo -- e o mesmo tipo de bug que ja custou caro aqui (o ollama tambem
    devolve raciocinio em campo separado).
    """
    corpo = {"query": {"text": pedido}, "generationSpec": {"modelId": modelo}}
    req = urllib.request.Request(url, headers=cab,
                                 data=json.dumps(corpo).encode())
    bruto = urllib.request.urlopen(req).read().decode()
    partes = []
    for pedaco in json.loads(bruto):
        for r in pedaco.get("answer", {}).get("replies", []):
            c = r.get("groundedContent", {}).get("content", {})
            if c.get("thought"):
                continue
            partes.append(c.get("text", ""))
    return "".join(partes)


def gerar(cab, url, pedido):
    """Tenta os professores em ordem de qualidade. Devolve (texto, modelo).

    Cai pro proximo em 429 (cota), 503 (indisponivel) e 500. NAO cai em 400:
    erro de payload nosso nao melhora trocando de modelo, so esconde o bug.
    """
    ultimo = None
    for modelo in MODELOS:
        try:
            return _uma_chamada(cab, url, pedido, modelo), modelo
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503):
                raise
            ultimo = e
            print(f"      {modelo} indisponivel ({e.code}), caindo pro proximo")
    raise ultimo


def extrai_bloco(texto, marca="python"):
    if f"```{marca}" in texto:
        return texto.split(f"```{marca}", 1)[1].split("```", 1)[0].strip()
    if "```" in texto:
        return texto.split("```", 1)[1].split("```", 1)[0].strip()
    return texto.strip()


def portao(codigo, testes):
    """Executa de verdade. Sem juiz opinando: passou ou nao passou."""
    ns = {}
    try:
        exec(codigo, ns)
        exec(testes, ns)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


PROMPT = """Gere UM exemplo de treino para ensinar um modelo pequeno a escrever Python utilitario.
Tarefa: {alvo}

Responda em DOIS blocos de codigo, nesta ordem e nada mais:

```python
# a funcao, autocontida, com os imports que precisar
```

```python
# so linhas 'assert' que testam a funcao acima.
# sem comentario, sem print, sem "exemplo de uso".
# inclua caso de borda: entrada vazia e de um elemento.
```"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alvo", required=True,
                   help="tarefa estreita a ensinar (task 09: 'codigo em geral' nao funciona)")
    p.add_argument("--n", type=int, default=10, help="quantos exemplos tentar")
    a = p.parse_args()

    cab, url = _sessao()
    DIR_DATASETS.mkdir(exist_ok=True)
    saida = DIR_DATASETS / f"{a.alvo.replace(' ', '-')[:40]}-{date.today()}.jsonl"

    aprovados = rejeitados = falhou_api = 0
    t0 = time.time()
    with open(saida, "a") as f:
        for i in range(a.n):
            pedido = PROMPT.format(alvo=f"{a.alvo} (variacao {i + 1})")
            try:
                texto, usado = gerar(cab, url, pedido)
            except Exception as e:
                falhou_api += 1
                print(f"  [{i+1}/{a.n}] ERRO API: {str(e)[:80]}")
                continue
            blocos = texto.split("```")
            codigos = [extrai_bloco("```" + b) for b in blocos if b.strip().startswith("python")]
            if len(codigos) < 2:
                rejeitados += 1
                print(f"  [{i+1}/{a.n}] REJEITADO: nao vieram os 2 blocos")
                continue
            codigo, testes = codigos[0], codigos[1]
            ok, erro = portao(codigo, testes)
            if ok:
                aprovados += 1
                f.write(json.dumps({"alvo": a.alvo, "professor": usado, "codigo": codigo,
                                    "testes": testes}, ensure_ascii=False) + "\n")
                f.flush()  # incremental: matar no meio nao perde o que passou
                print(f"  [{i+1}/{a.n}] aprovado  ({usado})")
            else:
                rejeitados += 1
                print(f"  [{i+1}/{a.n}] REJEITADO no portao: {erro[:70]}")

    print(f"\ngerados {a.n} | aprovados {aprovados} | rejeitados {rejeitados} "
          f"| erro de api {falhou_api} | {time.time() - t0:.0f}s")
    print(f"dataset: {saida}")


if __name__ == "__main__":
    main()
