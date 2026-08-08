"""Mede o TETO de capacidade dos modelos locais, nao o formato da saida.

Por que assim: o teste de tool-call media se ele acerta um formato — util pra
confiabilidade, inutil pra saber se ele e mais inteligente. Aqui cada problema
tem teste que EXECUTA. O modelo escreve a funcao, o teste roda, passou ou nao
passou. Nao tem juiz opinando, nao tem nota parcial, nao da pra fraudar com
resposta bonita.

Os problemas sobem de nivel de proposito (ver escala no historico do projeto):
  N2 = padrao com variacao, qualquer programador faz em minutos
  N3 = exige raciocinio de varios passos / estado / caso de borda
  N4 = regra INVENTADA, que nao da pra recuperar da memoria (ver n4_novos.py)
Se um modelo passa N2 e trava em N3, o teto dele esta entre os dois.

O N4 foi TROCADO em 2026-07-19 (task 03). Antes eram sudoku_solver,
match_wildcard e calculadora_completa — LeetCode classico, decorado no
pre-treino. O granite4:micro fazia 3/3 neles e errava `formata_moeda` no N2: a
escada estava invertida, e o N4 media o que o modelo VIU, nao o que ele
RACIOCINA. Os novos tem regra de negocio inventada, e por isso a especificacao e
a unica fonte da resposta.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from n4_novos import PROBLEMAS_N4 as _N4_NOVOS  # noqa: E402

MODELOS = ["qwen2.5-coder:3b", "granite4:micro"]
AQUI = Path(__file__).parent
TEMPO_LIMITE = 1800         # 6000 tokens a 3.5 tok/s = ate ~29min por problema

PROBLEMAS = [
    dict(
        nivel="N2", nome="agrupa_por_chave",
        pedido="Escreva a funcao Python `agrupa_por_chave(pares)` (use exatamente esses nomes de funcao e parametro) que recebe uma "
               "lista de tuplas (chave, valor) e devolve um dict mapeando cada "
               "chave para a lista de valores daquela chave, preservando a ordem "
               "de aparicao dos valores.",
        testes="""
assert agrupa_por_chave([("a",1),("b",2),("a",3)]) == {"a":[1,3],"b":[2]}
assert agrupa_por_chave([]) == {}
assert agrupa_por_chave([("x",None)]) == {"x":[None]}
""",
    ),
    dict(
        nivel="N2", nome="valida_parenteses",
        pedido="Escreva a funcao Python `valida_parenteses(s)` (use exatamente esses nomes de funcao e parametro) que devolve True se "
               "a string tem parenteses (), colchetes [] e chaves {} balanceados e "
               "corretamente aninhados, e False caso contrario. Ignore outros "
               "caracteres.",
        testes="""
assert valida_parenteses("(a[b]{c})") is True
assert valida_parenteses("([)]") is False
assert valida_parenteses("") is True
assert valida_parenteses("(((") is False
assert valida_parenteses("]") is False
""",
    ),
    dict(
        nivel="N2", nome="filtra_e_soma_positivos",
        pedido="Escreva a funcao Python `filtra_e_soma_positivos(lista)` (use exatamente esses nomes de funcao e parametro) que recebe "
               "uma lista de numeros (inteiros ou floats), filtra apenas os numeros "
               "estritamente positivos e devolve a soma deles. Se nao houver "
               "positivos ou a lista estiver vazia, devolva 0.",
        testes="""
assert filtra_e_soma_positivos([1, -2, 3.5, 0, -4]) == 4.5
assert filtra_e_soma_positivos([]) == 0
assert filtra_e_soma_positivos([-1, -2, -3]) == 0
assert filtra_e_soma_positivos([0.0, 5, 5]) == 10
""",
    ),
    dict(
        nivel="N2", nome="formata_moeda",
        pedido="Escreva a funcao Python `formata_moeda(valor)` (use exatamente esses nomes de funcao e parametro) que recebe um numero "
               "float e o formata como uma string de moeda no formato brasileiro "
               "(ex: 'R$ 1.234,56'). Use separador de milhar como ponto e separador "
               "decimal como virgula, sempre com duas casas decimals. Arredonde para "
               "2 casas decimais se necessario.",
        testes="""
assert formata_moeda(1234.567) == "R$ 1.234,57"
assert formata_moeda(0.5) == "R$ 0,50"
assert formata_moeda(-100.0) == "R$ -100,00"
assert formata_moeda(1000000) == "R$ 1.000.000,00"
""",
    ),
    dict(
        nivel="N2", nome="busca_anagramas",
        pedido="Escreva a funcao Python `busca_anagramas(palavras)` (use exatamente esses nomes de funcao e parametro) que recebe uma "
               "lista de strings e as agrupa em listas de anagramas (palavras que "
               "usam as mesmas letras na mesma quantidade). Retorne uma lista de "
               "listas, onde cada sublista contem as palavras que sao anagramas "
               "entre si. A ordem das sublistas e das palavras dentro delas deve "
               "ser preservada conforme aparecem originalmente.",
        testes="""
assert busca_anagramas(["roma", "amor", "casa", "asac", "ramo", "boca"]) == [["roma", "amor", "ramo"], ["casa", "asac"], ["boca"]]
assert busca_anagramas([]) == []
assert busca_anagramas(["a"]) == [["a"]]
""",
    ),
    dict(
        nivel="N2", nome="inverte_palavras",
        pedido="Escreva a funcao Python `inverte_palavras(frase)` (use exatamente esses nomes de funcao e parametro) que recebe uma "
               "string e inverte a ordem das letras de cada palavra individualmente, "
               "mantendo a ordem original das palavras e preservando os espacos "
               "extras originais.",
        testes="""
assert inverte_palavras("cafe com leite") == "efac moc etiel"
assert inverte_palavras("") == ""
assert inverte_palavras("  muito   espaco ") == "  otium   ocapse "
""",
    ),
    dict(
        nivel="N2", nome="frequencia_caracteres",
        pedido="Escreva a funcao Python `frequencia_caracteres(s)` (use exatamente esses nomes de funcao e parametro) que recebe uma "
               "string e retorna um dicionario com a frequencia de TODOS os caracteres "
               "na string (incluindo espacos, pontuacao e quaisquer caracteres especiais), "
               "ignorando diferencas de maiuscula/minuscula (converta tudo para minusculo).",
        testes="""
assert frequencia_caracteres("Abba") == {"a": 2, "b": 2}
assert frequencia_caracteres("") == {}
assert frequencia_caracteres("a-b_a!") == {"a": 2, "-": 1, "b": 1, "_": 1, "!": 1}
""",
    ),
    dict(
        nivel="N2", nome="encontra_duplicados",
        pedido="Escreva a funcao Python `encontra_duplicados(lista)` (use exatamente esses nomes de funcao e parametro) que recebe uma "
               "lista e retorna uma nova lista ordenada com os elementos que aparecem "
               "mais de uma vez na lista de entrada. Cada elemento duplicado deve "
               "aparecer apenas uma vez na lista retornada.",
        testes="""
assert encontra_duplicados([3, 1, 3, 2, 1, 1, 4]) == [1, 3]
assert encontra_duplicados([]) == []
assert encontra_duplicados([1, 2, 3]) == []
""",
    ),
    dict(
        nivel="N2", nome="compacta_lista",
        pedido="Escreva a funcao Python `compacta_lista(lista)` (use exatamente esses nomes de funcao e parametro) que recebe uma lista "
               "e remove elementos consecutivos duplicados, deixando apenas o primeiro "
               "elemento de cada sequencia repetida consecutiva. Preserve a ordem "
               "dos elementos.",
        testes="""
assert compacta_lista([1, 1, 2, 2, 3, 1, 1, 1, 4]) == [1, 2, 3, 1, 4]
assert compacta_lista([]) == []
assert compacta_lista([1]) == [1]
""",
    ),
    dict(
        nivel="N2", nome="parse_csv_linha",
        pedido="Escreva a funcao Python `parse_csv_linha(linha)` (use exatamente esses nomes de funcao e parametro) que recebe uma "
               "string representando uma unica linha de um arquivo CSV (campos "
               "separados por virgula). Campos podem opcionalmente estar entre "
               "aspas duplas, as quais devem ser removidas do resultado. Se o "
               "campo contiver virgulas mas estiver entre aspas duplas, a virgula "
               "nao deve separar o campo. Se a linha for vazia, devolva uma "
               "lista vazia `[]`.",
        testes="""
assert parse_csv_linha('nome,"idade, anos",cidade') == ["nome", "idade, anos", "cidade"]
assert parse_csv_linha('') == []
assert parse_csv_linha('a,b,"c"') == ["a", "b", "c"]
""",
    ),
    dict(
        nivel="N3", nome="intervalos_livres",
        pedido="Escreva a funcao Python `intervalos_livres(ocupados, inicio, fim)` (use exatamente esses nomes de funcao e parametros). "
               "`ocupados` e uma lista de tuplas (a, b) com a < b, possivelmente "
               "desordenada e com sobreposicoes. Devolva a lista ordenada dos "
               "intervalos LIVRES dentro da janela [inicio, fim], como tuplas. "
               "Intervalos vazios nao entram no resultado.",
        testes="""
assert intervalos_livres([(2,4),(6,8)], 0, 10) == [(0,2),(4,6),(8,10)]
assert intervalos_livres([(1,5),(3,7)], 0, 10) == [(0,1),(7,10)]
assert intervalos_livres([], 0, 5) == [(0,5)]
assert intervalos_livres([(0,10)], 0, 10) == []
assert intervalos_livres([(-5,2),(8,20)], 0, 10) == [(2,8)]
""",
    ),
    dict(
        nivel="N3", nome="parse_config",
        pedido="Escreva a funcao Python `parse_config(texto)` (use exatamente esses nomes de funcao e parametro) que interpreta um "
               "formato INI simplificado: linhas '[secao]' abrem secao, linhas "
               "'chave = valor' definem valor dentro da secao atual, linhas "
               "comecando com # ou vazias sao ignoradas, espacos em volta de chave "
               "e valor sao removidos. Chaves antes de qualquer secao vao para a "
               "secao ''. Devolva dict de dicts. Se o valor for so digitos, "
               "converta para int.",
        testes="""
assert parse_config("a = 1") == {"": {"a": 1}}
assert parse_config("[s]\\n# c\\n\\nx = oi ") == {"s": {"x": "oi"}}
assert parse_config("[a]\\np=1\\n[b]\\np=2") == {"a":{"p":1},"b":{"p":2}}
assert parse_config("") == {}
assert parse_config("[s]\\nk = 007") == {"s": {"k": 7}}
""",
    ),
    dict(
        nivel="N3", nome="soma_subvetor_maximo",
        pedido="Escreva a funcao Python `soma_subvetor_maximo(nums)` (use exatamente esses nomes de funcao e parametro) que encontra "
               "a soma contigua maxima em uma lista de inteiros (que pode conter "
               "numeros negativos). Se a lista for vazia, devolva 0.",
        testes="""
assert soma_subvetor_maximo([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert soma_subvetor_maximo([-1, -2, -3]) == -1
assert soma_subvetor_maximo([]) == 0
assert soma_subvetor_maximo([5]) == 5
""",
    ),
    dict(
        nivel="N3", nome="merge_intervalos",
        pedido="Escreva a funcao Python `merge_intervalos(intervalos)` (use exatamente esses nomes de funcao e parametro) que recebe "
               "uma lista de tuplas (inicio, fim) representando intervalos fechados "
               "e mescla todos os intervalos sobrepostos, retornando uma nova lista "
               "de tuplas ordenada pelo inicio de cada intervalo.",
        testes="""
assert merge_intervalos([(1,3), (2,6), (8,10), (15,18)]) == [(1,6), (8,10), (15,18)]
assert merge_intervalos([(1,4), (4,5)]) == [(1,5)]
assert merge_intervalos([]) == []
assert merge_intervalos([(5,5)]) == [(5,5)]
""",
    ),
    dict(
        nivel="N3", nome="maior_retangulo_histograma",
        pedido="Escreva a funcao Python `maior_retangulo_histograma(alturas)` (use exatamente esses nomes de funcao e parametro) que "
               "recebe uma lista de alturas de barras de um histograma (largura "
               "de cada barra e 1) e retorna a area do maior retangulo contido "
               "no histograma. Se a lista for vazia, retorne 0.",
        testes="""
assert maior_retangulo_histograma([2, 1, 5, 6, 2, 3]) == 10
assert maior_retangulo_histograma([]) == 0
assert maior_retangulo_histograma([2, 4]) == 4
assert maior_retangulo_histograma([1, 1, 1, 1]) == 4
""",
    ),
    dict(
        nivel="N3", nome="valida_ip",
        pedido="Escreva a funcao Python `valida_ip(ip)` (use exatamente esses nomes de funcao e parametro) que recebe uma string e "
               "verifica se ela e um endereco IPv4 valido. Um IPv4 valido tem o "
               "formato 'A.B.C.D' onde cada parte (A, B, C, D) e um numero inteiro "
               "de 0 a 255 e nao pode ter zeros a esquerda (por exemplo, '192.168.01.1' "
               "e invalido, mas '192.168.1.1' e valido; '192.168.1' ou '192.168.1.1.1' "
               "sao invalidos).",
        testes="""
assert valida_ip("192.168.1.1") is True
assert valida_ip("192.168.01.1") is False
assert valida_ip("256.100.10.1") is False
assert valida_ip("12.34.56") is False
assert valida_ip("a.b.c.d") is False
""",
    ),
    dict(
        nivel="N3", nome="calculadora_rpn",
        pedido="Escreva a funcao Python `calculadora_rpn(expressao)` (use exatamente esses nomes de funcao e parametro) que avalia uma "
               "expressao em Notacao Polonesa Reversa (RPN) representada por uma "
               "string contendo numeros inteiros e os operadores '+', '-', '*', "
               "'//' (divisao inteira). Os operandos e operadores sao separados "
               "por espacos. Retorne o resultado como um inteiro. Se a expressao "
               "for invalida, vazia ou nao puder ser avaliada, levante ValueError.",
        testes="""
assert calculadora_rpn("3 4 + 2 * 7 -") == 7
assert calculadora_rpn("10 3 //") == 3
try:
    calculadora_rpn("")
    assert False, "Deveria dar ValueError"
except ValueError:
    pass
try:
    calculadora_rpn("1 +")
    assert False, "Deveria dar ValueError"
except ValueError:
    pass
""",
    ),
    dict(
        nivel="N3", nome="rle_compress",
        pedido="Escreva a funcao Python `rle_compress(texto)` (use exatamente esses nomes de funcao e parametro) que faz a compressao "
               "Run-Length Encoding (RLE) de uma string. Ela deve substituir sequencias "
               "de caracteres repetidos consecutivos por uma unica letra seguida "
               "do numero de repeticoes (ex: 'AAAABBBCC' vira 'A4B3C2'). Se o "
               "caractere nao se repetir (frequencia 1), nao coloque o numero 1 "
               "(ex: 'A' vira 'A'). Se a string for vazia, retorne string vazia.",
        testes="""
assert rle_compress("AAAABBBCCDA") == "A4B3C2DA"
assert rle_compress("") == ""
assert rle_compress("XYZ") == "XYZ"
""",
    ),
    dict(
        nivel="N3", nome="encontra_caminho_matriz",
        pedido="Escreva a funcao Python `encontra_caminho_matriz(matriz)` (use exatamente esses nomes de funcao e parametro) que "
               "encontra um caminho da celula inicial (0,0) ate a celula final "
               "(N-1, M-1) em uma matriz bidimensional (lista de listas) de 0s e 1s, "
               "onde 0 representa caminho livre e 1 representa obstaculo. Retorne "
               "uma lista de tuplas (linha, coluna) representando as posicoes do "
               "caminho desde (0,0) ate (N-1, M-1), ou None se nao houver caminho. "
               "Voce so pode se mover para baixo ou para a direita.",
        testes="""
assert encontra_caminho_matriz([[0, 0, 1], [1, 0, 0], [1, 1, 0]]) == [(0,0), (0,1), (1,1), (1,2), (2,2)]
assert encontra_caminho_matriz([[0, 1], [1, 0]]) is None
assert encontra_caminho_matriz([[0]]) == [(0,0)]
""",
    ),
    dict(
        nivel="N3", nome="janela_deslizante_max",
        pedido="Escreva a funcao Python `janela_deslizante_max(nums, k)` (use exatamente esses nomes de funcao e parametros) que "
               "recebe uma lista de inteiros e um tamanho de janela `k`, e devolve "
               "uma lista com o valor maximo em cada janela deslizante de tamanho "
               "`k`. Se a lista `nums` for vazia ou `k` for maior que o tamanho "
               "da lista (ou k <= 0), devolva uma lista vazia `[]`.",
        testes="""
assert janela_deslizante_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert janela_deslizante_max([1], 1) == [1]
assert janela_deslizante_max([], 3) == []
assert janela_deslizante_max([1, 2], 3) == []
""",
    ),
    dict(
        nivel="N3", nome="resolve_expressao_simples",
        pedido="Escreva a funcao Python `resolve_expressao_simples(expr)` (use exatamente esses nomes de funcao e parametro) que "
               "calcula o valor de uma expressao matematica representada por uma "
               "string contendo apenas numeros inteiros nao negativos e os operadores "
               "'+', '-', '*' (sem parenteses). Respeite a precedencia de operadores "
               "(* antes de + e -). Desconsidere espacos na string.",
        testes="""
assert resolve_expressao_simples(" 3 + 5 * 2 - 4 ") == 9
assert resolve_expressao_simples("10") == 10
assert resolve_expressao_simples("2*3*4") == 24
""",
    ),
    dict(
        nivel="N3", nome="banco_simplificado",
        pedido="Escreva a funcao Python `banco_simplificado(operacoes)` (use exatamente esses nomes de funcao e parametro) que processa "
               "uma lista de strings contendo instrucoes de um banco ficticio: "
               "'CRIAR conta' (cria uma conta com saldo 0, se nao existir), "
               "'DEPOSITAR conta valor' (deposita se a conta existir), "
               "'SACAR conta valor' (saca se a conta existir e tiver saldo suficiente), "
               "'TRANSFERIR origem destino valor' (transfere o valor da conta origem "
               "para a conta destino se ambas existirem e a origem tiver saldo suficiente). "
               "Retorne um dicionario mapeando cada conta existente ao seu saldo final "
               "(contas sem operacoes validas de criacao nao entram no dicionario). "
               "Todos os valores sao inteiros positivos.",
        testes="""
assert banco_simplificado(["CRIAR c1", "DEPOSITAR c1 100", "CRIAR c2", "TRANSFERIR c1 c2 40", "SACAR c1 10"]) == {"c1": 50, "c2": 40}
assert banco_simplificado(["DEPOSITAR c1 50"]) == {}
assert banco_simplificado([]) == {}
""",
    ),
    *_N4_NOVOS,
]


def gerar(modelo, pedido):
    """Chama o ollama. Sem streaming: so interessa a saida final e o tempo."""
    corpo = json.dumps({
        "model": modelo,
        "prompt": pedido + "\n\nResponda APENAS com o codigo Python da funcao, "
                           "dentro de um bloco ```python. Sem explicacao.",
        "stream": False,
        # O ollama devolve o raciocinio num campo SEPARADO ("thinking"), nao
        # dentro de <think> no texto. Mas o num_predict conta os dois juntos:
        # o modelo gastava o orcamento inteiro pensando e "response" chegava
        # VAZIA -> NameError na hora de testar, e a bancada media o meu limite
        # em vez da capacidade dele. Desligar o raciocinio resolve na raiz e
        # ainda deixa a comparacao justa (todos respondem direto).
        "think": False,
        "options": {"temperature": 0, "num_predict": 3000},
    })
    t0 = time.time()
    r = subprocess.run(
        ["curl", "-s", "-m", str(TEMPO_LIMITE), "http://127.0.0.1:11434/api/generate",
         "-d", corpo],
        capture_output=True, text=True)
    dur = time.time() - t0
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return "", dur, 0
    saida = d.get("response", "")
    # Alguns modelos ainda embutem <think> no texto mesmo com think=False.
    saida = re.sub(r"<think>.*?</think>", "", saida, flags=re.S)
    if not saida.strip():
        # Distingue "escreveu codigo errado" de "nao escreveu codigo nenhum".
        # Sem isso os dois viram o mesmo ERR e a medicao mente — aconteceu 2x.
        print(f"    ! resposta VAZIA (motivo={d.get('done_reason')}, "
              f"pensou={len(d.get('thinking') or '')} chars)")
    return saida, dur, d.get("eval_count", 0)


def extrai_codigo(saida):
    blocos = re.findall(r"```(?:python)?\s*(.*?)```", saida, re.S)
    return (blocos[0] if blocos else saida).strip()


def testa(codigo, testes):
    """Roda num processo separado: codigo de modelo pequeno trava em loop as
    vezes, e um exec() aqui derrubaria a bancada inteira."""
    script = codigo + "\n\n" + testes + "\nprint('PASSOU')\n"
    try:
        r = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False, "travou (loop infinito?)"
    if "PASSOU" in r.stdout:
        return True, ""
    err = (r.stderr.strip().splitlines() or ["sem erro"])[-1]
    return False, err[:110]


def main():
    # `python3 teto.py granite4:micro` roda so um modelo. Sem isto a bancada
    # tentava o qwen2.5-coder:3b, que foi apagado do ollama a pedido do dono, e
    # gastava a rodada inteira pra devolver ERR em tudo.
    modelos = sys.argv[1:] or MODELOS
    placar = {}
    for modelo in modelos:
        print(f"\n{'='*66}\n{modelo}\n{'='*66}")
        acertos, tempo_total, tokens = 0, 0.0, 0
        por_nivel = {}
        resultados = []
        for p in PROBLEMAS:
            saida, dur, ntok = gerar(modelo, p["pedido"])
            tempo_total += dur
            tokens += ntok
            codigo = extrai_codigo(saida)
            ok, err = testa(codigo, p["testes"])
            acertos += ok
            nivel = por_nivel.setdefault(p["nivel"], {"acertos": 0, "total": 0})
            nivel["total"] += 1
            nivel["acertos"] += int(ok)
            resultados.append({
                "nivel": p["nivel"],
                "nome": p["nome"],
                "ok": bool(ok),
                "seg": round(dur, 1),
                "tokens": ntok,
                "erro": err,
                "resposta_vazia": not saida.strip(),
                "codigo_extraido": codigo,
            })
            marca = "OK " if ok else "ERR"
            print(f"  [{marca}] {p['nivel']} {p['nome']:20s} {dur:6.1f}s  {err}")
        vel = tokens / tempo_total if tempo_total else 0
        placar[modelo] = dict(acertos=acertos, total=len(PROBLEMAS),
                              por_nivel=por_nivel, seg=round(tempo_total, 1),
                              tok_s=round(vel, 1), resultados=resultados)
        print(f"  --> {acertos}/{len(PROBLEMAS)} · {tempo_total:.0f}s no total · "
              f"{vel:.1f} tok/s")

    print(f"\n{'#'*66}\n# PLACAR (N2 = basico, N3 = varios passos)\n{'#'*66}")
    for m, r in placar.items():
        print(f"  {m:34s} {r['acertos']}/{r['total']}  {r['tok_s']:>5.1f} tok/s")
    json.dump(placar, open(AQUI / "resultado_teto.json", "w"), indent=2)


if __name__ == "__main__":
    main()
