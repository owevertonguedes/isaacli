"""Os 3 N4 novos da bancada — regras INVENTADAS, nao LeetCode (task 03).

O DEFEITO QUE ISTO CONSERTA
---------------------------
Os N4 antigos eram `sudoku_solver`, `match_wildcard` e `calculadora_completa`:
LeetCode classico, decorado no pre-treino. O granite4:micro fazia 3/3 neles e ao
mesmo tempo errava `formata_moeda` no N2. A escada estava invertida — o N4 media
o que o modelo VIU, nao o que ele RACIOCINA.

O CRITERIO DE "ORIGINAL"
------------------------
Nao e conceito novo (nao existe algoritmo inedito num teste de 30 linhas). E
COMBINACAO ARBITRARIA DE REGRAS: um domino inventado onde a especificacao e a
unica fonte da resposta. O modelo nao pode recuperar isto da memoria porque nao
existe em lugar nenhum — ele tem que ler a regra e seguir. O sinal de que esta
certo e nao dar pra achar no Google.

O RISCO DESTE CAMINHO, E COMO ELE FOI TRATADO
---------------------------------------------
Regra inventada corre o risco de ficar AMBIGUA, e ambiguidade reprova modelo
certo — o erro mais caro deste projeto. Contramedida: cada regra que o teste
cobre esta dita explicitamente no enunciado, inclusive as chatas (empate, lista
vazia, o que levanta erro). Se um assert cobre um comportamento, o enunciado
precisa declarar aquele comportamento. Nada de "e obvio".

Cada problema vem com GABARITO (tem que passar) e uma solucao INGENUA (tem que
falhar) — validados por `validar_n4.py`.
"""

PROBLEMAS_N4 = [
    dict(
        nivel="N4", nome="ratear_custo",
        pedido=(
            "Escreva a funcao Python `ratear_custo(pesos, total)` (use exatamente esses "
            "nomes de funcao e parametros).\n"
            "`pesos` e um dict que mapeia nome (string) para peso (inteiro). `total` e um "
            "inteiro em centavos que deve ser dividido INTEIRAMENTE entre os nomes, "
            "proporcionalmente ao peso de cada um. A funcao devolve um dict nome -> "
            "centavos (inteiros), e a soma dos valores devolvidos tem que ser exatamente "
            "igual a `total`.\n"
            "Regras, nesta ordem:\n"
            "1. Cada nome recebe primeiro a parte inteira (arredondada PARA BAIXO) de "
            "total * peso / soma_dos_pesos.\n"
            "2. Sobra a diferenca entre `total` e a soma dessas partes inteiras. Distribua "
            "essa sobra de 1 em 1 centavo, dando cada centavo ao nome com o maior RESTO "
            "fracionario da divisao do passo 1. Em caso de empate no resto, o centavo vai "
            "para o nome menor em ordem alfabetica. Cada nome pode receber no maximo 1 "
            "centavo de sobra.\n"
            "3. Nome com peso 0 recebe 0 e NAO participa da distribuicao da sobra, mesmo "
            "que ainda haja centavos para distribuir.\n"
            "4. Se `pesos` for vazio, se a soma dos pesos for 0, se algum peso for "
            "negativo, ou se `total` for negativo, levante ValueError.\n"
            "5. Se `total` for 0, devolva todos os nomes com valor 0."
        ),
        testes='''
# proporcional simples, sem sobra
assert ratear_custo({"a": 1, "b": 1}, 10) == {"a": 5, "b": 5}

# sobra de 1 centavo vai pro maior resto: x tem resto 4/5, y tem 1/5
assert ratear_custo({"x": 2, "y": 3}, 7) == {"x": 3, "y": 4}

# empate no resto (todos 1/3) -> desempate alfabetico, so 1 centavo cada
assert ratear_custo({"a": 1, "b": 1, "c": 1}, 10) == {"a": 4, "b": 3, "c": 3}

# 2 de sobra, empate triplo -> os dois alfabeticamente menores
assert ratear_custo({"a": 1, "b": 1, "c": 1}, 11) == {"a": 4, "b": 4, "c": 3}

# peso 0 fica de fora da sobra mesmo havendo centavo pra distribuir
assert ratear_custo({"a": 0, "b": 1, "c": 1}, 3) == {"a": 0, "b": 2, "c": 1}

# a soma tem que fechar com o total, sempre
r = ratear_custo({"p": 7, "q": 11, "r": 13}, 1000)
assert sum(r.values()) == 1000

r = ratear_custo({"um": 1, "dois": 2, "tres": 3, "quatro": 4}, 33)
assert sum(r.values()) == 33

# total 0
assert ratear_custo({"a": 5, "b": 5}, 0) == {"a": 0, "b": 0}

# erros
for entrada in [({}, 10), ({"a": 0}, 10), ({"a": -1, "b": 2}, 10), ({"a": 1}, -5)]:
    try:
        ratear_custo(*entrada)
        assert False, "deveria ter levantado ValueError para " + repr(entrada)
    except ValueError:
        pass
''',
    ),
    dict(
        nivel="N4", nome="simular_esteira",
        pedido=(
            "Escreva a funcao Python `simular_esteira(n, comandos)` (use exatamente esses "
            "nomes de funcao e parametros), que simula um braco mecanico sobre uma esteira "
            "CIRCULAR.\n"
            "A esteira tem `n` posicoes numeradas de 0 a n-1, todas vazias no inicio. O "
            "braco comeca na posicao 0, com a mao vazia, e com o sentido de movimento "
            "igual a +1 (ou seja, andar aumenta o indice).\n"
            "`comandos` e uma lista de strings. Os comandos sao:\n"
            "- 'POE X' : coloca na posicao atual um item cujo rotulo e X (a parte depois "
            "do espaco). Se a posicao atual ja tiver um item, levante ValueError.\n"
            "- 'ANDA k': anda k posicoes (k inteiro >= 0) no sentido atual, dando a volta "
            "na esteira (circular).\n"
            "- 'GIRA'  : inverte o sentido atual (de +1 para -1, ou de -1 para +1).\n"
            "- 'PEGA'  : se a mao ja estiver cheia, levante ValueError. Se a posicao atual "
            "estiver vazia, nao faca nada. Caso contrario, tire o item da esteira e "
            "coloque na mao.\n"
            "- 'SOLTA' : se a mao estiver vazia, levante ValueError. Se a posicao atual "
            "estiver livre, deixe o item ali. Se estiver ocupada, procure a proxima "
            "posicao livre andando no SENTIDO ATUAL a partir da posicao atual e deixe o "
            "item na primeira que encontrar; o braco NAO se move nesse caso. Se nao "
            "houver nenhuma posicao livre na esteira inteira, levante ValueError.\n"
            "Qualquer comando que nao seja um desses levanta ValueError.\n"
            "A funcao devolve a tupla (posicao_do_braco, item_na_mao, esteira), onde "
            "item_na_mao e None se a mao estiver vazia, e esteira e uma lista de n "
            "elementos com o rotulo de cada posicao ou None se estiver vazia."
        ),
        testes='''
# basico: poe, anda, poe
assert simular_esteira(3, ["POE a", "ANDA 1", "POE b"]) == (1, None, ["a", "b", None])

# circular: andar mais que o tamanho da a volta
assert simular_esteira(3, ["ANDA 4"]) == (1, None, [None, None, None])

# GIRA inverte, e o indice da a volta pra tras
assert simular_esteira(3, ["GIRA", "ANDA 1"]) == (2, None, [None, None, None])

# pegar e segurar
assert simular_esteira(2, ["POE a", "PEGA"]) == (0, "a", [None, None])

# PEGA em posicao vazia nao faz nada (nao e erro)
assert simular_esteira(2, ["PEGA"]) == (0, None, [None, None])

# SOLTA em posicao ocupada procura a proxima livre NO SENTIDO ATUAL, e o braco
# NAO se move. Estes dois casos sao identicos menos pelo sentido, de proposito:
# quem ignorar o sentido na busca acerta um e erra o outro.
#
# esteira [a, b, _, _], braco volta pra 0 segurando d, sentido +1:
# 0 ocupado, 1 ocupado, 2 livre -> d vai pra 2
assert simular_esteira(4, ["POE a", "ANDA 1", "POE b", "ANDA 2", "POE d",
                           "PEGA", "ANDA 1", "SOLTA"]) \\
       == (0, None, ["a", "b", "d", None])

# mesma coisa, mas com GIRA antes do SOLTA: andando pra tras a partir do 0,
# a primeira livre e a 3
assert simular_esteira(4, ["POE a", "ANDA 1", "POE b", "ANDA 2", "POE d",
                           "PEGA", "ANDA 1", "GIRA", "SOLTA"]) \\
       == (0, None, ["a", "b", None, "d"])

# esteira cheia e mao cheia -> nao ha onde soltar
try:
    simular_esteira(2, ["POE a", "ANDA 1", "POE b", "PEGA", "ANDA 1", "POE c", "SOLTA"])
    assert False, "deveria ter levantado ValueError (esteira sem posicao livre)"
except ValueError:
    pass

# POE em posicao ocupada
try:
    simular_esteira(2, ["POE a", "POE b"])
    assert False, "deveria ter levantado ValueError (posicao ocupada)"
except ValueError:
    pass

# PEGA com a mao cheia
try:
    simular_esteira(2, ["POE a", "PEGA", "ANDA 1", "POE b", "PEGA"])
    assert False, "deveria ter levantado ValueError (mao cheia)"
except ValueError:
    pass

# SOLTA com a mao vazia
try:
    simular_esteira(2, ["SOLTA"])
    assert False, "deveria ter levantado ValueError (mao vazia)"
except ValueError:
    pass

# comando desconhecido
try:
    simular_esteira(2, ["VOA 3"])
    assert False, "deveria ter levantado ValueError (comando desconhecido)"
except ValueError:
    pass

# lista de comandos vazia
assert simular_esteira(2, []) == (0, None, [None, None])
''',
    ),
    dict(
        nivel="N4", nome="ler_plano",
        pedido=(
            "Escreva a funcao Python `ler_plano(texto)` (use exatamente esses nomes de "
            "funcao e parametro) que interpreta um formato de arquivo proprio chamado "
            "PLANO e devolve um dict.\n"
            "O texto e composto por blocos. Um bloco comeca numa linha "
            "'tarefa: NOME' (sem recuo) e continua nas linhas seguintes que estiverem "
            "RECUADAS por espacos, no formato 'chave: valor'. Sao aceitas exatamente duas "
            "chaves recuadas: 'duracao' e 'depende'.\n"
            "Regras:\n"
            "- 'duracao' vem no formato de tempo: '3h', '45m' ou '1h30m'. Converta para um "
            "total em minutos (inteiro). Se a tarefa nao declarar duracao, use 0.\n"
            "- 'depende' e uma lista de nomes separados por virgula; apare os espacos em "
            "volta de cada nome. Se a tarefa nao declarar depende, ou se o valor for vazio, "
            "use lista vazia. Preserve a ordem em que os nomes aparecem.\n"
            "- Uma linha cujo primeiro caractere nao-branco e '#' e um comentario e deve "
            "ser ignorada por completo. Linhas em branco tambem sao ignoradas.\n"
            "- Uma linha que termina com o caractere '\\\\' continua na linha seguinte: "
            "junte as duas removendo a '\\\\' e o recuo da linha seguinte, colocando um "
            "unico espaco entre elas.\n"
            "- Se o mesmo nome de tarefa aparecer duas vezes, levante ValueError.\n"
            "- Se alguma tarefa depender de um nome que nao e uma tarefa declarada no "
            "texto, levante ValueError.\n"
            "- Se aparecer uma chave recuada diferente de 'duracao' ou 'depende', levante "
            "ValueError. Se aparecer uma linha recuada antes de qualquer 'tarefa:', "
            "levante ValueError.\n"
            "A funcao devolve um dict que mapeia o nome da tarefa para um dict com as "
            "chaves 'duracao_min' (inteiro) e 'depende' (lista de strings)."
        ),
        testes='''
p = ler_plano("""tarefa: comprar
  duracao: 30m
tarefa: lixar
  duracao: 1h30m
  depende: comprar
tarefa: pintar
  duracao: 3h
  depende: comprar, lixar
""")
assert p == {
    "comprar": {"duracao_min": 30, "depende": []},
    "lixar": {"duracao_min": 90, "depende": ["comprar"]},
    "pintar": {"duracao_min": 180, "depende": ["comprar", "lixar"]},
}

# tarefa sem nenhuma chave recuada
assert ler_plano("tarefa: sozinha\\n") == {"sozinha": {"duracao_min": 0, "depende": []}}

# comentarios e linhas em branco sao ignorados, inclusive comentario recuado
assert ler_plano("""# plano de teste

tarefa: a
  # este comentario nao conta
  duracao: 2h

""") == {"a": {"duracao_min": 120, "depende": []}}

# continuacao de linha com barra invertida
p = ler_plano("""tarefa: a
tarefa: b
tarefa: c
  depende: a, \\\\
    b
""")
assert p["c"]["depende"] == ["a", "b"]

# depende vazio
assert ler_plano("tarefa: a\\n  depende:\\n")["a"]["depende"] == []

# texto vazio
assert ler_plano("") == {}

# tarefa duplicada
try:
    ler_plano("tarefa: a\\ntarefa: a\\n")
    assert False, "deveria ter levantado ValueError (tarefa duplicada)"
except ValueError:
    pass

# dependencia que nao existe
try:
    ler_plano("tarefa: a\\n  depende: fantasma\\n")
    assert False, "deveria ter levantado ValueError (dependencia inexistente)"
except ValueError:
    pass

# chave recuada desconhecida
try:
    ler_plano("tarefa: a\\n  cor: azul\\n")
    assert False, "deveria ter levantado ValueError (chave desconhecida)"
except ValueError:
    pass

# linha recuada antes de qualquer tarefa
try:
    ler_plano("  duracao: 1h\\n")
    assert False, "deveria ter levantado ValueError (recuo sem tarefa)"
except ValueError:
    pass
''',
    ),
]


# --- gabaritos: a solucao correta. Se um destes falhar, o problema esta errado.

GABARITOS = {
    "ratear_custo": '''
def ratear_custo(pesos, total):
    if not pesos:
        raise ValueError("pesos vazio")
    if total < 0:
        raise ValueError("total negativo")
    if any(p < 0 for p in pesos.values()):
        raise ValueError("peso negativo")
    soma = sum(pesos.values())
    if soma == 0:
        raise ValueError("soma dos pesos e zero")

    saida = {}
    restos = []
    for nome, peso in pesos.items():
        bruto = total * peso
        saida[nome] = bruto // soma
        if peso > 0:
            restos.append((bruto % soma, nome))

    sobra = total - sum(saida.values())
    # maior resto primeiro; empate -> nome alfabeticamente menor
    restos.sort(key=lambda r: (-r[0], r[1]))
    for _, nome in restos[:sobra]:
        saida[nome] += 1
    return saida
''',

    "simular_esteira": '''
def simular_esteira(n, comandos):
    esteira = [None] * n
    pos = 0
    mao = None
    sentido = 1

    for cmd in comandos:
        if cmd == "GIRA":
            sentido = -sentido
        elif cmd == "PEGA":
            if mao is not None:
                raise ValueError("mao cheia")
            if esteira[pos] is not None:
                mao = esteira[pos]
                esteira[pos] = None
        elif cmd == "SOLTA":
            if mao is None:
                raise ValueError("mao vazia")
            destino = None
            for passo in range(n):
                p = (pos + sentido * passo) % n
                if esteira[p] is None:
                    destino = p
                    break
            if destino is None:
                raise ValueError("sem posicao livre")
            esteira[destino] = mao
            mao = None
        elif cmd.startswith("POE "):
            if esteira[pos] is not None:
                raise ValueError("posicao ocupada")
            esteira[pos] = cmd[4:]
        elif cmd.startswith("ANDA "):
            k = int(cmd[5:])
            if k < 0:
                raise ValueError("k negativo")
            pos = (pos + sentido * k) % n
        else:
            raise ValueError("comando desconhecido: " + cmd)

    return (pos, mao, esteira)
''',

    "ler_plano": '''
def ler_plano(texto):
    # 1) junta as continuacoes de linha
    linhas = []
    pendente = None
    for bruta in texto.split("\\n"):
        parte = bruta.rstrip("\\n")
        if pendente is not None:
            parte = pendente + " " + parte.strip()
            pendente = None
        if parte.rstrip().endswith("\\\\"):
            pendente = parte.rstrip()[:-1].rstrip()
            continue
        linhas.append(parte)
    if pendente is not None:
        linhas.append(pendente)

    plano = {}
    atual = None
    for linha in linhas:
        if not linha.strip():
            continue
        if linha.lstrip().startswith("#"):
            continue
        recuada = linha[:1].isspace()
        conteudo = linha.strip()

        if not recuada:
            if not conteudo.startswith("tarefa:"):
                raise ValueError("linha nao recuada que nao e tarefa: " + conteudo)
            nome = conteudo[len("tarefa:"):].strip()
            if nome in plano:
                raise ValueError("tarefa duplicada: " + nome)
            plano[nome] = {"duracao_min": 0, "depende": []}
            atual = nome
        else:
            if atual is None:
                raise ValueError("linha recuada antes de qualquer tarefa")
            if ":" not in conteudo:
                raise ValueError("linha recuada sem chave: " + conteudo)
            chave, _, valor = conteudo.partition(":")
            chave = chave.strip()
            valor = valor.strip()
            if chave == "duracao":
                plano[atual]["duracao_min"] = _minutos(valor)
            elif chave == "depende":
                plano[atual]["depende"] = [x.strip() for x in valor.split(",") if x.strip()]
            else:
                raise ValueError("chave desconhecida: " + chave)

    for nome, dados in plano.items():
        for dep in dados["depende"]:
            if dep not in plano:
                raise ValueError("dependencia inexistente: " + dep)
    return plano


def _minutos(txt):
    import re
    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?", txt.strip())
    if not m or (m.group(1) is None and m.group(2) is None):
        raise ValueError("duracao invalida: " + txt)
    horas = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    return horas * 60 + mins
''',
}


# --- solucoes INGENUAS: o atalho plausivel. Cada uma TEM que ser reprovada,
# senao o problema nao esta medindo raciocinio de varios passos.

INGENUAS = {
    # arredonda e reparte a sobra sem ordenar por resto, e esquece o peso 0
    "ratear_custo": '''
def ratear_custo(pesos, total):
    soma = sum(pesos.values())
    saida = {n: total * p // soma for n, p in pesos.items()}
    sobra = total - sum(saida.values())
    for nome in list(saida)[:sobra]:
        saida[nome] += 1
    return saida
''',

    # ignora que SOLTA procura a proxima livre no sentido atual
    "simular_esteira": '''
def simular_esteira(n, comandos):
    esteira = [None] * n
    pos, mao, sentido = 0, None, 1
    for cmd in comandos:
        if cmd == "GIRA":
            sentido = -sentido
        elif cmd == "PEGA":
            mao, esteira[pos] = esteira[pos], None
        elif cmd == "SOLTA":
            esteira[pos] = mao
            mao = None
        elif cmd.startswith("POE "):
            esteira[pos] = cmd[4:]
        elif cmd.startswith("ANDA "):
            pos = (pos + sentido * int(cmd[5:])) % n
    return (pos, mao, esteira)
''',

    # nao trata continuacao de linha nem valida dependencia
    "ler_plano": '''
def ler_plano(texto):
    plano, atual = {}, None
    for linha in texto.split("\\n"):
        if not linha.strip() or linha.strip().startswith("#"):
            continue
        if linha.startswith("tarefa:"):
            atual = linha.split(":", 1)[1].strip()
            plano[atual] = {"duracao_min": 0, "depende": []}
        else:
            chave, valor = linha.strip().split(":", 1)
            if chave == "duracao":
                plano[atual]["duracao_min"] = int(valor.strip().rstrip("hm"))
            elif chave == "depende":
                plano[atual]["depende"] = valor.strip().split(",")
    return plano
''',
}
