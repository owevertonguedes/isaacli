"""Dataset pra ensinar o Qwen2.5-Coder a emitir <tool_call>, nao ```json.

O modelo JA acerta o conteudo (nome da funcao e argumentos corretos, verificado).
Erra so o involucro. Entao o dataset nao ensina raciocinio nenhum — ensina uma
gramatica de saida. Por isso da pra ser pequeno e ainda funcionar.

Formato: exatamente o chat template do Qwen2.5, senao o treino ensina o formato errado.
"""
import json
import random

FERRAMENTAS = [
    {"name": "read_file", "description": "Le o conteudo de um arquivo de texto.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}},
    {"name": "write_file", "description": "Escreve (sobrescreve) um arquivo de texto.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                     "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "append_file", "description": "Acrescenta texto no fim de um arquivo.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                     "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "list_dir", "description": "Lista arquivos e pastas de um diretorio.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": []}},
    {"name": "run_command", "description": "Executa um comando de shell e devolve a saida.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]}},
    {"name": "git_status", "description": "Mostra o status do repositorio git.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]

ARQUIVOS = ["notas.txt", "config.json", "README.md", "index.html", "app.py",
            "estilo.css", "dados.csv", "lista.txt", "main.js", "TODO.md"]
PASTAS = [".", "src", "jogos", "docs", "testes"]
TEXTOS = ["ola mundo", "linha nova", "projeto iniciado", "TODO: revisar",
          "versao 2", "concluido", "nome: Isaac", "status: ok"]
COMANDOS = ["ls -la", "git status --short", "python3 app.py", "npm test",
            "git add .", "cat README.md", "pwd", "git log --oneline -5"]

# (pedido do usuario, ferramenta, argumentos) — variados no jeito de pedir,
# porque o modelo precisa generalizar o FORMATO, nao decorar frases.
def exemplos():
    ex = []
    for a in ARQUIVOS:
        ex.append((random.choice([
            f"Leia o arquivo {a}", f"me mostre o conteudo de {a}",
            f"o que tem dentro de {a}?", f"abre o {a} pra mim",
            f"da uma olhada em {a}"]), "read_file", {"path": a}))
    for a in ARQUIVOS:
        t = random.choice(TEXTOS)
        ex.append((random.choice([
            f"Crie o arquivo {a} com o texto: {t}",
            f"escreva '{t}' em {a}", f"salve {a} contendo {t}",
            f"gera um {a} com o conteudo {t}"]), "write_file", {"path": a, "content": t}))
    for a in ARQUIVOS[:6]:
        t = random.choice(TEXTOS)
        ex.append((random.choice([
            f"Acrescente '{t}' no fim de {a}", f"adiciona a linha {t} em {a}",
            f"poe {t} no final do {a}"]), "append_file", {"path": a, "content": t}))
    for p in PASTAS:
        ex.append((random.choice([
            f"Liste os arquivos de {p}", f"o que tem na pasta {p}?",
            f"mostra o conteudo do diretorio {p}"]), "list_dir", {"path": p}))
    for c in COMANDOS:
        ex.append((random.choice([
            f"Execute: {c}", f"roda o comando {c}", f"pode rodar {c}?"]),
            "run_command", {"cmd": c}))
    for _ in range(6):
        ex.append((random.choice([
            "Qual o status do git?", "mostra o git status",
            "tem coisa pra commitar?", "o repositorio esta limpo?"]), "git_status", {}))
    return ex


SYS = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools>:
<tools>
{ferramentas}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def montar(seed=0):
    random.seed(seed)
    linhas = []
    for pedido, fn, args in exemplos():
        # Mostra um subconjunto de ferramentas por exemplo (inclusive a certa),
        # pra ele nao aprender "a lista sempre e essa".
        outras = [f for f in FERRAMENTAS if f["name"] != fn]
        random.shuffle(outras)
        conjunto = [f for f in FERRAMENTAS if f["name"] == fn] + outras[:random.randint(2, 4)]
        random.shuffle(conjunto)
        sig = "\n".join(json.dumps({"type": "function", "function": f}, ensure_ascii=False)
                        for f in conjunto)
        chamada = json.dumps({"name": fn, "arguments": args}, ensure_ascii=False)
        texto = (
            f"<|im_start|>system\n{SYS.format(ferramentas=sig)}<|im_end|>\n"
            f"<|im_start|>user\n{pedido}<|im_end|>\n"
            f"<|im_start|>assistant\n<tool_call>\n{chamada}\n</tool_call><|im_end|>"
        )
        linhas.append({"text": texto})
    return linhas


if __name__ == "__main__":
    dados = montar()
    with open("treino.jsonl", "w") as f:
        for d in dados:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(dados)} exemplos -> treino.jsonl")
    print("\n--- amostra ---")
    print(dados[0]["text"][-400:])
