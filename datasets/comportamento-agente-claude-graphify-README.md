# Dataset comportamento-agente-claude-graphify

Arquivo:

```text
datasets/comportamento-agente-claude-graphify-2026-07-20.jsonl
```

Objetivo: primeiro pacote curado usando o historico/fluxo do Claude Code,
Graphify e aprendizados locais do projeto para ensinar comportamento de agente,
nao conhecimento bruto.

## Conteudo

```text
24 exemplos
30 tool-calls alvo
```

Cobertura:

- ler `CONTEXTO.md`/tasks antes de agir;
- nao executar pergunta exploratoria;
- verificar antes de declarar sucesso;
- registrar relatorios grandes em `reports/`;
- nao enviar historico bruto para nuvem;
- nao repetir ciclo T4 sem dataset novo;
- manter adapter separado sem merge;
- usar erro cru para corrigir tentativa;
- consultar Graphify antes de editar;
- cair para busca local quando nao ha Graphify;
- reforcar commit assinado;
- reforcar persistencia/task05;
- responder com medicao, nao impressao.

## Conversa vs codigo

Este primeiro pacote **nao** treinou em cima de codigo bruto dos projetos nem em
arquivos inteiros. Ele contem principalmente exemplos de conversa/decisao e
alguns snippets pequenos dentro de tool-calls:

- miolos de funcoes da task05 (`saveTransactions`, `loadTransactions`,
  `deleteTransaction`);
- trechos CSS/JS pequenos entre marcadores;
- comandos Git canônicos;
- comandos Graphify canônicos.

Ou seja: o Granite viu exemplos de codigo pequenos e verificaveis, mas nao viu a
"arvore genealogica" inteira dos projetos como corpus de codigo. Para o proximo
pacote, se quisermos ensinar mais codigo, usar somente micro-edicoes com contexto
minimo e teste/checagem esperada, nunca arquivo real inteiro.

## Primeira medicao

Treinado no run:

```text
reports/lora/claude-graphify-s160/report.json
lora_runs/oficina/claude-graphify-s160-lite.tgz
```

Resultado:

```text
task05_save: 1/5 -> 5/5
commit_signature: 2/4 -> 2/4
intent_question: 3/5 -> 3/5
graphify_navigation: 2/5 -> 3/5
```

Veredito: **nao aprovado**. O pacote mostrou que Graphify entrou parcialmente no
comportamento, mas o adapter ainda gerou comando errado
`graphify-out/graph.json find...`, nao `graphify query ...`, e no fluxo de commit
emitiu tool-call JSON malformado com ferramenta inexistente.

## Proxima correcao

Antes de novo treino:

- adicionar checagem de JSON/tool-call valido na avaliacao;
- reforcar exemplos canônicos de `graphify query "..."`
- reforcar exemplos de commit com trailer Git e `git log -1 --format=%B`;
- reduzir exemplos vagos que ensinem texto sem acao verificavel;
- manter dataset pequeno, mas com alvos mais consistentes.
