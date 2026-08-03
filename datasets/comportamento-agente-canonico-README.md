# Dataset comportamento-agente-canonico

Arquivo ativo:

```text
datasets/comportamento-agente-canonico-2026-07-20.jsonl
```

Este dataset substitui, no pacote de treino ativo, o dataset
`comportamento-agente-claude-graphify-2026-07-20.jsonl`, que fica preservado
como tentativa reprovada.

Motivo: a corrida `claude-graphify-s160` melhorou pouco o fluxo Graphify, mas
continuou `2/4` em commit e gerou tool-call JSON malformado com ferramenta
inexistente. O alvo canônico ensina diretamente:

- commit assinado como trailer Git textual (`Co-Authored-By` ou `Signed-off-by`);
- verificação obrigatória com `git log -1 --format=%B` antes de declarar sucesso;
- proibição de `git commit -S` quando o pedido é assinatura textual do Isaac;
- `graphify query "... " --graph graphify-out/graph.json --budget 700`;
- pergunta exploratória sem execução de comandos;
- edição entre marcadores para task 05.

Antes de qualquer novo ciclo T4, validar:

```bash
python3 tool_harness/validar_datasets_lora.py
```

Relatório esperado:

```text
reports/datasets/lora-datasets-ativos-2026-07-20.json
```
