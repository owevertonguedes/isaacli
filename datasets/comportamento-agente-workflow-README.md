# Dataset comportamento-agente-workflow

Arquivo ativo:

```text
datasets/comportamento-agente-workflow-2026-07-20.jsonl
```

Este dataset substitui no pacote ativo os exemplos antigos que tratavam
assinatura textual de commit como requisito padrao. A decisao atual e:

- autoria/coautoria e responsabilidade estrutural do Isaac CLI/app;
- o modelo aprende fluxo: interpretar pedido, escolher ferramenta valida,
  montar mensagem, respeitar push proibido/permitido e verificar sucesso;
- assinatura textual so aparece quando o usuario pede literalmente assinatura no
  texto/titulo/corpo da mensagem.

Pacote ativo em 2026-07-20:

```text
task05-andaime-marcadores-gemini-2026-07-20.jsonl
intencao-pergunta-vs-execucao-isaac-2026-07-20.jsonl
comportamento-agente-workflow-2026-07-20.jsonl
```

Validar antes de treino:

```bash
python3 tool_harness/validar_datasets_lora.py
```

Resultado atual:

```text
rows: 23
issues: 0
```

Corrida medida:

```text
workflow-s80
```

Veredito: nao aprovado. Ver:

```text
reports/lora/workflow-s80/report.json
reports/lora/workflow-s80/reavaliacao-metrica.md
```
