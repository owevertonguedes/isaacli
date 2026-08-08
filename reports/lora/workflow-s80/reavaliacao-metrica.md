# Reavaliacao da metrica workflow-s80

Data: 2026-07-20.

Motivo: a primeira versao da avaliacao `commit_workflow` contava strings como
`git commit` e `git status` mesmo quando elas apareciam em texto explicativo,
fora de `<tool_call>`. Isso inflava o placar antes/depois.

Regra corrigida: comandos Git so contam quando aparecem dentro de tool-call
valido para `run_command`.

Resultado estrito sobre o `report.json` ja salvo:

```text
commit_workflow:
  antes:  5/9
  depois: 3/9

commit_literal_signature:
  antes:  4/7
  depois: 4/7
```

Leitura:

- `workflow-s80` nao e adapter aprovado;
- task05 continuou aprendendo, mas commit regrediu na metrica correta;
- o modelo voltou a emitir ferramenta inexistente (`git`) em vez de
  `run_command`;
- nao rodar novo ciclo em cima desse pacote sem ampliar exemplos de workflow e
  manter a avaliacao estrita.
