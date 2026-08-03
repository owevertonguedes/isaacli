# Dataset: intencao pergunta vs execucao

Data: 2026-07-20.

Objetivo: ensinar o Isaac a nao executar comandos quando o usuario esta apenas
fazendo uma pergunta exploratoria.

Sessao real que motivou o dataset:

```text
tool_harness/cli_sessoes/2026-07-20-142104-047ad3.jsonl
```

Falha observada:

- pedido do usuario: perguntou que tipo de minijogo o Isaac seria capaz de fazer;
- comportamento ruim: Isaac rodou `git status`, tentou `git init` e tentou
  instalar pacotes com `pip`;
- resultado: 7 comandos, 6 falhas.

Regra que o dado deve ensinar:

- Pergunta exploratoria nao autoriza comando.
- Quando o usuario pergunta "o que voce faria?", "que tipo voce consegue fazer?"
  ou "da para fazer?", responda em texto e aguarde escolha.
- So crie arquivos quando o usuario pedir explicitamente para criar/fazer/gerar.
- Para minijogo local em HTML/CSS/JS, nao usar `pip`, `npm`, internet nem
  `git init` sem pedido explicito.

Este dataset ainda nao prova aprendizado nos pesos. Ele e uma semente para
SFT/LoRA junto com os datasets de task 05 e commit assinado.
