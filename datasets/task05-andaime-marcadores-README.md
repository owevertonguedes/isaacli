# Dataset task05-andaime-marcadores

Dataset pequeno para ensinar comportamento de tool-call em edicao por marcadores
no Isaac/Granite.

Origem:

- task: `tasks/pending/15-gerar-dataset-task05-com-gemini-aprovacao-exportacao.md`
- professor: `gemini-3.5-flash`
- data: 2026-07-20
- entrada enviada ao professor: resumo sanitizado da task 05, autorizado pelo
  usuario, contendo apenas nomes de ferramentas, marcadores, erros observados e
  exemplos genericos de HTML/JS.

Escopo ensinado:

- usar `replace_between` com `path`, `start_marker`, `end_marker` e `content`;
- chamar `checar_arquivo` depois do patch;
- escrever somente o miolo quando o marcador fica dentro de funcao existente;
- nao redeclarar `function saveTransactions`, `function loadTransactions` ou
  `function deleteTransaction`;
- nao usar placeholder como `javascript aqui`;
- nao colocar `<style>` dentro de marcador CSS;
- nao colocar `<script>`, `<body>` ou `<html>` dentro de marcador JS;
- finalizar com resposta curta depois de `checar_arquivo` retornar OK.

Curadoria:

- A saida bruta do professor foi revisada antes de entrar no dataset.
- Exemplos com biblioteca externa ou sem atualizacao correta de estado foram
  descartados/substituidos por versoes validas.
- Casos de recuperacao de erro nao incluem uma chamada errada emitida pelo
  assistant como alvo de SFT; o erro aparece no pedido do usuario, e o assistant
  emite apenas a chamada corrigida.

Estado:

Este dataset ainda nao prova aprendizado nos pesos. Ele e uma semente para SFT
ou LoRA posterior e precisa ser medido antes/depois na mesma task 05.
