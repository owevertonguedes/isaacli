# Dataset: commit com assinatura do Isaac

Data: 2026-07-20.

Objetivo estreito: ensinar/avaliar se o Isaac interpreta pedidos de commit
assinado sem o usuario explicar comandos Git.

Formato correto para esta bancada:

```text
Co-Authored-By: Isaac <isaac-local@localhost>
Signed-off-by: Isaac <isaac-local@localhost>
```

O email pode variar; o que importa é ser trailer Git em linha própria no corpo,
com `Isaac <email>`. `Isaac (AI)` no título não conta. `git commit -S` também não
conta quando o pedido é assinatura textual do Isaac; `-S` é assinatura GPG do
usuário.

Por que este dataset existe:

- `2026-07-19-235850-c1daf7`: o Isaac fez `status` -> `add` -> `commit`, mas
  colocou `Isaac (AI)` no titulo em vez de trailer verificavel.
- `2026-07-20-001036-f1c0b5`: tentou `Co-Authored-By`, mas sem aspas; o sandbox
  recusou `<` como operador sem shell. Depois fez commit sem assinatura e tentou
  `git commit -S`, confundindo assinatura textual com GPG.
- `2026-07-20-001145-b907da`: mesmo com regra curta no system prompt, fez commit
  com corpo mas ainda sem trailer e sem `git log` de verificação.

Regra que o dado deve ensinar:

- Para commit novo assinado: `git commit -m "Titulo" -m "Motivo" -m
  "Co-Authored-By: Isaac <email>"` ou `git commit -m "Titulo" -m "Motivo" -m
  "Signed-off-by: Isaac <email>"`.
- Para corrigir commit ja criado: usar `git commit --amend` com a mesma estrutura
  de `-m`.
- Depois do commit, verificar `git log -1 --format=%B`.
- Se algum comando falhar, a resposta final deve dizer que falhou; nao pode
  declarar sucesso.

Este arquivo ainda nao prova aprendizado nos pesos. E dado curado + bancada
minima. Se o prompt/andaime continuar falhando, o proximo passo e treinar LoRA ou
criar uma ferramenta de commit assinado que reduza a superficie de erro.
