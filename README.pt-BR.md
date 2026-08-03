# local-llm-field-notes

*[Read in English](README.md).*

**Um registro público, com números, de um experimento sobre rodar agentes de
código em LLM local — em hardware fraco de propósito.**

Hardware: notebook com **GTX 1650 (4 GB VRAM)** e 15 GB de RAM. Não é máquina de
IA, e era esse o ponto — descobrir o que dá pra fazer com o que se tem, não com o
que seria ideal.

> **É um experimento, não um produto.** Não está em manutenção e não foi feito pra
> produção. O que tem valor aqui são as **medições** e os **erros de método
> documentados** — use como referência e copie os pedaços que servirem.
> Licença [AGPLv3](LICENSE): livre pra usar, estudar e modificar. Oferecer como
> serviço fechado exige licença comercial ([detalhes](LICENSING.md)).

---

## Pra que este repositório serve

A pergunta testada foi: *um LLM pequeno, rodando local, vira um agente de código
confiável sem depender de nuvem gerenciada?*

A resposta completa está abaixo, mas o uso prático deste repositório independe
dela. Ele serve para **quatro coisas concretas**, e provavelmente pelo menos uma
se aplica a você:

### 1. Decidir se vale a pena rodar local, sem gastar o fim de semana descobrindo

Os números de VRAM, velocidade e qualidade já estão medidos aqui. Se você está
com hardware parecido e pensando em montar isso, os
[achados técnicos](#achados-técnicos-que-custaram-tempo) respondem em cinco
minutos o que custou dias. Especialmente o penhasco de VRAM e o fato de que
**MoE economiza computação, não memória** — que é onde a maioria dos posts erra.

### 2. Um checklist de bugs de medição, aplicável a qualquer avaliação de LLM

Esta é a parte mais transferível, e **vale igual pra quem avalia modelo de
nuvem**. São seis modos de falha em que a *régua* mente, não o modelo — e todos
produzem exatamente o mesmo `ERR` na tela que uma falha real produziria.

Se você mantém um eval, um LLM-as-judge ou um benchmark interno, rode a
[lista](#o-aprendizado-mais-caro-a-régua-mente-antes-do-modelo) contra ele. É
barato e a chance de achar alguma coisa é alta.

### 3. Um sandbox pronto pra executar código de agente sem confiar nele

[`tool_harness/tools.py`](tool_harness/tools.py) e
[`isaac_cli.py`](tool_harness/isaac_cli.py) trazem contenção em três camadas
independentes: **execve direto** (sem shell, então não há injeção por `;` ou
`$()`), **allowlist curta** de binários, e **`bwrap`** com o disco todo
somente-leitura, rede fechada e só a pasta de trabalho gravável. Foi testado com
uma isca plantada fora da pasta, pra ver se escapava.

Isso é reaproveitável em qualquer projeto que execute código gerado por modelo —
local ou de nuvem — e é a parte do repositório que menos envelhece.

### 4. Um molde de bancada de avaliação que não se engana sozinha

[`bancada/`](bancada/) tem problemas com `assert` que **executam de verdade**, e —
a parte que importa — um validador que confere duas coisas em cada problema: que
o gabarito passa, **e que a solução ingênua é reprovada**. Uma régua que aprova a
solução ingênua não está medindo nada, e você só descobre isso testando a régua.

O mesmo vale pra [`validar_datasets_lora.py`](tool_harness/validar_datasets_lora.py),
que barra exemplo com ferramenta inexistente ou que declara sucesso sem evidência
— antes que ele entre no treino e vire alucinação nos pesos.

---

## A resposta que o experimento deu

**Sim para especialização e confiabilidade. Não para capacidade bruta.**

Com 4 GB de VRAM, o modelo é a peça que não se move: um 3B não vira um modelo de
fronteira, e isso é limite de escala, não de esforço. Boa parte do trabalho aqui
acabou sendo infraestrutura em volta de uma peça travada.

Mas a divisão que ficou clara — e que é o enquadramento mais útil do projeto todo:

> **Inteligência bruta você não coloca.** Vem do pré-treino, custa milhões, você
> baixa pronta. **Especialização e confiabilidade você coloca** — e isso está
> provado abaixo, com número.

Medir um projeto desses pela primeira coluna garante frustração. Pela segunda, ele
entrega.

---

## O que funcionou (com medição)

### Ensinar tool-call a um modelo que não sabia: 0/8 → 6/8

O `Qwen2.5-Coder-3B` parecia simplesmente incapaz de chamar ferramenta. Não era.

`<tool_call>` é o **token único `151657`**. Quem escolhe o token de saída é a
`lm_head` — e ela estava fora do `target_modules` do LoRA. O modelo estava sendo
treinado em tudo, menos na camada que decide emitir a tag. Com
`modules_to_save=["lm_head"]`, a loss caiu para ~1e-06 e a tag apareceu.

**Se você vai treinar tool-calling em LoRA, esse detalhe sozinho pode te poupar
dias.** E a lição generaliza: distinga *"o modelo não sabe"* de *"a camada que
serve não entende"*. Cave até a causa-raiz antes de descartar um modelo.

Código em [`qwen_tools_lora/`](qwen_tools_lora/).

### Destilação com portão mecânico: 22 pedidos → 16 aprovados

Um modelo maior gera exemplos, e um portão que **executa o código** decide se o
exemplo entra no dataset. Exemplo rejeitado é **descartado, nunca consertado à
mão** — erro que você conserta vira alucinação nos pesos, porque você ensinou o
modelo a produzir algo que ele não produziria sozinho.

Taxa de aprovação de ~73% é um número útil pra planejar: conte com perder um
terço do que o professor gerar.

### Andaime rende mais que modelo maior: `pass@1` 40% vs `pass@8` 75%

Essa lacuna é capacidade **já presente nos pesos** que não aparece numa tentativa
só. O teto é fixo *por passagem única*; o teto do **sistema** (modelo +
retentativa + verificador + decomposição em passos) não é.

Consequência prática, e talvez a mais acionável daqui: **quando o hardware é o
teto, investir em andaime rende mais que trocar de modelo.**

---

## O que não funcionou

Vale tanto quanto o resto — evita que você repita:

- **Nenhum adapter treinado foi aprovado.** O melhor subiu um fluxo de 1/5 para
  5/5 mas deixou outro em 4/6, ainda emitindo nome de ferramenta inexistente.
  Nada foi fundido no peso base. Os relatórios de cada corrida estão em
  [`reports/lora/`](reports/lora/), **inclusive os reprovados** — que é onde está
  a informação honesta.
- **Empilhar regra em system prompt não ensina modelo pequeno.** Medido: zero
  melhora. O que muda comportamento em modelo pequeno é LoRA e andaime externo,
  não instrução mais longa.
- **"Pedir pro modelo se revisar" não sobe capacidade.** `pass@1` foi de 40% para
  65%, mas `pass@8` foi de 75% para **77%**. Melhora confiabilidade, não o teto —
  o que ainda é útil, desde que você não confunda uma coisa com a outra.
- **Deixar o modelo "aprender sozinho lendo a internet"** nunca foi implementado,
  de propósito: model collapse e prompt injection. A versão que funciona é
  professor confiável + portão de qualidade automático antes do dado entrar.

---

## O aprendizado mais caro: a régua mente antes do modelo

Em um único dia apareceram **seis bugs de medição**. Cinco faziam o resultado
parecer *pior* do que era. O sexto fazia parecer *melhor*.

1. `skip_special_tokens=True` apagava `<tool_call>` da string que era avaliada.
   O modelo acertava; a régua não via.
2. Dataset ensinando ferramentas que não existiam → treina alucinação nos pesos.
3. Orçamento de saída consumido pelo raciocínio → resposta vazia com **HTTP 200**.
   Zero erro na tela, zero conteúdo.
4. Um caminho de resposta do Ollama devolveu o raciocínio em campo **separado**
   (`thinking`), não em `<think>`. A regex de limpeza nunca teve o que limpar.
5. Gabarito errado na bancada: `match_wildcard('acdcb', 'a*c?b')` estava marcado
   como `True`; o correto é `False`. **Um assert errado reprova um modelo certo.**
6. E o pior: a régua de avaliação de LoRA casava **substring sobre o texto
   gerado** em vez de executar o agente e conferir o estado final. Ela inflava o
   placar — e um juiz frouxo faz o modelo aprender a fraudar o juiz.

> **A regra que fica: quando o resultado for ruim, suspeite da régua antes do
> modelo.** `O modelo errou` e `eu medi errado` produzem o mesmo `ERR` na tela.
> Faça dump da saída crua antes de reportar qualquer veredito.

Se você levar uma única coisa daqui, que seja essa. Ela não tem nada a ver com
modelo local — vale pra qualquer avaliação de LLM que você faça.

---

## Achados técnicos que custaram tempo

- **Um desencontro Granite 4/Ollama entre endpoints apareceu no experimento, mas
  não reproduziu depois.** Na época, o `/api/chat` nativo devolvia tool calls e o
  compat OpenAI `/v1/chat/completions` devolvia `content=""` sem `tool_calls` e
  sem erro. Reteste em 2026-07-27 com Ollama `0.30.10` e o `granite4:micro` local
  devolveu tool calls nos dois endpoints. Trate isso como lição de diagnóstico —
  faça dump da resposta crua e registre versões — não como acusação de bug atual
  no Ollama.
- **Acima de ~4 GB de VRAM o desempenho não degrada, despenca.** Um modelo de
  6,6 GB rodou a **3,5 tok/s** nesta máquina — inutilizável. É penhasco, não
  ladeira. Planeje o modelo pelo que **cabe**, não pelo que é bom.
- **MoE economiza computação, não memória.** Os pesos todos ficam residentes. Um
  modelo anunciado por blog como "cabe em 16 GB" media 24 GB reais. Confira o
  tamanho do arquivo na fonte, não no post.
- **Arquitetura eficiente ganha de modelo maior quando o hardware é o teto.** Um
  modelo de 2,1 GB bem projetado (híbrido Mamba-2/Transformer) bateu um de 6,6 GB.
  Antes de pedir "modelo maior", pergunte se existe *modelo melhor projetado do
  mesmo tamanho*.
- **Destilação de raciocínio não transfere entre domínios.** Um modelo de 8B
  destilado para raciocínio fez **1/4** numa bancada de código — pior que um 3B.
- **Cuidado com capacidade que já vinha de fábrica.** Um dos modelos testados já
  chamava ferramenta nativamente (`ollama show` → `capabilities: tools`). O
  wrapper apenas *conectou*; não *ensinou* nada. **Não confunda integração com
  aprendizado** — é fácil se convencer de que você ensinou algo que já estava lá.

---

## Mapa do repositório

| pasta | o que tem |
|---|---|
| [`bancada/`](bancada/) | problemas de código com `assert` que executam, e o validador que testa o próprio gabarito |
| [`tool_harness/`](tool_harness/) | o agente: CLI, ferramentas, sandbox em três camadas, validador de dataset e os testes de cada peça |
| [`qwen_tools_lora/`](qwen_tools_lora/) | o treino de tool-calling que funcionou (0/8 → 6/8), com a correção da `lm_head` |
| [`datasets/`](datasets/) | 30 exemplos curados e validados, com README explicando o critério de cada conjunto |
| [`finetune_test/`](finetune_test/) | scripts de LoRA e a comparação antes/depois |
| [`reports/`](reports/) | medições brutas de todas as corridas, aprovadas e reprovadas |

Uma ressalva de tamanho: a bancada tem 25 problemas e foi feita pra uma pergunta
específica. **Não é benchmark sério, não compare modelos publicamente com ela.**
O que vale copiar é o método, não o placar.

---

## Se você quer uma ferramenta pronta, não um registro

Estas são mantidas e testadas, e vão te levar mais longe mais rápido do que
remontar o que está aqui:

| ferramenta | pra quê |
|---|---|
| [Ollama](https://ollama.com) | rodar os modelos. Daemon leve; é a base que este projeto usa |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | mais controle sobre quantização e offload de camadas; melhor quando a VRAM é o gargalo |
| [Aider](https://aider.chat) | agente de código no terminal, com Git integrado. Maduro |
| [OpenCode](https://github.com/sst/opencode) / [Continue](https://continue.dev) | agente no terminal e no editor, com suporte a modelo local |
| [LM Studio](https://lmstudio.ai) | se você prefere interface gráfica a terminal |
| [Unsloth](https://github.com/unslothai/unsloth) | treinar LoRA em GPU pequena, bem mais eficiente que o caminho usado aqui |
| [PEFT](https://github.com/huggingface/peft) + [TRL](https://github.com/huggingface/trl) | o caminho padrão de fine-tuning, se quiser montar você mesmo |

Elas resolvem o *rodar*. Este repositório resolve o *medir* — e as duas coisas se
complementam bem.

---

## Contribuições

**Use, estude e adapte à vontade**, copie pedaços, publique em cima. Licença
[AGPLv3](LICENSE): derivados continuam abertos sob a mesma licença, inclusive
quando servidos pela rede. Para oferecer isto como serviço fechado, existe
licença comercial: veja [LICENSING.md](LICENSING.md).

Sugestões, correções e issues são bem-vindas — em especial se você encontrar mais
algum bug de medição que passou batido, porque esse é o tipo de erro que a gente
só acha com outro par de olhos.

Só saiba de antemão: **não estou mantendo isto**, então resposta pode demorar ou
não vir, e partes do código foram escritas pra responder uma pergunta e deixadas
como estavam depois. Espere pontas soltas. Se a sua ideia for boa, ela
provavelmente merece um repositório próprio em vez de um PR aqui.
