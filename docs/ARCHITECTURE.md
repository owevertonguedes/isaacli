# Isaac CLI: arquitetura atual

Este documento é o ponto de entrada técnico para pessoas e agentes de IA. Ele
descreve o código existente; não é uma promessa de que a organização atual seja
a organização desejada.

## Leitura rápida

Leia nesta ordem antes de alterar o projeto:

1. `AGENTS.md`, quando disponível no ambiente de desenvolvimento;
2. este documento;
3. o diff atual (`git diff`) e o estado do Git;
4. somente os módulos relacionados à tarefa;
5. os testes correspondentes em `tool_harness/testar_*.py`.

Logs e sessões antigas são evidência histórica, não estado atual. Revalide
processos, configuração, modelos instalados e resultados de testes ao vivo.

## Fluxo principal

```text
isaacli (launcher)
  -> tool_harness/isaac_cli.py (argumentos, REPL e sessão)
     -> setup_ollama.py (modelos, contexto, esforço e API keys)
     -> agent.py (loop mensagens -> modelo -> tool calls -> modelo)
        -> tools.py (schemas e ferramentas)
           -> execucao.py (comando sem shell, allowlist, aprovação e bwrap)
     -> terminal_ui.py (tela alternativa, seletores e histórico interno)
```

O launcher de raiz apenas encontra o Python e entrega a execução ao harness. O
workspace escolhido pelo usuário vira o limite das ferramentas de arquivo e do
executor. Configuração, segredos e sessões ficam fora do workspace trabalhado.

## Mapa dos módulos ativos

| Arquivo | Responsabilidade atual | Observação |
| --- | --- | --- |
| `tool_harness/isaac_cli.py` | argumentos, REPL, comandos `/`, sessões, apresentação, permissões e ciclo do Ollama | É o maior ponto de acoplamento atual e deve ser o primeiro alvo do inventário de responsabilidades. |
| `tool_harness/terminal_ui.py` | tela alternativa, menus, prompt ocupado e visualizador do histórico | Não deve habilitar mouse reporting globalmente, pois isso bloqueia seleção/cópia nativa do terminal. |
| `tool_harness/setup_ollama.py` | setup local, catálogo recomendado, modelos locais, contexto, reasoning e API OpenAI-compatible | Contexto é configuração por requisição; não deve criar cópias Ollama `16k/32k`. |
| `tool_harness/agent.py` | chamadas Ollama/API, streaming, normalização e loop de ferramentas | Ollama usa `/api/chat`; APIs remotas usam `/chat/completions`. |
| `tool_harness/tools.py` | schemas e implementação das ferramentas do agente | `fetch_url` é a leitura web geral; comandos continuam sem rede no sandbox. |
| `tool_harness/execucao.py` | classificação, aprovação e execução confinada de programas | Nunca adicionar shell, pipes ou redirecionamento como atalho de UI. |
| `tool_harness/config.py` | config pública e segredos locais | API keys ficam em `secrets.json` com modo `0600`, fora do Git. |
| `tool_harness/i18n.py`, `locales/` | textos do setup em português e inglês | Novas chaves precisam existir nos dois idiomas. |
| `tool_harness/model_catalog.json` | pequena curadoria de recomendações | Não representa os modelos instalados; esses vêm ao vivo do Ollama local. |

Há código de experimentação, treino e interfaces anteriores no repositório. Não
presuma que ele participa do CLI atual apenas por existir; confirme imports e o
fluxo de execução antes de movê-lo ou removê-lo.

## Estado e dados locais

- Configuração: `~/.config/isaacli/config.json` (ou `XDG_CONFIG_HOME`).
- Segredos: `~/.config/isaacli/secrets.json`.
- Sessões: `tool_harness/cli_sessoes/*.jsonl` no estado atual.
- Feedback: `tool_harness/feedback/*.jsonl` no estado atual.
- Coordenação do Ollama gerenciado: diretório de runtime do usuário ou `/tmp`.

Novas sessões usam UUIDv4. IDs antigos baseados em data continuam aceitos para
retomada. O comando de retomada usa `isaacli` quando esta instalação está no
`PATH`; caso contrário mostra o launcher absoluto realmente executável.

## Terminal e encerramento

O REPL usa o buffer alternativo para não misturar a conversa com o histórico do
shell. Quando a conversa excede a tela, a roda abre o transcript como viewport
integrado e o percorre nas duas direções; `/history` também o abre. ↑/↓ no prompt
pertencem ao histórico de mensagens digitadas. Mouse reporting só fica ativo
quando há conteúdo rolável; nesse estado, Shift+arrastar preserva a seleção nativa.
Menus de tela inteira devem
sempre redesenhar a conversa recente ao retornar ao REPL.

No Ptyxis/VTE, o buffer alternativo não possui scrollback nativo. O viewport usa
o protocolo SGR do mouse para distinguir a roda das setas. Não reative DEC 1007:
ele transforma roda e ↑ na mesma sequência e recria o bug de navegação.

Teclas recebidas durante a inicialização são descartadas antes do primeiro
prompt. Durante geração, a entrada fica sem eco e é descartada ao terminar.
`Ctrl+C` no prompt encerra o REPL; sinais repetidos durante o cleanup são
ignorados até a coordenação do servidor Ollama ser concluída.

Ollama iniciado pelo Isaac é compartilhado entre sessões do Isaac. A última
sessão registrada encerra somente o servidor que o próprio Isaac iniciou. Um
servidor preexistente pertence ao usuário e não deve ser finalizado pelo app.

## Contratos de modelos

- Ollama: API nativa de chat, tools obrigatórias e `options.num_ctx` por chamada.
- API remota: contrato OpenAI-compatible Chat Completions com streaming e
  function calling. `reasoning_effort` é opcional e pode ser desativado.
- APIs nativas incompatíveis (por exemplo, formatos próprios de outros
  provedores) exigirão adapters explícitos; não espalhe condicionais por nome de
  provedor dentro do REPL.

O menu `/model` separa origem, modelo, contexto e esforço. Recomendação não é
sinônimo de instalação: recomendações vêm do JSON curado; modelos locais vêm da
consulta ao servidor Ollama.

Se um provedor avaliar tokens, mas devolver conteúdo e tool calls vazios ao
receber vários schemas, o agente tenta novamente com uma única ferramenta
selecionada pela intenção do pedido. Esse fallback é baseado na capacidade
observada, não no nome do modelo.

Métricas de geração só são comparáveis quando contexto, prompt e ferramentas
também são equivalentes. Um benchmark curto com `num_ctx=4096` não representa o
custo de uma sessão de agente com `num_ctx=32768`, histórico e schemas de
ferramentas: além da geração, o modelo precisa processar todo esse prefixo e um
KV cache maior pode deixar de caber integralmente na GPU.

## Limites de segurança que a reorganização deve preservar

- execução de um único programa, sem shell;
- workspace como limite de arquivos e comandos;
- aprovação antes de mutações não autorizadas;
- nenhuma exposição de API keys em config, logs ou saída;
- URLs públicas validadas contra destinos locais/privados;
- processos filhos presos ao ciclo correto e cleanup idempotente;
- texto do modelo sanitizado antes de chegar ao terminal.

## Verificação

```bash
rtk python3 tool_harness/testar_cli.py
rtk python3 tool_harness/testar_agent_config.py
rtk python3 tool_harness/testar_setup.py
rtk git diff --check
```

Se `execucao.py` mudar, execute também `testar_execucao.py` fora de um sandbox
aninhado, porque `bwrap` precisa criar sua interface loopback.

## Próxima etapa: organização

A próxima sessão deve começar por um inventário, não por mover arquivos às
cegas. A dívida visível é que `isaac_cli.py` concentra apresentação, aplicação,
sessões, comandos e lifecycle. A reorganização deve definir limites testáveis
para, no mínimo:

- comandos internos;
- sessões e persistência;
- providers/modelos;
- apresentação e entrada do terminal;
- ciclo de processos;
- política de permissões.

Antes de cada movimento, registre imports e consumidores, mantenha uma camada de
compatibilidade quando necessário e rode a suíte. Não misture reorganização
mecânica com mudança de comportamento no mesmo passo.
