# isaacli

**Um agente de código para linha de comando, local em primeiro lugar, construído
e medido para modelos que cabem em 4 GB de VRAM.**

*[Read in English](README.md)*

A maioria dos harnesses de agente pressupõe um modelo de fronteira na nuvem.
Aponte um deles para um modelo local pequeno e ele costuma fazer uma de duas
coisas: inventar nomes de ferramenta que não existem, ou descrever o trabalho em
vez de fazê-lo. O `isaacli` foi construído ao contrário: a pergunta inicial foi
quais modelos rodam bem em hardware modesto, e o harness foi desenhado em torno
do que eles precisam para funcionar de forma confiável. Não há restrição a 4 GB:
qualquer modelo servido pelo Ollama com chamada de ferramenta nativa, ou
qualquer endpoint compatível com OpenAI, funciona do mesmo jeito.

Ele lê e escreve arquivos, roda comandos de terminal dentro de um sandbox em
camadas com teto de recursos, e conclui a tarefa ou diz que falhou. Nada sai da
máquina, a menos que você configure uma API remota.

> [AGPLv3](LICENSE). Livre para usar, estudar e modificar. Oferecer como serviço
> de rede fechado exige licença comercial: veja [LICENSING.md](LICENSING.md).

## Começando

Requer [Ollama](https://ollama.com), Python 3.10+ e `bwrap` (`bubblewrap`) para
o sandbox.

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli

./isaacli setup                    # escolhe modelo, contexto e esforço de raciocínio

./isaacli                          # REPL interativo
./isaacli "rode git status e me diga o que está pendente"
./isaacli --workspace /caminho/do/projeto
./isaacli --resume <id-da-sessão>
```

A primeira execução interativa abre o setup automaticamente quando não existe
perfil. O setup também configura qualquer endpoint compatível com OpenAI (Groq,
por exemplo); a chave de API fica em `~/.config/isaacli/secrets.json` com modo
`0600`, nunca no workspace nem no log da sessão.

A interface fala inglês e português do Brasil, escolhidos no setup e trocáveis a
qualquer momento com `/language`.

Dentro do REPL, `/help` lista todos os comandos. Veja
**[docs/USAGE.md](docs/USAGE.md)** para os comandos, os modos de permissão, a
retomada de sessões e o fluxo de setup em detalhe (em inglês).

## Por que funciona

Nada de exótico. Quatro decisões, cada uma delas um modo de falha evitado:

1. **`/api/chat` nativa**, não uma camada de tradução compatível com OpenAI. O
   Ollama descarta `options.num_ctx` no endpoint `/v1` compatível e o respeita
   no nativo, então a camada de compatibilidade custa a janela de contexto antes
   de gerar um único token.
2. **Um schema de ferramentas curto.** Sete ferramentas de arquivo e terminal,
   para a lista caber no que um modelo pequeno consegue segurar e comparar.
3. **Um modelo com chamada de ferramenta nativa**, escolhido por medição e não
   por contagem de parâmetros. O raciocínio está em
   [`Modelfile.isaac-granite.tmpl`](tool_harness/Modelfile.isaac-granite.tmpl).
4. **`num_ctx` e `temperature` definidos explicitamente**, para viajarem com o
   perfil em vez de depender de como o servidor foi iniciado. O contexto padrão
   do Ollama trunca o schema de ferramentas em silêncio, e um modelo que não
   enxerga suas ferramentas inventa ferramentas plausíveis.

## O sandbox

A execução de comandos é contida em três camadas independentes, em
[`tool_harness/execution.py`](tool_harness/execution.py):

- **execve direto** para o que roda sozinho, então o modelo não injeta por `;`,
  `&&` ou `$()` sem você ver
- **uma lista de permitidos curta**, que decide o que roda *sem perguntar*; o
  resto é mostrado a você e roda se você aprovar
- **`bwrap`** com o disco inteiro em somente leitura e apenas a pasta de
  trabalho gravável. A rede fica fechada para o que roda sozinho; um comando que
  o usuário aprova ganha rede, então `git clone` e afins funcionam depois do sim

Além das três camadas, todo comando também roda sob um **teto de cgroup**
(`systemd-run --user --scope`: memória, número de processos e CPU), então um
comando descontrolado morre do próprio peso em vez de consumir a máquina. Se o
`systemd-run` não estiver instalado, o comando ainda roda e a saída avisa com
um `NOTE:`, em vez de ficar sem limite em silêncio.

Todo comando também carrega um **filtro seccomp** que nega as chamadas de
sistema que nenhum build ou teste precisa: namespaces e montagens aninhados,
carga de módulo, `kexec`, o chaveiro do kernel, `bpf`, `userfaultfd`, `ptrace`.
O primeiro grupo é a razão de o filtro existir: sem ele, um comando dentro da
jaula ainda consegue chamar `unshare(CLONE_NEWUSER)` e ganhar um conjunto
completo de capabilities dentro do namespace novo. Ele não é hermético e não
afirma ser: o `clone3` ainda alcança um user namespace, porque o seccomp não
consegue ler as flags atrás do ponteiro de struct dele e negar a chamada
quebraria threads no `python3` e no `pytest`. O filtro é só de x86_64 e avisa
com um `NOTE:` nas outras arquiteturas.

As ferramentas de arquivo se recusam a escapar da própria raiz, inclusive por
caminho absoluto e `..`. As duas coisas são testadas tentando escapar de
verdade, com iscas plantadas fora do diretório, em
[`check_sandbox.py`](tests/check_sandbox.py) e
[`check_execution.py`](tests/check_execution.py), em vez de conferir se a
mensagem de recusa apareceu. O mesmo arquivo prova o teto de cgroup e o filtro
seccomp pelo efeito: um consumidor de memória morre por OOM, um loop de fork é
bloqueado, e as chamadas negadas são tentadas de dentro do sandbox e precisam
voltar `EPERM`. Um controle roda a mesma sonda sem o filtro e separa as
chamadas que o filtro nega das que já falhavam por falta de capabilities,
imprimindo as duas listas, para a seção não levar crédito por uma recusa que
não causou.

O `bwrap`, o teto de cgroup e o filtro seccomp são o que a aprovação nunca
contorna, e não precisam contornar mais nada: o que você aprova roda, inclusive
`push --force`, programas fora da lista e uma linha com pipe, que vai para
`sh -c` dentro da mesma jaula. As duas primeiras camadas decidem o que acontece
*sem perguntar*, não o que você tem
permissão de fazer na sua própria máquina. Esta parte é reaproveitável sozinha,
em qualquer projeto que execute código gerado por modelo, local ou na nuvem.

## Limites honestos

- Um modelo de 2 GB é um modelo de 2 GB. Capacidade bruta vem do pré-treino e
  você baixa isso pronto. O que este repositório acrescenta é confiabilidade e
  especialização, não inteligência.
- O alvo é trabalho de arquivo e terminal através de um schema de ferramentas
  pequeno e fixo. Não é uma tentativa de substituir aquilo para que Aider ou
  Codex foram feitos: edição por diff em repositório grande, integração profunda
  com git, ou conduzir modelos de fronteira na nuvem.

## Documentação

- [docs/USAGE.md](docs/USAGE.md): comandos, permissões, sessões, setup
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): mapa dos módulos, fluxo e invariantes
- [CONTRIBUTING.md](CONTRIBUTING.md): ambiente, testes e regras de convivência

## Contribuindo

Issues e pull requests são bem-vindos, em especial reproduções em outro hardware
e, mais ainda, mais um erro de medição que passou batido.

Ao enviar um pull request você concorda em licenciar sua contribuição sob AGPLv3
e conceder ao mantenedor o direito de incluí-la em licenças comerciais deste
projeto. Veja [LICENSING.md](LICENSING.md).
