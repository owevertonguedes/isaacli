# isaacli

**Um agente de código para linha de comando, local em primeiro lugar, projetado para modelos que cabem em 4 GB de VRAM, sem limitar você a eles.**

*[Read in English](README.md)*

O `isaacli` lê e edita arquivos, executa comandos em um sandbox Linux em camadas e continua trabalhando até concluir a tarefa ou informar claramente a falha. Ele foi construído em torno de modelos locais pequenos, em vez de pressupor um modelo de fronteira na nuvem, mas a mesma interface também aceita modelos maiores no Ollama e APIs compatíveis com OpenAI.

Nada é enviado a um modelo remoto, a menos que você configure um. Leituras da web usam rotas explícitas e restritas; comandos de terminal fora do fluxo de aprovação permanecem offline por padrão.

> [AGPLv3](LICENSE). Livre para usar, estudar e modificar. Oferecer o isaacli como serviço de rede fechado exige licença comercial; veja [LICENSING.md](LICENSING.md).

## Começando

Requer [Ollama](https://ollama.com), Python 3.10+ e `bwrap` (`bubblewrap`).

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli
./isaacli install

isaacli setup
isaacli
```

A instalação adiciona um link do usuário em `~/.local/bin`; não precisa de `sudo` nem sobrescreve um comando existente. O launcher também funciona em terminais Flatpak, como o VS Code do Flathub, executando no host onde ficam o Ollama e as dependências do sandbox.

O setup escolhe idioma da interface, motor, modelo, contexto e esforço de raciocínio. Ele pode usar um modelo local no Ollama com chamada de ferramenta nativa ou um endpoint configurável compatível com OpenAI. Chaves de API ficam fora do workspace em um arquivo de segredos com modo `0600`.

Pontos de entrada úteis:

```bash
isaacli "rode git status e explique o que está pendente"
isaacli --workspace /caminho/do/projeto
isaacli --resume <id-da-sessão>
isaacli uninstall
```

Veja [Instalação](docs/INSTALLATION.md) para detalhes de Flatpak, recuperação e as opções de purge com confirmação explícita.

## O que ele oferece

- Ferramentas de arquivo, web e terminal expostas por um schema compacto, adequado a modelos menores.
- Pedidos interativos de permissão com autorizações únicas, por workspace ou globais.
- Interfaces em inglês e português do Brasil, alternáveis com `/language`.
- Seleção de modelo, contexto e raciocínio sem criar modelos duplicados no Ollama.
- Sessões JSONL retomáveis, histórico de saída dos comandos e avaliação de tarefas.
- Perfis remotos compatíveis com OpenAI, endpoint configurável e ID exato do modelo.

Dentro do REPL, digite `/` para abrir a paleta ou `/help` para listar todos os comandos. [Uso](docs/USAGE.md) explica setup, permissões, sessões, comportamento do terminal e a referência completa de comandos.

## Segurança e privacidade

Comandos autônomos executam sem shell, dentro do `bwrap`, com o workspace como único local gravável. Comandos fora da política padrão são mostrados antes da execução; depois da aprovação, podem usar shell ou rede, mas continuam dentro dos mesmos limites de sistema de arquivos, recursos e chamadas de sistema.

Os tetos de recursos usam `systemd-run --user`; o filtro seccomp funciona apenas em x86_64. Quando alguma dessas camadas não está disponível, o isaacli informa a limitação em vez de afirmar uma proteção inexistente. O sandbox limita comandos gerados pelo modelo, mas não é uma barreira de segurança contra o usuário, root ou malware já executado pela mesma conta.

Sessões e avaliações locais podem conter prompts, respostas, caminhos, comandos e resultados de ferramentas. O Git ignora esses arquivos, mas atualmente eles são armazenados como texto simples. Um provedor remoto recebe a conversa e os resultados de ferramentas incluídos em requisições posteriores. Leia [Segurança e privacidade](docs/SECURITY.md) antes de usar dados sensíveis ou um endpoint remoto.

## Limites honestos

- Um modelo pequeno continua sendo um modelo pequeno: o harness melhora confiabilidade e uso de ferramentas, não o conhecimento ou a capacidade de raciocínio do modelo.
- O projeto tem como alvo trabalho com arquivos e terminal por meio de um conjunto compacto de ferramentas; não é uma IDE completa nem um framework genérico de automação de navegador.
- O sandbox reduz dano acidental e escape do workspace, mas nenhum agente local deve ser tratado como barreira contra uma máquina ou conta comprometida.

## Documentação

- [Uso](docs/USAGE.md): setup, comandos, permissões, sessões e comportamento do terminal
- [Arquitetura](docs/ARCHITECTURE.md): mapa dos módulos, fluxo de dados e invariantes de implementação
- [Instalação](docs/INSTALLATION.md): instalação, Flatpak, remoção, purge e recuperação
- [Segurança e privacidade](docs/SECURITY.md): dados armazenados, APIs remotas e limites atuais
- [Contribuindo](CONTRIBUTING.md): ambiente de desenvolvimento, testes e regras de convivência

## Contribuindo

Issues e pull requests são bem-vindos, especialmente falhas reproduzíveis em outro hardware ou com outros modelos capazes de usar ferramentas.

Ao enviar um pull request você concorda em licenciar sua contribuição sob AGPLv3 e conceder ao mantenedor o direito de incluí-la em licenças comerciais deste projeto. Veja [LICENSING.md](LICENSING.md).
