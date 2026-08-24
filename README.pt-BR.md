# isaacli

**Um agente de código para linha de comando, local em primeiro lugar, projetado para modelos que cabem em 4 GB de VRAM, sem limitar você a eles.**

*[Read in English](README.md)*

O `isaacli` lê e edita arquivos, executa comandos em um sandbox Linux em camadas e continua trabalhando até concluir a tarefa ou informar claramente a falha. Ele foi construído em torno de modelos locais pequenos, em vez de pressupor um modelo de fronteira na nuvem, mas a mesma interface também aceita modelos maiores no Ollama e APIs compatíveis com OpenAI.

Nada é enviado a um modelo remoto, a menos que você configure um. Leituras da web usam rotas explícitas e restritas; comandos de terminal fora do fluxo de aprovação permanecem offline por padrão.

> [AGPLv3](LICENSE). Livre para usar, estudar e modificar. Oferecer o isaacli como serviço de rede fechado exige licença comercial; veja [LICENSING.md](LICENSING.md).

## Começando

Requer Python 3.10+ e `bwrap` (`bubblewrap`). Um motor não é pré-requisito: o setup lista as quatro origens e cada linha diz o que esta máquina já tem. O [Ollama](https://ollama.com) é usado onde estiver instalado, o isaacli instala o llama.cpp por você e serve um GGUF que você já tenha, e as outras duas origens são remotas.

```bash
git clone <url-do-repositorio>
cd isaacli
./isaacli install

isaacli setup
isaacli kaggle
isaacli
```

A instalação adiciona um link do usuário em `~/.local/bin`; não precisa de `sudo` nem sobrescreve um comando existente. O launcher também funciona em terminais Flatpak, como o VS Code do Flathub, executando no host onde ficam o Ollama e as dependências do sandbox.

O setup escolhe idioma da interface, motor, modelo, contexto e esforço de raciocínio. O motor pode ser um modelo local no Ollama com chamada de ferramenta nativa, um arquivo GGUF desta máquina servido pelo llama.cpp (que o isaacli instala no diretório do seu usuário se você não o tiver), um endpoint configurável compatível com OpenAI ou a integração guiada com GPU do Kaggle. Chaves de API ficam fora do workspace em um arquivo de segredos com modo `0600`.

O Kaggle está disponível no setup, em `isaacli kaggle`, em `/kaggle` e no menu de provedores de `/model`. O fluxo compartilhado guarda qualquer quantidade de contas do Kaggle, mostra a quota restante de cada uma e deixa a seleção manual com o usuário. Os assets preparados são privados e pertencem à conta selecionada. Quando eles não existem, o fluxo anuncia o custo de arranque medido, oferece preparação somente em CPU e mantém disponível o lançamento autocontido. Cada envio com GPU exige confirmação naquele momento, e pergunta por quanto tempo aquele kernel pode viver: o próprio kernel segura esse teto e o Kaggle segura o mesmo número, então uma janela que sumiu não deixa um kernel rodando por horas. Ele também se encerra sozinho depois de cinco minutos sem sinal de vida da janela que o subiu. O Kaggle é um serviço de terceiros cujas sessões podem cair, e seus termos não foram feitos pensando em notebooks como servidores persistentes de API.

Uma sessão do Kaggle gasta quota por relógio de parede enquanto o kernel está vivo, então o isaacli assume a vida inteira do kernel que ele mesmo subiu: fechar o programa encerra esse kernel, abrir de novo reaproveita o endpoint se ele ainda responde, o que não custa nada, e subir outro nunca é automático, porque é aí que o gasto acontece. O `isaacli kaggle --stop` encerra uma sessão de fora, e o `uninstall --purge --kaggle` lista o que ficou na conta e pergunta antes de apagar qualquer coisa. Duas janelas do isaacli podem dividir um kernel: quem encerra é a última a fechar. O que você escolheu da última vez, a conta, o modelo e a precisão exata, fica guardado e volta como uma tecla só, e trocar é a outra opção da mesma tela.

Endpoint em `localhost` não pede chave de API, porque um servidor que você mesmo roda não tem chave a exigir. Para esses, o setup também se oferece para subir o servidor por você: informe o comando que o inicia (por exemplo `llama-server -m /caminho/modelo.gguf -c 8192`) e o isaacli sobe ele quando uma sessão abre, compartilha entre sessões simultâneas e derruba quando a última fecha. Deixe o comando vazio para continuar subindo à mão. É assim que você aponta o isaacli para qualquer peso que tenha baixado, do Hugging Face ou de onde for: o modelo fica inteiramente na sua máquina.

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
- Instruções de projeto do `AGENTS.md` na raiz do workspace, limitadas e subordinadas ao contrato de segurança interno.
- Perfis remotos compatíveis com OpenAI, endpoint configurável e ID exato do modelo.
- Gerência opcional do ciclo de vida de um servidor local compatível com OpenAI, para que um llama-server ou equivalente suba junto com a sua sessão e caia com a última.

Dentro do REPL, digite `/` para abrir a paleta ou `/help` para listar todos os comandos. [Uso](docs/USAGE.md) explica setup, permissões, sessões, comportamento do terminal e a referência completa de comandos.

## Segurança e privacidade

Comandos autônomos executam sem shell, dentro do `bwrap`, com o workspace como único local gravável. Comandos fora da política padrão são mostrados antes da execução; depois da aprovação, podem usar shell ou rede, mas continuam dentro dos mesmos limites de sistema de arquivos, recursos e chamadas de sistema.

O que o sandbox alcança além dos diretórios de sistema é o seu próprio PATH, montado somente para leitura, para que a cadeia de ferramentas de um projeto funcione dentro dele. Como o PATH de quem iniciou o isaacli só é o do seu shell de login quando você o inicia de um terminal, o isaacli também lê uma vez por sessão o PATH do seu shell de login e une os dois; isso executa o seu profile, nunca amplia o que pode ser montado, e quando não é possível ler a saída diz isso em vez de perder a sua cadeia de ferramentas em silêncio.

Os tetos de recursos usam `systemd-run --user`; o filtro seccomp funciona apenas em x86_64. Quando alguma dessas camadas não está disponível, o isaacli informa a limitação em vez de afirmar uma proteção inexistente. O sandbox limita comandos gerados pelo modelo, mas não é uma barreira de segurança contra o usuário, root ou malware já executado pela mesma conta.

Sessões e avaliações locais podem conter prompts, respostas, caminhos, comandos e resultados de ferramentas. O Git ignora esses arquivos, mas atualmente eles são armazenados como texto simples. Um provedor remoto recebe a conversa e os resultados de ferramentas incluídos em requisições posteriores. Leia [Segurança e privacidade](docs/SECURITY.md) antes de usar dados sensíveis ou um endpoint remoto.

## Limites honestos

- Um modelo pequeno continua sendo um modelo pequeno: o harness melhora confiabilidade e uso de ferramentas, não o conhecimento ou a capacidade de raciocínio do modelo. Medido no hardware deste projeto em agosto de 2026, um modelo de código de 3B servido por endpoint compatível com OpenAI devolveu as chamadas de ferramenta como bloco JSON em markdown, em vez do formato nativo que o próprio template dele declara, então ele descreveu o arquivo que pretendia escrever e não escreveu nenhum. O harness avisou que nenhuma alteração foi confirmada, que é o comportamento correto e não substitui um modelo capaz de chamar ferramenta.
- O Ollama é o motor recomendado por ser um instalador único com catálogo de modelos e por já trazer o backend CUDA, não por ser mais rápido. Os dois foram medidos um contra o outro no hardware deste projeto em 2026-08-21, mesmo GGUF, mesmo contexto, mesmo prompt, com a ordem invertida: Ollama 39,36 tok/s contra llama-server 37,34, uma diferença de 5,4%, com a dispersão do próprio llama-server entre duas rodadas em 2,81 tok/s. A diferença está dentro do ruído da máquina, então nenhuma afirmação de velocidade é feita para nenhum dos dois, nem na interface nem aqui. A vazão por modelo nesta placa está em [Medições](docs/benchmarks/README.md).
- A decodificação especulativa existe no llama.cpp e foi medida aqui, não suposta. Na mesma placa e modelo ela não deu ganho com rascunho por n-grama e ficou 45% mais lenta com um modelo rascunho de 0,5B. A técnica é desenhada para alvo grande com rascunho minúsculo, que não é a configuração que cabe numa placa de 4 GB, então trate qualquer ganho publicado como sendo de outro hardware até medir o seu.
- O projeto tem como alvo trabalho com arquivos e terminal por meio de um conjunto compacto de ferramentas; não é uma IDE completa nem um framework genérico de automação de navegador.
- O sandbox reduz dano acidental e escape do workspace, mas nenhum agente local deve ser tratado como barreira contra uma máquina ou conta comprometida.

## Documentação

- [Uso](docs/USAGE.md): setup, comandos, permissões, sessões e comportamento do terminal
- [Arquitetura](docs/ARCHITECTURE.md): mapa dos módulos, fluxo de dados e invariantes de implementação
- [Instalação](docs/INSTALLATION.md): instalação, Flatpak, remoção, purge e recuperação
- [Segurança e privacidade](docs/SECURITY.md): dados armazenados, APIs remotas e limites atuais
- [Medições](docs/benchmarks/README.md): o que cada modelo recomendado fez na máquina em que foi medido, e em que data
- [Contribuindo](CONTRIBUTING.md): ambiente de desenvolvimento, testes e regras de convivência

## Contribuindo

Issues e pull requests são bem-vindos, especialmente falhas reproduzíveis em outro hardware ou com outros modelos capazes de usar ferramentas.

Ao enviar um pull request você concorda em licenciar sua contribuição sob AGPLv3 e conceder ao mantenedor o direito de incluí-la em licenças comerciais deste projeto. Veja [LICENSING.md](LICENSING.md).
