"""UI terminal sem dependencias: setas em TTY, numeros como fallback."""
from contextlib import contextmanager
import os
import re
import shutil
import sys


_ALT_DEPTH = 0


def interativo(input_fn=input):
    return input_fn is input and sys.stdin.isatty() and sys.stdout.isatty()


def limpar(input_fn=input):
    if interativo(input_fn):
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()


def sutil(texto, input_fn=input):
    return f"\033[2m{texto}\033[0m" if interativo(input_fn) else texto


@contextmanager
def tela_com_scrollback(input_fn=input):
    """Limpa o chat na entrada/saída, mas preserva o scrollback do terminal."""
    ativo = interativo(input_fn)
    if ativo:
        # Buffer principal: diferente de 1049h, o terminal mantém as linhas que
        # saem pelo topo e o usuário pode subir para copiar conteúdo antigo.
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()
    try:
        yield
    finally:
        if ativo:
            sys.stdout.write("\033[?25h\033[0m\033[H\033[2J")
            sys.stdout.flush()


@contextmanager
def tela_alternativa(input_fn=input):
    """Isola o wizard do histórico visível do terminal."""
    global _ALT_DEPTH
    ativo = interativo(input_fn)
    primeira = ativo and _ALT_DEPTH == 0
    if ativo:
        _ALT_DEPTH += 1
    if primeira:
        sys.stdout.write("\033[?1049h\033[H\033[2J")
        sys.stdout.flush()
    try:
        yield
    finally:
        if ativo:
            _ALT_DEPTH -= 1
        if primeira:
            sys.stdout.write("\033[?25h\033[0m\033[?1049l")
            sys.stdout.flush()


def selecionar(titulo, opcoes, input_fn=input, prompt="Selecione: ", invalido=None,
               inicial=0, desabilitados=None):
    if not opcoes:
        raise ValueError("select requires at least one option")
    desabilitados = set(desabilitados or ())
    selecionaveis = [i for i in range(len(opcoes)) if i not in desabilitados]
    if not selecionaveis:
        raise ValueError("select requires at least one enabled option")
    if not interativo(input_fn):
        print(titulo)
        numero_para_indice = []
        for i, opcao in enumerate(opcoes):
            if i in desabilitados:
                print(f"  {opcao}")
            else:
                numero_para_indice.append(i)
                print(f"  {len(numero_para_indice)}) {opcao}")
        while True:
            valor = input_fn(prompt).strip()
            try:
                numero = int(valor) - 1
            except ValueError:
                numero = -1
            if 0 <= numero < len(numero_para_indice):
                return numero_para_indice[numero]
            print(invalido or f"Escolha um número de 1 a {len(numero_para_indice)}.")

    import termios
    import tty

    fd = sys.stdin.fileno()
    anterior = termios.tcgetattr(fd)
    indice = min(selecionaveis, key=lambda i: abs(i - inicial))
    def mover(direcao):
        posicao = selecionaveis.index(indice)
        return selecionaveis[(posicao + direcao) % len(selecionaveis)]

    def renderizar():
        # Redesenhar a tela inteira também funciona quando uma opção longa quebra
        # automaticamente em mais de uma linha física.
        tamanho = shutil.get_terminal_size((80, 24))
        largura = max(tamanho.columns, 20)
        ansi = re.compile(r"\x1b\[[0-9;]*m")
        linhas_titulo = sum(
            max(1, (len(ansi.sub("", linha)) + largura - 1) // largura)
            for linha in titulo.splitlines()
        )
        capacidade = max(5, tamanho.lines - linhas_titulo - 4)
        inicio = max(0, indice - capacidade // 2)
        fim = min(len(opcoes), inicio + capacidade)
        inicio = max(0, fim - capacidade)
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.write(titulo.replace("\n", "\r\n") + "\r\n")
        if inicio:
            sys.stdout.write(f"   \033[2m↑ {inicio} opção(ões) acima\033[0m\r\n")
        for posicao in range(inicio, fim):
            opcao = opcoes[posicao]
            if posicao in desabilitados:
                sys.stdout.write(f"   \033[2m{opcao}\033[0m\r\n")
                continue
            cursor = "❯" if posicao == indice else " "
            destaque = "\033[1;36m" if posicao == indice else ""
            reset = "\033[0m" if destaque else ""
            sys.stdout.write(f" {cursor} {destaque}{opcao}{reset}\r\n")
        if fim < len(opcoes):
            sys.stdout.write(
                f"   \033[2m↓ {len(opcoes) - fim} opção(ões) abaixo\033[0m\r\n"
            )
        sys.stdout.flush()

    with tela_alternativa(input_fn):
        try:
            tty.setraw(fd)
            sys.stdout.write("\033[?25l")
            renderizar()
            while True:
                tecla = os.read(fd, 1)
                if tecla in (b"\r", b"\n"):
                    return indice
                if tecla in (b"k", b"K"):
                    indice = mover(-1)
                    renderizar()
                elif tecla in (b"j", b"J"):
                    indice = mover(1)
                    renderizar()
                elif tecla == b"\x1b":
                    sequencia = os.read(fd, 2)
                    if sequencia == b"[A":
                        indice = mover(-1)
                        renderizar()
                    elif sequencia == b"[B":
                        indice = mover(1)
                        renderizar()
                elif tecla == b"\x03":
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, anterior)
            sys.stdout.write("\033[?25h\033[0m")
            sys.stdout.flush()


def status(texto=None):
    """Atualiza uma linha discreta no rodapé sem mover o cursor de escrita."""
    if not (sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"):
        return
    conteudo = f"\033[2m{texto}\033[0m" if texto else ""
    sys.stdout.write(f"\0337\033[999;1H\033[2K{conteudo}\0338")
    sys.stdout.flush()


def selecionar_inline(opcoes, atalhos=None, input_fn=input, inicial=0):
    """Menu por setas sem limpar a conversa que já está na tela."""
    if not opcoes:
        raise ValueError("inline select requires at least one option")
    atalhos = atalhos or {}
    if not interativo(input_fn):
        for i, opcao in enumerate(opcoes, 1):
            print(f"  {i}) {opcao}")
        valor = input_fn("Escolha [Enter/w/g/n]: ").strip().lower()
        if valor == "":
            return inicial
        if valor in atalhos:
            return atalhos[valor]
        try:
            indice = int(valor) - 1
        except ValueError:
            return len(opcoes) - 1
        return indice if 0 <= indice < len(opcoes) else len(opcoes) - 1

    import termios
    import tty

    fd = sys.stdin.fileno()
    anterior = termios.tcgetattr(fd)
    indice = min(max(inicial, 0), len(opcoes) - 1)

    def desenhar(subir=False):
        if subir:
            sys.stdout.write(f"\033[{len(opcoes)}A")
        for posicao, opcao in enumerate(opcoes):
            cursor = "❯" if posicao == indice else " "
            destaque = "\033[1;36m" if posicao == indice else ""
            reset = "\033[0m" if destaque else ""
            sys.stdout.write(f"\r\033[2K {cursor} {destaque}{opcao}{reset}\r\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\033[?25l")
        desenhar()
        while True:
            tecla = os.read(fd, 1)
            if tecla in (b"\r", b"\n"):
                break
            if tecla.decode(errors="ignore").lower() in atalhos:
                indice = atalhos[tecla.decode(errors="ignore").lower()]
                break
            if tecla in (b"k", b"K"):
                indice = (indice - 1) % len(opcoes)
                desenhar(subir=True)
            elif tecla in (b"j", b"J"):
                indice = (indice + 1) % len(opcoes)
                desenhar(subir=True)
            elif tecla == b"\x1b":
                sequencia = os.read(fd, 2)
                if sequencia == b"[A":
                    indice = (indice - 1) % len(opcoes)
                    desenhar(subir=True)
                elif sequencia == b"[B":
                    indice = (indice + 1) % len(opcoes)
                    desenhar(subir=True)
            elif tecla == b"\x03":
                raise KeyboardInterrupt
        sys.stdout.write(f"\033[{len(opcoes)}A")
        for _ in opcoes:
            sys.stdout.write("\r\033[2K\033[1B")
        sys.stdout.write(f"\033[{len(opcoes)}A\r\033[2K")
        # O terminal ainda está em raw: \n sozinho não retorna à coluna zero.
        sys.stdout.write(f"Permissão: {opcoes[indice]}\r\n")
        sys.stdout.flush()
        return indice
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, anterior)
        sys.stdout.write("\033[?25h\033[0m")
        sys.stdout.flush()
