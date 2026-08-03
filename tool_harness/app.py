#!/usr/bin/env python3
"""OFICINA — a bancada onde o modelo trabalha.

NOMES (arrumado em 2026-07-19, estava confuso antes):
  - "Oficina" e ESTE APP (a janela, as ferramentas, a sandbox, o juiz).
  - "isaac"  e o MODELO que roda dentro dela (hoje: granite4:micro, da IBM).
  Trocar o modelo NAO muda o nome do app — e esse o ponto de separar os dois.

Ciclo de vida (o ponto principal, e a causa de ollama zumbi se feito errado):
  - ao abrir: se o ollama NAO estava rodando, este app sobe ele e ANOTA isso
  - ao fechar: descarrega o modelo da VRAM e mata o ollama SO SE foi ele que subiu
  - se o ollama ja estava rodando antes (voce subiu na mao), o app nao encosta nele
  - abrir o app NAO carrega o modelo: ele so sobe na primeira pergunta. Assim
    fechar sem usar nao deixa 2GB presos na memoria.

UI (rumo corrigido em 2026-07-19, task 01): a Oficina NAO e um VSCode e nao vai
virar um. O isaac roda no editor que o usuario ja usa; aqui e onde ele brinca com
os proprios arquivos e o usuario ASSISTE. Consequencia: menos superficie, mais
texto corrido, saida crua sempre — interface enfeitada esconde o que o modelo
faz, terminal mostra.

  - conversa em streaming (token a token) em cima
  - barra de autonomia: "Iniciar aprendizado" e "Rodar ciclo do juiz" viram
    BOTAO, com Parar. Antes exigiam alguem digitando comando no terminal.
  - terminal embaixo: chamada de ferramenta e saida crua dos lotes no mesmo
    lugar, sem resumir e sem esconder erro atras de icone
  - lateral: diario dos arquivos que o isaac TOCOU nesta sessao (nao e navegador
    de disco), historico de conversas e o grafico de evolucao
"""
import atexit
import datetime
import gi
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import cairo
from pathlib import Path

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk, Pango, Gdk  # noqa: E402

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI))
import agent  # noqa: E402
import processos  # noqa: E402
import tools  # noqa: E402

APP_NOME = "Oficina"                                  # o app
MODELO = os.environ.get("AGENTE_MODELO", "isaac")     # o modelo dentro dele
URL_SAUDE = "http://127.0.0.1:11434/api/tags"
URL_CARREGADOS = "http://127.0.0.1:11434/api/ps"      # quem esta ocupando memoria

# Base sobre a qual o `isaac` e montado. Se o granite4 for o unico modelo bom que
# roda nesta maquina, vale saber quando a IBM publica versao nova.
BASE_REPO, BASE_TAG = "granite4", "micro"
URL_MANIFESTO = f"https://registry.ollama.ai/v2/library/{BASE_REPO}/manifests/{BASE_TAG}"
MANIFESTO_LOCAL = (Path.home() / ".ollama/models/manifests/registry.ollama.ai"
                   / "library" / BASE_REPO / BASE_TAG)

# Pastas que nao entram na foto inicial da pasta de trabalho.
IGNORAR = {".git", "node_modules", "__pycache__", ".venv", "venv"}


# --- ciclo de vida do ollama -------------------------------------------------

class Ollama:
    def __init__(self):
        self.proc = None
        self.subimos = False

    @staticmethod
    def no_ar():
        try:
            urllib.request.urlopen(URL_SAUDE, timeout=2)
            return True
        except Exception:
            return False

    def garantir(self, log):
        if self.no_ar():
            log("ollama ja estava rodando — nao vou mexer nele ao sair.")
            return True
        log("subindo o ollama...")
        self.proc = subprocess.Popen(
            ["ollama", "serve"],
            env={**os.environ, "OLLAMA_CONTEXT_LENGTH": "8192"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,  # grupo proprio: da pra matar tudo junto
        )
        self.subimos = True
        for _ in range(40):
            if self.no_ar():
                log("ollama pronto.")
                return True
            time.sleep(0.25)
        log("ERRO: ollama nao respondeu a tempo.")
        return False

    def encerrar(self):
        # Descarrega o modelo da VRAM mesmo que o servidor seja de outro dono.
        try:
            subprocess.run(["ollama", "stop", MODELO], capture_output=True, timeout=15)
        except Exception:
            pass
        if not (self.subimos and self.proc):
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass


OLLAMA = Ollama()


def _encerrar_tudo():
    """Fechar a Oficina para TUDO — regra do projeto.

    Os lotes vem primeiro: `aprender.py` gasta credito por chamada de rede, entao
    um orfao dele custa dinheiro, enquanto o ollama zumbi so custa VRAM.
    """
    processos.matar_todos()
    OLLAMA.encerrar()


atexit.register(_encerrar_tudo)  # cobre saida normal e excecao


def _morrer(signum, _frame):
    """atexit NAO roda quando o processo recebe sinal (kill, logout, fim de sessao).

    Sem isto o ollama fica zumbi segurando VRAM — verificado na pratica. E o lote
    de aprendizado fica gastando credito com a janela ja fechada.
    """
    _encerrar_tudo()
    os._exit(128 + signum)


for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_s, _morrer)


# --- conversas ---------------------------------------------------------------

DIR_CONVERSAS = AQUI / "conversas"
DIR_CONVERSAS.mkdir(parents=True, exist_ok=True)
CONFIG_COLAB = AQUI / "colab_config.json"


def gerar_id_conversa():
    hoje = datetime.date.today().strftime("%Y-%m-%d")
    sufixo = "".join(random.choices("0123456789abcdef", k=3))
    return f"{hoje}-{sufixo}"


def salvar_conversa_atomica(caminho, dados):
    dir_parent = os.path.dirname(caminho)
    os.makedirs(dir_parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=dir_parent, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, caminho)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def carregar_colab_host():
    try:
        return json.loads(CONFIG_COLAB.read_text(encoding="utf-8")).get("cf_host", "")
    except Exception:
        return ""


def salvar_colab_host(host):
    CONFIG_COLAB.write_text(json.dumps({"cf_host": host}, ensure_ascii=False, indent=2),
                            encoding="utf-8")


# --- interface ---------------------------------------------------------------

class Janela(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, default_width=980, default_height=680)
        self.ocupado = False
        self.arquivo_tocado = None  # ultimo path que o Isaac mexeu (pra destacar)
        # Diario de arquivos: caminho -> "novo" | "alterado". `existiam` e a foto
        # da pasta no comeco da sessao; sem ela nao da pra distinguir os dois.
        self.tocados = {}
        self.existiam = set()
        self.lote = None            # o Processo de lote rodando agora (so um)
        self.contagem = {}

    # -- header: seletor de pasta + titulo com a pasta atual
        raiz = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        cab = Adw.HeaderBar()
        self.titulo = Adw.WindowTitle(title=APP_NOME, subtitle=str(tools.SANDBOX_ROOT))
        cab.set_title_widget(self.titulo)
        botao_pasta = Gtk.Button(icon_name="folder-open-symbolic",
                                 tooltip_text="Escolher a pasta de trabalho")
        botao_pasta.connect("clicked", self.escolher_pasta)
        cab.pack_start(botao_pasta)
        self.status = Gtk.Label(label="iniciando…")
        self.status.add_css_class("dim-label")
        cab.pack_end(self.status)

        # Controle explícito do modelo: carregar / descarregar da memória
        controles_modelo = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self.btn_iniciar = Gtk.Button(icon_name="media-playback-start-symbolic",
                                      tooltip_text="Carregar o modelo na memória")
        self.btn_iniciar.connect("clicked", self.carregar)
        self.btn_iniciar.add_css_class("flat")

        self.btn_parar = Gtk.Button(icon_name="media-playback-stop-symbolic",
                                    tooltip_text="Descarregar o modelo da memória (liberar VRAM)")
        self.btn_parar.connect("clicked", self.descarregar)
        self.btn_parar.add_css_class("flat")

        # Indicador do MODELO, separado do status da tarefa: diz se ele esta
        # ocupando memoria agora. Sem isso nao da pra saber se fechar o app vai
        # deixar 2GB presos — foi pedido explicito depois de um susto com zumbi.
        self.luz = Gtk.Label(label="○ parado")
        self.luz.add_css_class("dim-label")
        self.luz.set_tooltip_text("Estado do modelo na memoria")
        self.luz.set_margin_end(6)

        controles_modelo.append(self.luz)
        controles_modelo.append(self.btn_iniciar)
        controles_modelo.append(self.btn_parar)

        cab.pack_end(controles_modelo)
        GLib.timeout_add_seconds(3, self.atualizar_luz)
        raiz.append(cab)

        # -- corpo: arvore de arquivos + conversas a esquerda | conversa + ferramentas a direita
        painel = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, vexpand=True,
                           position=220, shrink_start_child=False)

        # Painel lateral esquerdo com Arquivos e Histórico de Conversas
        lateral = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        
        # Diario de arquivos TOCADOS nesta sessao — nao e navegador de disco.
        # Antes era uma arvore do disco inteiro que despejava o arquivo clicado
        # no painel da conversa. Virou visualizador sem querer, e visualizador
        # nao e o proposito da Oficina (task 01): o que interessa e ver o que o
        # isaac mexeu enquanto trabalhava.
        rotulo_arq = Gtk.Label(label="Arquivos tocados nesta sessão", xalign=0)
        rotulo_arq.add_css_class("dim-label")
        rotulo_arq.set_margin_start(12)
        rotulo_arq.set_margin_top(6)
        lateral.append(rotulo_arq)

        self.arvore = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.arvore.add_css_class("navigation-sidebar")
        self.arvore.set_activate_on_single_click(True)
        self.arvore.connect("row-activated", self.abrir_da_arvore)
        rolagem_arvore = Gtk.ScrolledWindow(child=self.arvore, vexpand=True)
        lateral.append(rolagem_arvore)

        # Seção de Histórico de Conversas
        rotulo_conv = Gtk.Label(label="Histórico de Conversas", xalign=0)
        rotulo_conv.add_css_class("dim-label")
        rotulo_conv.set_margin_start(12)
        rotulo_conv.set_margin_top(12)
        lateral.append(rotulo_conv)

        self.lista_conv = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self.lista_conv.add_css_class("navigation-sidebar")
        self.lista_conv.set_activate_on_single_click(True)
        self.lista_conv.connect("row-activated", self.abrir_conversa_antiga)
        rolagem_conv = Gtk.ScrolledWindow(child=self.lista_conv, vexpand=True)
        lateral.append(rolagem_conv)

        # Seção de Evolução do Isaac (Gráfico)
        rotulo_graf = Gtk.Label(label="Evolução do Isaac", xalign=0)
        rotulo_graf.add_css_class("dim-label")
        rotulo_graf.set_margin_start(12)
        rotulo_graf.set_margin_top(12)
        lateral.append(rotulo_graf)

        self.grafico = Gtk.DrawingArea()
        self.grafico.set_content_height(140)
        self.grafico.set_margin_start(12)
        self.grafico.set_margin_end(12)
        self.grafico.set_margin_bottom(12)
        self.grafico.set_draw_func(self.desenhar_grafico, None)
        lateral.append(self.grafico)

        painel.set_start_child(lateral)

        direita = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Barra de ferramentas da conversa (ID e botões de ação)
        barra_chat = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        barra_chat.set_margin_start(12); barra_chat.set_margin_end(12)
        barra_chat.set_margin_top(6); barra_chat.set_margin_bottom(6)
        
        lbl_id = Gtk.Label(label="ID:")
        lbl_id.add_css_class("dim-label")
        barra_chat.append(lbl_id)
        
        self.entry_id = Gtk.Entry(editable=False, can_focus=True, hexpand=False, width_chars=16)
        self.entry_id.set_valign(Gtk.Align.CENTER)
        self.entry_id.add_css_class("flat")
        barra_chat.append(self.entry_id)
        
        espaco = Gtk.Box(hexpand=True)
        barra_chat.append(espaco)
        
        self.btn_nova = Gtk.Button(label="Nova Conversa", icon_name="document-new-symbolic")
        self.btn_nova.connect("clicked", self.nova_conversa)
        barra_chat.append(self.btn_nova)
        
        self.btn_copiar = Gtk.Button(label="Copiar Conversa", icon_name="edit-copy-symbolic")
        self.btn_copiar.connect("clicked", self.copiar_conversa_clipboard)
        barra_chat.append(self.btn_copiar)
        
        direita.append(barra_chat)

        self.buf = Gtk.TextView(editable=False, cursor_visible=False,
                                wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.buf.set_left_margin(12)
        self.buf.set_right_margin(12)
        self.buf.set_top_margin(12)
        rolagem = Gtk.ScrolledWindow(vexpand=True, child=self.buf)
        direita.append(rolagem)

        # -- barra de autonomia: o que antes exigia me chamar pra digitar comando.
        # Fica ACIMA do terminal de proposito: botao e saida colados, pra ficar
        # obvio que a coisa que apareceu ali veio do botao que ele clicou.
        barra_auto = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        barra_auto.set_margin_start(12); barra_auto.set_margin_end(12)
        barra_auto.set_margin_top(6)

        self.btn_aprender = Gtk.Button(label="Iniciar aprendizado")
        self.btn_aprender.set_tooltip_text(
            "Dispara aprender.py: o professor de nuvem gera exemplos, o portão "
            "mecânico executa cada um, e só o que passa entra no dataset.")
        self.btn_aprender.connect("clicked", self.iniciar_aprendizado)
        barra_auto.append(self.btn_aprender)

        self.alvo_aprender = Gtk.Entry(
            hexpand=True, text="funcoes utilitarias de texto em Python",
            placeholder_text="o que ensinar (tarefa ESTREITA)")
        self.alvo_aprender.set_tooltip_text(
            "Tarefa estreita. 'código em geral' não funciona — está medido.")
        barra_auto.append(self.alvo_aprender)

        barra_auto.append(Gtk.Label(label="n:"))
        self.n_aprender = Gtk.SpinButton.new_with_range(1, 200, 1)
        self.n_aprender.set_value(10)
        self.n_aprender.set_tooltip_text("Quantos exemplos tentar gerar")
        barra_auto.append(self.n_aprender)

        self.btn_juiz = Gtk.Button(label="Rodar ciclo do juiz")
        self.btn_juiz.set_tooltip_text(
            "Dispara loop_juiz.py: o isaac escreve, o juiz avalia requisito por "
            "requisito, e cada ciclo vira um ponto no gráfico ao lado.")
        self.btn_juiz.connect("clicked", self.iniciar_juiz)
        barra_auto.append(self.btn_juiz)

        self.btn_parar_lote = Gtk.Button(label="Parar")
        self.btn_parar_lote.add_css_class("destructive-action")
        self.btn_parar_lote.set_sensitive(False)
        self.btn_parar_lote.set_tooltip_text("Mata o lote em andamento sem fechar o app")
        self.btn_parar_lote.connect("clicked", self.parar_lote)
        barra_auto.append(self.btn_parar_lote)

        direita.append(barra_auto)

        barra_lora = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        barra_lora.set_margin_start(12); barra_lora.set_margin_end(12)
        barra_lora.set_margin_top(6)

        rotulo_colab = Gtk.Label(label="Colab:")
        rotulo_colab.add_css_class("dim-label")
        barra_lora.append(rotulo_colab)

        self.colab_host = Gtk.Entry(
            hexpand=True,
            text=carregar_colab_host(),
            placeholder_text="CF_HOST do Colab atual")
        self.colab_host.set_tooltip_text(
            "Cole so o host trycloudflare.com atual. Ex: logical-...trycloudflare.com")
        barra_lora.append(self.colab_host)

        barra_lora.append(Gtk.Label(label="steps:"))
        self.steps_lora = Gtk.SpinButton.new_with_range(1, 5000, 1)
        self.steps_lora.set_value(120)
        self.steps_lora.set_tooltip_text("Steps do ciclo LoRA no T4")
        barra_lora.append(self.steps_lora)

        barra_lora.append(Gtk.Label(label="ckpt:"))
        self.save_steps_lora = Gtk.SpinButton.new_with_range(1, 500, 1)
        self.save_steps_lora.set_value(20)
        self.save_steps_lora.set_tooltip_text("Salvar checkpoint a cada N steps")
        barra_lora.append(self.save_steps_lora)

        self.btn_lora = Gtk.Button(label="Iniciar ciclos T4")
        self.btn_lora.set_tooltip_text(
            "Roda ciclos_lora_t4.py: envia o seed pack para o Colab/T4, treina "
            "LoRA com checkpoints, copia adapter/relatorio para lora_runs/ e "
            "reports/lora/. Nao funde o adapter.")
        self.btn_lora.connect("clicked", self.iniciar_ciclos_lora)
        barra_lora.append(self.btn_lora)

        direita.append(barra_lora)

        # -- terminal: chamadas de ferramenta E saida crua dos lotes, no mesmo
        # lugar. Sao a mesma coisa do ponto de vista dele — "o que a máquina está
        # fazendo agora" — e dois painéis separados só escondiam metade.
        linha_term = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        linha_term.set_margin_start(12); linha_term.set_margin_end(12)
        rotulo_fer = Gtk.Label(label="terminal", xalign=0)
        rotulo_fer.add_css_class("dim-label")
        linha_term.append(rotulo_fer)
        self.placar = Gtk.Label(label="", xalign=0)
        self.placar.add_css_class("dim-label")
        self.placar.add_css_class("monospace")
        linha_term.append(self.placar)
        direita.append(linha_term)

        self.painel_tools = Gtk.TextView(editable=False, cursor_visible=False,
                                         wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.painel_tools.set_left_margin(12)
        self.painel_tools.add_css_class("monospace")
        # Redimensionavel: quando o lote esta rodando, o terminal e o que importa,
        # e 110px fixos escondiam justamente o traceback que ele precisa ler.
        rolagem_tools = Gtk.ScrolledWindow(child=self.painel_tools,
                                           min_content_height=180, vexpand=True)
        direita.append(rolagem_tools)

        linha = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        linha.set_margin_start(12); linha.set_margin_end(12)
        linha.set_margin_top(6); linha.set_margin_bottom(12)
        self.entrada = Gtk.Entry(hexpand=True, placeholder_text="peça algo… (ex: crie notas.txt com uma lista)")
        self.entrada.connect("activate", self.enviar)
        self.botao = Gtk.Button(label="Enviar")
        self.botao.add_css_class("suggested-action")
        self.botao.connect("clicked", self.enviar)
        linha.append(self.entrada); linha.append(self.botao)
        direita.append(linha)

        painel.set_end_child(direita)
        raiz.append(painel)

        self.set_content(raiz)

        # Inicializa a conversa ativa
        self.conversa_id = None
        self.conversa_dados = None
        self.conversa_leitura = False
        self.nova_conversa()

        self.log(f"pasta de trabalho: {tools.SANDBOX_ROOT}\n")
        self.fotografar_pasta()
        self.montar_diario()
        threading.Thread(target=self.subir, daemon=True).start()

    # -- helpers de UI (sempre voltar pra thread principal com idle_add)
    def _anexar(self, textview, txt):
        b = textview.get_buffer()
        b.insert(b.get_end_iter(), txt)
        textview.scroll_to_mark(b.create_mark(None, b.get_end_iter(), False), 0, False, 0, 0)
        return False

    def log(self, txt):
        GLib.idle_add(self._anexar, self.buf, txt if txt.endswith("\n") else txt + "\n")

    def token(self, pedaco):
        """Streaming: cada pedaco de texto do modelo, sem quebra de linha forcada."""
        GLib.idle_add(self._anexar, self.buf, pedaco)

    def log_tool(self, txt):
        GLib.idle_add(self._anexar, self.painel_tools, txt + "\n")

    # Saida crua no terminal. Mesmo destino do log_tool — nome separado so pra
    # deixar claro na chamada que aquilo veio de um lote, nao de ferramenta.
    terminal = log_tool

    def set_status(self, txt):
        GLib.idle_add(lambda: self.status.set_label(txt) or False)

    # -- diario de arquivos tocados (lateral)
    def fotografar_pasta(self):
        """Foto do que JA existia, pra depois saber o que e novo de verdade.

        Roda ao abrir e ao trocar de pasta. Se falhar (pasta ilegivel), a foto
        fica vazia e tudo aparece como 'novo' — errar pra mais e melhor que
        esconder um arquivo que o isaac criou.
        """
        self.existiam = set()
        try:
            for caminho in self._varrer(tools.SANDBOX_ROOT, 0):
                self.existiam.add(caminho)
        except OSError:
            pass

    def _varrer(self, pasta, profundidade):
        if profundidade > 4:
            return
        try:
            filhos = sorted(pasta.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for f in filhos:
            if f.name.startswith(".") or f.name in IGNORAR:
                continue
            if f.is_dir():
                yield from self._varrer(f, profundidade + 1)
            else:
                yield f.resolve()

    def anotar_toque(self, caminho):
        """Registra que o isaac mexeu neste arquivo. Chamado a cada ferramenta."""
        caminho = Path(caminho).resolve()
        if caminho in self.tocados:
            return  # ja classificado; o primeiro toque e que define novo/alterado
        self.tocados[caminho] = "alterado" if caminho in self.existiam else "novo"

    def montar_diario(self):
        """Lista SO o que o isaac tocou nesta sessao, mais recente em cima."""
        def _():
            while (filho := self.arvore.get_first_child()) is not None:
                self.arvore.remove(filho)

            if not self.tocados:
                vazio = Gtk.Label(label="   (o isaac ainda não tocou em nada)", xalign=0)
                vazio.add_css_class("dim-label")
                self.arvore.append(vazio)
                return False

            for caminho, estado in reversed(list(self.tocados.items())):
                try:
                    nome = str(caminho.relative_to(tools.SANDBOX_ROOT))
                except ValueError:
                    nome = str(caminho)
                marca = "＋" if estado == "novo" else "±"
                texto = f"{marca} {nome}"
                rotulo = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.MIDDLE)
                if caminho == self.arquivo_tocado:
                    rotulo.set_markup(f"<b>{GLib.markup_escape_text(texto)}</b> ●")
                else:
                    rotulo.set_text(texto)
                rotulo.set_tooltip_text(f"{estado} — {caminho}")
                # Precisa ser ListBoxRow de verdade pra emitir row-activated;
                # Label solto so era embrulhado e o clique morria.
                linha_arv = Gtk.ListBoxRow(child=rotulo)
                linha_arv.caminho = caminho
                self.arvore.append(linha_arv)
            return False
        GLib.idle_add(_)

    def abrir_da_arvore(self, _lista, linha):
        """Clique num arquivo do diario: ecoa o caminho completo no terminal.

        NAO despeja o conteudo, de proposito. A Oficina nao e editor — o usuario
        edita no VSCode que ja usa. O que ele precisa daqui e o caminho pra abrir
        la, entao e isso que sai.
        """
        caminho = getattr(linha, "caminho", None)
        if caminho is None:
            return
        estado = self.tocados.get(caminho, "?")
        try:
            tam = f"{caminho.stat().st_size} bytes"
        except OSError:
            tam = "sumiu do disco"
        self.terminal(f"{caminho}  [{estado}, {tam}]")

    # -- seletor de pasta de trabalho
    def escolher_pasta(self, *_):
        dialogo = Gtk.FileDialog(title="Pasta de trabalho do Isaac")
        dialogo.select_folder(self, None, self._pasta_escolhida)

    def _pasta_escolhida(self, dialogo, resultado):
        try:
            pasta = Path(dialogo.select_folder_finish(resultado).get_path())
        except GLib.Error:
            return  # cancelou
        # Guarda-corpo: a raiz do agente nunca pode ser a home inteira nem /.
        # Com a raiz escolhivel, o confinamento do _safe() e a UNICA protecao —
        # entao a raiz precisa ser uma pasta de projeto, nao a vida toda do usuario.
        proibidas = {Path.home(), Path("/"), Path.home() / "Documents", Path.home() / "Documentos"}
        if pasta in proibidas:
            self.log(f"‼ recusei {pasta}: escolha uma pasta de PROJETO, nao a home/raiz inteira.")
            return
        tools.SANDBOX_ROOT = pasta
        # Pasta nova, sessao nova pro diario: os arquivos tocados na pasta
        # anterior nao dizem nada sobre esta, e misturar as duas listas mentiria.
        self.arquivo_tocado = None
        self.tocados = {}
        self.titulo.set_subtitle(str(pasta))
        self.log(f"\n▸ pasta de trabalho agora e: {pasta}")
        self.fotografar_pasta()
        self.montar_diario()

    def checar_atualizacao(self):
        """Compara o digest do granite4 local com o do registro do Ollama.

        Roda em thread e SO AVISA — nunca baixa sozinho. Baixar 2GB sem o dono
        pedir e exatamente o tipo de surpresa que este projeto evita. Falha de
        rede e silenciosa: sem internet o app tem que abrir normal.
        """
        def trabalho():
            try:
                local = json.loads(MANIFESTO_LOCAL.read_text())["config"]["digest"]
            except Exception:
                return                      # modelo nao instalado; subir() ja avisa
            try:
                req = urllib.request.Request(URL_MANIFESTO, headers={
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    remoto = json.load(r)["config"]["digest"]
            except Exception:
                return                      # offline: nao e erro, nao polui a tela
            if remoto != local:
                GLib.idle_add(self.log,
                    f"\n⬆ saiu versao nova do {BASE_REPO}:{BASE_TAG}.\n"
                    f"  atualizar:  ollama pull {BASE_REPO}:{BASE_TAG}\n"
                    f"  depois:     ollama create {MODELO} -f Modelfile.{MODELO}\n"
                    f"  (o segundo passo e necessario: o '{MODELO}' aponta pra "
                    f"versao antiga ate ser recriado)\n")
        threading.Thread(target=trabalho, daemon=True).start()

    def atualizar_luz(self):
        """Pergunta ao ollama QUEM esta carregado agora. Roda a cada 3s.

        Nao usa `ollama ps` por subprocesso: abrir processo a cada 3 segundos
        pesa e pisca. A API /api/ps responde a mesma coisa de graca.
        """
        try:
            with urllib.request.urlopen(URL_CARREGADOS, timeout=1) as r:
                carregados = json.load(r).get("models") or []
        except Exception:
            self.luz.set_label("○ ollama fora")
            return True
        meu = next((m for m in carregados if m.get("name", "").startswith(MODELO)), None)
        if meu:
            gb = (meu.get("size") or 0) / (1024 ** 3)
            self.luz.set_label(f"● {MODELO} · {gb:.1f}GB")
        else:
            self.luz.set_label("○ parado")
        return True        # True = continua repetindo

    def desenhar_grafico(self, area, cr, width, height, user_data):
        # Limpa fundo com um tom cinza escuro ou claro bem sutil, dependendo do tema
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.05)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        diario = tools.SANDBOX_ROOT / "diario_juiz.json"
        dados = []
        if diario.exists():
            try:
                dados = json.loads(diario.read_text(encoding="utf-8"))
            except Exception:
                dados = []
                
        if not dados:
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.6)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(10)
            text = "Nenhum ciclo executado ainda"
            extents = cr.text_extents(text)
            cr.move_to((width - extents.width) / 2, (height + extents.height) / 2)
            cr.show_text(text)
            return

        pad_left = 30
        pad_right = 15
        pad_top = 15
        pad_bottom = 20
        
        chart_w = width - pad_left - pad_right
        chart_h = height - pad_top - pad_bottom
        
        ciclos = [d["ciclo"] for d in dados]
        valores = [d["requisitos_cumpridos"] for d in dados]
        totais = [d.get("total_requisitos", 6) for d in dados]
        max_reqs = max(totais) if totais else 6
        
        n_ciclos = max(5, max(ciclos) if ciclos else 5)
        
        # Linhas de grade horizontais e rótulos do eixo Y
        for r in range(max_reqs + 1):
            y = pad_top + chart_h - (r / max_reqs) * chart_h
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.15)
            cr.move_to(pad_left, y)
            cr.line_to(width - pad_right, y)
            cr.stroke()
            
            if r % 2 == 0 or r == max_reqs:
                cr.set_source_rgba(0.5, 0.5, 0.5, 0.7)
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                cr.set_font_size(8)
                cr.move_to(pad_left - 16, y + 3)
                cr.show_text(str(r))
                
        # Linhas de grade verticais e rótulos do eixo X (ciclos)
        for c in range(1, n_ciclos + 1):
            x = pad_left + ((c - 1) / (n_ciclos - 1)) * chart_w
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.08)
            cr.move_to(x, pad_top)
            cr.line_to(x, height - pad_bottom)
            cr.stroke()
            
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.7)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            cr.set_font_size(8)
            lbl = f"C{c}"
            extents = cr.text_extents(lbl)
            cr.move_to(x - extents.width / 2, height - 6)
            cr.show_text(lbl)
            
        # Eixos X e Y
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.3)
        cr.set_line_width(1.2)
        cr.move_to(pad_left, pad_top)
        cr.line_to(pad_left, height - pad_bottom)
        cr.line_to(width - pad_right, height - pad_bottom)
        cr.stroke()
        
        # Plota a linha da evolução
        if len(dados) > 0:
            cr.set_line_width(2.2)
            # Verde esmeralda elegante
            cr.set_source_rgb(0.18, 0.76, 0.49)
            
            points = []
            for d in dados:
                c = d["ciclo"]
                val = d["requisitos_cumpridos"]
                tot = d.get("total_requisitos", 6)
                
                x = pad_left + ((c - 1) / (n_ciclos - 1)) * chart_w
                y = pad_top + chart_h - (val / tot) * chart_h
                points.append((x, y))
                
            cr.move_to(points[0][0], points[0][1])
            for pt in points[1:]:
                cr.line_to(pt[0], pt[1])
            cr.stroke()
            
            # Marcadores de pontos
            for pt in points:
                cr.set_source_rgb(0.18, 0.76, 0.49)
                cr.arc(pt[0], pt[1], 3.5, 0, 2 * 3.14159)
                cr.fill()
                
                cr.set_source_rgb(1.0, 1.0, 1.0)
                cr.arc(pt[0], pt[1], 1.5, 0, 2 * 3.14159)
                cr.fill()

    # -- lotes de autonomia (aprender / juiz) ---------------------------------

    def _lote_travado(self):
        """Um lote por vez. Dois `aprender.py` juntos so dobram o gasto de credito."""
        if self.lote is not None and self.lote.vivo():
            self.terminal(f"‼ já tem um lote rodando ({self.lote.nome}). "
                          f"Clique em Parar antes de começar outro.")
            return True
        return False

    def _disparar(self, nome, argv, contagem, contador, placar):
        self.contagem = contagem
        self.terminal("")
        self.terminal(f"$ {' '.join(['python3', '-u'] + argv)}")
        self.terminal(f"  (pasta de trabalho: {tools.SANDBOX_ROOT})")

        def on_linha(txt):
            self.terminal(txt)
            if contador(txt, self.contagem):
                GLib.idle_add(lambda: self.placar.set_label(placar(self.contagem)) or False)

        def on_fim(proc, codigo):
            fim = "interrompido pelo usuário" if proc.parando else f"código de saída {codigo}"
            self.terminal(f"— {proc.nome} terminou: {fim} —")
            self.lote = None
            GLib.idle_add(self._botoes_lote, False)
            # O gráfico le o diario do disco; so tem ponto novo depois do fim.
            GLib.idle_add(self.grafico.queue_draw)
            self.montar_diario()

        self.lote = processos.Processo(nome, argv, tools.SANDBOX_ROOT, on_linha, on_fim)
        if self.lote.iniciar():
            self._botoes_lote(True)
            self.placar.set_label(placar(self.contagem))
        else:
            self.lote = None

    def _botoes_lote(self, rodando):
        self.btn_aprender.set_sensitive(not rodando)
        self.btn_juiz.set_sensitive(not rodando)
        self.btn_lora.set_sensitive(not rodando)
        self.btn_parar_lote.set_sensitive(rodando)
        return False

    def iniciar_aprendizado(self, *_):
        if self._lote_travado():
            return
        alvo = self.alvo_aprender.get_text().strip()
        if not alvo:
            self.terminal("‼ diga O QUE ensinar antes de iniciar (tarefa estreita).")
            return
        n = int(self.n_aprender.get_value())
        self._disparar(
            "aprender.py",
            ["aprender.py", "--alvo", alvo, "--n", str(n)],
            {"gerados": 0, "aprovados": 0, "rejeitados": 0, "erro_api": 0},
            processos.contar_aprendizado,
            lambda c: (f"gerados {c['gerados']}/{n} · aprovados {c['aprovados']} · "
                       f"rejeitados {c['rejeitados']} · erro de api {c['erro_api']}"),
        )

    def iniciar_juiz(self, *_):
        if self._lote_travado():
            return
        self._disparar(
            "loop_juiz.py",
            ["loop_juiz.py", "--modelo", MODELO],
            {"ciclos": 0, "cumpridos": 0, "total": 0},
            processos.contar_juiz,
            lambda c: (f"ciclos {c['ciclos']} · último: "
                       f"{c['cumpridos']}/{c['total']} requisitos"
                       if c["ciclos"] else "aguardando o primeiro ciclo…"),
        )

    def iniciar_ciclos_lora(self, *_):
        if self._lote_travado():
            return
        host = self.colab_host.get_text().strip()
        host = host.removeprefix("https://").removeprefix("http://").strip("/")
        if not host:
            self.terminal("‼ cole o CF_HOST do Colab antes de iniciar ciclos T4.")
            return
        salvar_colab_host(host)
        steps = int(self.steps_lora.get_value())
        save_steps = int(self.save_steps_lora.get_value())
        if save_steps > steps:
            self.terminal("‼ ckpt nao pode ser maior que steps.")
            return
        self._disparar(
            "ciclos_lora_t4.py",
            ["ciclos_lora_t4.py", "--cf-host", host,
             "--steps", str(steps), "--save-steps", str(save_steps)],
            {"task05_save": "?", "commit_workflow": "?", "commit_literal_signature": "?",
             "commit_signature": "?", "intent_question": "?", "graphify_navigation": "?",
             "concluido": 0},
            processos.contar_lora,
            lambda c: (f"LoRA T4 · task05 {c['task05_save']} · "
                       f"commit {c.get('commit_workflow') if c.get('commit_workflow') != '?' else c.get('commit_signature', '?')} · "
                       f"assinatura literal {c.get('commit_literal_signature', '?')} · intenção {c['intent_question']} · "
                       f"graphify {c['graphify_navigation']}"
                       if c["concluido"] else
                       f"LoRA T4 rodando · steps {steps} · ckpt {save_steps}"),
        )

    def parar_lote(self, *_):
        if self.lote is None or not self.lote.vivo():
            self._botoes_lote(False)
            return
        self.terminal(f"▸ parando {self.lote.nome}…")
        # Em thread: matar o grupo pode levar ate 5s no SIGTERM, e travar a
        # janela nesse tempo faria parecer que o app pendurou.
        threading.Thread(target=self.lote.parar, daemon=True).start()

    def carregar(self, *_):
        """Carrega o modelo na memoria."""
        def _carregar_e_atualizar():
            try:
                # OLLAMA preloads the model when we send an empty /api/generate request
                data = json.dumps({"model": MODELO}).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:11434/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    r.read() # Consumes response
                self.log(f"▸ modelo {MODELO} carregado com sucesso.")
            except Exception as e:
                self.log(f"‼ erro ao carregar o modelo: {e}")
            # Immediately update the status on the main UI thread
            GLib.idle_add(self.atualizar_luz)

        threading.Thread(target=_carregar_e_atualizar, daemon=True).start()
        self.log(f"\n▸ pedi pro ollama carregar {MODELO} na memoria.")

    def descarregar(self, *_):
        """Tira o modelo da memoria sem fechar o app."""
        def _descarregar_e_atualizar():
            try:
                subprocess.run(["ollama", "stop", MODELO], capture_output=True, timeout=20)
            except Exception as e:
                self.log(f"‼ erro ao parar o modelo: {e}")
            # Wait a brief moment for Ollama to stop and then immediately update UI
            time.sleep(0.5)
            GLib.idle_add(self.atualizar_luz)

        threading.Thread(target=_descarregar_e_atualizar, daemon=True).start()
        self.log(f"\n▸ pedi pro ollama descarregar {MODELO} da memoria.")

    def subir(self):
        tools.SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
        if not OLLAMA.garantir(self.log):
            self.set_status("erro")
            return
        # Confere que o modelo EXISTE antes de anunciar que esta pronto. Sem
        # isso o app dizia "pronto" e so quebrava na primeira pergunta, com o
        # ollama ja no ar sem ninguem usando — o caminho do zumbi.
        try:
            with urllib.request.urlopen(URL_SAUDE, timeout=3) as r:
                nomes = [m["name"] for m in json.load(r).get("models", [])]
        except Exception:
            nomes = []
        if not any(n.startswith(MODELO) for n in nomes):
            self.set_status("modelo ausente")
            self.log(f"‼ o modelo '{MODELO}' nao existe no ollama.\n"
                     f"  crie com:  ollama create {MODELO} -f Modelfile.{MODELO}\n")
            return
        self.set_status(MODELO)
        self.log("pronto. o que vamos fazer?\n")
        self.checar_atualizacao()

    def enviar(self, *_):
        if self.conversa_leitura:
            return
        pedido = self.entrada.get_text().strip()
        if not pedido or self.ocupado:
            return
        self.entrada.set_text("")
        self.ocupado = True
        self.botao.set_sensitive(False)
        self.set_status("pensando…")
        self.log(f"\n▸ você: {pedido}\n◂ {MODELO}: ")
        threading.Thread(target=self.trabalhar, args=(pedido,), daemon=True).start()

    def _tool_antes(self, nome, args):
        """Mostra o que VAI rodar, antes de rodar.

        Importa mesmo pro executar_comando: ele pode levar ate o teto de tempo, e
        sem isto o usuario ficaria olhando pra uma tela parada sem saber o que a
        maquina esta fazendo — que e exatamente o que a Oficina existe pra evitar.
        """
        try:
            dados = json.loads(args) if isinstance(args, str) else (args or {})
        except json.JSONDecodeError:
            dados = {}
        if nome == "executar_comando":
            self.terminal(f"⟩ rodando: {dados.get('cmd', args)}")
        else:
            self.terminal(f"⟩ {nome}({str(args)[:80]})")

    def _tool_ao_vivo(self, nome, args, resultado, via):
        """Chamado pelo agente NO MOMENTO em que cada ferramenta roda (thread de trabalho)."""
        if nome == "executar_comando":
            # Saida de comando vai INTEIRA (o execucao.py ja cortou no teto e
            # avisou). Cortar de novo em 100 chars aqui esconderia justamente o
            # traceback que o usuario quer ler — o oposto do "terminal bruto".
            self.terminal(resultado)
        else:
            self.log_tool(f"⚙ {nome}({str(args)[:80]}) → {resultado[:100]}")
        try:
            caminho = json.loads(args).get("path") if isinstance(args, str) else (args or {}).get("path")
            if caminho:
                self.arquivo_tocado = (tools.SANDBOX_ROOT / caminho).resolve()
                # SO escrita entra no diario. read_file e list_dir nao "tocam" em
                # nada — listar leitura junto encheria a lista de ruido e
                # escondia justamente o que ele mexeu.
                if nome in ("write_file", "append_file"):
                    self.anotar_toque(self.arquivo_tocado)
        except (json.JSONDecodeError, AttributeError):
            pass


        # Gravação incremental da chamada de ferramenta
        if self.conversa_dados and self.conversa_dados["trocas"]:
            self.conversa_dados["trocas"][-1]["chamadas"].append({
                "nome": nome,
                "args": args,
                "resultado": resultado
            })
            self.salvar_conversa_atual()

        self.montar_diario()

    def trabalhar(self, pedido):
        try:
            conh = AQUI / "conhecimento.md"
            if conh.exists():
                agent.CONHECIMENTO_FERRAMENTAS = conh.read_text()

            # Prepara a troca atual nos dados da conversa
            nova_troca = {
                "pedido": pedido,
                "resposta": "",
                "chamadas": [],
                "timestamp": time.time()
            }
            if self.conversa_dados is not None:
                self.conversa_dados["trocas"].append(nova_troca)
                self.salvar_conversa_atual()

            # Chama o agente passando o histórico
            hist = self.conversa_dados["mensagens"] if self.conversa_dados else None
            r = agent.rodar(pedido, MODELO, verbose=False,
                            on_token=self.token, on_tool=self._tool_ao_vivo,
                            on_tool_antes=self._tool_antes,
                            historico=hist)
            self.log("")  # fecha a linha do streaming

            # Atualiza a resposta final e salva
            final_res = r.get("final", "") if r else ""
            if self.conversa_dados and self.conversa_dados["trocas"]:
                self.conversa_dados["trocas"][-1]["resposta"] = final_res
                self.salvar_conversa_atual()
                
            # Verifica o tamanho do contexto e avisa se estiver muito longo
            if hist:
                char_total = sum(len(m.get("content") or "") for m in hist)
                if char_total > 20000:
                    self.log(f"\n⚠️ Aviso: Histórico longo ({char_total} caracteres). "
                             f"Considere clicar em 'Nova Conversa' para reiniciar o contexto se notar lentidão ou falhas.")
        except Exception as e:
            self.log(f"‼ erro: {e}")
            if self.conversa_dados and self.conversa_dados["trocas"]:
                self.conversa_dados["trocas"][-1]["resposta"] = f"‼ erro: {e}"
                self.salvar_conversa_atual()
        finally:
            self.ocupado = False
            GLib.idle_add(lambda: self.botao.set_sensitive(not self.conversa_leitura) or False)
            self.set_status(MODELO)
            self.montar_diario()
            # Atualiza a lista lateral para mostrar a nova conversa/preview
            self.carregar_lista_conversas()
            # Força o redesenho do gráfico de evolução do Isaac
            GLib.idle_add(self.grafico.queue_draw)

    # -- gerenciamento de conversas e persistência -----------------------------

    def salvar_conversa_atual(self):
        if not self.conversa_id or self.conversa_leitura:
            return
        caminho = DIR_CONVERSAS / f"{self.conversa_id}.json"
        try:
            salvar_conversa_atomica(caminho, self.conversa_dados)
        except Exception as e:
            self.log(f"\n‼ erro ao salvar conversa no disco: {e}")

    def carregar_lista_conversas(self):
        def _():
            # Limpa itens antigos da lista
            while (filho := self.lista_conv.get_first_child()) is not None:
                self.lista_conv.remove(filho)
            
            if not DIR_CONVERSAS.exists():
                return False
                
            arquivos = []
            try:
                for f in DIR_CONVERSAS.iterdir():
                    if f.is_file() and f.suffix == ".json":
                        arquivos.append(f)
                arquivos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            except OSError:
                pass
                
            for f in arquivos:
                cid = f.stem
                try:
                    with open(f, 'r', encoding='utf-8') as file_obj:
                        dados = json.load(file_obj)
                    preview = ""
                    if dados.get("trocas"):
                        preview = dados["trocas"][-1].get("pedido", "")[:30]
                    label_text = f"{cid} ({dados.get('modelo', MODELO)})"
                    if preview:
                        label_text += f"\n   {preview}..."
                except Exception:
                    label_text = cid
                
                rotulo = Gtk.Label(label=label_text, xalign=0, ellipsize=Pango.EllipsizeMode.END)
                linha_conv = Gtk.ListBoxRow(child=rotulo)
                linha_conv.conversa_id = cid
                self.lista_conv.append(linha_conv)
            return False
        GLib.idle_add(_)

    def abrir_conversa_antiga(self, _lista, linha):
        cid = getattr(linha, "conversa_id", None)
        if not cid:
            return
        
        caminho = DIR_CONVERSAS / f"{cid}.json"
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
        except Exception as e:
            self.log(f"\n‼ erro ao abrir conversa {cid}: {e}")
            return
            
        self.conversa_id = cid
        self.conversa_dados = dados
        self.conversa_leitura = True
        
        self.entry_id.set_text(cid)
        self.entrada.set_sensitive(False)
        self.botao.set_sensitive(False)
        
        # Limpa o TextView principal e reescreve a conversa
        self.buf.get_buffer().set_text("")
        self.painel_tools.get_buffer().set_text("")
        self.log(f"--- CONVERSA HISTÓRICA: {cid} (Modo Leitura) ---")
        self.log(f"Modelo: {dados.get('modelo', 'desconhecido')}\n")
        
        for troca in dados.get("trocas", []):
            self.log(f"\n▸ você: {troca.get('pedido')}")
            for cham in troca.get("chamadas", []):
                self.log_tool(f"⚙ {cham.get('nome')}({str(cham.get('args'))[:80]}) → {cham.get('resultado')[:100]}")
            self.log(f"◂ {dados.get('modelo', 'modelo')}: {troca.get('resposta')}")
            
        self.log("\n[Fim do histórico. Clique em 'Nova Conversa' para voltar a interagir.]")

    def nova_conversa(self, *_):
        self.conversa_leitura = False
        self.conversa_id = gerar_id_conversa()
        self.conversa_dados = {
            "id": self.conversa_id,
            "modelo": MODELO,
            "timestamp_criacao": time.time(),
            "trocas": [],
            "mensagens": [
                {"role": "system", "content": agent.CONHECIMENTO_FERRAMENTAS}
            ]
        }
        
        self.entry_id.set_text(self.conversa_id)
        self.entrada.set_sensitive(True)
        self.botao.set_sensitive(True)
        
        self.buf.get_buffer().set_text("")
        self.painel_tools.get_buffer().set_text("")
        
        self.log(f"Nova conversa iniciada: {self.conversa_id}")

        self.salvar_conversa_atual()
        self.carregar_lista_conversas()

    def obter_transcript_markdown(self):
        if not self.conversa_dados:
            return ""
        
        md = []
        md.append(f"# Conversa {self.conversa_dados.get('id')}")
        md.append(f"- **Modelo:** {self.conversa_dados.get('modelo')}")
        md.append(f"- **Data/Hora:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.conversa_dados.get('timestamp_criacao', time.time())))}")
        md.append("")
        
        for i, troca in enumerate(self.conversa_dados.get("trocas", []), 1):
            md.append(f"## Turno {i}")
            md.append(f"**Usuário:** {troca.get('pedido')}")
            md.append("")
            
            chamadas = troca.get("chamadas", [])
            if chamadas:
                md.append("<details>")
                md.append(f"<summary>⚙️ Ferramentas executadas ({len(chamadas)})</summary>")
                md.append("")
                for cham in chamadas:
                    md.append(f"**Ferramenta:** `{cham.get('nome')}`")
                    md.append("**Argumentos:**")
                    md.append(f"```json\n{cham.get('args')}\n```")
                    md.append("**Resultado:**")
                    md.append(f"```\n{cham.get('resultado')}\n```")
                    md.append("---")
                md.append("</details>")
                md.append("")
            
            md.append(f"**Isaac:** {troca.get('resposta')}")
            md.append("")
            
        return "\n".join(md)

    def copiar_conversa_clipboard(self, *_):
        markdown_text = self.obter_transcript_markdown()
        if not markdown_text:
            return
        
        display = Gdk.Display.get_default()
        if display:
            clipboard = display.get_clipboard()
            provider = Gdk.ContentProvider.new_for_value(markdown_text)
            clipboard.set_content(provider)
            self.log("\n[✓ Transcript copiado para o clipboard em Markdown!]")


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id="dev.local.AgenteLocal")

    def do_activate(self):
        w = Janela(self)
        w.connect("close-request", self.sair)
        w.present()

    def sair(self, *_):
        _encerrar_tudo()
        self.quit()
        return False


if __name__ == "__main__":
    sys.exit(App().run(sys.argv))
