"""Roda os scripts de lote (aprender.py, loop_juiz.py) como subprocesso, com a
saida CRUA voltando linha a linha pra tela.

Por que existe (task 01): tudo aqui ja funcionava por terminal. O que faltava era
o usuario conseguir disparar sozinho, sem me chamar pra digitar comando. Entao
isto nao "melhora" nada — so vira botao o que ja rodava.

REGRA DO CICLO DE VIDA: fechar a Oficina mata os filhos. `aprender.py` faz
chamada de rede em lote; processo orfao continua gastando credito com a janela
ja fechada. Por isso cada filho nasce em SESSAO PROPRIA (start_new_session) —
assim da pra matar o GRUPO inteiro, e nao so o `python3` da ponta, que deixaria
netos vivos.

Saida sem buffer (`python3 -u`): sem isso o pipe segura ~4KB e a tela fica parada
por minutos enquanto o lote roda. O usuario precisa VER acontecendo.
"""
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# Todo processo vivo disparado pela Oficina. O app varre isto ao fechar.
_VIVOS = []
_TRAVA = threading.Lock()


class Processo:
    """Um script de lote rodando, com a saida sendo entregue linha a linha."""

    def __init__(self, nome, argv, raiz, on_linha, on_fim):
        self.nome = nome
        self.argv = argv
        self.raiz = raiz
        self._on_linha = on_linha
        self._on_fim = on_fim
        self.proc = None
        self.parando = False

    def iniciar(self):
        env = dict(os.environ)
        # O subprocesso tem que trabalhar na MESMA pasta que o usuario escolheu
        # no header, senao o lote roda numa raiz e a arvore mostra outra.
        env["AGENTE_RAIZ"] = str(self.raiz)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", *self.argv],
                cwd=str(AQUI),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # erro e saida na MESMA corrente, em ordem
                text=True,
                bufsize=1,
                start_new_session=True,     # grupo proprio: da pra matar tudo junto
            )
        except OSError as e:
            self._on_linha(f"‼ nao consegui iniciar {self.nome}: {e}")
            self._on_fim(self, -1)
            return False

        with _TRAVA:
            _VIVOS.append(self)
        threading.Thread(target=self._drenar, daemon=True).start()
        return True

    def _drenar(self):
        """Le a saida ate o fim. Nada de resumir ou filtrar: o usuario quer o cru."""
        try:
            for linha in self.proc.stdout:
                self._on_linha(linha.rstrip("\n"))
        except (OSError, ValueError):
            pass  # pipe fechado por parar() — esperado, nao e erro
        codigo = self.proc.wait()
        with _TRAVA:
            if self in _VIVOS:
                _VIVOS.remove(self)
        self._on_fim(self, codigo)

    def vivo(self):
        return self.proc is not None and self.proc.poll() is None

    def parar(self):
        """SIGTERM no GRUPO, e SIGKILL se teimar. Nunca so no processo da ponta."""
        if not self.vivo():
            return
        self.parando = True
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except OSError:
                pass
        except (OSError, ProcessLookupError):
            pass


def matar_todos():
    """Chamado ao fechar a Oficina. Idempotente: pode rodar duas vezes sem doer."""
    with _TRAVA:
        atuais = list(_VIVOS)
    for p in atuais:
        p.parar()


def contar_aprendizado(linha, contagem):
    """Le os contadores da saida do aprender.py sem mudar o aprender.py.

    Casa com o que ele ja imprime: '[3/10] aprovado (gemini-...)',
    '[4/10] REJEITADO ...', '[5/10] ERRO API: ...'.
    """
    t = linha.strip()
    if "] aprovado" in t:
        contagem["aprovados"] += 1
    elif "] REJEITADO" in t:
        contagem["rejeitados"] += 1
    elif "] ERRO API" in t:
        contagem["erro_api"] += 1
    else:
        return False
    contagem["gerados"] = (contagem["aprovados"] + contagem["rejeitados"]
                           + contagem["erro_api"])
    return True


def contar_juiz(linha, contagem):
    """Le 'Requisitos Cumpridos: 4/6' da saida do loop_juiz.py."""
    t = linha.strip()
    marca = "Requisitos Cumpridos:"
    if marca not in t:
        return False
    try:
        fracao = t.split(marca, 1)[1].strip().split()[0]
        cumpridos, total = fracao.split("/")
        contagem["cumpridos"] = int(cumpridos)
        contagem["total"] = int(total)
        contagem["ciclos"] += 1
        return True
    except (ValueError, IndexError):
        return False


def contar_lora(linha, contagem):
    """Lê o resumo final de ciclos_lora_t4.py.

    O treino imprime muita saida crua do Trainer; o placar so precisa do resumo
    final antes/depois para nao tentar interpretar barra de progresso.
    """
    t = linha.strip()
    for chave in (
        "task05_save",
        "commit_workflow",
        "commit_literal_signature",
        "commit_signature",
        "intent_question",
        "graphify_navigation",
    ):
        if t.startswith(chave + ":"):
            contagem[chave] = t.split(":", 1)[1].strip()
            return True
    if "ciclo concluido" in t:
        contagem["concluido"] = 1
        return True
    return False
