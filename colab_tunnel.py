# === CÉLULA ÚNICA: abre SSH pro Claude via cloudflared. Nada de projeto antigo. ===
# Cole numa célula do Colab (runtime: T4 GPU) e rode. Ela SEGURA rodando de propósito —
# o túnel morre se a célula terminar. Deixe a aba aberta.

import os, re, subprocess, time, urllib.request

PUBKEY = ("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKdkfPhZ7XaBvvauzIxG+uQNSQClbm36"
          "+HzspKAK7O0L claude-code-tribe")

# --- 0) confirma que a GPU é a que esperamos (senão o treino não cabe) ---
gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "!!! NENHUMA GPU — vá em Ambiente de execução > Alterar tipo > T4 !!!")

# --- 1) limpa restos de execução anterior ---
os.system("pkill -f cloudflared 2>/dev/null; pkill -x sshd 2>/dev/null; sleep 2")

# --- 2) autoriza a chave ---
os.makedirs("/root/.ssh", exist_ok=True)
with open("/root/.ssh/authorized_keys", "w") as f:   # com `with`: garante flush/close
    f.write(PUBKEY + "\n")
os.chmod("/root/.ssh/authorized_keys", 0o600)

# --- 3) sshd ---
# `apt-get update` antes: sem ele o install falha silenciosamente em runtime novo,
# e aí o sshd nunca sobe e o túnel aponta pra porta morta.
os.system("apt-get update -qq >/dev/null 2>&1")
os.system("apt-get install -y -qq openssh-server >/dev/null 2>&1")
os.system("ssh-keygen -A")
os.makedirs("/var/run/sshd", exist_ok=True)
with open("/etc/ssh/sshd_config", "w") as f:
    f.write(
        "Port 22\n"
        "PermitRootLogin prohibit-password\n"   # só chave, nem por acidente senha
        "PubkeyAuthentication yes\n"
        "PasswordAuthentication no\n"
        "UsePAM no\n"
        "ClientAliveInterval 60\n"              # não derrubar sessão ociosa longa
        "ClientAliveCountMax 10\n"
    )
os.system("/usr/sbin/sshd")
time.sleep(2)
porta = subprocess.run("ss -tlnp | grep ':22'", shell=True,
                       capture_output=True, text=True).stdout.strip()
print("sshd na 22:", porta or "!!! NÃO SUBIU — rode a célula de novo !!!")
if not porta:
    raise SystemExit("sshd não subiu; sem isso o túnel não serve pra nada.")

# --- 4) cloudflared ---
if not os.path.exists("/usr/local/bin/cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64", "/usr/local/bin/cloudflared")
    os.chmod("/usr/local/bin/cloudflared", 0o755)

# --- 5) sobe o túnel e segura a célula viva ---
proc = subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "ssh://localhost:22",
     "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

achou = False
for line in proc.stdout:
    m = re.search(r"https://([-\w]+\.trycloudflare\.com)", line)
    if m and not achou:
        achou = True
        print("\n" + "=" * 60)
        print(f"CF_HOST={m.group(1)}")
        print("=" * 60)
        print("Copie a linha CF_HOST acima e mande pro Claude.")
        print("Deixe esta célula rodando — se ela parar, o túnel cai.\n")
    elif not achou:
        print(line, end="")   # só mostra ruído até achar a URL
