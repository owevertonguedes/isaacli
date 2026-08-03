#!/usr/bin/env bash
# Conecta neste PC ao Colab pelo túnel cloudflared.
#   ./conectar_colab.sh <algo>.trycloudflare.com  [comando remoto...]
# Sem comando, abre sessão interativa. Com comando, roda e sai (bom pra automação).
set -euo pipefail

HOST="${1:?uso: $0 <CF_HOST> [comando]}"; shift || true
CHAVE="$HOME/.ssh/colab_tribe"

[[ -f "$CHAVE" ]] || { echo "chave $CHAVE não existe"; exit 1; }
command -v cloudflared >/dev/null || { echo "cloudflared não instalado"; exit 1; }

# ProxyCommand é obrigatório: o trycloudflare.com não fala SSH direto, o
# cloudflared traduz a conexão. Sem isso dá "connection reset" e parece firewall.
SSH_OPTS=(
  -i "$CHAVE"
  -o "ProxyCommand=cloudflared access ssh --hostname %h"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ServerAliveInterval=30
)

# O Colab só injeta o caminho do driver NVIDIA no kernel do notebook — a sessão
# SSH nasce sem ele, e aí torch.cuda.is_available() dá False com a GPU anexada
# (/dev/nvidia0 existe). Exportar aqui evita cair nessa em todo comando.
ENV_CUDA='export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH;
          export PATH=/opt/bin:/usr/local/cuda/bin:$PATH;'

if [[ $# -eq 0 ]]; then
  exec ssh -t "${SSH_OPTS[@]}" "root@$HOST" "$ENV_CUDA exec bash -l"
else
  exec ssh "${SSH_OPTS[@]}" "root@$HOST" "$ENV_CUDA $*"
fi
