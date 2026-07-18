#!/usr/bin/env bash
# Install cloudflare-warp, register, set proxy (SOCKS5) mode, connect, verify warp=on.
. "$(dirname "$0")/_lib.sh"

run "
  # keyring needs gnupg present (else 'sqv' keyring-parse error on apt update)
  apt-get install -y -qq gnupg >/dev/null
  curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ bookworm main' > /etc/apt/sources.list.d/cloudflare-client.list
  apt-get update -qq
  apt-get install -y -qq cloudflare-warp >/dev/null
  warp-cli --accept-tos registration new  >/dev/null 2>&1 || true
  warp-cli --accept-tos mode proxy         >/dev/null 2>&1
  warp-cli --accept-tos proxy port $WARP_SOCKS_PORT >/dev/null 2>&1
  warp-cli --accept-tos connect            >/dev/null 2>&1
  sleep 5
  echo '== verify (expect warp=on) =='
  curl -s -x socks5h://127.0.0.1:$WARP_SOCKS_PORT --max-time 10 https://1.1.1.1/cdn-cgi/trace | grep -E 'ip=|warp='"
