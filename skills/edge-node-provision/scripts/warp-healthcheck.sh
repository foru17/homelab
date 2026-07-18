#!/usr/bin/env bash
# WARP proxy watchdog: WARP can show "Connected" while the SOCKS proxy stops egressing.
# Checks warp=on through the SOCKS proxy; restarts warp-svc + reconnects if not.
set -euo pipefail
PROXY_ADDR="${PROXY_ADDR:-socks5h://127.0.0.1:40000}"
TRACE_URL="${TRACE_URL:-https://www.cloudflare.com/cdn-cgi/trace}"
TIMEOUT="${CURL_TIMEOUT:-15}"

log() { logger -t warp-healthcheck "$*"; echo "warp-healthcheck: $*"; }

if ! ss -lnt | grep -q '127.0.0.1:40000'; then
  log "SOCKS listener missing; restarting warp-svc"
elif trace="$(curl -k -sS --max-time "$TIMEOUT" --proxy "$PROXY_ADDR" "$TRACE_URL" 2>/dev/null)" && grep -q 'warp=on' <<<"$trace"; then
  exit 0   # healthy
else
  log "warp not on via proxy; recovering"
fi

systemctl restart warp-svc 2>/dev/null || true
sleep 8
warp-cli --accept-tos connect 2>/dev/null || true
