#!/usr/bin/env bash
# Optional: install Nezha + Komari agents pointing at YOUR panels (placeholders in config.env).
# Both are skipped unless the relevant config.env fields are set. No panel URLs/keys ship here.
. "$(dirname "$0")/_lib.sh"

# --- Komari (auto-discovery) ---
if [ -n "${KOMARI_ENDPOINT:-}" ] && [ -n "${KOMARI_DISCOVERY_KEY:-}" ]; then
  run "curl -sSL https://raw.githubusercontent.com/komari-monitor/komari-agent/main/install.sh \
        | bash -s -- --endpoint '$KOMARI_ENDPOINT' --auto-discovery '$KOMARI_DISCOVERY_KEY' 2>&1 | tail -3
       systemctl is-active komari-agent 2>/dev/null || true"
  echo "Komari: agent installed. Rename/group the node in your panel (it auto-registers as 'Auto-<hostname>')."
else
  echo "Komari: skipped (set KOMARI_ENDPOINT + KOMARI_DISCOVERY_KEY to enable)."
fi

# --- Nezha (agent v2, official one-liner) ---
if [ -n "${NEZHA_SERVER:-}" ] && [ -n "${NEZHA_SECRET:-}" ]; then
  [ -n "${NEZHA_UUID:-}" ] || NEZHA_UUID="$(run 'cat /proc/sys/kernel/random/uuid')"
  run "curl -L https://raw.githubusercontent.com/nezhahq/scripts/main/agent/install.sh -o /tmp/nezha.sh 2>/dev/null
       env NZ_SERVER='$NEZHA_SERVER' NZ_TLS='$NEZHA_TLS' NZ_CLIENT_SECRET='$NEZHA_SECRET' NZ_UUID='$NEZHA_UUID' \
         bash /tmp/nezha.sh 2>&1 | tail -4 || echo 'nezha install script changed? see references/pitfalls.md for the manual binary+config method'
       systemctl is-active nezha-agent 2>/dev/null || true"
  echo "Nezha: agent installed (uuid $NEZHA_UUID). Name/group the node in your panel."
else
  echo "Nezha: skipped (set NEZHA_SERVER + NEZHA_SECRET to enable)."
fi
