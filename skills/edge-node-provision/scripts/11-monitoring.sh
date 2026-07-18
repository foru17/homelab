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

# --- Nezha (agent v2): download binary + write config.yml + systemd (robust across arch;
#     the one-liner installer changes often, so we do it explicitly). ---
if [ -n "${NEZHA_SERVER:-}" ] && [ -n "${NEZHA_SECRET:-}" ]; then
  ARCH="$(box_arch)"
  [ -n "${NEZHA_UUID:-}" ] || NEZHA_UUID="$(run 'cat /proc/sys/kernel/random/uuid')"
  run "set -e
    mkdir -p /opt/nezha/agent
    ver=\$(curl -sL https://api.github.com/repos/nezhahq/agent/releases/latest | grep -oE '\"tag_name\": *\"[^\"]+' | head -1 | grep -oE 'v[0-9.]+')
    url=https://github.com/nezhahq/agent/releases/download/\$ver/nezha-agent_linux_${ARCH}.zip
    curl -fsSL \"\$url\" -o /tmp/nezha-agent.zip
    (command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip >/dev/null))
    unzip -oq /tmp/nezha-agent.zip -d /opt/nezha/agent && rm -f /tmp/nezha-agent.zip
    chmod +x /opt/nezha/agent/nezha-agent
    cat > /opt/nezha/agent/config.yml <<CFG
client_secret: $NEZHA_SECRET
server: $NEZHA_SERVER
tls: ${NEZHA_TLS:-false}
uuid: $NEZHA_UUID
disable_auto_update: true
disable_force_update: true
CFG
    cat > /etc/systemd/system/nezha-agent.service <<'SVC'
[Unit]
Description=Nezha Agent
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
ExecStart=/opt/nezha/agent/nezha-agent -c /opt/nezha/agent/config.yml
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
SVC
    systemctl daemon-reload
    systemctl enable --now nezha-agent
    sleep 3; systemctl is-active nezha-agent"
  echo "Nezha: agent installed (uuid $NEZHA_UUID, arch $ARCH). Name/group the node in your panel."
else
  echo "Nezha: skipped (set NEZHA_SERVER + NEZHA_SECRET to enable)."
fi
