#!/usr/bin/env bash
# Install the WARP health-check watchdog (systemd timer, every 60s).
. "$(dirname "$0")/_lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# adjust the proxy port in the shipped script to your configured one
sed "s#127.0.0.1:40000#127.0.0.1:$WARP_SOCKS_PORT#g" "$HERE/warp-healthcheck.sh" | run "cat > /usr/local/bin/warp-healthcheck.sh"
push "$HERE/warp-healthcheck.service" /etc/systemd/system/warp-healthcheck.service
push "$HERE/warp-healthcheck.timer"   /etc/systemd/system/warp-healthcheck.timer
run 'chmod +x /usr/local/bin/warp-healthcheck.sh
     systemctl daemon-reload
     systemctl enable --now warp-healthcheck.timer 2>&1 | tail -1
     systemctl is-active warp-healthcheck.timer'
