#!/usr/bin/env bash
# Install aaPanel International (7.x) headless, then print the default credentials.
. "$(dirname "$0")/_lib.sh"

run 'cd /root && curl -sSO https://www.aapanel.com/script/install_7.0_en.sh && ls -la install_7.0_en.sh'
echo "== installing aaPanel in background (5-10 min) =="
run 'cd /root && (echo y | bash install_7.0_en.sh aapanel > /root/aapanel-install.log 2>&1 &) && echo started'

echo "== waiting for panel binary =="
until run 'test -f /www/server/panel/BT-Panel && ! pgrep -f install_7.0 >/dev/null'; do sleep 15; done
echo "== aaPanel installed. Default credentials (change in 05-harden): =="
run 'grep -iE "Address|username|password" /root/aapanel-install.log | grep -ivE "wget|Resolving|Connecting|HTTP request|Saving|node.aapanel|systemd" | tail -6'
