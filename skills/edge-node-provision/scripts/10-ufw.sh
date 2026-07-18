#!/usr/bin/env bash
# Allow web + panel + aaPanel FTP/passive ports. Reads the real panel port from the box.
. "$(dirname "$0")/_lib.sh"

run 'P=$(cat /www/server/panel/data/port.pl 2>/dev/null | tr -d "[:space:]"); P=${P:-8888}
     for p in 22 80 443 "$P" 888; do ufw allow "$p"/tcp >/dev/null 2>&1; done
     ufw allow 39000:40000/tcp >/dev/null 2>&1
     yes | ufw enable >/dev/null 2>&1 || true
     ufw status | grep -E "22|80|443|$P|888|39000"'
