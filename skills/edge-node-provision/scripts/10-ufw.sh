#!/usr/bin/env bash
# Allow web + panel + aaPanel FTP/passive ports.
. "$(dirname "$0")/_lib.sh"

run "for p in 22 80 443 $AAPANEL_PORT 888; do ufw allow \$p/tcp >/dev/null 2>&1; done
     ufw allow 39000:40000/tcp >/dev/null 2>&1
     yes | ufw enable >/dev/null 2>&1 || true
     ufw status | grep -E '22|80|443|$AAPANEL_PORT|888|39000'"
