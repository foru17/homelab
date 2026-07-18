#!/usr/bin/env bash
# Harden aaPanel: set username/password, secret entrance, bind panel domain, enable CDN_PROXY
# (required so the panel works behind an nginx reverse proxy).
. "$(dirname "$0")/_lib.sh"

[ -n "${AAPANEL_PASS:-}" ] || AAPANEL_PASS="$(gen_pass 20)"
echo "AAPANEL PASSWORD (save to your password manager): $AAPANEL_PASS"

run "cd /www/server/panel
  /www/server/panel/pyenv/bin/python tools.py panel '$AAPANEL_PASS' >/dev/null 2>&1 && echo password-set
  # username: set directly in the panel DB (tools.py username is interactive)
  /www/server/panel/pyenv/bin/python -c \"import sqlite3;c=sqlite3.connect('/www/server/panel/data/default.db');c.execute(\\\"update users set username='$AAPANEL_USER'\\\");c.commit();print('username ->',c.execute('select username from users').fetchall())\"
  echo '$AAPANEL_ENTRANCE' > data/admin_path.pl && echo entrance-set
  echo '$SUB_PANEL.$DOMAIN_ROOT' > data/domain.conf && echo domain-bound
  echo 'CDN_PROXY=True' > config/cdn.conf && echo cdn-proxy-on
  /etc/init.d/bt restart >/dev/null 2>&1; sleep 3
  ss -lnt | grep -q ':$AAPANEL_PORT' && echo panel-listening || echo 'WARN: panel not listening on $AAPANEL_PORT (check with: bt 14)'"
echo "NOTE: reverse-proxy access at https://$SUB_PANEL.$DOMAIN_ROOT$AAPANEL_ENTRANCE (after 09-aapanel-sites)."
echo "NOTE: direct IP:port returns 403 by design (domain binding). Restart panel with 'bt 1', never 'systemctl restart btpanel'."
