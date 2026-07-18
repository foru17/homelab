#!/usr/bin/env bash
# Install 3x-ui and set the panel to listen on 127.0.0.1:XUI_PORT with your path/creds.
. "$(dirname "$0")/_lib.sh"

[ -n "${XUI_PASS:-}" ] || XUI_PASS="$(gen_pass 18)"
echo "3X-UI PASSWORD (save to your password manager): $XUI_PASS"

run "echo -e 'n' | bash <(curl -Ls https://raw.githubusercontent.com/mhsanaei/3x-ui/master/install.sh) >/dev/null 2>&1 || true
  /usr/local/x-ui/x-ui setting -username '$XUI_USER' -password '$XUI_PASS' -port $XUI_PORT -webBasePath '$XUI_PATH' 2>&1 | tail -4
  /usr/local/x-ui/x-ui setting -listenIP 127.0.0.1 2>&1 | tail -1
  systemctl restart x-ui; sleep 4
  echo '== status =='; systemctl is-active x-ui
  ss -lnt | grep -q ':$XUI_PORT' && echo panel-listening-$XUI_PORT || echo 'WARN: panel not listening'"
