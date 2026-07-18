#!/usr/bin/env bash
# Push 09-aapanel-sites.py and run it with aaPanel's bundled python + your config.
. "$(dirname "$0")/_lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

push "$HERE/09-aapanel-sites.py" /root/_aapanel-sites.py
run "DOMAIN_ROOT='$DOMAIN_ROOT' SUB_PANEL='$SUB_PANEL' SUB_XUI='$SUB_XUI' SUB_COVER='$SUB_COVER' \
     WS_PATH_DIRECT='$WS_PATH_DIRECT' WS_PATH_WARP='$WS_PATH_WARP' \
     PORT_DIRECT='$PORT_DIRECT' PORT_WARP='$PORT_WARP' XUI_PORT='$XUI_PORT' \
     /www/server/panel/pyenv/bin/python /root/_aapanel-sites.py
     rm -f /root/_aapanel-sites.py
     /etc/init.d/nginx reload 2>&1 | tail -1"
