#!/usr/bin/env bash
# Push 08-xui-bootstrap.py to the box and run it with your config injected as env vars.
. "$(dirname "$0")/_lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -n "${XUI_UUID:-}" ] || { echo "ERROR: set XUI_UUID in config.env (uuidgen)"; exit 1; }

push "$HERE/08-xui-bootstrap.py" /root/_xui-bootstrap.py
run "XUI_UUID='$XUI_UUID' XUI_SUBID='$XUI_SUBID' XUI_EMAIL='node-main' \
     PORT_DIRECT='$PORT_DIRECT' PORT_WARP='$PORT_WARP' \
     WS_PATH_DIRECT='$WS_PATH_DIRECT' WS_PATH_WARP='$WS_PATH_WARP' \
     WARP_SOCKS_PORT='$WARP_SOCKS_PORT' \
     python3 /root/_xui-bootstrap.py; rm -f /root/_xui-bootstrap.py"
