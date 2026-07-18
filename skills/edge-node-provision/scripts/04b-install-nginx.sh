#!/usr/bin/env bash
# Install nginx 1.24 via aaPanel's install_soft.sh (compile; a few minutes).
. "$(dirname "$0")/_lib.sh"

run 'cd /www/server/panel/install && (bash install_soft.sh 0 install nginx 1.24 > /root/nginx-install.log 2>&1 &) && echo started'
echo "== waiting for nginx binary (compile) =="
until run 'test -x /www/server/nginx/sbin/nginx'; do sleep 20; done
run '/www/server/nginx/sbin/nginx -v 2>&1'
