#!/usr/bin/env bash
# Unmatched server_name -> return 444 (drop). Also drop the landing page.
. "$(dirname "$0")/_lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"

# landing page
sed "s/__NODE_NAME__/$NODE_NAME/g" "$HERE/../templates/landing.html" | run "mkdir -p /www/wwwroot/$DOMAIN_ROOT && cat > /www/wwwroot/$DOMAIN_ROOT/index.html"

run "cat > /www/server/panel/vhost/nginx/default_444.conf <<'CONF'
server {
    listen 443 ssl default_server;
    server_name _;
    ssl_certificate /www/server/panel/vhost/cert/$DOMAIN_ROOT/fullchain.pem;
    ssl_certificate_key /www/server/panel/vhost/cert/$DOMAIN_ROOT/privkey.pem;
    return 444;
}
CONF
/www/server/nginx/sbin/nginx -t && /etc/init.d/nginx reload 2>&1 | tail -1"
