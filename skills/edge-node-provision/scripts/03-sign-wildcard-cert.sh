#!/usr/bin/env bash
# Issue a wildcard cert for *.<DOMAIN_ROOT> on the box via acme.sh DNS-01 (Cloudflare),
# deploy into the aaPanel BT cert dirs, and register a renew --reloadcmd.
# Fleet alternative (sign once, scp to many): references/centralized-certs.md
. "$(dirname "$0")/_lib.sh"

SITES=("$DOMAIN_ROOT" "$SUB_COVER.$DOMAIN_ROOT" "$SUB_XUI.$DOMAIN_ROOT" "$SUB_PANEL.$DOMAIN_ROOT")
SITES_CSV=$(IFS=,; echo "${SITES[*]}")

run "export CF_Token='$CF_API_TOKEN'
  curl -s https://get.acme.sh | sh -s email=admin@$PARENT_ZONE >/dev/null 2>&1 || true
  ~/.acme.sh/acme.sh --set-default-ca --server letsencrypt >/dev/null 2>&1 || true
  ~/.acme.sh/acme.sh --issue --dns dns_cf -d '$DOMAIN_ROOT' -d '*.$DOMAIN_ROOT' --keylength ec-256 --server letsencrypt 2>&1 | tail -4

  # deploy into BT cert dirs + reload nginx
  SRC=~/.acme.sh/${DOMAIN_ROOT}_ecc
  for s in ${SITES_CSV//,/ }; do
    d=/www/server/panel/vhost/cert/\$s; mkdir -p \$d
    install -m 644 \$SRC/fullchain.cer \$d/fullchain.pem
    install -m 600 \$SRC/${DOMAIN_ROOT}.key \$d/privkey.pem
  done
  ( /www/server/nginx/sbin/nginx -t && /www/server/nginx/sbin/nginx -s reload ) 2>/dev/null || echo 'nginx not up yet (fine — reload after aaPanel install)'

  # register reloadcmd so renewals re-deploy automatically
  ~/.acme.sh/acme.sh --install-cert -d '$DOMAIN_ROOT' --ecc \
    --reloadcmd \"for s in ${SITES_CSV//,/ }; do d=/www/server/panel/vhost/cert/\\\$s; install -m 644 \\\$SRC/fullchain.cer \\\$d/fullchain.pem; install -m 600 \\\$SRC/${DOMAIN_ROOT}.key \\\$d/privkey.pem; done; /www/server/nginx/sbin/nginx -s reload\" 2>&1 | tail -2
  echo cert-done"
