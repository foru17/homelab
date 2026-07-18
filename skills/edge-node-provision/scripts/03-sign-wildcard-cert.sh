#!/usr/bin/env bash
# Issue a wildcard cert for *.<DOMAIN_ROOT> on the box via acme.sh DNS-01,
# deploy into the aaPanel BT cert dirs, and register a renew --reloadcmd.
# DNS provider is set by ACME_DNS_PLUGIN (default dns_cf); creds are passed through from config.env.
# Fleet alternative (sign once, scp to many): references/centralized-certs.md
. "$(dirname "$0")/_lib.sh"

PLUGIN="${ACME_DNS_PLUGIN:-dns_cf}"
# Export whichever acme.sh DNS credential vars are set in config.env (only non-empty ones).
CRED_EXPORTS=""
for v in CF_API_TOKEN CF_Zone_ID DP_Id DP_Key Ali_Key Ali_Secret AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY GD_Key GD_Secret; do
  [ -n "${!v:-}" ] && CRED_EXPORTS+="export $v='${!v}'; "
done
# acme.sh's Cloudflare plugin expects CF_Token, so alias it from CF_API_TOKEN.
[ -n "${CF_API_TOKEN:-}" ] && CRED_EXPORTS+="export CF_Token='${CF_API_TOKEN}'; "

SITES=("$DOMAIN_ROOT" "$SUB_COVER.$DOMAIN_ROOT" "$SUB_XUI.$DOMAIN_ROOT" "$SUB_PANEL.$DOMAIN_ROOT")
SITES_CSV=$(IFS=,; echo "${SITES[*]}")

run "$CRED_EXPORTS
  curl -s https://get.acme.sh | sh -s email=admin@$PARENT_ZONE >/dev/null 2>&1 || true
  ~/.acme.sh/acme.sh --set-default-ca --server letsencrypt >/dev/null 2>&1 || true
  ~/.acme.sh/acme.sh --issue --dns $PLUGIN -d '$DOMAIN_ROOT' -d '*.$DOMAIN_ROOT' --keylength ec-256 --server letsencrypt 2>&1 | tail -4

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
