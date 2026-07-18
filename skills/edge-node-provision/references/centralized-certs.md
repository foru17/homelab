# Centralized wildcard certs (fleet option)

`03-sign-wildcard-cert.sh` signs the wildcard **on the box** — simplest for a single node.
For a fleet you may prefer to sign every wildcard on **one** host and push to the rest, so
the Cloudflare DNS token lives in exactly one place and renewals are centralized.

## Model

```
        one signer host (acme.sh + CF token)
        ┌───────────────────────────────────┐
        │  acme.sh --dns dns_cf              │  issues *.nodeA.example.com,
        │  --reloadcmd distribute.sh         │         *.nodeB.example.com, …
        └───────────────┬───────────────────┘
                        │ scp fullchain+key on every renew
          ┌─────────────┼─────────────┐
        nodeA         nodeB         nodeC       (BT cert dirs + nginx reload)
```

## Signer setup (once)

```bash
export CF_Token='<token>'
acme.sh --set-default-ca --server letsencrypt
acme.sh --issue --dns dns_cf -d nodeA.example.com -d '*.nodeA.example.com' --keylength ec-256
```

## Distribution reloadcmd

A small script the signer runs after each renewal — for each target it scp's the cert into
the aaPanel cert dirs and reloads nginx. Register it once:

```bash
acme.sh --install-cert -d nodeA.example.com --ecc --reloadcmd "/root/acme-deploy/distribute.sh"
```

`distribute.sh` (sketch — adapt hosts/sites to yours; **no real hosts here**):

```bash
TARGETS=(
  "<nodeA_ip>:root:nodeA.example.com,<cover>.nodeA.example.com,<xui>.nodeA.example.com,<panel>.nodeA.example.com"
)
SRC=/root/.acme.sh/nodeA.example.com_ecc
for entry in "${TARGETS[@]}"; do
  host="${entry%%:*}"; rest="${entry#*:}"; user="${rest%%:*}"; sites="${rest#*:}"
  scp -o BatchMode=yes "$SRC/fullchain.cer" "$user@$host:/tmp/.fc.pem"
  scp -o BatchMode=yes "$SRC/nodeA.example.com.key" "$user@$host:/tmp/.pk.pem"
  ssh "$user@$host" "IFS=,; for s in \$sites; do d=/www/server/panel/vhost/cert/\$s; mkdir -p \$d;
      install -m 644 /tmp/.fc.pem \$d/fullchain.pem; install -m 600 /tmp/.pk.pem \$d/privkey.pem; done;
      rm -f /tmp/.fc.pem /tmp/.pk.pem; /www/server/nginx/sbin/nginx -s reload"
done
```

## Trust + notes

- Put the signer's SSH **public** key in each target's `authorized_keys`; the signer needs write to the BT cert dirs + nginx reload only.
- To add a region to an existing wildcard, `--issue` again with the extra `-d` names (SAN change requires a re-issue, not `--renew`), then extend `TARGETS`.
- Keep the CF token on the signer only; targets never see it.
