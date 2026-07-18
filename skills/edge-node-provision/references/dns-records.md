# DNS records this node needs (any provider)

The wildcard cert (`03-sign-wildcard-cert.sh`) proves domain control via **DNS-01** using
whatever `ACME_DNS_PLUGIN` you set — acme.sh supports Cloudflare, DNSPod, Aliyun, Route53,
GoDaddy, and dozens more. That part is provider-agnostic.

The only thing that is *not* automated for non-Cloudflare providers is creating the host
**A/AAAA records**. That's just two names. Create them in your DNS provider's console/API:

| Name | Type | Value | Proxy/CDN |
|---|---|---|---|
| `<DOMAIN_ROOT>` (e.g. `nodeA.example.com`) | A | `SERVER_IP` | **off** (DNS-only / grey cloud) |
| `*.<DOMAIN_ROOT>` | A | `SERVER_IP` | **off** |
| `<DOMAIN_ROOT>` | AAAA | `SERVER_IPV6` | off (skip if no IPv6) |
| `*.<DOMAIN_ROOT>` | AAAA | `SERVER_IPV6` | off |

Notes:
- Keep them **unproxied** — the node terminates its own TLS and needs the real client IP path; a proxy/CDN in front would break the WS cover and the panel anti-scan logic.
- TTL 300 is fine.
- The wildcard `*.<DOMAIN_ROOT>` covers the landing / cover / panel subdomains, so you don't add per-subdomain records.

## Cloudflare shortcut

If your zone is on Cloudflare, skip the manual step — `02-cf-dns.py` upserts all four
records for you via the API (`CF_API_TOKEN` with `Zone:DNS:Edit`). Run it from wherever the
token lives; it never touches the VPS.

## Picking the acme.sh plugin + creds

Set `ACME_DNS_PLUGIN` and the matching credential vars in `config.env` (acme.sh reads them
from the environment; `03-sign-wildcard-cert.sh` forwards whichever are set):

| Provider | `ACME_DNS_PLUGIN` | env vars |
|---|---|---|
| Cloudflare | `dns_cf` | `CF_API_TOKEN` (forwarded as `CF_Token`) |
| DNSPod | `dns_dp` | `DP_Id`, `DP_Key` |
| Aliyun | `dns_ali` | `Ali_Key`, `Ali_Secret` |
| AWS Route53 | `dns_aws` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| GoDaddy | `dns_gd` | `GD_Key`, `GD_Secret` |
| others | see acme.sh dnsapi | per its README |

Full list: <https://github.com/acmesh-official/acme.sh/wiki/dnsapi>
