---
name: edge-node-provision
description: Provision a fresh Linux VPS into a well-managed edge node — aaPanel (international) + nginx, 3x-ui (Xray) with a direct inbound and a Cloudflare-WARP "clean-exit" inbound, a wildcard TLS cert via acme.sh DNS-01, Cloudflare wildcard DNS, a WARP health-check watchdog, and optional Nezha/Komari monitoring agents. Use when the user says they bought/rented a new VPS and wants a standard, repeatable deployment (panel + proxy + WARP landing + TLS + DNS + passwordless SSH). Debian/Ubuntu targets.
---

# Edge Node Provision

Turn a bare Debian/Ubuntu VPS into a fully managed edge node with one guided workflow. This skill captures a battle-tested deployment: control-panel + proxy + WARP clean-exit + wildcard TLS + DNS + monitoring, plus every pitfall learned the hard way.

**What it builds**

| Layer | Component | Result |
|---|---|---|
| Panel | aaPanel International (7.x) + nginx | `<SUB_PANEL>.<root>` reverse-proxied, secret-path entrance, hardened |
| Proxy | 3x-ui (Xray) | `<SUB_XUI>.<root>` panel; two inbounds behind nginx (a direct cover path + a WARP cover path) |
| Clean exit | cloudflare-warp (proxy mode) | the WARP-path inbound egresses through Cloudflare WARP |
| TLS | acme.sh DNS-01 (Cloudflare) | wildcard `*.<root>` cert, auto-renew + reload |
| DNS | Cloudflare API | `<root>` + `*.<root>` A/AAAA (DNS-only) |
| Cover | nginx vhosts | landing page + node cover + default-server `444` |
| Watchdog | systemd timer | WARP proxy self-heal every 60s |
| Monitoring | Nezha + Komari agents (optional) | node appears on your panels |
| Access | SSH key + optional shortcut | passwordless login |

> **This is a template, not a turnkey binary.** You supply your own domain, Cloudflare token, server IP, and passwords. No secrets, domains, or routing rules are baked in.

## Inputs to collect first

Fill a working copy of `scripts/config.env.example` → `config.env` (git-ignored by you). Collect these before touching the box:

- `SERVER_IP` / `SERVER_IPV6` — the new VPS public addresses (amd64 or arm64)
- `SSH_USER` / `SSH_KEY` — `root` or a sudo user (scripts escalate via `sudo`), and the private key
- `TZ` — server timezone (default `UTC`)
- `DOMAIN_ROOT` — the sub-zone this node owns, e.g. `nodeA.example.com` (you get `*.nodeA.example.com`)
- `CF_API_TOKEN` — Cloudflare token with **Zone:DNS:Edit** on the parent zone (`example.com`)
- `CF_ZONE_ID` — the parent zone id (script can look it up from the token)
- `SUB_PANEL` / `SUB_XUI` / `SUB_COVER` — the subdomain labels for the three planes. **If your zone is public, choose non-obvious labels** — don't reuse a guessable convention or people can enumerate your control planes from DNS.
- `AAPANEL_USER` / `AAPANEL_ENTRANCE` — desired panel username + secret entrance path (e.g. `/somesecret`)
- `XUI_USER` / `XUI_PATH` — 3x-ui username + webBasePath (e.g. `/somesecret/`)
- passwords — generate strong ones at runtime; **store them in your password manager**, never in git
- monitoring (optional) — Nezha `SERVER:PORT` + shared key; Komari endpoint + auto-discovery key

## Workflow

Run steps in order. Each script is idempotent and reads `config.env`. SSH into the box as root (or a sudo user) for on-box steps.

### 0. Verify access & prep
```bash
scripts/00-preflight.sh          # confirms SSH, records specs, sets timezone, installs base pkgs (curl/gnupg/ufw)
```
Also drop **your** public key into the box's `~/.ssh/authorized_keys` for passwordless login, and (optional) add an entry to your SSH shortcut/config.

### 1. Cloudflare wildcard DNS
```bash
python3 scripts/02-cf-dns.py     # upserts <root> + *.<root> A/AAAA -> SERVER_IP/IPV6 (proxied=false)
```
Run it wherever the CF token lives (your laptop or a designated host); the token never needs to touch the VPS.

### 2. Wildcard TLS cert (acme.sh DNS-01)
```bash
scripts/03-sign-wildcard-cert.sh   # on-box acme.sh --dns dns_cf, issues *.<root> (ECDSA P-256, Let's Encrypt)
```
Installs the cert into the BT cert dirs and registers a renew `--reloadcmd`. (For a fleet, see `references/centralized-certs.md` to sign on one host and scp to the rest.)

### 3. aaPanel International + nginx
```bash
scripts/04-install-aapanel.sh    # official install_7.0_en.sh (headless); waits for panel binary
scripts/05-harden-aapanel.sh     # sets username/password/entrance, binds panel domain, CDN_PROXY=True
scripts/04b-install-nginx.sh     # install_soft.sh 0 install nginx 1.24 (aaPanel-managed)
```

### 4. Cloudflare WARP (clean exit)
```bash
scripts/06-install-warp.sh       # cloudflare-warp, proxy mode SOCKS5 127.0.0.1:40000, verify warp=on
scripts/06b-warp-watchdog.sh     # installs warp-healthcheck.sh + systemd timer (self-heal every 60s)
```

### 5. 3x-ui (Xray) + inbounds
```bash
scripts/07-install-3xui.sh       # installs 3x-ui, sets panel 127.0.0.1:14533 + XUI_PATH + creds
scripts/08-xui-bootstrap.sh      # adds direct + WARP inbounds (your WS_PATH_*) + client + xray template
```

### 6. aaPanel sites (panel-native, UI-manageable)
```bash
scripts/09-aapanel-sites.sh      # AddSite + SetSSL + CreateProxy + HttpToHttps for 4 domains, via panel classes
scripts/09b-default-444.sh       # unmatched server_name -> return 444; drop landing page
```

### 7. Firewall + monitoring (optional)
```bash
scripts/10-ufw.sh                # allow 22/80/443 + panel port + aaPanel 888/39000-40000
scripts/11-monitoring.sh         # optional: install Nezha + Komari agents (placeholders in config.env)
```

### 8. Smoke test
```bash
scripts/12-smoke-test.sh         # WS 101 on both cover paths; panels 200; landing 200; e2e exit IPs (direct vs WARP)
```
Expected: the direct-path egress = `SERVER_IP`, the WARP-path egress = a Cloudflare WARP IP (`warp=on`).

## Domains this creates

With `DOMAIN_ROOT=nodeA.example.com` and the **example** labels `SUB_COVER=edge`, `SUB_XUI=gateway`, `SUB_PANEL=console` (pick your own):

| Domain | Role | Backend |
|---|---|---|
| `nodeA.example.com` | landing page | static |
| `<SUB_COVER>.nodeA.example.com` | node cover: `<WS_PATH_DIRECT>`→direct, `<WS_PATH_WARP>`→WARP | 127.0.0.1:57707 / 57708 |
| `<SUB_XUI>.nodeA.example.com` | 3x-ui panel | 127.0.0.1:14533 |
| `<SUB_PANEL>.nodeA.example.com` | aaPanel | 127.0.0.1:\<panel port\> |
| `*.nodeA.example.com` (default) | unmatched → `return 444` | — |

> The labels above are just examples. **Never publish the real labels you pick** — on a public zone they are the map to your control planes.

## Read this before you start

`references/pitfalls.md` — the non-obvious failures (aaPanel service model, anti-scan fake 404, new-schema 3x-ui client tables, WARP keyring/gnupg, nginx http2 syntax, etc.). Skipping it costs hours.

## Privacy / safety notes

- Never commit `config.env`, cert keys, panel/xui passwords, or any UUID/token. Add them to your password manager.
- The scripts contain **no** real domains, IPs, subscription rules, or routing policy — only placeholders. Keep it that way when you adapt them.
- WARP mode here is for **your own** clean egress; it is not a bypass for anyone else's controls.
