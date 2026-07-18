# edge-node-provision

An **agent-agnostic runbook** for provisioning a bare Debian/Ubuntu VPS into a well-managed edge node. Drive it with **any** coding agent — Claude Code, Codex, Kimi, Cursor, Cline, … — or run it by hand. The AI only orchestrates ordered, plain bash/python steps; nothing is model-specific.

**aaPanel (International) + nginx · 3x-ui (Xray) with a direct inbound + a Cloudflare-WARP clean-exit inbound · wildcard TLS (acme.sh DNS-01, any DNS provider) · wildcard DNS · a WARP self-heal watchdog · optional Nezha/Komari monitoring · passwordless SSH.**

It encodes a real, repeatable deployment plus every non-obvious pitfall (`references/pitfalls.md`) so the next node takes minutes, not an afternoon.

## Use it with your agent

- **Any agent / human** — the source of truth is the ordered steps in `SKILL.md` (or the one-liner block below). Point the agent at this folder and say *"provision this VPS as an edge node following SKILL.md."*
- **Claude Code** — copy into a skills dir; it auto-loads `SKILL.md`:
  ```bash
  cp -r edge-node-provision ~/.claude/skills/            # user-level
  # or  cp -r edge-node-provision <project>/.claude/skills/
  ```
- **Codex / agents that read `AGENTS.md`** — `AGENTS.md` in this folder points them at the same workflow. Just open the folder in the agent.
- **Kimi / Cursor / Cline / others** — open this folder and tell the agent to follow `SKILL.md`. The scripts are ordinary; no plugin format needed.

## Run the steps (any agent, or by hand)

Everything is plain bash/python:

```bash
cd skills/edge-node-provision/scripts
cp config.env.example config.env      # fill in YOUR values (git-ignore it)
./00-preflight.sh
python3 02-cf-dns.py
./03-sign-wildcard-cert.sh
./04-install-aapanel.sh && ./05-harden-aapanel.sh && ./04b-install-nginx.sh
./06-install-warp.sh && ./06b-warp-watchdog.sh
./07-install-3xui.sh && ./08-xui-bootstrap.sh
./09-aapanel-sites.sh && ./09b-default-444.sh
./10-ufw.sh && ./11-monitoring.sh
./12-smoke-test.sh
```

## Requirements

- Target: Debian 12/13 or Ubuntu 22.04+ VPS, **amd64 or arm64**. Log in as **root or a sudo user** (scripts escalate via `sudo` automatically when `SSH_USER` isn't root).
- A domain in **Cloudflare** + an API token with `Zone:DNS:Edit`.
- Local: `bash`, `python3`, `ssh/scp`.

Notes: the aaPanel port is read from the box (the installer randomizes it — leave `AAPANEL_PORT` empty). Set `TZ` to your preferred server timezone (default `UTC`).

## What it is NOT

- Not a bypass tool for anyone else's network controls — WARP mode here is your own clean egress.
- Not opinionated about routing/subscriptions — it stops at "node is up, TLS valid, both egress paths verified." Your client config / subscription layer is yours and stays private.

## Security

- `config.env`, cert keys, panel/3x-ui passwords, UUIDs and tokens are **yours** — keep them out of git and in a password manager. `.gitignore` in this folder already excludes `config.env`.
- On a public DNS zone, **choose non-obvious subdomain labels and cover paths** (`SUB_*`, `WS_PATH_*`) and don't publish them — they are the map to your control planes.

## Layout

```
edge-node-provision/
├── SKILL.md                 # the guided workflow (Claude Code reads this)
├── README.md                # this file
├── scripts/
│   ├── config.env.example   # copy -> config.env, fill in
│   ├── _lib.sh              # shared ssh/scp helpers
│   ├── 00-preflight.sh … 12-smoke-test.sh
│   ├── 08-xui-bootstrap.py  # runs on box (schema-adaptive client tables)
│   ├── 09-aapanel-sites.py  # runs on box (panel-native site builder)
│   └── warp-healthcheck.{sh,service,timer}
├── templates/landing.html
└── references/
    ├── pitfalls.md          # read this first
    └── centralized-certs.md # fleet cert signing
```
