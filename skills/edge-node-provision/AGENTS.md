# AGENTS.md — edge-node-provision

Instructions for any coding agent (Codex, Kimi, Cursor, Cline, Claude Code, …) asked to
provision a VPS into a managed edge node using this folder. This is a thin pointer; the full
workflow lives in `SKILL.md`.

## What you're doing

Turn a bare Debian/Ubuntu VPS (amd64 or arm64) into: aaPanel International + nginx, 3x-ui
with a direct inbound + a Cloudflare-WARP clean-exit inbound, a wildcard TLS cert (acme.sh
DNS-01), wildcard DNS, a WARP watchdog, optional monitoring agents, passwordless SSH.

## How to run

1. **Read `references/pitfalls.md` first.** Several steps fail in non-obvious ways; that file is the map.
2. Ask the user for the inputs listed in `SKILL.md` → "Inputs to collect first", then fill
   `scripts/config.env` from `scripts/config.env.example`. Generate strong passwords at runtime.
3. Execute `scripts/00-preflight.sh` … `scripts/12-smoke-test.sh` **in numeric order** (see
   `SKILL.md` → Workflow for the exact sequence and what each does). Each script is idempotent
   and reads `config.env`.
4. Confirm the smoke test: WS `101` on both cover paths; panels/landing `200`; direct egress =
   `SERVER_IP`, WARP egress = a Cloudflare WARP IP with `warp=on`.

## Rules (do not skip)

- **Secrets stay out of git.** Never commit `config.env`, cert keys, passwords, UUIDs, or tokens.
  Tell the user to store passwords in their password manager. `.gitignore` already excludes them.
- **On a public DNS zone, pick non-obvious `SUB_*` labels and `WS_PATH_*` cover paths**, and don't
  print/publish the real values — they map to the user's control planes.
- **Don't hardcode** domains, IPs, ports, arch, or timezone — they're all `config.env` variables.
- **Destructive/edge actions** (ufw enable, panel restarts): the scripts handle these safely, but
  confirm with the user before re-running steps on a box that's already serving traffic.
- WARP mode here is the user's own clean egress — not a bypass of anyone else's controls.

## Environment notes

- Runs the box steps over SSH as root, or a sudo user (scripts escalate automatically).
- DNS: Cloudflare has a turnkey record script (`02-cf-dns.py`); any acme.sh DNS provider works
  for the cert via `ACME_DNS_PLUGIN` (see `references/dns-records.md`).
- Local tooling needed: `bash`, `python3`, `ssh`/`scp`.
