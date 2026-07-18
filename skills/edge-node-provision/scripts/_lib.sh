#!/usr/bin/env bash
# Shared helpers. Every script sources this, which sources config.env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/config.env" ] || { echo "ERROR: copy config.env.example -> config.env and fill it in"; exit 1; }
# shellcheck disable=SC1091
. "$HERE/config.env"

# --- config sanity (fail fast) ---
require_vars() {
  local miss=()
  for v in "$@"; do [ -n "${!v:-}" ] || miss+=("$v"); done
  if [ ${#miss[@]} -gt 0 ]; then echo "ERROR: config.env missing: ${miss[*]}" >&2; exit 1; fi
}
require_vars SERVER_IP SSH_USER DOMAIN_ROOT
if [ -n "${SSH_KEY:-}" ]; then
  [ -f "$SSH_KEY" ] || { echo "ERROR: SSH_KEY not found: $SSH_KEY" >&2; exit 1; }
  chmod 600 "$SSH_KEY" 2>/dev/null || true
fi

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
[ -n "${SSH_KEY:-}" ] && SSH_OPTS+=(-i "$SSH_KEY")

# Non-root login -> escalate on the box via sudo. Redirects/pipes must run as root too,
# so wrap the whole command string in `sudo bash -c`.
_ROOT_LOGIN=1; [ "$SSH_USER" != "root" ] && _ROOT_LOGIN=0

# run <remote command...> — execute on the VPS (as root, escalating if needed)
run() {
  if [ "$_ROOT_LOGIN" = 1 ]; then
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER_IP" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER_IP" "sudo bash -c $(printf '%q' "$*")"
  fi
}

# push <localfile> <remotepath> — copy to the box (stages through /tmp + sudo install if non-root)
push() {
  if [ "$_ROOT_LOGIN" = 1 ]; then
    scp "${SSH_OPTS[@]}" "$1" "$SSH_USER@$SERVER_IP:$2"
  else
    local stage="/tmp/.stage.$$.$(basename "$2")"
    scp "${SSH_OPTS[@]}" "$1" "$SSH_USER@$SERVER_IP:$stage"
    run "install -D -m 644 '$stage' '$2'; rm -f '$stage'"
  fi
}

gen_pass() { python3 -c "import secrets,string;print(''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range($1)))"; }

# panel_port — read the aaPanel port the installer actually assigned (do not trust config guesses)
panel_port() { run 'cat /www/server/panel/data/port.pl 2>/dev/null | tr -d "[:space:]"'; }

# box_arch — normalized arch token: amd64 | arm64
box_arch() { run 'case "$(uname -m)" in x86_64) echo amd64;; aarch64|arm64) echo arm64;; *) uname -m;; esac'; }
