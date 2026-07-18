#!/usr/bin/env bash
# Shared helpers. Every script sources this, which sources config.env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/config.env" ] || { echo "ERROR: copy config.env.example -> config.env and fill it in"; exit 1; }
# shellcheck disable=SC1091
. "$HERE/config.env"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
[ -n "${SSH_KEY:-}" ] && SSH_OPTS+=(-i "$SSH_KEY")

# run <remote command...> — execute on the VPS
run() { ssh "${SSH_OPTS[@]}" "$SSH_USER@$SERVER_IP" "$@"; }
# push <localfile> <remotepath>
push() { scp "${SSH_OPTS[@]}" "$1" "$SSH_USER@$SERVER_IP:$2"; }
# pipe local stdin to a remote file
put() { run "cat > $1"; }

gen_pass() { python3 -c "import secrets,string;print(''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range($1)))"; }
