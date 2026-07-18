#!/usr/bin/env bash
# Verify SSH, record specs, set timezone, install base packages, authorize your key.
. "$(dirname "$0")/_lib.sh"

echo "== SSH + specs =="
run 'hostname; . /etc/os-release; echo "$PRETTY_NAME"; nproc; free -m | awk "/Mem/{print \$2\"MB\"}"; df -h / | tail -1'

echo "== timezone + base packages =="
run 'timedatectl set-timezone Asia/Shanghai 2>/dev/null || true
     export DEBIAN_FRONTEND=noninteractive
     apt-get update -qq
     apt-get install -y -qq curl wget gnupg ufw ca-certificates >/dev/null
     echo base-packages-ok'

# Authorize YOUR public key for passwordless login (edit path if needed).
MYPUB="$(cat "$HOME/.ssh/id_ed25519.pub" 2>/dev/null || cat "$HOME/.ssh/id_rsa.pub" 2>/dev/null || true)"
if [ -n "$MYPUB" ]; then
  run "mkdir -p ~/.ssh && chmod 700 ~/.ssh
       grep -qF '$MYPUB' ~/.ssh/authorized_keys 2>/dev/null || echo '$MYPUB' >> ~/.ssh/authorized_keys
       chmod 600 ~/.ssh/authorized_keys; echo authorized-key-added"
else
  echo "WARN: no local public key found; add yours to the box manually for passwordless login"
fi
echo "preflight done"
