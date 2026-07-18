# Pitfalls (read before you run)

Every item here cost real debugging time on a live deployment. They are ordered by how likely they are to bite you.

## aaPanel

- **Never `systemctl restart btpanel`.** `btpanel.service` is `enabled` but its `ExecStart` is `/etc/init.d/bt start` (SysV); the systemd unit shows `active: inactive (dead)` while the panel process (`webserver`) runs fine. `systemctl restart btpanel` stops the panel and it does **not** come back → your reverse-proxied panel returns **502**. To restart use `bt 1` or `/etc/init.d/bt start`. Same for nginx (`/etc/init.d/nginx`, not `systemctl`).
- **Reverse-proxy needs `CDN_PROXY=True`** in `/www/server/panel/config/cdn.conf`, or the panel rejects proxied requests. Restart the panel after setting it.
- **Anti-scan fake 404/403.** aaPanel returns a *fake* nginx 404 to requests whose UA lacks `Mozilla` or that originate from `127.0.0.1`, and **403** to direct `IP:port` access once a panel domain is bound. So: `curl` tests of the entrance look "broken" but a real browser over the public domain works. To curl-verify, use `-A "Mozilla/5.0 …"` through the public hostname (not localhost).
- **Build sites through the panel, never hand-write vhosts.** A hand-written `/www/server/panel/vhost/nginx/*.conf` works in nginx but is invisible/unmanageable in the panel UI. Use the panel classes (`AddSite`/`SetSSL`/`CreateProxy`/`HttpToHttps`) — that's what `09-aapanel-sites.py` does. The panel HTTP API can't be driven by curl (CSRF + anti-scan); drive the classes with the panel's bundled `pyenv` python.
- **nginx 1.24 uses `listen 443 ssl http2;`** — it does **not** support the newer `http2 on;` directive; that syntax fails `nginx -t`.

## 3x-ui (newer versions, Xray 26.x)

- **Clients live in separate tables, not `inbound.settings.clients`.** Writing clients into `inbound.settings` leaves the runtime config with `clients: null` (auth fails silently). You must populate `clients` + `client_inbounds` (+ `client_traffics`). `08-xui-bootstrap.py` detects the schema and does the right thing.
- **`client_traffics.email` is GLOBALLY UNIQUE.** One client shared across two inbounds gets **one** `client_traffics` row, not one per inbound — inserting two rows with the same email throws `UNIQUE constraint failed`.
- **Auth is by UUID**, so a mismatch between the traffic-stat email label and the running config email is cosmetic, not a fault.
- **VMESS ≠ VLESS when testing.** The inbounds here are **vmess+ws**. A VLESS test client is correctly rejected (`websocket close 1000`) — that is not a server bug. Test with a vmess client (`security: auto`), tls+ws, `Host` header = the cover hostname.
- **Panel API `/login` may 403** on newer builds (security middleware). Do DB-direct edits + `systemctl restart x-ui` instead of the HTTP API.

## Cloudflare WARP

- **`gpg` must be installed first.** On a minimal Debian, `gpg --dearmor` for the repo keyring fails with `gpg: command not found`, and the later `apt update` then errors `Failed to parse keyring … sqv returned an error code`. `apt-get install -y gnupg` before adding the repo.
- Use the **`bookworm`** repo line even on Debian 13 (trixie) — the bookworm package works.
- **WARP can lie about being connected.** The control plane shows `Connected` while the SOCKS proxy stops egressing. The watchdog (`warp-healthcheck.*`) checks `warp=on` *through the proxy* and restarts `warp-svc` when it isn't.

## nginx / cover

- WS upgrade must return **101** with `--http1.1` + the `Upgrade`/`Connection` headers. If you test with HTTP/2 you'll get 400 (HTTP/2 doesn't carry the Upgrade frame).
- Default server for unmatched `server_name` should `return 444` so scanners hitting the bare IP / wrong host get dropped.

## Monitoring

- **Komari** auto-discovery registers the node with an auto-generated name (`Auto-<hostname>`). Rename/group it in your panel afterward (panel-specific; often a small DB update or a UI action). The agent falls back to a v1 WebSocket if the panel has no v2 API — that's normal.
- **Nezha** agent install one-liners change over time. If the scripted install fails, drop the `nezha-agent` binary + a `config.yml` (server, `tls`, `client_secret`, `uuid`) and a small systemd unit — the agent connects out to `SERVER:PORT` (gRPC), so only outbound is needed.

## SSH / access

- Some providers ship a `.pem`/`.ppk` bundle; use the `.pem`. `chmod 600` it or ssh refuses it.
- Fresh boxes may run fail2ban; repeated failed auth from a dynamic IP can lock you out. Get your key in early (step 0) so you're not relying on password auth.
