#!/usr/bin/env python3
"""Runs ON the box. Adds two vmess+ws inbounds (direct + WARP), a client, and an xray
template with a WARP outbound + routing. Reads config from environment variables.

Schema-adaptive: newer 3x-ui (Xray 26.x) stores clients in separate tables
(clients / client_inbounds / client_traffics) and IGNORES inbound.settings.clients
(runtime shows clients=null). Older 3x-ui embeds clients in inbound.settings.
This handles both.
"""
import json, os, sqlite3, subprocess, sys, time

DB = os.environ.get("XUI_DB", "/etc/x-ui/x-ui.db")
UUID = os.environ["XUI_UUID"]
SUBID = os.environ.get("XUI_SUBID", "")
EMAIL = os.environ.get("XUI_EMAIL", "node-main")
PORT_DIRECT = int(os.environ["PORT_DIRECT"])
PORT_WARP = int(os.environ["PORT_WARP"])
WS_PATH_DIRECT = os.environ["WS_PATH_DIRECT"]
WS_PATH_WARP = os.environ["WS_PATH_WARP"]
WARP_SOCKS_PORT = int(os.environ.get("WARP_SOCKS_PORT", "40000"))

INBOUNDS = [
    ("node-direct", PORT_DIRECT, WS_PATH_DIRECT, "inbound-%d" % PORT_DIRECT),
    ("node-warp",   PORT_WARP,   WS_PATH_WARP,   "inbound-warp"),
]

TEMPLATE = {
    "log": {"access": "/var/log/v2ray/access.log", "error": "/var/log/v2ray/error.log", "loglevel": "warning"},
    "api": {"services": ["HandlerService", "LoggerService", "StatsService"], "tag": "api"},
    "inbounds": [{"listen": "127.0.0.1", "port": 62789, "protocol": "dokodemo-door",
                  "settings": {"address": "127.0.0.1"}, "tag": "api"}],
    "outbounds": [
        {"protocol": "freedom", "settings": {}},
        {"tag": "warp-outbound", "protocol": "freedom",
         "settings": {"domainStrategy": "UseIPv6v4"}, "proxySettings": {"tag": "warp-socks"}},
        {"tag": "warp-socks", "protocol": "socks",
         "settings": {"servers": [{"address": "127.0.0.1", "port": WARP_SOCKS_PORT}]}},
        {"protocol": "blackhole", "settings": {}, "tag": "blocked"},
    ],
    "policy": {"system": {"statsInboundDownlink": True, "statsInboundUplink": True}},
    "routing": {"rules": [
        {"inboundTag": ["api"], "outboundTag": "api", "type": "field"},
        {"inboundTag": ["inbound-warp", "inbound-127.0.0.1:%d" % PORT_WARP],
         "outboundTag": "warp-outbound", "type": "field"},
        {"ip": ["geoip:private"], "outboundTag": "blocked", "type": "field"},
        {"protocol": ["bittorrent"], "outboundTag": "blocked", "type": "field"},
    ]},
    "stats": {},
    "dns": {"servers": ["1.1.1.1"], "queryStrategy": "UseIP"},
}


def cols(cur, table):
    return {r[1] for r in cur.execute("PRAGMA table_info(%s)" % table).fetchall()}


def has_table(cur, name):
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def inbound_settings(embed_client):
    clients = [{"id": UUID, "email": EMAIL, "alterId": 0}] if embed_client else []
    return json.dumps({"clients": clients, "disableInsecureEncryption": False})


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    ic = cols(cur, "inbounds")
    new_schema = has_table(cur, "clients") and has_table(cur, "client_inbounds")
    existing_ports = {r[0] for r in cur.execute("SELECT port FROM inbounds").fetchall()}
    now = int(time.time() * 1000)

    for remark, port, path, tag in INBOUNDS:
        if port in existing_ports:
            print("  inbound :%d exists, skip" % port); continue
        row = {
            "up": 0, "down": 0, "total": 0, "remark": remark, "enable": 1, "expiry_time": 0,
            "listen": "127.0.0.1", "port": port, "protocol": "vmess",
            "settings": inbound_settings(embed_client=not new_schema),
            "stream_settings": json.dumps({"network": "ws", "security": "none",
                                           "wsSettings": {"path": path, "headers": {}}}),
            "sniffing": json.dumps({"enabled": True, "destOverride": ["http", "tls"]}),
            "tag": tag,
        }
        keys = [k for k in row if k in ic]
        if "user_id" in ic:
            keys = ["user_id"] + keys
            vals = [1] + [row[k] for k in keys if k != "user_id"]
        else:
            vals = [row[k] for k in keys]
        cur.execute("INSERT INTO inbounds (%s) VALUES (%s)" % (",".join(keys), ",".join("?" * len(keys))), vals)
        print("  inbound :%d %s added" % (port, path))

    if new_schema:
        ids = {p: cur.execute("SELECT id FROM inbounds WHERE port=?", (p,)).fetchone()[0]
               for p in (PORT_DIRECT, PORT_WARP)}
        crow = cur.execute("SELECT id FROM clients WHERE uuid=?", (UUID,)).fetchone()
        if crow:
            cid = crow[0]
        else:
            cur.execute("INSERT INTO clients (email,sub_id,uuid,security,flow,enable,group_name,total_gb,expiry_time,limit_ip,reset,created_at,updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (EMAIL, SUBID, UUID, "auto", "", 1, "", 0, 0, 0, 0, now, now))
            cid = cur.lastrowid
        for p, iid in ids.items():
            if not cur.execute("SELECT 1 FROM client_inbounds WHERE client_id=? AND inbound_id=?", (cid, iid)).fetchone():
                cur.execute("INSERT INTO client_inbounds (client_id,inbound_id,created_at) VALUES (?,?,?)", (cid, iid, now))
        # client_traffics.email is GLOBALLY UNIQUE -> one row per email even across inbounds
        if not cur.execute("SELECT 1 FROM client_traffics WHERE email=?", (EMAIL,)).fetchone():
            cur.execute("INSERT INTO client_traffics (inbound_id,enable,email,up,down,expiry_time,total,reset) VALUES (?,?,?,?,?,?,?,?)",
                        (ids[PORT_DIRECT], 1, EMAIL, 0, 0, 0, 0, 0))
        print("  client %s linked to both inbounds (new schema)" % EMAIL)

    cur.execute("UPDATE settings SET value=? WHERE key='xrayTemplateConfig'", (json.dumps(TEMPLATE, ensure_ascii=False),))
    if cur.rowcount == 0:
        cur.execute("INSERT INTO settings (key, value) VALUES ('xrayTemplateConfig', ?)", (json.dumps(TEMPLATE, ensure_ascii=False),))
    conn.commit(); conn.close()
    print("  xrayTemplateConfig written")

    subprocess.run(["systemctl", "restart", "x-ui"], check=True)
    time.sleep(6)
    print("OK — 3x-ui bootstrap complete")


if __name__ == "__main__":
    main()
