#!/usr/bin/env python3
"""Upsert <DOMAIN_ROOT> + *.<DOMAIN_ROOT> A/AAAA records in Cloudflare (DNS-only).
Run wherever the CF token lives (your laptop / a bastion) — the token never touches the VPS.
Reads config.env from the same dir. No real domains/IPs are baked in.
"""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env():
    env = {}
    path = os.path.join(HERE, "config.env")
    if not os.path.exists(path):
        sys.exit("ERROR: copy config.env.example -> config.env")
    for line in open(path):
        m = re.match(r'\s*([A-Z0-9_]+)\s*=\s*"?([^"#\n]*)"?', line)
        if m:
            env[m.group(1)] = m.group(2).strip()
    return env


E = load_env()
API = "https://api.cloudflare.com/client/v4"
TOKEN = E["CF_API_TOKEN"]
ROOT = E["DOMAIN_ROOT"]
IPV4 = E.get("SERVER_IP", "")
IPV6 = E.get("SERVER_IPV6", "")
NAMES = [ROOT, "*." + ROOT]


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method)
    r.add_header("Authorization", "Bearer " + TOKEN)
    r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=20) as f:
        return json.load(f)


def zone_id():
    zid = E.get("CF_ZONE_ID", "")
    if zid:
        return zid
    parent = E["PARENT_ZONE"]
    res = req("GET", "/zones?name=%s" % parent)
    if not res.get("result"):
        sys.exit("ERROR: zone %s not found for this token" % parent)
    return res["result"][0]["id"]


ZONE = zone_id()


def upsert(rtype, content):
    if not content:
        return
    existing = req("GET", "/zones/%s/dns_records?per_page=200&type=%s" % (ZONE, rtype))["result"]
    by_name = {r["name"]: r for r in existing}
    for name in NAMES:
        payload = {"type": rtype, "name": name, "content": content, "ttl": 300, "proxied": False}
        if name in by_name:
            res = req("PUT", "/zones/%s/dns_records/%s" % (ZONE, by_name[name]["id"]), payload)
            print("UPDATE %-4s %-28s -> %s  %s" % (rtype, name, content, "OK" if res.get("success") else res.get("errors")))
        else:
            res = req("POST", "/zones/%s/dns_records" % ZONE, payload)
            print("CREATE %-4s %-28s -> %s  %s" % (rtype, name, content, "OK" if res.get("success") else res.get("errors")))


upsert("A", IPV4)
upsert("AAAA", IPV6)
print("DNS done for *.%s (DNS-only / grey cloud)" % ROOT)
