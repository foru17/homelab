#!/usr/bin/env python3
"""Runs ON the box with aaPanel's bundled python:
     /www/server/panel/pyenv/bin/python /root/_aapanel-sites.py
Registers 4 domains as panel-native sites (AddSite + SetSSL + CreateProxy + HttpToHttps)
so they are visible/manageable in the aaPanel UI (never hand-write vhosts — the UI won't
see those). Reads subdomain labels + cover paths from environment variables.
"""
import json, os, sys

sys.path.insert(0, "/www/server/panel")
sys.path.insert(0, "/www/server/panel/class")
import public          # noqa: E402
import panelSite       # noqa: E402

ROOT = os.environ["DOMAIN_ROOT"]
SUB_PANEL = os.environ["SUB_PANEL"]
SUB_XUI = os.environ["SUB_XUI"]
SUB_COVER = os.environ["SUB_COVER"]
WS_DIRECT = os.environ["WS_PATH_DIRECT"]
WS_WARP = os.environ["WS_PATH_WARP"]
PORT_DIRECT = os.environ["PORT_DIRECT"]
PORT_WARP = os.environ["PORT_WARP"]
XUI_PORT = os.environ.get("XUI_PORT", "14533")
PANEL_PORT = public.readFile("/www/server/panel/data/port.pl").strip()

D_LANDING = ROOT
D_COVER = "%s.%s" % (SUB_COVER, ROOT)
D_XUI = "%s.%s" % (SUB_XUI, ROOT)
D_PANEL = "%s.%s" % (SUB_PANEL, ROOT)

site = panelSite.panelSite()

SITES = [
    (D_LANDING, "/www/wwwroot/" + D_LANDING, "edge landing"),
    (D_COVER,   "/www/wwwroot/" + D_COVER,   "node cover"),
    (D_XUI,     "/www/wwwroot/" + D_XUI,     "3x-ui panel proxy"),
    (D_PANEL,   "/www/wwwroot/" + D_PANEL,   "aaPanel proxy"),
]
PROXIES = [
    ("xui-panel", D_XUI,   "/",       "http://127.0.0.1:%s" % XUI_PORT,   D_XUI),
    ("aapanel",   D_PANEL, "/",       "https://127.0.0.1:%s" % PANEL_PORT, D_PANEL),
    ("cover-d",   D_COVER, WS_DIRECT, "http://127.0.0.1:%s" % PORT_DIRECT, D_COVER),
    ("cover-w",   D_COVER, WS_WARP,   "http://127.0.0.1:%s" % PORT_WARP,   D_COVER),
]


def obj(d):
    return public.to_dict_obj(d)


def main():
    for name, path, ps in SITES:
        get = obj({"webname": json.dumps({"domain": name, "domainlist": [], "count": 0}),
                   "path": path, "port": "80", "version": "00", "ps": ps,
                   "type_id": "0", "type": "static", "ftp": "false", "sql": "false"})
        print("AddSite", name, "->", site.AddSite(get).get("siteStatus", "?"))
    for name, _p, _ps in SITES:
        cdir = "/www/server/panel/vhost/cert/" + name
        key = public.readFile(cdir + "/privkey.pem"); csr = public.readFile(cdir + "/fullchain.pem")
        if key and csr:
            print("SetSSL", name, "->", site.SetSSL(obj({"siteName": name, "key": key, "csr": csr})).get("status"))
    for pname, sname, pdir, target, todomain in PROXIES:
        get = obj({"proxyname": pname, "sitename": sname, "proxydir": pdir, "proxysite": target,
                   "todomain": todomain, "type": "1", "cache": "0", "subfilter": "[]",
                   "advanced": "0", "cachetime": "1", "nocheck": "1"})
        print("CreateProxy", sname, pdir, "->", site.CreateProxy(get).get("status"))
    for name, _p, _ps in SITES:
        print("HttpToHttps", name, "->", site.HttpToHttps(obj({"siteName": name})).get("status"))
    print("DONE — reload nginx:  /etc/init.d/nginx reload")


if __name__ == "__main__":
    main()
