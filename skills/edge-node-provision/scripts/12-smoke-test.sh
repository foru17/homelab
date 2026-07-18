#!/usr/bin/env bash
# End-to-end smoke test. Run from anywhere with DNS resolving to the box.
. "$(dirname "$0")/_lib.sh"

COVER="$SUB_COVER.$DOMAIN_ROOT"
echo "== cover WS upgrade (expect 101 each) =="
for P in "$WS_PATH_DIRECT" "$WS_PATH_WARP"; do
  curl -sk --http1.1 --max-time 8 -o /dev/null -w "$P -> %{http_code}\n" \
    -H "Connection: Upgrade" -H "Upgrade: websocket" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" -H "Sec-WebSocket-Version: 13" \
    "https://$COVER$P"
done

echo "== landing + panels (expect 200) =="
curl -sk --max-time 8 -o /dev/null -w "landing -> %{http_code}\n" "https://$DOMAIN_ROOT/"
curl -sk --max-time 8 -o /dev/null -w "3x-ui   -> %{http_code}\n" "https://$SUB_XUI.$DOMAIN_ROOT$XUI_PATH"
curl -sk --max-time 8 -A "Mozilla/5.0 (Macintosh)" -o /dev/null -w "aaPanel -> %{http_code}\n" "https://$SUB_PANEL.$DOMAIN_ROOT$AAPANEL_ENTRANCE"

echo "== unmatched host (expect 000/444, connection dropped) =="
curl -sk --max-time 6 -o /dev/null -w "random -> %{http_code}\n" "https://nope-$RANDOM.$DOMAIN_ROOT/" || true

echo "== e2e egress via temp xray client (direct vs WARP) =="
run "cat > /tmp/_tc.json <<JSON
{ \"inbounds\":[
   {\"port\":10801,\"listen\":\"127.0.0.1\",\"protocol\":\"socks\",\"tag\":\"a\",\"settings\":{\"udp\":true}},
   {\"port\":10802,\"listen\":\"127.0.0.1\",\"protocol\":\"socks\",\"tag\":\"b\",\"settings\":{\"udp\":true}}],
  \"outbounds\":[
   {\"tag\":\"od\",\"protocol\":\"vmess\",\"settings\":{\"vnext\":[{\"address\":\"$COVER\",\"port\":443,\"users\":[{\"id\":\"$XUI_UUID\",\"security\":\"auto\"}]}]},\"streamSettings\":{\"network\":\"ws\",\"security\":\"tls\",\"tlsSettings\":{\"serverName\":\"$COVER\"},\"wsSettings\":{\"path\":\"$WS_PATH_DIRECT\",\"headers\":{\"Host\":\"$COVER\"}}}},
   {\"tag\":\"ow\",\"protocol\":\"vmess\",\"settings\":{\"vnext\":[{\"address\":\"$COVER\",\"port\":443,\"users\":[{\"id\":\"$XUI_UUID\",\"security\":\"auto\"}]}]},\"streamSettings\":{\"network\":\"ws\",\"security\":\"tls\",\"tlsSettings\":{\"serverName\":\"$COVER\"},\"wsSettings\":{\"path\":\"$WS_PATH_WARP\",\"headers\":{\"Host\":\"$COVER\"}}}}],
  \"routing\":{\"rules\":[{\"type\":\"field\",\"inboundTag\":[\"a\"],\"outboundTag\":\"od\"},{\"type\":\"field\",\"inboundTag\":[\"b\"],\"outboundTag\":\"ow\"}]} }
JSON
cd /usr/local/x-ui/bin
XRAY=\$(ls xray-linux-* 2>/dev/null | head -1); [ -n \"\$XRAY\" ] || XRAY=\$(ls xray* 2>/dev/null | head -1)
./\$XRAY -c /tmp/_tc.json >/tmp/_tc.log 2>&1 &
XPID=\$!; sleep 4
echo -n 'direct egress: '; curl -s -x socks5h://127.0.0.1:10801 --max-time 12 https://1.1.1.1/cdn-cgi/trace | grep -E 'ip=|warp=' | tr '\n' ' '; echo
echo -n 'WARP egress:   '; curl -s -x socks5h://127.0.0.1:10802 --max-time 12 https://1.1.1.1/cdn-cgi/trace | grep -E 'ip=|warp=' | tr '\n' ' '; echo
kill \$XPID 2>/dev/null; rm -f /tmp/_tc.json /tmp/_tc.log"
echo "Expected: direct egress ip=$SERVER_IP warp=off ; WARP egress ip=<cloudflare> warp=on"
