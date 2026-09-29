#!/bin/sh
# Per-node latency / throughput benchmark (Linux or OpenWrt busybox).
# Usage:
#   mkdir -p /tmp/lc && cp config.example.yaml /tmp/lc/config.yaml   # fill in YOUR proxies (never commit it)
#   printf '28901 line-a\n28902 line-b\n' > /tmp/lc/nodes.txt          # "listener-port name", one per node
#   CORE=/path/to/mihomo sh line-bench.sh
# Env: CORE (mihomo binary), NOGROUP=1 on an OpenClash router, SKIP_DL=1 latency only (use for home-broadband exits).
# Prints: name|new-conn p50 ms|new-conn max|fail/20|1-stream Mbps|4-stream Mbps|warm p50|warm max
# new-conn = each request opens a new connection through the node (includes proxy handshakes: 2-4x RTT).
# warm = 10 requests reusing one connection (first dropped) ~= real round-trip latency. Run >= 3 rounds, take medians.
CORE=${CORE:-/etc/openclash/core/clash_meta}
URL204=http://www.gstatic.com/generate_204
DL=https://hkg.download.datapacket.com/100mb.bin
cd /tmp/lc || exit 1
if [ "${NOGROUP:-0}" = 1 ]; then
  # On an OpenClash router only the core's group (65534) bypasses OpenClash's own OUTPUT redirect; run the test
  # core the same way, or its connections to the entry nodes get intercepted and proxied a second time.
  start-stop-daemon -S -b -m -p /tmp/lc/core.pid -c root:nogroup -x "$CORE" -- -d /tmp/lc -f /tmp/lc/config.yaml
  sleep 4; P=$(cat /tmp/lc/core.pid)
else
  "$CORE" -d /tmp/lc -f /tmp/lc/config.yaml >/tmp/lc/core.log 2>&1 &
  P=$!
  sleep 4
fi
while read -r port name; do
  x="socks5h://127.0.0.1:$port"
  curl -s -o /dev/null -m 8 -x "$x" $URL204 </dev/null
  i=0; fail=0; : > lat
  while [ $i -lt 20 ]; do
    r=$(curl -s -o /dev/null -m 6 -x "$x" -w '%{http_code} %{time_total}' $URL204 </dev/null)
    set -- $r
    if [ "$1" = 204 ]; then awk -v t="$2" 'BEGIN{printf "%.0f\n", t*1000}' >> lat; else fail=$((fail+1)); fi
    i=$((i+1))
  done
  st=$(sort -n lat | awk '{a[NR]=$1} END{ if(NR) printf "%s|%s", a[int((NR+1)/2)], a[NR]; else print "-|-" }')
  args=""; i=0; while [ $i -lt 11 ]; do args="$args -o /dev/null $URL204"; i=$((i+1)); done
  warm=$(curl -s -m 30 -x "$x" -w '%{time_total}\n' $args </dev/null | awk 'NR>1{printf "%.0f\n", $1*1000}' | sort -n |
    awk '{a[NR]=$1} END{ if(NR) printf "%s|%s", a[int((NR+1)/2)], a[NR]; else print "-|-" }')
  if [ "${SKIP_DL:-0}" = 1 ]; then echo "$name|$st|$fail/20|-|-|$warm"; continue; fi
  s1=$(curl -s -o /dev/null -m 25 -x "$x" -r 0-52428799 -w '%{speed_download}' $DL </dev/null)
  pids=""
  for k in 0 1 2 3; do
    curl -s -o /dev/null -m 25 -x "$x" -r $((k*15000000))-$(((k+1)*15000000-1)) -w '%{speed_download}\n' $DL </dev/null > "s4.$k" &
    pids="$pids $!"
  done
  wait $pids   # only the downloads; a bare `wait` would also wait for the mihomo core forever
  s4=$(cat s4.0 s4.1 s4.2 s4.3 | awk '{s+=$1} END{printf "%.0f", s*8/1e6}')
  echo "$name|$st|$fail/20|$(awk -v s="$s1" 'BEGIN{printf "%.0f", s*8/1e6}')|$s4|$warm"
done < nodes.txt
kill $P 2>/dev/null
