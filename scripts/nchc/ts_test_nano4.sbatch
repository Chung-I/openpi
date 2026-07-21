#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ts_test
#SBATCH --output=/work/roboleon1295/openpi/logs/ts_test_%j.log
set -x
TS=/work/roboleon1295/tailscale
SOCK=/work/roboleon1295/ts.sock
PORT=18080
echo "NODE=$(hostname)"
rm -f "$SOCK"
"$TS/tailscaled" --tun=userspace-networking --state=/work/roboleon1295/ts.state --socket="$SOCK" >/work/roboleon1295/openpi/logs/tsd_${SLURM_JOB_ID}.log 2>&1 &
TSD=$!
sleep 5
"$TS/tailscale" --socket="$SOCK" up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" --hostname=nchc-mem-serve --reset 2>&1
echo "TS_UP_EXIT=$?"
for i in $(seq 1 20); do
  IP=$("$TS/tailscale" --socket="$SOCK" ip -4 2>/dev/null | head -1)
  [ -n "$IP" ] && { echo "TAILSCALE_IP=$IP"; break; }
  echo "[$i] waiting for tailscale ip..."; sleep 3
done
"$TS/tailscale" --socket="$SOCK" status 2>&1 | head
python3 -c "import http.server,socketserver; socketserver.TCPServer((\"127.0.0.1\",$PORT), http.server.SimpleHTTPRequestHandler).serve_forever()" >/dev/null 2>&1 &
SRV=$!
echo "SERVER_LISTENING 127.0.0.1:$PORT pid=$SRV"
echo "READY_FOR_DEVBOX_TEST host=nchc-mem-serve ip=$IP port=$PORT"
sleep 480
kill $SRV $TSD 2>/dev/null
"$TS/tailscale" --socket="$SOCK" logout 2>/dev/null
echo "DONE"
