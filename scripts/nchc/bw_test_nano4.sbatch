#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=bw_test
#SBATCH --output=/work/roboleon1295/openpi/logs/bw_test_%j.log
set -x
CML=cml18.csie.ntu.edu.tw
echo "NODE=$(hostname)"
echo "=== 1. outbound SSH/22 compute -> cml18 ==="
ssh -o BatchMode=yes -o ConnectTimeout=15 "$CML" "hostname; echo CML18_REACHED" 2>&1 || { echo "SSH_TO_CML18_FAILED (port 22 blocked?)"; exit 0; }
echo "=== 2. UPLOAD compute -> cml18 (client->server dir over reverse tunnel), 300MB ==="
dd if=/dev/zero bs=1M count=300 status=none | ssh -o BatchMode=yes "$CML" "dd of=/dev/null bs=1M 2>&1 | tail -2"
echo "=== 3. DOWNLOAD cml18 -> compute, 300MB ==="
ssh -o BatchMode=yes "$CML" "dd if=/dev/zero bs=1M count=300 status=none" | dd of=/dev/null bs=1M 2>&1 | tail -2
echo "=== done ==="
