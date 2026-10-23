#!/usr/bin/env bash
# Copy code (and cached pools / base policies) to the pod, or fetch results back.
#   experiments/minicraft/pod/sync.sh up   <ip> <port>
#   experiments/minicraft/pod/sync.sh down <ip> <port>      # outputs/minicraft -> local outputs/pod/
set -euo pipefail
dir=$1 ip=$2 port=$3
ssh_cmd="ssh -p $port -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
root=$(git rev-parse --show-toplevel)
if [ "$dir" = up ]; then
  rsync -rlptz -e "$ssh_cmd" --exclude .venv --exclude .git --exclude 'outputs/*' --exclude '*.pdf' --exclude __pycache__ \
    "$root"/ root@"$ip":/workspace/gft/
  rsync -rlptz -e "$ssh_cmd" --relative "$root"/./outputs/minicraft/pools "$root"/./outputs/minicraft/base \
    root@"$ip":/workspace/gft/
else
  mkdir -p "$root"/outputs/pod
  rsync -rlptz -e "$ssh_cmd" --exclude '*.ckpt' root@"$ip":/workspace/gft/outputs/minicraft/ "$root"/outputs/pod/
fi
