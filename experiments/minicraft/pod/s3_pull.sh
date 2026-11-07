#!/usr/bin/env bash
# Download results from the network volume through RunPod's S3-compatible API: no pod needed.
#   bash -ic 'experiments/minicraft/pod/s3_pull.sh [tree] [subdir] [dest]'
#   tree:   folder on the volume (gft = new game, gft_old = old curriculum); default gft
#   subdir: path under <tree>/outputs/minicraft/ (default: everything)
#   dest:   local folder (default outputs/pod/<tree>/<subdir>)
# Needs RUNPOD_S3_ACCESS_KEY / RUNPOD_S3_SECRET_KEY (from ~/.bashrc, hence bash -ic). Checkpoints (*.ckpt) are skipped.
set -euo pipefail
tree=${1:-gft} sub=${2:-} vol=yy9bsgna2b
root=$(git rev-parse --show-toplevel)
dest=${3:-$root/outputs/pod/$tree/$sub}
mkdir -p "$dest"
AWS_ACCESS_KEY_ID=$RUNPOD_S3_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$RUNPOD_S3_SECRET_KEY \
  aws s3 sync --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io --exclude '*.ckpt' --only-show-errors \
  "s3://$vol/$tree/outputs/minicraft/$sub" "$dest"
echo "pulled s3://$vol/$tree/outputs/minicraft/$sub -> $dest"
