#!/bin/sh
# GFT and GRPO side by side on one pod, every flag at train.py's default run setting (docs/STATUS.md 2026-11-06).
# The base is built first (cached per architecture), then both arms run to BUDGET feedbacks. Optional env:
# BUDGET, EVAL, SEED, ARCH (e.g. d=512,ch=64,trunk=4), O (output dir), TAG (file name suffix).
#   cd /workspace/gft && sh experiments/minicraft/pod/jobs_main.sh > /workspace/main.log 2>&1
set -x
P=/workspace/gft/.venv/bin/python
O=${O:-outputs/minicraft/main}
A=${ARCH:+--arch $ARCH}
C="--budget ${BUDGET:-100000} --eval-every ${EVAL:-5000} --seed ${SEED:-0} $A"
mkdir -p $O
$P experiments/minicraft/train.py --arm gft $C --prepare-only > $O/prepare.log 2>&1
$P experiments/minicraft/train.py --arm gft $C --compile --threads 4 --out $O/gft${TAG}.json > $O/gft${TAG}.log 2>&1 &
$P experiments/minicraft/train.py --arm grpo $C --threads 3 --out $O/grpo${TAG}.json > $O/grpo${TAG}.log 2>&1 &
wait
