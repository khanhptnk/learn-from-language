"""Time GFT and PPO rounds (with the per-phase breakdown each learner reports), to find hot spots before long runs.

  .venv/bin/python experiments/minicraft/profile_rounds.py --base random --rounds 4 --compile
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, has_heldout  # noqa: E402

from gft.minicraft.core import WorldConfig  # noqa: E402
from gft.minicraft.learn import GFT, PPO, CurriculumConfig, GFTConfig, PPOConfig, cached_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="random")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gft-only", action="store_true")
    ap.add_argument("--masked-claims", default="0", help="comma-separated values to compare")
    ap.add_argument("--epochs-aux", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg, dev = WorldConfig(), a.device
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    base = base_policy(cfg, dev, seed=0, kind=a.base)
    cur = CurriculumConfig(enabled=False)
    runs = [(f"gft masked={m}", lambda m=int(m): GFT(cfg, pool, copy.deepcopy(base), dev,
                                                    GFTConfig(masked_claims=m, epochs_aux=a.epochs_aux), curriculum=cur))
            for m in a.masked_claims.split(",")]
    if not a.gft_only:
        runs.append(("ppo", lambda: PPO(cfg, pool, copy.deepcopy(base), dev, PPOConfig(), curriculum=cur)))
    for name, make in runs:
        learner = make()
        if a.compile:
            for m in [learner.clean, learner.aux] if name.startswith("gft") else [learner.model]:
                m.forward = torch.compile(m.forward)
        for r in range(a.rounds):
            torch.cuda.synchronize() if dev.startswith("cuda") else None
            t = time.time()
            info = learner.round()
            n = learner.g.episodes + learner.g.stage2 if name.startswith("gft") else learner.p.episodes
            print(f"{name} round {r}: {time.time() - t:5.1f}s for {n} episodes  "
                  + "  ".join(f"{k} {v:.1f}" for k, v in info["time"].items()), flush=True)


if __name__ == "__main__":
    main()
