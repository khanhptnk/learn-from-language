"""Is the entropy floor capping success? Final policies (request only) on fresh worlds, per goal, sampled at
temperature 1.0 (as trained and evaluated), 0.7, 0.5, 0.3 and greedy, same worlds for every temperature. A big gain
at low temperature means the policy's mode is right and its sampling noise (entropy held near 0.8 nats by the
controller) costs success. (docs/STATUS.md 2026-11-04)
  .venv/bin/python experiments/minicraft/check_temperature.py <policy .pt> [<policy .pt> ...]"""
import sys

import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.goals import GOAL_ID, MAX_GOALS, Requests
from gft.minicraft.learn import CleanPolicy, policy_actions, run_episodes

torch.set_num_threads(8)
cfg = WorldConfig(success_needs_done=True)
NAMES = ["gather wood", "dig clay", "make planks", "make rope", "make bricks", "make a pickaxe", "build a raft"]
N = 64
TEMPS = [1.0, 0.7, 0.5, 0.3, 0.0]
g = np.full((len(NAMES) * N, MAX_GOALS), -1)
g[:, 0] = np.repeat([GOAL_ID[x] for x in NAMES], N)
req = Requests(g, np.random.default_rng(123).integers(2 ** 31, 2 ** 32 - 1, len(g)))   # unseen world seeds
for path in sys.argv[1:]:
    pol = CleanPolicy(cfg.size, pixels=True, exact_text=True)
    pol.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    pol.eval()
    print(f"\n{path}\n  {'temp':6s} {'all':>5s}  " + "  ".join(f"{x.split()[-1]:>7s}" for x in NAMES), flush=True)
    for T in TEMPS:
        torch.manual_seed(0)
        _, _, _, _, valid, sat, _ = run_episodes(cfg, req, lambda e, t: policy_actions(
            pol, e, req, "cpu", None, greedy=T == 0, temperature=max(T, 1e-6)), record=False)
        per = [sat[req.goals[:, 0] == GOAL_ID[x], 0].mean() for x in NAMES]
        print(f"  {'greedy' if T == 0 else T:<6} {sat[:, 0].mean():5.2f}  " + "  ".join(f"{p:7.2f}" for p in per) +
              f"   steps {valid.sum(1).mean():.0f}", flush=True)
