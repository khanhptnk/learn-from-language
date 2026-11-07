"""Does a policy learn shorter successful paths? (pace teacher, docs/STATUS.md 2026-11-05) Final policies (request
only, temperature 1, as evaluated) and the expert on the same fresh worlds, per goal: success, and the steps of the
successful episodes -- mean, and paired against the expert on worlds both solved (excess steps over the expert).
  .venv/bin/python experiments/minicraft/check_path_length.py <name>=<policy .pt> [...]"""
import sys

import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.expert import expert_action
from gft.minicraft.goals import GOAL_ID, MAX_GOALS, Requests
from gft.minicraft.learn import CleanPolicy, policy_actions, run_episodes

torch.set_num_threads(8)
cfg = WorldConfig(success_needs_done=True)
NAMES = ["gather wood", "gather fiber", "dig clay", "make planks", "make rope", "make bricks"]
N = 128
g = np.full((len(NAMES) * N, MAX_GOALS), -1)
g[:, 0] = np.repeat([GOAL_ID[x] for x in NAMES], N)
req = Requests(g, np.random.default_rng(123).integers(2 ** 31, 2 ** 32 - 1, len(g)))   # unseen world seeds


def run(act):
    torch.manual_seed(0)
    _, _, _, _, valid, sat, _ = run_episodes(cfg, req, act, record=False)
    return sat[:, 0], valid.sum(1)


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    pol = CleanPolicy(cfg.size, pixels=True, exact_text=True)
    new = pol.state_dict()
    for k, v in sd.items():                       # a policy from before the pace words: zero rows for them
        if v.dim() == 2 and new[k].shape[0] > v.shape[0] and new[k].shape[1] == v.shape[1]:
            sd[k] = torch.cat([v, v.new_zeros(new[k].shape[0] - v.shape[0], v.shape[1])])
    pol.load_state_dict(sd)
    return pol.eval()


ex_ok, ex_len = run(lambda e, t: np.array([expert_action(e, b, req.goals[b]) if not e.done[b] else 0
                                           for b in range(len(req))]))
res = {"expert": (ex_ok, ex_len)}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    pol = load(path)
    with torch.no_grad():
        res[name] = run(lambda e, t: policy_actions(pol, e, req, "cpu", None))
gid = req.goals[:, 0]
print(f"{'':22s}" + "".join(f"{x.split()[-1]:>22s}" for x in NAMES))
print(f"{'':22s}" + "".join(f"{'succ  len  +over expert':>22s}" for _ in NAMES))
for name, (ok, L) in res.items():
    cells = []
    for x in NAMES:
        m = gid == GOAL_ID[x]
        both = m & ok & ex_ok
        over = (L[both] - ex_len[both]).mean() if both.any() else np.nan
        cells.append(f"{ok[m].mean():5.2f} {L[m & ok].mean() if (m & ok).any() else np.nan:5.1f} {over:+6.1f}")
    print(f"{name:22s}" + "".join(f"{c:>22s}" for c in cells), flush=True)
