"""Qualitative look at a GFT checkpoint: for a few training tasks with one goal, the event trace of the base policy, pi
(request prompt) and pi^fb (request + ideal feedback), with the teacher's feedback on each episode.

  .venv/bin/python experiments/minicraft/qualitative.py --ckpt outputs/minicraft/graph_attn/gft_s0.ckpt --goal 4 --n 4
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, has_heldout  # noqa: E402

from gft.minicraft.core import (ACTION_NAMES, CELL_NAMES, EV_BRIDGE, EV_COLLECT, EV_CRAFT, EV_CRAFT_FAIL, EV_DONE,  # noqa
                                EV_USE_FAIL, ITEM_NAMES, WorldConfig)
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import CleanPolicy, cached_pool, policy_actions, request_prompts, run_episodes  # noqa: E402
from gft.minicraft.teacher import compose, episode_facts  # noqa: E402


def trace(log, b):
    """Significant events of episode b: 't12 +wood', 't30 craft plank FAIL(no workbench)', 't60 DONE'."""
    out = []
    for t, ev in enumerate(log):
        typ = ev["type"][b]
        if typ == EV_COLLECT:
            out.append(f"t{t} +{ITEM_NAMES[ev['item'][b]]}")
        elif typ == EV_CRAFT:
            out.append(f"t{t} craft {ITEM_NAMES[ev['item'][b]]}")
        elif typ == EV_CRAFT_FAIL:
            why = (f"no {CELL_NAMES[ev['station'][b]]}" if ev["station"][b] >= 0
                   else f"no {ITEM_NAMES[ev['missing'][b]]}")
            out.append(f"t{t} craft {ITEM_NAMES[ev['item'][b]]} FAIL({why})")
        elif typ == EV_USE_FAIL:
            out.append(f"t{t} use FAIL(no {ITEM_NAMES[ev['tool'][b]]})")
        elif typ == EV_BRIDGE:
            out.append(f"t{t} bridge")
        elif typ == EV_DONE:
            out.append(f"t{t} DONE")
    return ", ".join(out) or "(nothing)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", type=int, default=4)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg, dev = WorldConfig(), a.device
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)["learner"]
    if "enc.q.weight" in st["clean"]:              # pixel policies have a patch conv; symbolic ones do not
        base = base_policy(cfg, dev, seed=0, kind="random", pixels="enc.patch.weight" in st["clean"])
    else:                                    # checkpoint from before cross-attention: pooled-text policy, base v6
        base = CleanPolicy(cfg.size, text_attn=False).to(dev)
        base.load_state_dict(torch.load("outputs/minicraft/base/base_v6_random_s0_size13.pt", map_location=dev))
    clean, aux = copy.deepcopy(base), copy.deepcopy(base)
    clean.load_state_dict(st["clean"])
    aux.load_state_dict(st["aux"])
    rng = np.random.default_rng(3)
    idx = rng.choice(np.nonzero((pool.goals[:, 0] == a.goal) & (pool.goals[:, 1] < 0))[0], size=a.n, replace=False)
    req = pool.requests(idx)
    star = [" ".join(s for _, _, s in compose(pool.ideal_facts(i, rng), "mixed", rng)) or "Okay." for i in idx]
    runs = {}
    for name, model, fb in (("base", base, None), ("pi", clean, None), ("pi^fb + ideal", aux, star)):
        p = request_prompts(model, req, dev, feedback=fb)
        r = np.random.default_rng(1)
        torch.manual_seed(1)
        runs[name] = run_episodes(cfg, req, lambda e, t: policy_actions(model, e, req, dev, r, prompt=p), record=False)
    for b in range(a.n):
        print(f"\n=== task {b}: request '{req.text(b)}' (goal: {GOALS[a.goal].phrase})")
        print(f"  ideal feedback given to pi^fb: {star[b]!r}")
        for name, (env, log, _, _, _, sat, mask) in runs.items():
            ok = bool((sat[b] | ~mask[b]).all())
            fbk = " ".join(s for _, _, s in compose(episode_facts(log, b, env, req.goals[b], sat[b]), "mixed", rng))
            print(f"  [{name}] {'SUCCESS' if ok else 'fail'} in {env.t[b]} steps: {trace(log, b)}")
            print(f"      teacher: {fbk!r}")


if __name__ == "__main__":
    main()
