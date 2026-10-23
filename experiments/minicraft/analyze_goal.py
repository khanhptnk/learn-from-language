"""What does a GFT checkpoint do on one goal? Rolls out pi (request prompt) and pi^fb (request + ideal feedback from
the success bank) on training-pool tasks whose only goal is --goal, and reports behaviour statistics.

  .venv/bin/python experiments/minicraft/analyze_goal.py --ckpt outputs/minicraft/graph/gft_s0.ckpt --goal 4
"""

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, has_heldout  # noqa: E402

from gft.minicraft.core import ITEM_NAMES, STONE_PICKAXE, WorldConfig  # noqa: E402
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import CleanPolicy, cached_pool, policy_actions, request_prompts, run_episodes  # noqa: E402
from gft.minicraft.teacher import compose, episode_facts  # noqa: E402


def stats(name, req, env, log, sat, mask, goal):
    facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
    item = GOALS[goal].item
    ended_done = np.mean([env.t[b] < env.cfg.max_steps for b in range(len(req))])
    print(f"  {name}: success {(sat | ~mask).all(1).mean():.2f} | steps {env.t.mean():.0f} | said DONE {ended_done:.2f}")
    if item >= 0:
        got = np.array([f.collected[item] + f.crafted[item] for f in facts])
        held = env.inv[:, item]
        print(f"    obtained {ITEM_NAMES[item]}: 0 -> {np.mean(got == 0):.2f}, 1 -> {np.mean(got == 1):.2f}, "
              f">=2 -> {np.mean(got >= 2):.2f}; held at end (mean) {held.mean():.2f}")
    print(f"    crafted stone pickaxe: {np.mean([f.crafted[STONE_PICKAXE] > 0 for f in facts]):.2f}; "
          f"distinct items made: {np.mean([len(set(f.collected) | set(f.crafted)) for f in facts]):.1f}")
    fails = Counter()
    for f in facts:
        for t in f.use_fail_tool:
            fails[f"no {ITEM_NAMES[t]}"] += 1
    print(f"    use failures: {dict(fails.most_common(4))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", type=int, default=4)
    ap.add_argument("--n", type=int, default=256)
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
    idx = np.nonzero((pool.goals[:, 0] == a.goal) & (pool.goals[:, 1] < 0))[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(idx, size=a.n)
    req = pool.requests(idx)
    print(f"goal: {GOALS[a.goal].phrase}; {len(np.unique(idx))} distinct pool tasks, {a.n} rollouts each policy")
    star = [" ".join(s for _, _, s in compose(pool.ideal_facts(i, rng), "mixed", rng)) or "Okay." for i in idx]
    print(f"  example ideal feedback: {star[0]!r}")
    for name, model, fb in (("base", base, None), ("pi", clean, None), ("pi^fb + ideal feedback", aux, star)):
        p = request_prompts(model, req, dev, feedback=fb)
        r = np.random.default_rng(1)
        torch.manual_seed(1)
        env, log, _, _, _, sat, mask = run_episodes(cfg, req, lambda e, t: policy_actions(model, e, req, dev, r, prompt=p),
                                                    record=False)
        stats(name, req, env, log, sat, mask, a.goal)


if __name__ == "__main__":
    main()
