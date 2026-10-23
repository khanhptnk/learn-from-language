"""Compare candidate base policies (train.py BASES) on unseen validation worlds: success on the true requests, and how
purposeful the behaviour is (what gets achieved, wasted actions, episode length).

  .venv/bin/python experiments/minicraft/check_base.py --bases easy,mixture,permuted --out outputs/minicraft/base_check.json

A good base for studying feedback learning moves competently but does not know which subgoals a request needs, so
its episodes differ mainly in which subgoals were achieved -- what the feedback describes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import BASES, base_policy, heldout_splits  # noqa: E402

from gft.minicraft.core import ITEM_NAMES, WorldConfig  # noqa: E402
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import policy_actions, request_prompts, run_episodes  # noqa: E402
from gft.minicraft.teacher import episode_facts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg, dev = WorldConfig(), a.device
    req = heldout_splits(a.n_eval, tune=True)["val_worlds"]
    record = {}
    for kind in a.bases.split(","):
        t0 = time.time()
        model = base_policy(cfg, dev, seed=0, kind=kind)
        built = time.time() - t0
        model.eval()
        prompt = request_prompts(model, req, dev)
        rng = np.random.default_rng(1)
        torch.manual_seed(1)
        env, log, _, _, _, sat, mask = run_episodes(
            cfg, req, lambda e, t: policy_actions(model, e, req, dev, rng, prompt=prompt), record=False)
        facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
        made = {ITEM_NAMES[i]: float(np.mean([f.collected[i] + f.crafted[i] > 0 for f in facts]))
                for i in range(len(ITEM_NAMES))}
        rec = dict(
            build_seconds=built,
            success=float((sat | ~mask).all(1).mean()), frac=float((sat & mask).sum() / mask.sum()),
            per_goal={g.phrase: float(sat[req.goals == i].mean()) for i, g in enumerate(GOALS) if (req.goals == i).any()},
            mean_steps=float(env.t.mean()),
            crossed=float(np.mean([f.crossed for f in facts])),
            distinct_items=float(np.mean([len(set(f.collected) | set(f.crafted)) for f in facts])),
            blocked_per_step=float(np.mean([f.blocked / max(f.steps, 1) for f in facts])),
            wasted_use_per_step=float(np.mean([f.wasted_uses / max(f.steps, 1) for f in facts])),
            failed_crafts_per_episode=float(np.mean([sum(f.craft_fail_station.values()) + sum(f.craft_fail_missing.values())
                                                     for f in facts])),
            share_making=made)
        record[kind] = rec
        print(f"== {kind} (built/loaded in {built:.0f}s)\n  success {rec['success']:.3f}  frac {rec['frac']:.3f}  "
              f"steps {rec['mean_steps']:.0f}  distinct items {rec['distinct_items']:.2f}  crossed {rec['crossed']:.2f}  "
              f"bumps/step {rec['blocked_per_step']:.3f}  empty uses/step {rec['wasted_use_per_step']:.3f}  "
              f"failed crafts/ep {rec['failed_crafts_per_episode']:.1f}\n  making: "
              + " ".join(f"{k}={v:.2f}" for k, v in made.items()) + "\n  per goal: "
              + " ".join(f"{k.split()[-1]}={v:.2f}" for k, v in rec["per_goal"].items()), flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
