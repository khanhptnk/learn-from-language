"""How much of the base policy's success is incidental? Runs the base for the full step budget (no early stop) and
compares success at ANY step (what the current success check counts: the episode ends as soon as all requirements
hold) with success at the END of the budget. Also reports the request-size mix.

  uv run python experiments/minicraft/check_success_timing.py --base random
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, heldout_splits  # noqa: E402

from gft.minicraft.core import MiniCraft, WorldConfig  # noqa: E402
from gft.minicraft.goals import requirement_status  # noqa: E402
from gft.minicraft.learn import policy_actions, request_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="random")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1)
    cfg, dev = WorldConfig(), a.device
    model = base_policy(cfg, dev, seed=0, kind=a.base).eval()
    req = heldout_splits(a.n, tune=True)["val_worlds"]
    prompt = request_prompts(model, req, dev)
    rng = np.random.default_rng(1)
    env = MiniCraft(cfg, req.seeds)
    any_ok = np.zeros(len(req), bool)
    first = np.full(len(req), -1)
    for t in range(cfg.max_steps):
        sat, mask = requirement_status(env, req)
        ok = (sat | ~mask).all(1)
        first[ok & ~any_ok] = t
        any_ok |= ok
        env.step(policy_actions(model, env, req, dev, rng, prompt=prompt))
    sat, mask = requirement_status(env, req)
    end_ok = (sat | ~mask).all(1)
    k = (req.goals >= 0).sum(1)
    print(f"requests: {len(req)}; goals per request: " + ", ".join(f"{i}: {np.mean(k == i):.2f}" for i in (1, 2, 3)))
    print(f"success at any step (current check): {any_ok.mean():.3f}; at the end of {cfg.max_steps} steps: "
          f"{end_ok.mean():.3f}; satisfied at some step but not at the end: {(any_ok & ~end_ok).mean():.3f}")
    for i in (1, 2, 3):
        m = k == i
        print(f"  {i}-goal requests: any step {any_ok[m].mean():.3f}, end {end_ok[m].mean():.3f}")
    print(f"first satisfied at step (median over successes): {int(np.median(first[any_ok]))}")


if __name__ == "__main__":
    main()
