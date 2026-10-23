"""Capacity check with DAgger: is the network able to solve every goal once behaviour cloning's compounding errors are
removed?

  .venv/bin/python experiments/minicraft/check_dagger.py --arch trunk=3 --out outputs/minicraft/arch/dagger_trunk3.json

Start from BC on the cached expert demos (built by check_bc.py), then each iteration: roll out the current policy on
fresh training requests, label every visited state with the expert's action, and train on that batch only (no
aggregation, like an on-policy update; --aggregate keeps all earlier data as in standard DAgger).
Per-goal success on the validation splits after each iteration. If this approaches the expert, the architecture is
sufficient and BC's gap is distribution shift; if it plateaus low, the network is the bottleneck.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from train import has_heldout, heldout_splits  # noqa: E402

from gft.minicraft.core import WorldConfig  # noqa: E402
from gft.minicraft.expert import ExpertPool  # noqa: E402
from gft.minicraft.goals import sample_requests  # noqa: E402
from gft.minicraft.learn import (CleanPolicy, GoalPrompts, evaluate, obs_tensor, policy_actions,  # noqa: E402
                                 request_prompts, run_episodes, stack_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="outputs/minicraft/demos/demos_n10000_noise0.1_v1.npz")
    ap.add_argument("--arch", default="trunk=3")
    ap.add_argument("--bc-steps", type=int, default=10000)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--aggregate", action="store_true", help="keep earlier data (standard DAgger)")
    ap.add_argument("--episodes", type=int, default=2000, help="policy rollouts labelled per iteration")
    ap.add_argument("--steps", type=int, default=2000, help="training steps per iteration")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy (GPU speed-up, same math)")
    ap.add_argument("--workers", type=int, default=0, help="processes for expert labelling (0 = all CPUs)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    cfg, dev = WorldConfig(), a.device
    rng = np.random.default_rng(0)
    z = np.load(a.demos)
    data = dict(grid=[z["grid"]], pf=[z["pf"]], inv=[z["inv"]], act=[z["act"]], goals=[z["goals"][z["b"]]])
    arch = {k: int(v) for k, v in (kv.split("=") for kv in a.arch.split(",") if kv)}
    model = CleanPolicy(cfg.size, **arch).to(dev)
    if a.compile:
        model.forward = torch.compile(model.forward)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    splits = heldout_splits(a.n_eval, tune=True)
    record = dict(args=vars(a), arch=arch, iters=[])
    t0 = time.time()

    def train(steps):
        """Training data lives on the device; minibatch indices are drawn there (no host slicing or copies)."""
        grid, pf, inv, act = (torch.as_tensor(np.concatenate(data[k]), device=dev) for k in ("grid", "pf", "inv", "act"))
        prompts = GoalPrompts(np.concatenate(data["goals"]), dev)
        losses = []
        for _ in range(steps):
            i = torch.randint(len(act), (a.batch,), device=dev)
            logits, _ = model(*obs_tensor(grid[i], pf[i], inv[i], dev), prompts.vectors(model, i))
            loss = F_.cross_entropy(logits, act[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        return float(torch.stack(losses[-200:]).mean()), len(act)

    def report(it, loss, n, extra=None):
        row = dict(iter=it, train_loss=loss, dataset_steps=n, seconds=time.time() - t0, **(extra or {}))
        for name, r in splits.items():
            ev = evaluate(model, cfg, r, dev, seed=1, detail=True)
            row[name] = {k: v for k, v in ev.items() if k != "episode_success"}
        record["iters"].append(row)
        pg = " ".join(f"{k.split()[-1]}={v[0]:.2f}" for k, v in row["val_worlds"]["per_goal"].items())
        print(f"iter {it}: loss {loss:.3f}, data {n} steps, worlds {row['val_worlds']['success']:.3f}, "
              f"combos {row['val_combos']['success']:.3f} ({time.time() - t0:.0f}s)\n    {pg}", flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(record, indent=1))

    expert = ExpertPool(a.workers or None)
    loss, n = train(a.bc_steps)
    report(0, loss, n)
    for it in range(1, a.iters + 1):
        req = sample_requests(rng, a.episodes, rng.integers(0, 2 ** 31, size=a.episodes),
                              combo_filter=lambda gs: not has_heldout(gs))
        labels = []
        prompt = request_prompts(model, req, dev)

        def act_fn(e, t):
            labels.append(expert(e, req.goals, np.nonzero(~e.done)[0]))
            return policy_actions(model, e, req, dev, rng, prompt=prompt)

        model.eval()
        _, _, snaps, acts, valid, sat, mask = run_episodes(cfg, req, act_fn)
        model.train()
        grid, pf, inv, _, b, t = stack_steps(snaps, acts, valid)
        lab = np.array(labels).T[b, t]
        agree = float((lab == acts[b, t]).mean())
        for k, v in (("grid", grid), ("pf", pf), ("inv", inv), ("act", lab), ("goals", req.goals[b])):
            data[k] = data[k] + [v] if a.aggregate else [v]
        loss, n = train(a.steps)
        report(it, loss, n, dict(rollout_success=float((sat | ~mask).all(1).mean()), expert_agreement=agree))
    expert.close()


if __name__ == "__main__":
    main()
