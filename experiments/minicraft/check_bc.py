"""Capacity check: can the policy network learn every goal by behaviour cloning on expert demos?

  uv run python experiments/minicraft/check_bc.py --episodes 2000 --steps 4000 --arch d=128,ch=32,crop=5,layers=3

BC on (slightly noisy) expert runs of requests over all goals (held-out goal pairs excluded, as in training), then
per-goal success on the *validation* splits (unseen worlds; held-out goal combinations), so architecture choice never
sees test data. If this fails, no learner can be expected to reach the deep goals with this network.
Writes a JSON summary to --out.
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
from train import parse_arch, has_heldout, heldout_splits  # noqa: E402

from gft.minicraft.core import N_WORLD_ACTIONS, WorldConfig  # noqa: E402
from gft.minicraft.expert import ExpertPool  # noqa: E402
from gft.minicraft.goals import sample_requests  # noqa: E402
from gft.minicraft.learn import CleanPolicy, GoalPrompts, evaluate, obs_tensor, run_episodes, stack_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--evals", default="1000,2000,4000")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arch", default="", help="CleanPolicy kwargs, e.g. kind=gvin,pixels=true or d=128,rec=8")
    ap.add_argument("--out", default="")
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy (GPU speed-up, same math)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cache", default="outputs/minicraft/demos")
    ap.add_argument("--workers", type=int, default=0, help="processes for expert demos (0 = all CPUs)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    cfg, dev = WorldConfig(), a.device
    rng = np.random.default_rng(0)
    t0 = time.time()
    cache = Path(a.cache) / f"demos_n{a.episodes}_noise{a.noise}_v1.npz"   # demos depend only on these and seed 0
    if cache.exists():
        z = np.load(cache)
        grid, pf, inv, act, b, ep_goals = (z[k] for k in ("grid", "pf", "inv", "act", "b", "goals"))
        demo_success = float(z["success"])
    else:
        req = sample_requests(rng, a.episodes, rng.integers(0, 2 ** 31, size=a.episodes),
                              combo_filter=lambda gs: not has_heldout(gs))
        expert = ExpertPool(a.workers or None)

        def act_fn(e, t):                       # same random stream as labelling one world at a time
            ex = expert(e, req.goals, np.nonzero(~e.done)[0])
            return np.array([ex[b] if rng.random() > a.noise else int(rng.integers(N_WORLD_ACTIONS)) for b in range(len(req))])

        _, _, snaps, acts, valid, sat, mask = run_episodes(cfg, req, act_fn)
        expert.close()
        grid, pf, inv, act, b, _ = stack_steps(snaps, acts, valid)
        ep_goals, demo_success = req.goals, float((sat | ~mask).all(1).mean())
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache.with_suffix(".tmp.npz"), grid=grid, pf=pf, inv=inv, act=act, b=b, goals=ep_goals,
                 success=demo_success)
        cache.with_suffix(".tmp.npz").rename(cache)
    print(f"demos: {a.episodes} episodes, {len(b)} steps, demo success {demo_success:.3f} ({time.time() - t0:.0f}s)",
          flush=True)
    held = np.random.default_rng(1).random(a.episodes) < 0.05            # demo episodes held out for BC val loss
    tr, va = np.nonzero(~held[b])[0], np.nonzero(held[b])[0]
    splits = heldout_splits(a.n_eval, tune=True)
    arch = parse_arch(a.arch)
    model = CleanPolicy(cfg.size, **arch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch {arch or 'default'}: {n_params} parameters", flush=True)
    record = dict(args=vars(a), arch=arch, params=n_params, evals=[])
    model.to(dev)
    if a.compile:
        model.forward = torch.compile(model.forward)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    grid, pf, inv = (torch.as_tensor(x, device=dev) for x in (grid, pf, inv))   # data on the device
    prompts = GoalPrompts(ep_goals[b], dev)               # request prompts, encoded per minibatch
    act = torch.as_tensor(act, device=dev)
    tr_t = torch.as_tensor(tr, device=dev)

    @torch.no_grad()
    def val_loss():
        out = []
        for j in range(0, len(va), 8192):
            i = va[j:j + 8192]
            lg = model(*obs_tensor(grid[i], pf[i], inv[i], dev), prompts.vectors(model, i))[0]
            out.append(F_.cross_entropy(lg, act[i],
                                        reduction="sum").item())
        return sum(out) / len(va)
    evals = set(map(int, a.evals.split(",")))
    t0 = time.time()
    losses = []
    for step in range(1, a.steps + 1):
        i = tr_t[torch.randint(len(tr_t), (a.batch,), device=dev)]
        x, iv = obs_tensor(grid[i], pf[i], inv[i], dev)
        logits, _ = model(x, iv, prompts.vectors(model, i))
        loss = F_.cross_entropy(logits, act[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach())
        if step in evals:
            vl, tl = val_loss(), float(torch.stack(losses[-200:]).mean())
            print(f"step {step}: BC loss {tl:.3f}, held-out demos {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
            row = dict(step=step, bc_loss=tl, val_loss=vl, train_seconds=time.time() - t0)
            record["evals"].append(row)
            for name, r in splits.items():
                ev = evaluate(model, cfg, r, dev, seed=1, detail=True)
                row[name] = {k: v for k, v in ev.items() if k != "episode_success"}
                pg = "  ".join(f"{k}: {v[0]:.2f}" for k, v in ev["per_goal"].items())
                print(f"  {name:10s} success {ev['success']:.3f}  frac {ev['frac']:.3f}  steps {ev['mean_steps']:.0f}\n"
                      f"    {pg}", flush=True)
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
