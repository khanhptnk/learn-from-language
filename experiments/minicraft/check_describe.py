"""Capacity check for pi^fb: can the policy network learn to follow feedback-style descriptions under ideal
conditions (behaviour cloning on expert demos whose prompt is the teacher's description of that very demo)?

  .venv/bin/python experiments/minicraft/check_describe.py --profile describe --out outputs/minicraft/describe.json

Demos: the expert carries out random plans on fresh worlds; each episode's prompt is a teacher message about it (no
request). Test, on training-pool tasks (worlds unseen in the demos): prompted with an ideal message for the task (from
the success bank), does the policy achieve the task's requirements? Control: prompted with the ideal message of a
different task. If this fails, GFT's pi^fb cannot work with this network / message format; if it works, the problem
is Stage I's data, not capacity. Diagnostic only -- not training data for any method.
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
from train import has_heldout  # noqa: E402

from gft.minicraft.core import N_WORLD_ACTIONS, WorldConfig  # noqa: E402
from gft.minicraft.expert import ExpertPool  # noqa: E402
from gft.minicraft.goals import Requests, sample_requests  # noqa: E402
from gft.minicraft.learn import (CleanPolicy, cached_pool, obs_tensor, on_device, policy_actions,  # noqa: E402
                                 prompt_tokens, rng_state, run_episodes, set_rng_state, stack_steps)
from gft.minicraft.teacher import PROFILES, compose, episode_facts  # noqa: E402

PROFILES.setdefault("describe", dict(descriptive=1.0))      # every descriptive claim, nothing else


def message(facts, profile, rng):
    return " ".join(s for _, _, s in compose(facts, profile, rng)) or "Okay."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="describe")
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--evals", default="2500,5000,10000")
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    cfg, dev = WorldConfig(), a.device
    rng = np.random.default_rng(123)
    t0 = time.time()
    beh = sample_requests(rng, a.episodes, rng.integers(0, 2 ** 31, size=a.episodes),
                          combo_filter=lambda gs: not has_heldout(gs))
    expert = ExpertPool(a.workers or None)

    def act_fn(e, t):
        ex = expert(e, beh.goals, np.nonzero(~e.done)[0])
        return np.array([ex[b] if rng.random() > a.noise else int(rng.integers(N_WORLD_ACTIONS)) for b in range(len(beh))])

    env, log, snaps, acts, valid, sat, mask = run_episodes(cfg, beh, act_fn)
    expert.close()
    texts = [message(episode_facts(log, b, env, beh.goals[b], sat[b]), a.profile, rng) for b in range(len(beh))]
    tok = prompt_tokens(texts, dev)                         # prompt = the description alone (no request)
    grid, pf, inv, act, b, _ = stack_steps(snaps, acts, valid)
    grid, pf, inv = on_device(dev, grid, pf, inv)
    act, b_t = torch.as_tensor(act, device=dev), torch.as_tensor(b, device=dev)
    print(f"demos: {a.episodes} episodes, {len(b)} steps ({time.time() - t0:.0f}s); example prompt: {texts[0]!r}",
          flush=True)

    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    idx = rng.choice(len(pool.seeds), size=a.n_eval)
    req = pool.requests(idx)
    ideal = [message(pool.ideal_facts(i, rng), a.profile, rng) for i in idx]
    other = [ideal[(k + 1) % len(ideal)] for k in range(len(ideal))]           # control: another task's message
    eval_prompts = {"ideal": prompt_tokens(ideal, dev), "other_task": prompt_tokens(other, dev)}

    model = CleanPolicy(cfg.size).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    record = dict(args=vars(a), evals=[])

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for name, tk in eval_prompts.items():
            p = model.encode_prompt(tk)
            r = np.random.default_rng(1)
            saved = rng_state()
            torch.manual_seed(1)
            _, _, _, _, _, s, m = run_episodes(cfg, req, lambda e, t: policy_actions(model, e, req, dev, r, prompt=p),
                                               record=False)
            set_rng_state(saved)
            out[name] = dict(success=float((s | ~m).all(1).mean()), frac=float((s & m).sum() / m.sum()))
        model.train()
        return out

    evals = set(map(int, a.evals.split(",")))
    losses, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        i = torch.randint(len(act), (a.batch,), device=dev)
        eps, inverse = torch.unique(b_t[i], return_inverse=True)
        lg, _ = model(*obs_tensor(grid[i], pf[i], inv[i], dev), model.encode_prompt(tok[eps])[inverse])
        loss = F_.cross_entropy(lg, act[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach())
        if step in evals:
            row = dict(step=step, bc_loss=float(torch.stack(losses[-200:]).mean()), seconds=time.time() - t0,
                       **evaluate())
            record["evals"].append(row)
            print(f"step {step}: BC loss {row['bc_loss']:.3f} | task success with ideal message "
                  f"{row['ideal']['success']:.3f} (frac {row['ideal']['frac']:.3f}) | with another task's message "
                  f"{row['other_task']['success']:.3f} (frac {row['other_task']['frac']:.3f}) ({row['seconds']:.0f}s)",
                  flush=True)
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
