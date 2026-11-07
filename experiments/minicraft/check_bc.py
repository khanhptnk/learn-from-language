"""Can behaviour cloning master the tasks? Request-conditioned BC (prompt = the request, no feedback) on noise-free
expert runs of every request without the excluded goals, pixel observations, success needing DONE (the train.py default);
then success per goal on fresh worlds: single-goal requests for every seen goal, and mixed 1-3 goal requests. The
expert on the same requests is the reference. Tests whether the network can represent and learn the crafting tasks
at all (the v3 run's arms stalled on the first crafting tier).
  uv run python experiments/minicraft/check_bc.py --episodes 5000 --steps 5000 --out outputs/minicraft/check_bc.json"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.expert import ExpertPool
from gft.minicraft.goals import GOALS, N_GOALS, MAX_GOALS, Requests, sample_requests
from gft.minicraft.learn import evaluate, pretrain_base, run_episodes

EXCLUDED = ("build a well", "make a sling")


def single_goal_requests(goals, n, rng):
    g = np.full((len(goals) * n, MAX_GOALS), -1, dtype=np.int64)
    g[:, 0] = np.repeat(goals, n)
    return Requests(g, rng.integers(2 ** 31, 2 ** 32, size=len(g)))


def expert_success(cfg, req):
    pool = ExpertPool()
    _, _, _, _, _, sat, mask = run_episodes(cfg, req, lambda e, t: pool(e, req.goals, np.nonzero(~e.done)[0]),
                                            record=False)
    pool.close()
    return {GOALS[q].phrase: float(sat[req.goals == q].mean()) for q in range(N_GOALS) if (req.goals == q).any()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=64, help="evaluation episodes per goal (single-goal requests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="outputs/minicraft/check_bc.json")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = WorldConfig(success_needs_done=True)
    excl = {q for q, g in enumerate(GOALS) if g.phrase in EXCLUDED}
    seen = [q for q in range(N_GOALS) if q not in excl]
    t0 = time.time()
    losses = []
    model = pretrain_base(cfg, a.device, n_episodes=a.episodes, goal_pool=seen, noise=0.0, steps=a.steps,
                          behaviour="request", combo_filter=lambda gs: not gs & excl, text=True, feedback_frac=0.0,
                          arch=dict(pixels=True), on_loss=lambda st, l: (losses.append((st, l)),
                                                                     print(f"step {st} loss {l:.4f}", flush=True)))
    t_train = time.time() - t0
    rng = np.random.default_rng(12345)
    single = single_goal_requests(seen, a.n_eval, rng)
    mixed = sample_requests(rng, 256, rng.integers(2 ** 31, 2 ** 32, size=256), goal_pool=seen,
                            combo_filter=lambda gs: not gs & excl)
    res = dict(episodes=a.episodes, steps=a.steps, train_seconds=t_train, loss=losses,
               bc_single=evaluate(model, cfg, single, a.device, detail=True),
               bc_single_greedy=evaluate(model, cfg, single, a.device, greedy=True, detail=True),
               bc_mixed=evaluate(model, cfg, mixed, a.device, detail=True),
               expert_single=expert_success(cfg, single))
    for k in ("bc_single", "bc_single_greedy", "bc_mixed"):
        for r in ("episode_success",):
            res[k].pop(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"trained {t_train:.0f}s; mixed success {res['bc_mixed']['success']:.2f}")
    print(f"{'goal':18s} {'BC':>6s} {'greedy':>7s} {'expert':>7s}")
    for q in seen:
        p = GOALS[q].phrase
        print(f"{p:18s} {res['bc_single']['per_goal'][p][0]:6.2f} {res['bc_single_greedy']['per_goal'][p][0]:7.2f} "
              f"{res['expert_single'][p]:7.2f}")
