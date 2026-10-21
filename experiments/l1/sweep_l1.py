"""L1 (tabular) experiments: per-arm hyperparameter tuning, then the regime map.

  uv run python experiments/l1/sweep_l1.py tune     # writes outputs/l1/tuned.json
  uv run python experiments/l1/sweep_l1.py regime   # uses tuned.json, writes outputs/l1/regime.json + plots

Metric: area under the success curve (mean success over evaluation points up to the budget),
and episodes to 50% / 90% success. The x-axis is environment episodes; GFT's Stage II rollouts
count as episodes (conservative for GFT).
"""

import argparse
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from gft.l1_tabular import Env, episodes_to, run

OUT = Path("outputs/l1")
BUDGET = 65536

# Hyperparameter grids per arm (tuned on the tuning settings, then frozen).
LRS = [30, 100, 300, 1000, 3000, 10000, 30000, 100000]
GRIDS = {
    "gft": [dict(beta=b, beta_unrep=bu, beta_distill=bd)
             for b in (0.01, 0.1) for bu in (10, 100, 1000) for bd in (1, 2, 4, 8)],
    "gft-unfactored": [dict(beta=b, beta_unrep=bu, beta_distill=bd)
                        for b in (0.01, 0.1) for bu in (10, 100, 1000) for bd in (1, 2, 4, 8)],
    "fbc-verifier": [dict(beta=b) for b in (0.003, 0.01, 0.03, 0.1, 0.5, 2)],
    "fbc-all_positive": [dict(beta=b) for b in (0.1, 0.5, 2, 8, 32)],
    **{f"{algo}-{rew}": [dict(lr=lr) for lr in LRS]
       for algo in ("grpo", "rloo") for rew in ("bin", "frac", "parse", "align")},
}
TUNE_ENVS = [Env(N=16, m=4, q=0.35), Env(N=64, m=8, q=0.35), Env(N=64, m=4, q=0.1)]
TUNE_SEEDS = [100, 101, 102]


def auc(curve):
    return float(np.mean([p["success"] for p in curve]))


def job(spec):
    arm, env, seed, overrides = spec
    r = run(arm, env, seed=seed, budget=BUDGET, overrides=overrides)
    c = r.curve
    return dict(arm=arm, env=asdict(env), seed=seed, overrides=overrides, auc=auc(c),
                to50=episodes_to(c, 0.5), to90=episodes_to(c, 0.9), final=c[-1]["success"],
                curve=[(p["env_episodes"], p["teacher_calls"], round(p["success"], 4), round(p["frac"], 4)) for p in c])


def pool_map(specs, procs):
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(job, specs, chunksize=4)


def tune(procs):
    specs = [(arm, env, s, o) for arm, grid in GRIDS.items() for o in grid for env in TUNE_ENVS for s in TUNE_SEEDS]
    t = time.time()
    res = pool_map(specs, procs)
    print(f"{len(specs)} runs in {time.time() - t:.0f}s")
    best = {}
    for arm, grid in GRIDS.items():
        scores = []
        for o in grid:
            per_env = [np.mean([r["auc"] for r in res if r["arm"] == arm and r["overrides"] == o
                                and r["env"] == asdict(env)]) for env in TUNE_ENVS]
            scores.append((float(np.mean(per_env)), o, per_env))
        score, o, per_env = max(scores, key=lambda x: x[0])
        best[arm] = o
        on_edge = o == grid[0] or o == grid[-1] or any(o.get(k) in (min(g[k] for g in grid), max(g[k] for g in grid))
                                                            for k in o if len(grid) > 1)
        print(f"{arm:20s} best={o} auc={score:.3f} per_env={[round(x, 3) for x in per_env]}"
              + ("  [at grid edge]" if on_edge else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tuned.json").write_text(json.dumps(best, indent=1))


REGIME = dict(N=[16, 64], m=[2, 4, 8], q=[0.02, 0.1, 0.35], reveal=[True, False], kappa=[0.25, 0.5, 1.0])
MAIN_ARMS = ["gft", "gft-unfactored", "fbc-verifier", "fbc-all_positive",
             "grpo-bin", "grpo-frac", "grpo-parse", "rloo-frac", "rloo-parse", "grpo-align"]


def regime(procs, seeds):
    best = json.loads((OUT / "tuned.json").read_text())
    specs = []
    for N, m, q, reveal, kappa in itertools.product(*REGIME.values()):
        env = Env(N=N, m=m, q=q, reveal=reveal, kappa_sat=kappa, kappa_viol=kappa)
        for arm in MAIN_ARMS:
            # Feedback content only matters to arms that read feedback; skip duplicate runs.
            if not reveal and arm != "gft":
                continue
            if kappa != 0.5 and not (arm.startswith("gft") or arm.endswith("parse") or arm == "fbc-all_positive"):
                continue  # arms that never see the teacher do not depend on kappa
            for s in seeds:
                specs.append((arm, env, s, best[arm]))
    t = time.time()
    res = pool_map(specs, procs)
    print(f"{len(specs)} runs in {time.time() - t:.0f}s")
    (OUT / "regime.json").write_text(json.dumps(res))
    summarize(res)


def summarize(res):
    """AUC table; arms that ignore kappa reuse their kappa = 0.5 runs."""
    rows = {}
    for r in res:
        e = r["env"]
        rows.setdefault((e["N"], e["m"], e["q"], e["kappa_sat"], e["reveal"]), {}).setdefault(r["arm"], []).append(r)
    cols = MAIN_ARMS + ["gft[p/f]"]
    print(f"{'N':>3} {'m':>2} {'q':>5} {'kap':>4} " + " ".join(f"{a[:11]:>11}" for a in cols))
    for N, m, q, kappa in itertools.product(REGIME["N"], REGIME["m"], REGIME["q"], REGIME["kappa"]):
        line = f"{N:>3} {m:>2} {q:>5} {kappa:>4} "
        for a in cols:
            arm, reveal = (a[:-5], False) if a.endswith("[p/f]") else (a, True)
            v = rows.get((N, m, q, kappa, reveal), {}).get(arm) or rows.get((N, m, q, 0.5, reveal), {}).get(arm, [])
            line += f"{np.mean([r['auc'] for r in v]):11.3f} " if v else f"{'':11} "
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "regime", "summary"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.what == "tune":
        tune(a.procs)
    elif a.what == "regime":
        regime(a.procs, list(range(a.seeds)))
    else:
        summarize(json.loads((OUT / "regime.json").read_text()))


if __name__ == "__main__":
    main()
