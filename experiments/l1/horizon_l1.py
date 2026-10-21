"""L1-H horizon sweep: how GFT vs RL scale with chain depth k (horizon H = m * k).

Sweep A  fixed outcome space N = 64, (b, k) in {(64,1), (8,2), (4,3), (2,6)}: depth only.
Sweep B  fixed branching b = 4, k in {1, 2, 3, 4} (N = 4 ... 256): longer tasks, more outcomes.
Every arm is tuned per cell on seeds 100-102 (AUC), then evaluated on seeds 0-4.

  uv run python experiments/l1/horizon_l1.py run      # outputs/l1/horizon.json
  uv run python experiments/l1/horizon_l1.py summary
"""

import argparse
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gft.l1_horizon import HEnv, run
from gft.l1_tabular import episodes_to

OUT = Path("outputs/l1")
LRS = [100, 300, 1000, 3000, 10000, 30000]
GRIDS = {
    "gft": [dict(beta=0.01, beta_unrep=bu, beta_distill=bd) for bu in (10, 100) for bd in (2, 4, 8)],
    "gft[p/f]": [dict(beta=0.01, beta_unrep=bu, beta_distill=bd) for bu in (10, 100) for bd in (2, 4, 8)],
    "fbc-verifier": [dict(beta=b) for b in (0.003, 0.03, 0.3)],
    **{arm: [dict(lr=lr) for lr in LRS] for arm in ("grpo-frac", "rloo-frac", "grpo-parse", "grpo-align")},
}
CELLS_A = [(b, k) for b, k in [(64, 1), (8, 2), (4, 3), (2, 6)]]
CELLS_B = [(4, k) for k in (1, 2, 3, 4)]
QS = [0.02, 0.1, 0.35]
MS = [4, 8]


def env_for(b, k, m, q, arm):
    return HEnv(b=b, k=k, m=m, q=q, reveal=not arm.endswith("[p/f]"))


def job(spec):
    arm, b, k, m, q, seed, o = spec
    c = run(arm.replace("[p/f]", ""), env_for(b, k, m, q, arm), seed=seed, overrides=o)
    return dict(arm=arm, b=b, k=k, m=m, q=q, seed=seed, overrides=o, auc=float(np.mean([p["success"] for p in c])),
                final=c[-1]["success"], to50=episodes_to(c, 0.5), to90=episodes_to(c, 0.9))


def cells():
    return sorted({(b, k, m, q) for (b, k) in CELLS_A + CELLS_B for m in MS for q in QS})


def main_run(procs):
    t = time.time()
    tune_specs = [(arm, b, k, m, q, s, o) for (b, k, m, q) in cells() for arm, grid in GRIDS.items()
                  for o in grid for s in (100, 101, 102)]
    with mp.get_context("fork").Pool(procs) as pool:
        tuned = pool.map(job, tune_specs, chunksize=2)
    print(f"tuning: {len(tune_specs)} runs in {time.time() - t:.0f}s", flush=True)
    best = {}
    for (b, k, m, q) in cells():
        for arm, grid in GRIDS.items():
            score = lambda o: np.mean([r["auc"] for r in tuned if (r["arm"], r["b"], r["k"], r["m"], r["q"]) ==
                                       (arm, b, k, m, q) and r["overrides"] == o])
            best[(arm, b, k, m, q)] = max(grid, key=score)
    eval_specs = [(arm, b, k, m, q, s, best[(arm, b, k, m, q)]) for (b, k, m, q) in cells() for arm in GRIDS
                  for s in range(5)]
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(job, eval_specs, chunksize=2)
    print(f"total {time.time() - t:.0f}s", flush=True)
    (OUT / "horizon.json").write_text(json.dumps(res))
    summary(res)


def summary(res):
    arms = list(GRIDS)
    for name, cl in (("A: fixed N=64, depth k varies", CELLS_A), ("B: fixed b=4, N=4^k grows", CELLS_B)):
        print(f"\n{name}.  Each cell: mean final success @65k | median episodes-to-90% (k = thousands, - = never)")
        print(f"{'m':>2} {'q':>5} {'b':>3} {'k':>2} {'H':>3}  " + " ".join(f"{a:>17}" for a in arms))
        for m, q in itertools.product(MS, QS):
            for b, k in cl:
                line = f"{m:>2} {q:>5} {b:>3} {k:>2} {m * k:>3}  "
                for a in arms:
                    rs = [r for r in res if (r["arm"], r["b"], r["k"], r["m"], r["q"]) == (a, b, k, m, q)]
                    t90 = np.median([r["to90"] for r in rs])
                    t90s = "-" if not np.isfinite(t90) else f"{t90 / 1000:.0f}k"
                    line += f"{np.mean([r['final'] for r in rs]):>11.2f} | {t90s:>3} "
                print(line)
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary"])
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.what == "run":
        main_run(a.procs)
    else:
        summary(json.loads((OUT / "horizon.json").read_text()))


if __name__ == "__main__":
    main()
