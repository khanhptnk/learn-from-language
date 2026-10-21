"""L1 experiments with incremental GLFT (batched Online GLFT, paper Algorithm 2) as the method under test.

Every arm is tuned per cell on seeds 100-102 (AUC of success over the 65k-episode budget) and
evaluated on seeds 0-4. RL-parse arms choose the parser (pessimistic / optimistic) as part of
tuning. References: batch GLFT (Algorithm 1 + replay, `glft`), oracle-reward RL, filtered BC.

  uv run python experiments/l1/online_l1.py regime    # N x m x q, reporter vs grader, kappa 0.5
  uv run python experiments/l1/online_l1.py horizon   # N = 64, depth k in {1, 2, 3, 6}, m in {4, 8}
  uv run python experiments/l1/online_l1.py intents   # 12 profiles x depth {1, 3, 6} x q
  uv run python experiments/l1/online_l1.py summary --name regime
"""

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from glft.l1_horizon import HEnv
from glft.l1_horizon import run as run_h
from glft.l1_intents import PROFILES, desc_share_profile
from glft.l1_online import run as run_online
from glft.l1_tabular import episodes_to

OUT = Path("outputs/l1_online")
PROFS = {**PROFILES, **{f"desc{s}": desc_share_profile(s) for s in (0.0, 0.1, 0.25, 0.5, 1.0)}}
LR = (100, 300, 1000, 3000)
GRIDS = {
    "glft-sgd": [dict(lr_aux=a, lr_clean=c, epochs=ep) for a in (1000, 3000) for c in (10, 30, 100)
                 for ep in (1, 2)],
    "glft-sgd-unf": [dict(lr_aux=a, lr_clean=c, epochs=ep) for a in (1000, 3000) for c in (10, 30, 100)
                     for ep in (1, 2)],
    "glft": [dict(beta=0.01, beta_unrep=bu, beta_distill=bd) for bu in (10, 100) for bd in (2, 4, 8)],
    "grpo-parse": [dict(lr=lr, parser=p) for lr in LR for p in ("pessimistic", "optimistic")],
    "rloo-parse": [dict(lr=lr, parser=p) for lr in LR for p in ("pessimistic", "optimistic")],
}
ORACLE_GRIDS = {  # do not read the teacher: tuned once per (env) cell, shared across profiles
    "rloo-frac": [dict(lr=lr) for lr in LR],
    "grpo-frac": [dict(lr=lr) for lr in LR],
    "fbc-verifier": [dict(beta=b) for b in (0.003, 0.03)],
    "grpo-align": [dict(lr=lr) for lr in (1000, 10000)],
}
SWEEPS = {
    "regime": dict(bk=[(16, 1), (64, 1)], m=[2, 4, 8], q=[0.02, 0.1, 0.35], profiles=["reporter", "grader"],
                   arms=["glft-sgd", "glft-sgd-unf", "glft", "grpo-parse", "rloo-parse"]),
    "horizon": dict(bk=[(64, 1), (8, 2), (4, 3), (2, 6)], m=[4, 8], q=[0.02, 0.1, 0.35], profiles=["reporter"],
                    arms=["glft-sgd", "glft", "grpo-parse", "rloo-parse"]),
    "intents": dict(bk=[(64, 1), (4, 3), (2, 6)], m=[4], q=[0.02, 0.1, 0.35], profiles=list(PROFS),
                    arms=["glft-sgd", "grpo-parse", "rloo-parse"]),
}


def job(spec):
    arm, prof, b, k, m, q, seed, o = spec
    env = HEnv(b=b, k=k, m=m, q=q)
    if arm in ORACLE_GRIDS:
        c = run_h(arm, env, seed=seed, budget=65536, eval_every=2048, overrides=o)
    else:
        c = run_online(arm, env, PROFS[prof], seed=seed, budget=65536, eval_every=2048, overrides=o)
    return dict(arm=arm, profile=prof, b=b, k=k, m=m, q=q, seed=seed, overrides=o,
                auc=float(np.mean([p["success"] for p in c])), final=c[-1]["success"],
                to90=episodes_to(c, 0.9), to90_teacher=episodes_to(c, 0.9, x="teacher_calls"))


def pool_map(specs, procs, label, log):
    out, t0, step = [], time.time(), max(1, len(specs) // 20)
    with mp.get_context("fork").Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, specs, chunksize=2), 1):
            out.append(r)
            if i % step == 0 or i == len(specs):
                el = time.time() - t0
                print(f"[{label}] {i}/{len(specs)} runs, {el:.0f}s elapsed, ~{el / i * (len(specs) - i):.0f}s left",
                      file=log, flush=True)
    return out


def sweep(name, procs):
    sw = SWEEPS[name]
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"{name}.progress", "w")
    envs = [(b, k, m, q) for (b, k) in sw["bk"] for m in sw["m"] for q in sw["q"]]
    cells = [(arm, p) + e for e in envs for p in sw["profiles"] for arm in sw["arms"]]
    cells += [(arm, "-") + e for e in envs for arm in ORACLE_GRIDS]
    grid = lambda arm: GRIDS.get(arm) or ORACLE_GRIDS[arm]
    tune_specs = [c + (s, o) for c in cells for o in grid(c[0]) for s in (100, 101, 102)]
    tuned = pool_map(tune_specs, procs, "tune", log)
    scores = {}
    for r in tuned:
        key = (r["arm"], r["profile"], r["b"], r["k"], r["m"], r["q"])
        scores.setdefault(key, {}).setdefault(json.dumps(r["overrides"], sort_keys=True), []).append(r["auc"])
    best = {key: json.loads(max(v, key=lambda o: np.mean(v[o]))) for key, v in scores.items()}
    res = pool_map([c + (s, best[c]) for c in cells for s in range(5)], procs, "eval", log)
    (OUT / f"{name}.json").write_text(json.dumps(res))
    with open(OUT / f"{name}.txt", "w") as f:
        summary(res, name, f)
    summary(res, name)


def summary(res, name, f=None):
    sw = SWEEPS[name]

    def cell(arm, prof, b, k, m, q, key):
        v = [r[key] for r in res if (r["arm"], r["profile"], r["b"], r["k"], r["m"], r["q"]) == (arm, prof, b, k, m, q)]
        return (np.median(v) if key.startswith("to") else np.mean(v)) if v else np.nan

    def fmt(arm, prof, *e):
        t = cell(arm, prof, *e, "to90")
        return f"{cell(arm, prof, *e, 'final'):.2f}|{'-' if not np.isfinite(t) else f'{t / 1000:.0f}k':>3}"

    cols = ["glft-sgd", "glft-sgd-unf", "glft", "grpo-parse", "rloo-parse"]
    print(f"[{name}] final success @65k | median episodes-to-90%.  Oracle refs need a verifier.", file=f)
    print(f"{'profile':>9} {'b':>3} {'k':>2} {'m':>2} {'H':>3} {'q':>5}  " + " ".join(f"{c:>15}" for c in cols)
          + "  | " + " ".join(f"{c:>12}" for c in ORACLE_GRIDS), file=f)
    for p in sw["profiles"]:
        for (b, k), m, q in itertools.product(sw["bk"], sw["m"], sw["q"]):
            e = (b, k, m, q)
            line = f"{p:>9} {b:>3} {k:>2} {m:>2} {m * k:>3} {q:>5}  " + " ".join(f"{fmt(c, p, *e):>15}" for c in cols)
            line += "  | " + " ".join(f"{fmt(c, '-', *e):>12}" for c in ORACLE_GRIDS)
            print(line, file=f)
        print(file=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=list(SWEEPS) + ["summary"])
    ap.add_argument("--name", default="regime")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.what == "summary":
        summary(json.loads((OUT / f"{a.name}.json").read_text()), a.name)
    else:
        sweep(a.what, a.procs)


if __name__ == "__main__":
    main()
