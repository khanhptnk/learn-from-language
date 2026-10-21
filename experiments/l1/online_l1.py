"""L1 experiments with incremental GFT (batched Online GFT, paper Algorithm 2) as the method under test.

Every arm is tuned per cell on seeds 100-102 (AUC of success over the 65k-episode budget) and
evaluated on seeds 0-4. RL-parse arms choose the parser (pessimistic / optimistic) as part of
tuning. References: batch GFT (Algorithm 1 + replay, `gft`), oracle-reward RL, filtered BC.

  uv run python experiments/l1/online_l1.py regime    # N x m x q, reporter vs grader, kappa 0.5
  uv run python experiments/l1/online_l1.py horizon   # N = 64, depth k in {1, 2, 3, 6}, m in {4, 8}
  uv run python experiments/l1/online_l1.py intents   # 12 profiles x depth {1, 3, 6} x q
  uv run python experiments/l1/online_l1.py gstar     # miscalibrated ideal feedback (needs intents.json)
  uv run python experiments/l1/online_l1.py summary --name regime
"""

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from gft.l1_horizon import HEnv
from gft.l1_horizon import run as run_h
from gft.l1_intents import PROFILES, desc_share_profile
from gft.l1_online import run as run_online
from gft.l1_tabular import episodes_to

OUT = Path("outputs/l1_online")
PROFS = {**PROFILES, **{f"desc{s}": desc_share_profile(s) for s in (0.0, 0.1, 0.25, 0.5, 1.0)}}
LR = (100, 300, 1000, 3000)
GRIDS = {
    "gft-sgd": [dict(lr_aux=1000, lr_clean=c, unm_lr_frac=u, stage2_grad="expected")
                 for c in (30, 100, 300, 1000) for u in (0.0, 0.1, 1.0)],
    "gft-sgd-unf": [dict(lr_aux=1000, lr_clean=c, unm_lr_frac=u, stage2_grad="expected")
                     for c in (30, 100, 300, 1000) for u in (0.0, 0.1, 1.0)],
    "gft": [dict(beta=0.01, beta_unrep=bu, beta_distill=bd) for bu in (10, 100) for bd in (2, 4, 8)],
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
                   arms=["gft-sgd", "gft-sgd-unf", "gft", "grpo-parse", "rloo-parse"]),
    "horizon": dict(bk=[(64, 1), (8, 2), (4, 3), (2, 6)], m=[4, 8], q=[0.02, 0.1, 0.35], profiles=["reporter"],
                    arms=["gft-sgd", "gft", "grpo-parse", "rloo-parse"]),
    "intents": dict(bk=[(64, 1), (4, 3), (2, 6)], m=[4], q=[0.02, 0.1, 0.35], profiles=list(PROFS),
                    arms=["gft-sgd", "grpo-parse", "rloo-parse"]),
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

    cols = ["gft-sgd", "gft-sgd-unf", "gft", "grpo-parse", "rloo-parse"]
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


GSTAR = dict(profiles=["reporter", "mixed"], bk=[(64, 1), (4, 3)], q=[0.02, 0.1],
             kappa=[-1.0, 0.25, 1.0], error=[0.0, 0.05, 0.1, 0.2, 0.4])


def gstar_sweep(procs):
    """GFT with an approximate ideal-feedback generator, at hyperparameters tuned for the calibrated one.

    gstar_kappa: G~ mentions each requirement with this probability (-1 = the teacher's rate).
    gstar_error: G~ describes a wrong outcome for a mentioned requirement with this probability.
    RL does not use G*, so its numbers are the profile-matched rows of intents.json.
    """
    log = open(OUT / "gstar.progress", "w")
    tuned = json.loads((OUT / "intents.json").read_text())
    best = {(r["profile"], r["b"], r["k"], r["q"]): r["overrides"] for r in tuned if r["arm"] == "gft-sgd"}
    specs = [("gft-sgd", p, b, k, 4, q, s, {**best[(p, b, k, q)], "gstar_kappa": kap, "gstar_error": err})
             for p in GSTAR["profiles"] for (b, k) in GSTAR["bk"] for q in GSTAR["q"]
             for kap in GSTAR["kappa"] for err in GSTAR["error"] for s in range(5)]
    res = pool_map(specs, procs, "gstar", log)
    (OUT / "gstar.json").write_text(json.dumps(res))
    rl = {}
    for r in tuned:
        if r["arm"] in ("grpo-parse", "rloo-parse"):
            rl.setdefault((r["profile"], r["b"], r["k"], r["q"], r["arm"]), []).append(r["final"])
    with open(OUT / "gstar.txt", "w") as f:
        for out in (f, None):
            print("GFT final success with approximate G* (tuned for calibrated G*); "
                  "rows: G~ mention rate (-1 = calibrated), columns: hallucinated-outcome rate", file=out)
            for p in GSTAR["profiles"]:
                for (b, k) in GSTAR["bk"]:
                    for q in GSTAR["q"]:
                        best_rl = max(np.mean(rl[(p, b, k, q, a)]) for a in ("grpo-parse", "rloo-parse"))
                        print(f"\n{p} H={4 * k} q={q}   (RL with the same feedback: {best_rl:.2f})", file=out)
                        print(f"  {'kappa~':>7} " + " ".join(f"err={e:<5}" for e in GSTAR["error"]), file=out)
                        for kap in GSTAR["kappa"]:
                            vals = [np.mean([r["final"] for r in res if (r["profile"], r["b"], r["k"], r["q"]) ==
                                             (p, b, k, q) and r["overrides"]["gstar_kappa"] == kap
                                             and r["overrides"]["gstar_error"] == e]) for e in GSTAR["error"]]
                            print(f"  {kap:>7} " + " ".join(f"{v:9.2f}" for v in vals), file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=list(SWEEPS) + ["summary", "gstar"])
    ap.add_argument("--name", default="regime")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.what == "summary":
        summary(json.loads((OUT / f"{a.name}.json").read_text()), a.name)
    elif a.what == "gstar":
        gstar_sweep(a.procs)
    else:
        sweep(a.what, a.procs)


if __name__ == "__main__":
    main()
