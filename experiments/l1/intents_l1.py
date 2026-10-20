"""L1-I intent sweep: which feedback intents carry GLFT's advantage over RL?

Profiles: reporter, grader, explainer, coach, scorer, vague, mixed, and a descriptive-share
sweep desc{s}. Cells: N = 64 outcomes reached by chains of depth k in {1, 3, 6} (b = 64, 4, 2;
horizon H = m k), m = 4, q in {0.02, 0.1, 0.35}.
Arms per profile (tuned per cell on seeds 100-102, evaluated on seeds 0-4):
  glft, grpo-parse, rloo-parse  (rewards parsed from the same mixed-intent message)
Profile-independent oracle references (L1 tuned settings): rloo-frac, grpo-frac, fbc-verifier, grpo-align.

  uv run python experiments/l1/intents_l1.py run
  uv run python experiments/l1/intents_l1.py summary
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
from glft.l1_intents import PROFILES, desc_share_profile, run
from glft.l1_tabular import episodes_to

OUT = Path("outputs/l1")
SHARES = [0.0, 0.1, 0.25, 0.5, 1.0]
PROFS = {**PROFILES, **{f"desc{s}": desc_share_profile(s) for s in SHARES}}
CELLS = list(itertools.product([(64, 1), (4, 3), (2, 6)], [0.02, 0.1, 0.35]))  # ((b, k), q)
M = 4
GRIDS = {
    "glft": [dict(beta=0.01, beta_unrep=bu, beta_distill=bd) for bu in (10, 100) for bd in (2, 4, 8)],
    "grpo-parse": [dict(lr=lr) for lr in (30, 100, 300, 1000, 3000, 10000)],
    "rloo-parse": [dict(lr=lr) for lr in (30, 100, 300, 1000, 3000, 10000)],
}
ORACLES = ["rloo-frac", "grpo-frac", "fbc-verifier", "grpo-align"]


def summarize_curve(c):
    return dict(auc=float(np.mean([p["success"] for p in c])), final=c[-1]["success"],
                to50=episodes_to(c, 0.5), to90=episodes_to(c, 0.9),
                to90_teacher=episodes_to(c, 0.9, x="teacher_calls"))


def job(spec):
    kind, arm, prof, bk, q, seed, o = spec
    env = HEnv(b=bk[0], k=bk[1], m=M, q=q)
    if kind == "oracle":
        c = run_h(arm, env, seed=seed, budget=65536, eval_every=2048, overrides=o)
    else:
        c = run(arm, env, PROFS[prof], seed=seed, overrides=o)
    return dict(kind=kind, arm=arm, profile=prof, N=list(bk), q=q, seed=seed, overrides=o, **summarize_curve(c))


def pool_map(specs, procs, label):
    """Parallel map that logs progress every ~5% so a long sweep can be checked mid-run."""
    out, t0, step = [], time.time(), max(1, len(specs) // 20)
    with mp.get_context("fork").Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, specs, chunksize=4), 1):
            out.append(r)
            if i % step == 0 or i == len(specs):
                el = time.time() - t0
                print(f"[{label}] {i}/{len(specs)} runs, {el:.0f}s elapsed, ~{el / i * (len(specs) - i):.0f}s left",
                      flush=True)
    return out


def rerun_profiles(procs, names):
    """Re-tune and re-evaluate GLFT only for some profiles (after a GLFT fix), with a wider grid."""
    grid = [dict(beta=0.01, beta_unrep=bu, beta_distill=bd, replay_unrep=rp)
            for bu in (0.1, 1, 10, 100) for bd in (2, 4, 8) for rp in (False, True)]
    specs = [("prof", "glft", p, N, q, s, o) for p in names for (N, q) in CELLS for o in grid for s in (100, 101, 102)]
    tuned = pool_map(specs, procs, "tune")
    best = {}
    for r in tuned:
        best.setdefault((r["profile"], tuple(r["N"]), r["q"]), {}).setdefault(
            json.dumps(r["overrides"], sort_keys=True), []).append(r["auc"])
    choice = {k: json.loads(max(v, key=lambda o: np.mean(v[o]))) for k, v in best.items()}
    new = pool_map([("prof", "glft", p, N, q, s, choice[(p, N, q)]) for p in names for (N, q) in CELLS
                    for s in range(5)], procs, "eval")
    res = [r for r in json.loads((OUT / "intents.json").read_text())
           if not (r["arm"] == "glft" and r["profile"] in names)] + new
    (OUT / "intents.json").write_text(json.dumps(res))
    summary(res)


def main_run(procs):
    t = time.time()
    specs = [("prof", arm, p, N, q, s, o) for p in PROFS for (N, q) in CELLS for arm, grid in GRIDS.items()
             for o in grid for s in (100, 101, 102)]
    tuned = pool_map(specs, procs, "tune")
    print(f"tuning: {len(specs)} runs in {time.time() - t:.0f}s", flush=True)
    best = {}
    for r in tuned:
        k = (r["arm"], r["profile"], tuple(r["N"]), r["q"], json.dumps(r["overrides"], sort_keys=True))
        best.setdefault(k[:4], {}).setdefault(k[4], []).append(r["auc"])
    choice = {k: json.loads(max(v, key=lambda o: np.mean(v[o]))) for k, v in best.items()}
    l1_tuned = json.loads((OUT / "tuned.json").read_text())
    specs = [("prof", arm, p, N, q, s, choice[(arm, p, N, q)]) for p in PROFS for (N, q) in CELLS
             for arm in GRIDS for s in range(5)]
    specs += [("oracle", arm, "-", N, q, s, l1_tuned[arm]) for (N, q) in CELLS for arm in ORACLES for s in range(5)]
    res = pool_map(specs, procs, "eval")
    print(f"total {time.time() - t:.0f}s", flush=True)
    (OUT / "intents.json").write_text(json.dumps(res))
    summary(res)


def summary(res):
    def cell(arm, prof, N, q, key):
        v = [r[key] for r in res if (r["arm"], r["profile"], tuple(r["N"]), r["q"]) == (arm, prof, N, q)]
        return np.median(v) if key.startswith("to") else np.mean(v)

    def fmt_t(x):
        return "-" if not np.isfinite(x) else f"{x / 1000:.0f}k"

    for N, q in CELLS:
        print(f"\n(b, k)={N} H={M * N[1]} q={q}   final success @65k episodes | median episodes-to-90%")
        orc = "  ".join(f"{a}: {cell(a, '-', N, q, 'final'):.2f}|{fmt_t(cell(a, '-', N, q, 'to90'))}" for a in ORACLES)
        print(f"  oracle references: {orc}")
        print(f"  {'profile':>10}  " + "  ".join(f"{a:>16}" for a in GRIDS))
        for p in PROFS:
            line = f"  {p:>10}  "
            for a in GRIDS:
                line += f"{cell(a, p, N, q, 'final'):>9.2f} | {fmt_t(cell(a, p, N, q, 'to90')):>4}  "
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "summary", "rerun"])
    ap.add_argument("--profiles", nargs="*", default=["scorer", "vague"])
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.what == "run":
        main_run(a.procs)
    elif a.what == "rerun":
        rerun_profiles(a.procs, a.profiles)
    else:
        summary(json.loads((OUT / "intents.json").read_text()))


if __name__ == "__main__":
    main()
