"""Run a grid of Phase 1 runs in parallel (one thread per process), skipping finished ones.

Example:
  uv run python experiments/phase1/sweep.py --name regime --arms gft grpo-frac grpo-parse \
      --grid n_values=4,16,64 m=2,4,8 --seeds 0 1 2 --budget 16384 --set-gft stage1_steps=20
"""

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import run, type_cast  # noqa: E402


def job(spec):
    out = Path(spec.pop("out"))
    if out.exists():
        return str(out), "skip", 0.0
    t = time.time()
    run(out=str(out), **spec)
    return str(out), "done", time.time() - t


def run_name(arm, cfg, seed):
    keys = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    return f"{arm}__{keys}__s{seed}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--grid", nargs="*", default=[], help="key=v1,v2,... over run() kwargs")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--fixed", nargs="*", default=[], help="key=value run() kwargs for all runs")
    # Per-arm-family learner overrides, e.g. --set-gft stage1_steps=20 --set-rl epochs=4
    ap.add_argument("--set-gft", nargs="*", default=[])
    ap.add_argument("--set-rl", nargs="*", default=[])
    ap.add_argument("--set-fbc", nargs="*", default=[])
    a = ap.parse_args()

    kv = lambda items: {k: type_cast(v) for k, v in (s.split("=", 1) for s in items)}
    grid = {k: [type_cast(x) for x in v.split(",")] for k, v in (s.split("=", 1) for s in a.grid)}
    fixed = kv(a.fixed)
    family = {"gft": kv(a.set_gft), "fbc": kv(a.set_fbc), "rl": kv(a.set_rl)}
    out_dir = Path("outputs/phase1") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_args.json").write_text(json.dumps(vars(a), indent=1))

    specs = []
    for arm in a.arms:
        fam = "gft" if arm == "gft" else "fbc" if arm.startswith("fbc-") else "rl"
        for combo in itertools.product(*grid.values()):
            cfg = dict(zip(grid.keys(), combo))
            for seed in a.seeds:
                specs.append(dict(arm=arm, seed=seed, budget=a.budget, overrides=family[fam], **fixed, **cfg,
                                  out=str(out_dir / run_name(arm, cfg, seed))))
    # Pretrain each distinct base policy once before fanning out (avoids duplicate work and races).
    from run import base_policy  # noqa: E402
    import torch  # noqa: E402
    from gft.envs.reqworld import ReqWorldConfig  # noqa: E402
    torch.set_num_threads(a.procs)
    bases = {(s.get("n_slots", 8), s.get("n_values", 4), s.get("m", 4), s.get("scramble", True), s.get("q", 0.35),
              s.get("systematic", 1.0), s["seed"]) for s in specs}
    for K, V, m, scr, q, sysm, seed in sorted(bases):
        base_policy(ReqWorldConfig(n_slots=K, n_values=V, m=m, scramble=scr), q, seed, fixed.get("d_model", 32),
                    fixed.get("n_layers", 1), fixed.get("pretrain_steps", 1000), fixed.get("model", "set"),
                    systematic=sysm)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, (path, status, secs) in enumerate(pool.imap_unordered(job, specs), 1):
            print(f"[{i}/{len(specs)} {time.time() - t0:6.0f}s] {status} {secs:5.1f}s {Path(path).name}", flush=True)


if __name__ == "__main__":
    main()
