"""L1 sanity checks.

capacity  (a) BC on correct demos: can pi reach ~100%?
          (b) Stage-I realizability: pi^fb fit on base rollouts, conditioned on G*:
              accuracy on reported slots (should be ~1) and unreported slots (should match the
              exact posterior, = the prior when kappa_sat = kappa_viol).
exact     Finite-sample GFT per-requirement accuracy per round vs the exact recursion,
          for several data/step budgets.

  uv run python experiments/phase1/sanity.py capacity --n-values 16
  uv run python experiments/phase1/sanity.py exact --n-values 16 --set episodes_per_round=1024
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run import base_policy, type_cast  # noqa: E402

from gft.algos.common import evaluate, execute, rollout, train_bc  # noqa: E402
from gft.algos.gft_exact import exact_slot_recursion  # noqa: E402
from gft.algos.gft_neural import GFT, GFTConfig  # noqa: E402
from gft.envs.reqworld import ReqWorldConfig, arg_for, sample_requests  # noqa: E402
from gft.feedback.logical_form import TeacherConfig, audit, ideal  # noqa: E402


def capacity(w, t, base, g, lr=1e-3):
    pol = base()
    opt = torch.optim.Adam(pol.parameters(), lr)
    for it in range(1, 6):
        req = sample_requests(w, 4096, g)
        ep = execute(w, req, req.slots.clone(), arg_for(w)[req.slots, req.values], torch.full((4096,), w.m))
        train_bc(pol, opt, ep, None, 200, 256, g)
        print(f"(a) expert BC steps={200 * it:4d} {fmt(evaluate(pol, w, 2000, g))}")
    pol = base()
    req = sample_requests(w, 8192, g)
    ep = rollout(pol, w, req, None, g)
    a0 = ep.sat.float().mean().item()
    msg = audit(req, ep.sat, ep.actual, w.n_values, t, g)
    opt = torch.optim.Adam(pol.parameters(), lr)
    target_unrep = a0 * (1 - t.kappa_sat) / ((1 - t.kappa_sat) * a0 + (1 - t.kappa_viol) * (1 - a0))
    for it in range(1, 6):
        train_bc(pol, opt, ep, msg, 200, 256, g)
        req2 = sample_requests(w, 4000, g)
        f = ideal(req2, w.n_values, t, g)
        ep2 = rollout(pol, w, req2, f, g)
        print(f"(b) pi^fb BC steps={200 * it:4d} acc|reported={ep2.sat[f.mask].float().mean():.3f} (target 1) "
              f"acc|unreported={ep2.sat[~f.mask].float().mean():.3f} (exact posterior {target_unrep:.3f}) "
              f"success|G*={ep2.sat.all(1).float().mean():.3f}  [base per-slot {a0:.3f}]")


def exact(w, t, base, g, overrides, rounds):
    pol = base()
    a0 = evaluate(pol, w, 4000, g)["frac_sample"]
    ref = exact_slot_recursion(a0, t.kappa_sat, t.kappa_viol, rounds)
    learner = GFT(pol, w, t, GFTConfig(**overrides), g)
    print(f"round  exact_a  stage2_frac  pi_frac  pi_success  (exact success = a^m)   overrides={overrides}")
    for r in range(1, rounds + 1):
        info = learner.round()
        ev = evaluate(pol, w, 2000, g)
        print(f"{r:5d}  {ref[r]:.3f}    {info['stage2_frac']:.3f}        {ev['frac_sample']:.3f}    "
              f"{ev['success_sample']:.3f}       {ref[r] ** w.m:.3f}   eps={learner.res.env_episodes}")


def fmt(d):
    return " ".join(f"{k}={v:.3f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["capacity", "exact"])
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n-values", type=int, default=16)
    ap.add_argument("--q", type=float, default=0.35)
    ap.add_argument("--systematic", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--hide-value", action="store_true")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    torch.set_num_threads(4)
    w = ReqWorldConfig(n_values=a.n_values, m=a.m)
    t = TeacherConfig(a.kappa, a.kappa, not a.hide_value)
    base = lambda: base_policy(w, a.q, 0, a.d_model, a.n_layers, 1000, "set", a.systematic)
    g = torch.Generator().manual_seed(123)
    if a.check == "capacity":
        capacity(w, t, base, g, a.lr)
    else:
        exact(w, t, base, g, {k: type_cast(v) for k, v in (s.split("=") for s in a.set)}, a.rounds)


if __name__ == "__main__":
    main()
