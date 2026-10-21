"""Phase 1: one learner on ReqWorld-C, logged against environment episodes.

Every arm starts from the same pretrained base policy (same seed => same weights), whose
per-requirement competence is ~q, so the initial success rate is ~q^m.

Example:
  uv run python experiments/phase1/run.py --arm gft --m 4 --seed 0 --budget 20000
Arms: gft | reinforce-{bin,frac,parse} | rloo-* | grpo-* | ppo-* | fbc-{verifier,all_positive}
"""

import argparse
import json
import time
from pathlib import Path

import torch

from gft.algos.common import Curve, evaluate, pretrain_base
from gft.algos.gft_neural import GFT, GFTConfig
from gft.algos.lang_baselines import FilteredBC, FilteredBCConfig
from gft.algos.rl_neural import PolicyGradient, RLConfig
from gft.envs.reqworld import ReqWorldConfig
from gft.feedback.logical_form import TeacherConfig
from gft.models.set_policy import SetPolicy
from gft.models.transformer import Policy


def build(arm, policy, wcfg, tcfg, gen, overrides):
    if arm == "gft":
        return GFT(policy, wcfg, tcfg, GFTConfig(**overrides), gen)
    if arm.startswith("fbc-"):
        return FilteredBC(policy, wcfg, tcfg, FilteredBCConfig(accept=arm[4:], **overrides), gen)
    algo, reward = arm.split("-")
    return PolicyGradient(policy, wcfg, tcfg, RLConfig(algo=algo, reward=reward, **overrides), gen)


MODELS = {"set": SetPolicy, "seq": Policy}


def base_policy(wcfg, q, seed, d_model, n_layers, pretrain_steps, model="set", systematic=1.0,
                cache_dir="outputs/phase1/base"):
    """Pretrained base policy, cached on disk so every arm starts from identical weights."""
    torch.manual_seed(seed)
    policy = MODELS[model](wcfg, d_model=d_model, n_layers=n_layers, value_head=True)
    key = (f"{model}_K{wcfg.n_slots}_V{wcfg.n_values}_m{wcfg.m}_scr{int(wcfg.scramble)}_w{wcfg.world_seed}"
           f"_q{q}_sys{systematic}_s{seed}_d{d_model}_l{n_layers}_p{pretrain_steps}.pt")
    path = Path(cache_dir) / key
    if path.exists():
        policy.load_state_dict(torch.load(path))
    else:
        pretrain_base(policy, wcfg, q, pretrain_steps, torch.Generator().manual_seed(seed), lr=3e-3,
                      systematic=systematic)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{time.time_ns()}")
        torch.save(policy.state_dict(), tmp)
        tmp.rename(path)
    return policy


def run(arm, m=4, n_slots=8, n_values=4, q=0.35, systematic=1.0, scramble=True, seed=0, budget=20000,
        eval_every=1024, n_eval=1000, pretrain_steps=1000, kappa_sat=0.5, kappa_viol=0.5, reveal_value=True,
        lie_rate=0.0, model="set", d_model=32, n_layers=1, overrides=None, out=None):
    torch.set_num_threads(1)
    wcfg = ReqWorldConfig(n_slots=n_slots, n_values=n_values, m=m, scramble=scramble)
    tcfg = TeacherConfig(kappa_sat, kappa_viol, reveal_value, lie_rate)
    policy = base_policy(wcfg, q, seed, d_model, n_layers, pretrain_steps, model, systematic)
    # Arms diverge only after this point; give each the same downstream randomness too.
    gen = torch.Generator().manual_seed(10_000 + seed)
    eval_gen = torch.Generator().manual_seed(20_000 + seed)
    learner = build(arm, policy, wcfg, tcfg, gen, overrides or {})
    curve = Curve()
    curve.log(learner.res, evaluate(policy, wcfg, n_eval, eval_gen), round=0)
    t0, rnd, next_eval = time.time(), 0, eval_every
    while learner.res.env_episodes < budget:
        info = learner.round()
        rnd += 1
        if learner.res.env_episodes >= next_eval or learner.res.env_episodes >= budget:
            curve.log(learner.res, evaluate(policy, wcfg, n_eval, eval_gen), round=rnd, **info)
            next_eval += eval_every
    result = {
        "arm": arm,
        "config": dict(m=m, n_slots=n_slots, n_values=n_values, q=q, systematic=systematic, scramble=scramble,
                       seed=seed, budget=budget,
                       kappa_sat=kappa_sat, kappa_viol=kappa_viol, reveal_value=reveal_value, lie_rate=lie_rate,
                       model=model, d_model=d_model, n_layers=n_layers, overrides=overrides or {}),
        "curve": curve.points,
        "seconds": time.time() - t0,
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n-values", type=int, default=4)
    ap.add_argument("--systematic", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--kappa-sat", type=float, default=0.5)
    ap.add_argument("--kappa-viol", type=float, default=0.5)
    ap.add_argument("--hide-value", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="learner overrides, key=value")
    ap.add_argument("--out")
    a = ap.parse_args()
    overrides = {k: type_cast(v) for k, v in (s.split("=") for s in a.set)}
    r = run(a.arm, m=a.m, n_values=a.n_values, systematic=a.systematic, q=a.q, seed=a.seed, budget=a.budget, kappa_sat=a.kappa_sat, kappa_viol=a.kappa_viol,
            reveal_value=not a.hide_value, overrides=overrides, out=a.out)
    for p in r["curve"]:
        print(f"ep={p['env_episodes']:6d} succ={p['success_sample']:.3f} greedy={p['success_greedy']:.3f} "
              f"frac={p['frac_sample']:.3f}")
    print(f"{r['seconds']:.1f}s")


def type_cast(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


if __name__ == "__main__":
    main()
