"""Tier 0: exact-population GLFT experiments.

E1  Section 7 non-monotone example: trajectory of p_t(C), Phi_t and bound (18).
E2  Mixture aliasing: feedback matched at round 0, success stuck.
E3  Bound tightness on random audit instances: observed vs (13), (25); empirical decay rate.
E4  Rounds to success vs (kappa, m), compared with the gamma = kappa/m scaling of the bound.
E5  Comparators (vanilla PG, natural PG, filtered BC) vs GLFT: success and diversity over C.

Usage: uv run python experiments/tier0/run_tier0.py [--out outputs/tier0]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glft.algos.glft_exact import bound_failure, bound_phi, glft_step, run_glft_exact
from glft.algos.pg_exact import run_filtered_bc, run_natural_pg, run_softmax_pg
from glft.envs.bandit import audit_instance, mixture_aliasing, random_audit_instance, section7_nonmonotone
from glft.feedback.audit_teacher import all_positive_accept
from glft.metrics import normalized_entropy_on


def rounds_to(success, threshold):
    hit = np.nonzero(success >= threshold)[0]
    return int(hit[0]) if hit.size else np.inf


def e1_section7(out):
    inst = section7_nonmonotone()
    T = 500
    run = run_glft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=T, pbar=inst.pbar)
    ts = np.arange(T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].semilogx(ts + 1, run.success, label="$p_t(\\mathcal{C})$")
    ax[0].semilogx(ts + 1, 1 - bound_failure(run.v[0], inst.gamma, ts), "--", label="1 - bound (18)")
    ax[0].set_xlabel("round t + 1")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend()
    ax[0].set_title("Section 7: success")
    ax[1].loglog(ts + 1, run.phi, label="$\\Phi_t$")
    ax[1].loglog(ts + 1, bound_phi(run.v[0], ts), "--", label="bound (13)")
    ax[1].set_xlabel("round t + 1")
    ax[1].legend()
    ax[1].set_title("Section 7: feedback KL")
    fig.tight_layout()
    fig.savefig(out / "e1_section7.png", dpi=150)
    plt.close(fig)
    return {
        "success_t0_t1_t10_t100": [float(run.success[t]) for t in (0, 1, 10, 100)],
        "min_success": float(run.success.min()),
        "argmin_success": int(run.success.argmin()),
        "rounds_to_recover_0.9": int(run.success.argmin()) + rounds_to(run.success[run.success.argmin():], 0.9),
        "phi_t0_t1": [float(run.phi[0]), float(run.phi[1])],
    }


def e2_aliasing(out):
    res = {}
    for p0 in [(0.1, 0.45, 0.45), (0.1, 0.6, 0.3), (0.1, 0.8, 0.1)]:
        inst = mixture_aliasing(p0)
        run = run_glft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=2000, pbar=inst.pbar)
        res[str(p0)] = {"success_final": float(run.success[-1]), "phi_final": float(run.phi[-1])}
    return res


def e3_tightness(out, n=500, T=2000, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(T + 1)
    checkpoints = [10, 100, 1000]
    ratio_fail = {c: [] for c in checkpoints}
    ratio_phi = {c: [] for c in checkpoints}
    slopes = []
    for _ in range(n):
        m = int(rng.integers(1, 5))
        inst = random_audit_instance(rng, m=m, n_opt=int(rng.integers(1, 4)), n_bad=int(rng.integers(1, 10)),
                                     kappa=rng.uniform(0.1, 1.0, size=m))
        run = run_glft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=T, pbar=inst.pbar)
        fail = 1 - run.success
        bf = bound_failure(run.v[0], inst.gamma, ts)
        bp = bound_phi(run.v[0], ts)
        for c in checkpoints:
            ratio_fail[c].append(fail[c] / bf[c])
            ratio_phi[c].append(run.phi[c] / bp[c])
        # Geometric rate: slope of log failure while it is still above float precision.
        live = np.nonzero(fail > 1e-12)[0]
        if live.size >= 5:
            slopes.append(float(np.polyfit(ts[live], np.log(fail[live]), 1)[0]))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for c in checkpoints:
        ax.hist(np.log10(np.clip(ratio_fail[c], 1e-300, None)), bins=50, alpha=0.5, label=f"T={c}")
    ax.set_xlabel("log10( observed failure / bound (25) )")
    ax.set_ylabel("instances")
    ax.legend()
    ax.set_title("Tightness of the optimality bound")
    fig.tight_layout()
    fig.savefig(out / "e3_tightness.png", dpi=150)
    plt.close(fig)
    q = lambda a: [float(np.quantile(a, x)) for x in (0.5, 0.9, 1.0)]
    return {
        "failure_over_bound25_quantiles_50_90_max": {c: q(ratio_fail[c]) for c in checkpoints},
        "phi_over_bound13_quantiles_50_90_max": {c: q(ratio_phi[c]) for c in checkpoints},
        "tail_log_failure_slope_quantiles_50_90_max": q(slopes) if slopes else None,
        "n_instances": n,
    }


def full_pattern_instance(m, kappa, opt_mass):
    """All 2^m satisfaction patterns, one execution each; p0(C) = opt_mass, rest uniform."""
    R = np.array([[(i >> j) & 1 for j in range(m)] for i in range(2 ** m)])
    opt = R.all(axis=1)
    p0 = np.where(opt, opt_mass, (1 - opt_mass) / (~opt).sum())
    return audit_instance(R, np.full(m, kappa), p0)


def e4_kappa_m(out, opt_mass=0.05, T=20000):
    """Bound (25) predicts rounds ~ (m/kappa)^2 log(1/p0(C)); compare with the observed scaling."""
    ms = [1, 2, 3, 4, 5, 6]
    kappas = [0.1, 0.2, 0.4, 0.7, 1.0]
    table = {}
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for m in ms:
        rounds = []
        for k in kappas:
            inst = full_pattern_instance(m, k, opt_mass)
            p, t = inst.p0, 0
            while p[inst.optimal].sum() < 0.9 and t < T:
                p, t = glft_step(p, inst.F, inst.g_star)[0], t + 1
            rounds.append(t)
        table[m] = dict(zip(map(str, kappas), rounds))
        ax.loglog(np.array(kappas) / m, rounds, "o-", label=f"m={m}")
    ax.set_xlabel("$\\gamma = \\kappa / m$")
    ax.set_ylabel("rounds to $p_t(\\mathcal{C}) \\geq 0.9$")
    ax.legend(fontsize=7)
    ax.set_title(f"Rounds vs separation gap, $p_0(\\mathcal{{C}})$={opt_mass}")
    fig.tight_layout()
    fig.savefig(out / "e4_kappa_m.png", dpi=150)
    plt.close(fig)
    return {"rounds_to_0.9": table, "opt_mass": opt_mass}


def e5_comparators(out, T=300):
    # m = 3 requirements, 3 optimal executions with unequal initial mass, plus all 7 bad patterns.
    m = 3
    kappa = np.full(m, 0.4)
    bad = np.array([[(i >> j) & 1 for j in range(m)] for i in range(2 ** m - 1)])
    R = np.vstack([np.ones((3, m), dtype=int), bad])
    p0 = np.concatenate([[0.06, 0.03, 0.01], np.full(len(bad), 0.9 / len(bad))])
    inst = audit_instance(R, kappa, p0)
    r_bin = inst.optimal.astype(float)
    r_audit = inst.F @ inst.s  # E[s(F) | x]: the information-matched scalar reward
    curves = {
        "GLFT": run_glft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=T).p,
        "filtered BC (partial audit)": run_filtered_bc(p0, all_positive_accept(R, kappa), T),
        "PG, R-bin, lr=10": run_softmax_pg(p0, r_bin, 10.0, T),
        "PG, R-audit, lr=10": run_softmax_pg(p0, r_audit, 10.0, T),
        "NPG, R-bin, lr=0.5": run_natural_pg(p0, r_bin, 0.5, T),
        "NPG, R-audit, lr=2": run_natural_pg(p0, r_audit, 2.0, T),
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    res = {}
    for name, ps in curves.items():
        succ = ps[:, inst.optimal].sum(axis=1)
        ent = np.array([normalized_entropy_on(p, inst.optimal) for p in ps])
        ax[0].semilogx(np.arange(T + 1) + 1, succ, label=name)
        ax[1].semilogx(np.arange(T + 1) + 1, ent, label=name)
        res[name] = {
            "rounds_to_0.9": rounds_to(succ, 0.9),
            "rounds_to_0.99": rounds_to(succ, 0.99),
            "final_opt_entropy": float(ent[-1]),
        }
    ax[0].set_title("success $p_t(\\mathcal{C})$")
    ax[1].set_title("normalized entropy over $\\mathcal{C}$")
    for a in ax:
        a.set_xlabel("round t + 1")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "e5_comparators.png", dpi=150)
    plt.close(fig)
    res["_initial_opt_entropy"] = float(normalized_entropy_on(p0, inst.optimal))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tier0")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "E1_section7": e1_section7(out),
        "E2_aliasing": e2_aliasing(out),
        "E3_tightness": e3_tightness(out),
        "E4_kappa_m": e4_kappa_m(out),
        "E5_comparators": e5_comparators(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
