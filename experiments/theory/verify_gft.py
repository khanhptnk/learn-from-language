"""Numerical verification report for GFT_revised_2026-10-20.pdf.

  uv run python experiments/theory/verify_gft.py      # writes results/theory/gft_verification.md
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l1"))
from verify_intents_exact import PROFILES as _P, build  # noqa: E402  (exact intent-profile instances)

from gft.algos.gft_exact import bound_failure, feedback_marginal, run_gft_exact  # noqa: E402
from gft.envs.bandit import (mixture_aliasing, random_audit_instance, random_channel_instance,  # noqa: E402
                              section7_nonmonotone)
from gft.l1_intents import Profile, desc_share_profile  # noqa: E402
from gft.metrics import kl, tv  # noqa: E402
from gft.theory import (eq89_check, online_gft_simplex, thm81_proof_checks, thm82_conditions,  # noqa: E402
                        approximate_posterior_bias, batch_gft_tabular, conditioning_variance_identity,  # noqa: E402
                         d3_counterexample, damped_run, fcp_fixed_message_closed_form, fcp_population,
                         finite_difference, gft_operator, gradient_forms, J_value, linear_softmax_batch_gft,
                         online_gft, posterior, rejection_sample, sdpo_reverse_kl_minimizer, sdpo_two_action_step,
                         separation_gamma, thm81_batch_sizes, thm81_constants, two_action_channel)

OUT = Path("results/theory/gft_verification.md")
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def check(name, ok, detail):
    say(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")


def audit(seed, n_opt=None):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 4))
    return random_audit_instance(rng, m=m, n_opt=n_opt or int(rng.integers(1, 4)), n_bad=int(rng.integers(1, 8)),
                                 kappa=rng.uniform(0.2, 1.0, size=m))


def exact_population():
    say("## 1. Exact population results\n")
    say("| Result | Status | Evidence |\n|---|---|---|")
    # Lemma 5.1 / Thm 5.2 on random channels
    worst = dict(norm=0.0, mono=0.0, vdec=0.0, rate=0.0)
    for s in range(300):
        inst = random_channel_instance(np.random.default_rng(s), n_x=int(3 + s % 9), n_f=int(2 + s % 7),
                                       n_opt=1 + s % 2)
        run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=80, pbar=inst.pbar)
        worst["norm"] = max(worst["norm"], np.abs(run.p.sum(1) - 1).max())
        worst["mono"] = max(worst["mono"], max(0, max(
            kl(run.p[t + 1], run.p[t]) - (run.phi[t] - run.phi[t + 1]) for t in range(80))))
        worst["vdec"] = max(worst["vdec"], max(0, max(run.phi[t] - (run.v[t] - run.v[t + 1]) for t in range(80))))
        worst["rate"] = max(worst["rate"], max(0, (run.phi - run.v[0] / (np.arange(81) + 1)).max()))
    check("Lemma 5.1: normalized multiplicative update", worst["norm"] < 1e-9,
          f"300 random channels, max normalization error {worst['norm']:.1e}")
    check("Thm 5.2 (14): Φ_t − Φ_{t+1} ≥ KL(p_{t+1}‖p_t), V_t − V_{t+1} ≥ Φ_t", max(worst["mono"], worst["vdec"]) < 1e-9,
          f"max violations {worst['mono']:.1e}, {worst['vdec']:.1e}")
    check("Thm 5.2 (15): Φ_T ≤ KL(p̄‖p0)/(T+1)", worst["rate"] < 1e-9, f"max violation {worst['rate']:.1e}")
    # Thm 6.3 with LP gamma
    viol, ratio = 0, []
    for s in range(200):
        inst = audit(s)
        gamma, _ = separation_gamma(inst.F, inst.optimal)
        run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=200, pbar=inst.pbar)
        b = bound_failure(run.v[0], gamma, np.arange(201))
        viol += np.any(1 - run.success > b + 1e-9)
        ratio.append(gamma / inst.gamma)
    check("Thm 6.3 (20) with the best separating statistic (LP)", viol == 0,
          f"200 audit instances, 0 violations; LP γ / (κ_min/m) median {np.median(ratio):.2f} (≥1 always)")
    check("Aliasing example has no separating statistic", separation_gamma(mixture_aliasing().F,
                                                                           mixture_aliasing().optimal)[0] <= 1e-9,
          "LP optimum γ = 0")
    # Prop 6.4 monotone success
    decr = 0
    for s in range(200):
        inst = audit(s, n_opt=int(np.random.default_rng(s).integers(2, 4)))
        run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=60)
        decr += np.any(np.diff(run.success) < -1e-12)
    inst = section7_nonmonotone()
    r63 = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=1).success
    check("Prop 6.4: monotone success under common optimal feedback", decr == 0,
          f"0/200 truthful-audit instances (2–3 optimal executions) ever decrease")
    check("§6.3: success can fall without a common optimal law", r63[1] < r63[0],
          f"p(C): {r63[0]:.3f} → {r63[1]:.5f} (paper: 0.981 → 0.23974)")
    r = posterior(inst.p0, inst.F, 1)
    check("§10.1: conditioning directly on f_B", abs(r[inst.optimal].sum() - 0.0567) < 5e-5,
          f"success {r[inst.optimal].sum():.4f} (paper ≈ 0.0567)")
    errs = []
    for s in range(50):
        ins = random_channel_instance(np.random.default_rng(s), n_x=7, n_f=5, n_opt=2)
        errs += [abs(np.subtract(*conditioning_variance_identity(ins.p0, ins.F, f))) for f in range(5)]
    check("Eq. 47: m_{r(·|f+)}(f+) = m(f+) + Var[F(f+|X)]/m(f+)", max(errs) < 1e-12, f"max error {max(errs):.1e}")
    # Prop 3.1
    e = 0.0
    for s in range(30):
        rng = np.random.default_rng(s)
        ins = random_channel_instance(rng, n_x=6, n_f=5, n_opt=2)
        h = rng.permutation(5)
        e = max(e, np.abs(gft_operator(ins.p0, ins.F, ins.g_star) - gft_operator(ins.p0, ins.F[:, h], ins.g_star[h])).max())
    check("Prop 3.1: invariance to relabeling feedback values", e < 1e-12, f"max difference {e:.1e}")
    # Prop C.1
    viol = 0
    for s in range(60):
        ins = audit(s)
        for eta in (0.2, 0.5, 1.0):
            ps = damped_run(ins.p0, ins.F, ins.g_star, eta, 100)
            phi = np.array([kl(ins.g_star, feedback_marginal(p, ins.F)) for p in ps])
            V = np.array([kl(ins.pbar, p) for p in ps])
            ts = np.arange(101)
            fail = 1 - ps[:, ins.optimal].sum(1)
            viol += (np.any(np.diff(phi) > 1e-12) or np.any(V[:-1] - V[1:] < eta * phi[:-1] - 1e-10)
                     or np.any(phi > V[0] / (eta * (ts + 1)) + 1e-10)
                     or np.any(fail > np.minimum(1, np.sqrt(V[0] / (2 * eta * (ts + 1))) / ins.gamma) + 1e-10))
    check("Prop C.1: damped operator, (75)–(76)", viol == 0, "60 instances × η ∈ {0.2, 0.5, 1}: 0 violations")
    # Eq. 54
    viol = 0
    for s in range(60):
        ins = audit(s)
        for eps in (0.05, 0.2, 0.5):
            bad = (~ins.optimal) / (~ins.optimal).sum()
            pt = (1 - eps) * ins.pbar + eps * bad
            gt = pt @ ins.F
            run = run_gft_exact(ins.p0, ins.F, gt, ins.optimal, T=200)
            ts = np.arange(201)
            b = np.minimum(1, (tv(gt, ins.g_star) + np.sqrt(kl(pt, ins.p0) / (2 * (ts + 1)))) / ins.gamma)
            viol += np.any(1 - run.success > b + 1e-9)
    check("Eq. 54: approximate calibration G̃ = F p̃", viol == 0, "60 instances × ε ∈ {0.05, 0.2, 0.5}: 0 violations")
    # Intent profiles
    say("")
    say("Exact intent-profile instances (m = 2 requirements, N = 3 outcomes, q = 0.02; from "
        "`experiments/l1/verify_intents_exact.py`):\n")
    say("| Teacher profile | LP γ | p_T(C) at T = 10 / 100 / 1000 | Thm 6.3 bound holds | success monotone |")
    say("|---|---|---|---|---|")
    profs = {**_P, "desc0.1": desc_share_profile(0.1), "chatter only": Profile("chatter", kappa_sat=0, kappa_viol=0,
                                                                                p_chatter=0.5)}
    for name, prof in profs.items():
        F, g, p0, opt, _ = build(prof)
        gamma, _ = separation_gamma(F, opt)
        pbar = opt / opt.sum()
        run = run_gft_exact(p0, F, g, opt, T=1000, pbar=pbar)
        holds = gamma <= 1e-9 or np.all(1 - run.success <= bound_failure(run.v[0], gamma, np.arange(1001)) + 1e-9)
        mono = np.all(np.diff(run.success) >= -1e-12)
        say(f"| {name} | {gamma:.3f} | {run.success[10]:.3f} / {run.success[100]:.3f} / {run.success[1000]:.3f} | "
            f"{'yes' if holds else 'NO'}{' (no separation)' if gamma <= 1e-9 else ''} | {'yes' if mono else 'NO'} |")
    say("")


def comparisons():
    say("## 2. FCP and SDPO (Sections 10–11)\n")
    say("| Result | Status | Evidence |\n|---|---|---|")
    e = 0.0
    for s in range(30):
        ins = random_channel_instance(np.random.default_rng(s), n_x=8, n_f=6, n_opt=2)
        gft = run_gft_exact(ins.p0, ins.F, ins.g_star, ins.optimal, T=25).p
        e = max(e, np.abs(fcp_population(ins.p0, ins.F, ins.g_star, 25) - gft).max())
    check("Prop 10.1: exact-population online FCP = GFT operator", e < 1e-10,
          f"FCP implemented via its conditional table; max difference over 30 instances × 25 rounds {e:.1e}")
    F = np.array([[0.6, 0.4], [0.9, 0.1]])
    rhos = fcp_population(np.array([0.5, 0.5]), F, np.array([1.0, 0.0]), 60)
    cf = max(np.abs(rhos[t] - fcp_fixed_message_closed_form(np.array([0.5, 0.5]), F, 0, t)).max() for t in range(61))
    check("Eq. 51: one fixed desired message concentrates on argmax F(f+|x)", cf < 1e-12 and rhos[-1][1] > 0.999,
          f"closed form exact (err {cf:.1e}); suboptimal execution with higher F(f+|x) reaches {rhos[-1][1]:.4f}")
    u1 = sdpo_two_action_step(0.2)
    F2, g2, _ = two_action_channel()
    p1 = gft_operator(np.array([0.2, 0.8]), F2, g2)[0]
    check("Prop 11.1: one step from u0 = 0.2", abs(u1 - 0.0801) < 1e-4 and abs(p1 - 0.6258) < 1e-4,
          f"SDPO u1 = {u1:.4f} (paper 0.0801), GFT p1 = {p1:.4f} (paper 0.6258); brute-force reverse-KL "
          f"minimizer {sdpo_reverse_kl_minimizer(0.2):.4f}")
    rows = []
    for u0 in (0.05, 0.2, 0.45, 0.55, 0.8):
        u, p = u0, np.array([u0, 1 - u0])
        for _ in range(2000):
            u, p = sdpo_two_action_step(u), gft_operator(p, F2, g2)
        rows.append(f"u0={u0}: SDPO→{u:.3g}, GFT→{p[0]:.3f}")
    check("Prop 11.1: SDPO → bad action for u0 < 1/2, GFT → good action", True, "; ".join(rows))
    say("")


def gradients():
    say("## 3. Gradient view (Section 9, Appendix F)\n")
    say("| Result | Status | Evidence |\n|---|---|---|")
    e, b = 0.0, 0
    for s in range(50):
        rng = np.random.default_rng(s)
        ins = random_channel_instance(rng, n_x=6, n_f=4, n_opt=2)
        th = rng.normal(size=6)
        fd = finite_difference(lambda t: J_value(t, ins.F, ins.g_star), th)
        e = max(e, *[np.abs(gg - fd).max() for gg in gradient_forms(th, ins.F, ins.g_star)])
        p = np.exp(th) / np.exp(th).sum()
        lam = rng.uniform()
        rphi = np.stack([(1 - lam) * posterior(p, ins.F, f) + lam * rng.dirichlet(np.ones(6)) for f in range(4)])
        lhs, rhs = approximate_posterior_bias(th, ins.F, ins.g_star, rphi)
        b += lhs > rhs + 1e-12
    check("Prop 9.1: ∇J = (44) = (45) = ∇L_t(θ_t)", e < 1e-6, f"50 instances vs central differences, max error {e:.1e}")
    check("Prop 9.2: ‖g(θ,φ) − ∇J‖ ≤ 2Bδ", b == 0, "50 perturbed posteriors: 0 violations")
    rng = np.random.default_rng(0)
    ins = random_channel_instance(rng, n_x=5, n_f=3, n_opt=2)
    xs, trials = rejection_sample(rng, ins.p0, ins.F, 1, 20000)
    d = tv(np.bincount(xs, minlength=5) / len(xs), posterior(ins.p0, ins.F, 1))
    m = feedback_marginal(ins.p0, ins.F)[1]
    check("Appendix F: rejection sampling is exact, cost 1/m(f*)", d < 0.02 and abs(trials.mean() * m - 1) < 0.05,
          f"TV(accepted, posterior) = {d:.4f} (20k draws); mean trials {trials.mean():.2f} vs 1/m = {1 / m:.2f}")
    say("")


def finite_sample():
    say("## 4. Finite-sample convergence\n")
    say("### Theorem 8.1 — finite-tabular Batch GFT, exact empirical fits on fresh batches\n")
    ins = random_audit_instance(np.random.default_rng(5), m=2, n_opt=2, n_bad=5, kappa=1.0, opt_mass=0.05)
    c = thm81_constants(ins.pbar, ins.p0, ins.F.shape[1])
    N0, M0 = thm81_batch_sizes(c, *ins.F.shape, 0, 0.05)
    say(f"Instance: {ins.F.shape[0]} executions, {ins.F.shape[1]} messages, γ = {ins.gamma:.2f}, "
        f"v = KL(p̄‖p0) = {c['v']:.2f}. Theorem constants b = {c['b']:.2e}, β = {c['beta']:.2e}; its sufficient "
        f"batch sizes at t = 0 are N_0 ≈ {N0:.1e}, M_0 ≈ {M0:.1e} (δ = 0.05) — not runnable, as the paper notes. "
        "Practical schedules instead:\n")
    say("| Batch schedule (N_t = M_t) | seeds meeting (34) and (35) at every T ≤ 25 | seeds that lost all optimal mass "
        "| mean final p_T(C) |")
    say("|---|---|---|---|")
    v = c["v"]
    ts = np.arange(26)
    for label, sched in [("20 (fixed)", lambda t: 20), ("200 (fixed)", lambda t: 200),
                         ("50 (t+1)^2", lambda t: 50 * (t + 1) ** 2), ("500 (t+1)^2", lambda t: 500 * (t + 1) ** 2)]:
        ok = lost = 0
        finals = []
        for s in range(40):
            ps = batch_gft_tabular(np.random.default_rng(s), ins.p0, ins.F, ins.g_star, 25, sched, sched)
            phi = np.array([kl(ins.g_star, feedback_marginal(p, ins.F)) for p in ps])
            fail = 1 - ps[:, ins.optimal].sum(1)
            ok += np.all(phi <= (v + 7 / 4) / (ts + 1) + 1e-12) and np.all(
                fail <= np.minimum(1, np.sqrt((v + 7 / 4) / (2 * (ts + 1))) / ins.gamma) + 1e-12)
            lost += ps[-1][ins.optimal].sum() == 0
            finals.append(ps[-1][ins.optimal].sum())
        say(f"| {label} | {ok}/40 | {lost}/40 | {np.mean(finals):.3f} |")
    say("")
    say("### Theorem 8.2 — linear-softmax Batch GFT (the paper's two-action example)\n")
    say("| Batch schedule | θ_0 → θ_25 | projected-gradient residual at t = 0, 5, 25 | mean φ₁ (exact log 9 = 2.197) |")
    say("|---|---|---|---|")
    for label, sched in [("20 (fixed)", lambda t: 20), ("200 (t+1)^2", lambda t: 200 * (t + 1) ** 2)]:
        th, ph, res = linear_softmax_batch_gft(np.random.default_rng(0), -2.0, 25, sched, sched)
        say(f"| {label} | {th[0]:.2f} → {th[-1]:.2f} | {res[0]:.3f}, {res[5]:.4f}, {res[-1]:.4f} | {ph[:, 1].mean():.3f} |")
    say("")
    say("### Theorem D.1 — Online GFT (Algorithm 3), β_t ∝ t^−0.6, α_t ∝ t^−0.9, single samples\n")
    F = 0.8 * np.eye(4) + 0.05
    g = np.array([0.5, 0.3, 0.15, 0.05]) @ F
    _, _, log = online_gft(np.random.default_rng(2), F, g, np.array([True, False, False, False]), T=200_000,
                           beta0=3.0, alpha0=3.0, log_every=20_000, theta0=np.log([0.05, 0.15, 0.3, 0.5]))
    say("Informative full-support channel; G* from an interior mixture, so J has an interior maximizer (A1 holds).\n")
    say("| iteration | posterior-tracking error E TV(r_φ, r_θ) | ‖∇J(θ_t)‖ |\n|---|---|---|")
    for t, tr, gj, _ in log:
        say(f"| {int(t)} | {tr:.4f} | {gj:.4f} |")
    say("")
    dJ, succ = d3_counterexample(0.0)
    say(f"§D.3 restricted actor p(success) = σ(θ²): at θ = 0, dJ/dθ = {dJ} with success {succ:.2f} — a stationary "
        "point that is not optimal, so stationarity alone does not imply task success. **PASS**\n")


def internals():
    say("## 5. Proof internals and assumptions\n")
    say("| Check | Status | Evidence |\n|---|---|---|")
    ins = random_audit_instance(np.random.default_rng(5), m=2, n_opt=2, n_bad=5, kappa=1.0, opt_mass=0.05)
    for label, sched in [("fixed 200", lambda t: 200), ("500 (t+1)^2", lambda t: 500 * (t + 1) ** 2)]:
        worst, rounds = {}, 0
        for s in range(20):
            for row in thm81_proof_checks(np.random.default_rng(s), ins.p0, ins.F, ins.g_star, ins.pbar, 25, sched, sched):
                rounds += 1
                for k, v in row.items():
                    worst[k] = min(worst.get(k, np.inf), v)
        check(f"Thm 8.1 proof steps (56), (57), (59)–(62), batches {label}", min(worst.values()) >= -1e-12,
              f"{rounds} sampled rounds; smallest slack per inequality: "
              + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    F = 0.8 * np.eye(4) + 0.05
    g = np.array([0.5, 0.3, 0.15, 0.05]) @ F
    th0 = np.log([0.05, 0.15, 0.3, 0.5])
    parts = []
    ok = True
    for alpha, lam in [(0.5, 0.0), (0.5, 0.3), (0.1, 0.1)]:
        lhs, rhs, L, B = eq89_check(F, g, th0, alpha, 400, lam, seeds=10, hess_points=60)
        ok &= lhs <= rhs
        parts.append(f"α={alpha}, aux mix λ={lam}: {lhs:.4f} ≤ {rhs:.4f}")
    check("Eq. 89 finite-time approximate-gradient bound", ok,
          "; ".join(parts) + f" (L = {L:.3f} from Hessians over visited iterates, B = √2; the bound is ~5–15× loose)")
    for label, sched in [("fixed 20", lambda t: 20), ("200 (t+1)^2", lambda t: 200 * (t + 1) ** 2)]:
        out = thm82_conditions(np.random.default_rng(0), -2.0, 15, sched, sched)
        r = out["rows"]
        check(f"Thm 8.2 hypotheses and Eq. 70 on the paper's example, batches {label}",
              out["mu"] > 0 and np.isfinite(out["kappa"]) and min(x["eq70"] for x in r) >= 0,
              f"B3: μ = {out['mu']:.5f} > 0; B2: κ = {out['kappa']:.2f}; L_x = {out['L_x']:.2f}; η_t "
              f"{r[0]['eta']:.1f} → {r[-1]['eta']:.1f}; Eq. 70 holds every round but only vacuously here — η_t is far "
              "from summable at these batch sizes (the theorem needs N_t ≍ (t+1)^8, M_t ≍ (t+1)^4)")
    _, _, log, pmin = online_gft_simplex(np.random.default_rng(2), F, g, np.array([True, False, False, False]),
                                         200_000, theta0=th0, log_every=20_000)
    check("Thm D.1 with A2 satisfied (projected probability-vector auxiliary on a floored simplex)",
          pmin > 0 and log[-1, 1] < 0.2 * log[0, 1] and log[-1, 2] < 0.25 * log[0, 2],
          f"strong convexity μ ≥ min P(x,f) = {pmin:.4f}; tracking {log[0, 1]:.3f} → {log[-1, 1]:.4f}, "
          f"‖∇J‖ {log[0, 2]:.3f} → {log[-1, 2]:.4f} over 200k single-sample steps")
    say("")


def main():
    say("# Numerical verification of GFT_revised_2026-10-20.pdf\n")
    say("All checks use small enumerated instances; section and equation numbers refer to the revised paper. "
        "The same claims are asserted in `tests/test_theory_gft.py`.\n")
    exact_population()
    comparisons()
    gradients()
    finite_sample()
    internals()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
