"""Checks of the results in GFT_revised_2026-10-20.pdf (section / equation numbers of that version)."""

import numpy as np
import pytest

from gft.algos.gft_exact import bound_failure, feedback_marginal, run_gft_exact
from gft.envs.bandit import random_audit_instance, random_channel_instance, section7_nonmonotone
from gft.metrics import kl, tv
from gft.theory import (approximate_posterior_bias, batch_gft_tabular, conditioning_variance_identity,
                         d3_counterexample, damped_run, fcp_fixed_message_closed_form, fcp_population,
                         finite_difference, gft_operator, gradient_forms, J_value, linear_softmax_batch_gft,
                         online_gft, posterior, rejection_sample, sdpo_reverse_kl_minimizer, sdpo_two_action_step,
                         separation_gamma, sigmoid, two_action_channel)


def audit(seed, n_opt=None):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 4))
    return random_audit_instance(rng, m=m, n_opt=n_opt or int(rng.integers(1, 4)), n_bad=int(rng.integers(1, 8)),
                                 kappa=rng.uniform(0.2, 1.0, size=m))


# ------------------------------------------------------------------ Prop. 6.4 and Section 6.3

@pytest.mark.parametrize("seed", range(100))
def test_prop64_monotone_success_under_common_optimal_feedback(seed):
    inst = audit(seed, n_opt=int(np.random.default_rng(seed).integers(2, 4)))
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=60)
    assert np.all(np.diff(run.success) >= -1e-12)


def test_section63_success_can_fall_without_common_optimal_feedback():
    inst = section7_nonmonotone()
    assert not np.allclose(inst.F[0], inst.F[1])          # optimal executions have different feedback laws
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=1)
    assert run.success[1] < run.success[0]


def test_section101_conditioning_on_fB_drops_success_to_5_67_percent():
    inst = section7_nonmonotone()
    r = posterior(inst.p0, inst.F, 1)                      # condition on f_B
    assert r[inst.optimal].sum() == pytest.approx(0.0567, abs=5e-5)


@pytest.mark.parametrize("seed", range(50))
def test_eq47_conditioning_raises_the_conditioned_message(seed):
    inst = random_channel_instance(np.random.default_rng(seed), n_x=7, n_f=5, n_opt=2)
    for f in range(5):
        lhs, rhs = conditioning_variance_identity(inst.p0, inst.F, f)
        assert lhs == pytest.approx(rhs, abs=1e-12)
        assert lhs >= feedback_marginal(inst.p0, inst.F)[f] - 1e-12


# ------------------------------------------------------------------ Prop. 3.1 relabeling

@pytest.mark.parametrize("seed", range(20))
def test_prop31_feedback_relabeling_invariance(seed):
    rng = np.random.default_rng(seed)
    inst = random_channel_instance(rng, n_x=6, n_f=5, n_opt=2)
    h = rng.permutation(5)
    F2, g2 = inst.F[:, h], inst.g_star[h]
    np.testing.assert_allclose(gft_operator(inst.p0, inst.F, inst.g_star), gft_operator(inst.p0, F2, g2), atol=1e-12)
    assert kl(inst.g_star, feedback_marginal(inst.p0, inst.F)) == pytest.approx(kl(g2, feedback_marginal(inst.p0, F2)))


# ------------------------------------------------------------------ Thm 6.3 with an LP-computed gamma

@pytest.mark.parametrize("seed", range(50))
def test_thm63_behavioral_bound_with_best_separating_statistic(seed):
    inst = audit(seed)
    gamma, s = separation_gamma(inst.F, inst.optimal)
    assert gamma >= inst.gamma - 1e-9                       # LP finds at least the paper's kappa_min / m
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=200, pbar=inst.pbar)
    assert np.all(1 - run.success <= bound_failure(run.v[0], gamma, np.arange(201)) + 1e-9)


def test_aliasing_example_has_no_separating_statistic():
    from gft.envs.bandit import mixture_aliasing
    inst = mixture_aliasing()
    gamma, _ = separation_gamma(inst.F, inst.optimal)
    assert gamma <= 1e-9


# ------------------------------------------------------------------ Prop. C.1 damped operator

@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("eta", [0.2, 0.5, 1.0])
def test_propC1_damped_update_rates(seed, eta):
    inst = audit(seed)
    T = 100
    ps = damped_run(inst.p0, inst.F, inst.g_star, eta, T)
    phi = np.array([kl(inst.g_star, feedback_marginal(p, inst.F)) for p in ps])
    V = np.array([kl(inst.pbar, p) for p in ps])
    assert np.all(np.diff(phi) <= 1e-12)
    assert np.all(V[:-1] - V[1:] >= eta * phi[:-1] - 1e-10)
    ts = np.arange(T + 1)
    assert np.all(phi <= V[0] / (eta * (ts + 1)) + 1e-10)
    fail = 1 - ps[:, inst.optimal].sum(1)
    assert np.all(fail <= np.minimum(1, np.sqrt(V[0] / (2 * eta * (ts + 1))) / inst.gamma) + 1e-10)


# ------------------------------------------------------------------ Eq. 54 approximate calibration

@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("eps", [0.05, 0.2])
def test_eq54_approximate_calibration_bound(seed, eps):
    inst = audit(seed)
    bad = (~inst.optimal) / (~inst.optimal).sum()
    p_tilde = (1 - eps) * inst.pbar + eps * bad
    g_tilde = p_tilde @ inst.F
    eps_G = tv(g_tilde, inst.g_star)
    run = run_gft_exact(inst.p0, inst.F, g_tilde, inst.optimal, T=200)
    ts = np.arange(201)
    bound = np.minimum(1, (eps_G + np.sqrt(kl(p_tilde, inst.p0) / (2 * (ts + 1)))) / inst.gamma)
    assert np.all(1 - run.success <= bound + 1e-9)


# ------------------------------------------------------------------ Prop. 10.1 and Eq. 51 (FCP)

@pytest.mark.parametrize("seed", range(30))
def test_prop101_fcp_population_equals_gft(seed):
    inst = random_channel_instance(np.random.default_rng(seed), n_x=8, n_f=6, n_opt=2)
    gft = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=25).p
    np.testing.assert_allclose(fcp_population(inst.p0, inst.F, inst.g_star, 25), gft, atol=1e-10)


def test_eq51_fixed_message_concentrates_on_max_likelihood_not_success():
    F = np.array([[0.6, 0.4],    # optimal
                  [0.9, 0.1]])   # suboptimal, but more likely to elicit f+ = 0
    rho0 = np.array([0.5, 0.5])
    rhos = fcp_population(rho0, F, np.array([1.0, 0.0]), 60)
    for t in (1, 10, 60):
        np.testing.assert_allclose(rhos[t], fcp_fixed_message_closed_form(rho0, F, 0, t), atol=1e-12)
    assert rhos[-1][1] > 0.999


# ------------------------------------------------------------------ Prop. 11.1 (SDPO counterexample)

def test_prop111_sdpo_numbers_and_closed_form():
    assert sdpo_two_action_step(0.2) == pytest.approx(0.0801, abs=1e-4)
    F, g, opt = two_action_channel()
    assert gft_operator(np.array([0.2, 0.8]), F, g)[0] == pytest.approx(0.6258, abs=1e-4)
    for u in (0.1, 0.2, 0.4, 0.7):
        assert sdpo_reverse_kl_minimizer(u) == pytest.approx(sdpo_two_action_step(u), abs=1e-4)


@pytest.mark.parametrize("u0", [0.05, 0.2, 0.45])
def test_prop111_sdpo_converges_to_bad_action_gft_to_good(u0):
    u, p = u0, np.array([u0, 1 - u0])
    F, g, _ = two_action_channel()
    for _ in range(2000):
        u = sdpo_two_action_step(u)
        p = gft_operator(p, F, g)
    assert u < 1e-6 and p[0] > 0.99                        # GFT converges sublinearly (Thm 6.3), SDPO to 0


# ------------------------------------------------------------------ Prop. 9.1 / 9.2 (gradients)

@pytest.mark.parametrize("seed", range(30))
def test_prop91_gradient_identities(seed):
    rng = np.random.default_rng(seed)
    inst = random_channel_instance(rng, n_x=6, n_f=4, n_opt=2)
    theta = rng.normal(size=6)
    g44, g45, g_sur = gradient_forms(theta, inst.F, inst.g_star)
    fd = finite_difference(lambda th: J_value(th, inst.F, inst.g_star), theta)
    for g_ in (g44, g45, g_sur):
        np.testing.assert_allclose(g_, fd, atol=1e-6)


@pytest.mark.parametrize("seed", range(30))
def test_prop92_gradient_bias_bound(seed):
    rng = np.random.default_rng(seed)
    inst = random_channel_instance(rng, n_x=6, n_f=4, n_opt=2)
    theta = rng.normal(size=6)
    p = np.exp(theta) / np.exp(theta).sum()
    lam = rng.uniform(0, 1)
    r_phi = np.stack([(1 - lam) * posterior(p, inst.F, f) + lam * rng.dirichlet(np.ones(6)) for f in range(4)])
    lhs, rhs = approximate_posterior_bias(theta, inst.F, inst.g_star, r_phi)
    assert lhs <= rhs + 1e-12


# ------------------------------------------------------------------ Appendix F rejection sampling

def test_appendixF_rejection_sampling_is_exact_with_inverse_marginal_cost():
    rng = np.random.default_rng(0)
    inst = random_channel_instance(rng, n_x=5, n_f=3, n_opt=2)
    xs, trials = rejection_sample(rng, inst.p0, inst.F, 1, 20000)
    emp = np.bincount(xs, minlength=5) / len(xs)
    assert tv(emp, posterior(inst.p0, inst.F, 1)) < 0.02
    assert trials.mean() == pytest.approx(1 / feedback_marginal(inst.p0, inst.F)[1], rel=0.05)


# ------------------------------------------------------------------ Thm 8.1 finite-tabular Batch GFT

def test_thm81_bounds_hold_with_growing_fresh_batches():
    """Eqs. 34-35: Phi_T <= (v + 7/4)/(T+1) and p_T(C^c) <= min(1, gamma^-1 sqrt((v + 7/4)/(2(T+1)))) for all T.
    The batch sizes here are far below the theorem's sufficient schedule (Eqs. 32-33)."""
    inst = random_audit_instance(np.random.default_rng(5), m=2, n_opt=2, n_bad=5, kappa=1.0, opt_mass=0.05)
    v = kl(inst.pbar, inst.p0)
    grow = lambda t: 500 * (t + 1) ** 2
    T = 25
    ts = np.arange(T + 1)
    for s in range(20):
        ps = batch_gft_tabular(np.random.default_rng(s), inst.p0, inst.F, inst.g_star, T, grow, grow)
        phi = np.array([kl(inst.g_star, feedback_marginal(p, inst.F)) for p in ps])
        assert np.all(phi <= (v + 7 / 4) / (ts + 1) + 1e-12)
        fail = 1 - ps[:, inst.optimal].sum(1)
        assert np.all(fail <= np.minimum(1, np.sqrt((v + 7 / 4) / (2 * (ts + 1))) / inst.gamma) + 1e-12)


def test_thm81_small_fixed_batches_can_lose_optimal_support():
    inst = random_audit_instance(np.random.default_rng(3), m=3, n_opt=1, n_bad=6, kappa=0.5, opt_mass=0.02)
    lost = 0
    for s in range(40):
        ps = batch_gft_tabular(np.random.default_rng(s), inst.p0, inst.F, inst.g_star, 40, lambda t: 20, lambda t: 20)
        lost += ps[-1][inst.optimal].sum() == 0
    assert lost > 0


# ------------------------------------------------------------------ Thm 8.2 linear-softmax Batch GFT

def test_thm82_linear_softmax_reaches_constrained_stationarity():
    grow = lambda t: 200 * (t + 1) ** 2
    thetas, phis, resid = linear_softmax_batch_gft(np.random.default_rng(0), -2.0, 25, grow, grow, bound=6.0)
    assert resid[-1] < 1e-2 and thetas[-1] > thetas[0] + 5
    # Stage I recovers the realizable posterior: phi = (theta_t, log 9) up to sampling error.
    assert phis[:, 1].mean() == pytest.approx(np.log(9), abs=0.15)
    assert np.abs(phis[:, 0] - thetas[:-1]).mean() < 0.3


# ------------------------------------------------------------------ Thm D.1 Online GFT and Section D.3

def test_thmD1_online_gft_tracks_posterior_and_gradient_vanishes():
    """Informative full-support channel; G* generated by an interior mixture, so J has an interior maximizer
    and the actor stays in a compact region (assumption A1)."""
    F = 0.8 * np.eye(4) + 0.05
    g = np.array([0.5, 0.3, 0.15, 0.05]) @ F
    theta0 = np.log(np.array([0.05, 0.15, 0.3, 0.5]))
    _, _, log = online_gft(np.random.default_rng(2), F, g, np.array([True, False, False, False]), T=100_000,
                           beta0=3.0, alpha0=3.0, log_every=10_000, theta0=theta0)
    track, gJ = log[:, 1], log[:, 2]
    assert track[-1] < 0.05 * track[0] and gJ[-1] < 0.1 * gJ[0]


def test_sectionD3_restricted_actor_stationary_point_is_not_optimal():
    dJ, success = d3_counterexample(0.0)
    assert dJ == 0.0 and success == pytest.approx(0.5)
