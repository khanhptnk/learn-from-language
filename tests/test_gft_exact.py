import numpy as np
import pytest

from gft.algos.gft_exact import (
    bound_failure,
    bound_phi,
    bound_quality,
    feedback_marginal,
    gft_step,
    run_gft_exact,
)
from gft.envs.bandit import (
    audit_instance,
    mixture_aliasing,
    random_audit_instance,
    random_channel_instance,
    section7_nonmonotone,
)
from gft.metrics import kl

TOL = 1e-10


def test_section7_numbers():
    inst = section7_nonmonotone()
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=1, pbar=inst.pbar)
    assert run.success[0] == pytest.approx(0.981, abs=1e-12)
    np.testing.assert_allclose(run.p[1], [0.19416, 0.04559, 0.76026], atol=5e-6)
    assert run.success[1] == pytest.approx(0.23974, abs=5e-6)
    assert run.phi[0] == pytest.approx(3.22886, abs=5e-6)
    assert run.phi[1] == pytest.approx(0.20864, abs=5e-6)


def test_section7_eventually_optimal():
    inst = section7_nonmonotone()
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=2000, pbar=inst.pbar)
    assert run.success[-1] > 0.95
    assert np.all(1 - run.success <= bound_failure(run.v[0], inst.gamma, np.arange(2001)) + TOL)


def test_mixture_aliasing_is_a_bad_fixed_point():
    inst = mixture_aliasing()
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=50)
    assert run.phi[0] == pytest.approx(0.0)
    np.testing.assert_allclose(run.success, 0.1)


@pytest.mark.parametrize("seed", range(200))
def test_update_identities_and_feedback_bounds(seed):
    rng = np.random.default_rng(seed)
    inst = random_channel_instance(rng, n_x=int(rng.integers(3, 12)), n_f=int(rng.integers(2, 10)),
                                   n_opt=int(rng.integers(1, 3)))
    T = 60
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=T, pbar=inst.pbar)
    # Lemma 4.1: normalized and witness support preserved.
    np.testing.assert_allclose(run.p.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(run.p[:, inst.pbar > 0] > 0)
    # Theorem 4.2, Eq. (12).
    for t in range(T):
        assert run.phi[t] - run.phi[t + 1] >= kl(run.p[t + 1], run.p[t]) - 1e-9
        assert run.v[t] - run.v[t + 1] >= run.phi[t] - 1e-9
    # Eq. (13).
    assert np.all(run.phi <= bound_phi(run.v[0], np.arange(T + 1)) + 1e-9)


def test_gft_step_matches_bayes_posterior_mixture():
    rng = np.random.default_rng(0)
    inst = random_channel_instance(rng, n_x=6, n_f=5, n_opt=2)
    p = inst.p0
    m = feedback_marginal(p, inst.F)
    posterior = p[:, None] * inst.F / m[None, :]          # r(x | f)
    expected = posterior @ inst.g_star                      # sum_f G*(f) r(x|f)
    np.testing.assert_allclose(gft_step(p, inst.F, inst.g_star)[0], expected, atol=1e-12)


def test_audit_statistic_mean_matches_eq23():
    R = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 0, 0]])
    kappa = np.array([0.3, 0.6, 0.9])
    inst = audit_instance(R, kappa, p0=np.full(4, 0.25))
    np.testing.assert_allclose(inst.F.sum(axis=1), 1.0)
    np.testing.assert_allclose(inst.F @ inst.s, (R * kappa).sum(axis=1) / 3)
    # Calibration: G* equals the feedback law of an optimal execution.
    np.testing.assert_allclose(inst.g_star, inst.F[0])


@pytest.mark.parametrize("seed", range(200))
def test_audit_bounds_eq25_eq27(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 5))
    kappa = float(rng.uniform(0.1, 1.0))
    inst = random_audit_instance(rng, m=m, n_opt=int(rng.integers(1, 4)), n_bad=int(rng.integers(1, 8)),
                                 kappa=kappa)
    T = 200
    run = run_gft_exact(inst.p0, inst.F, inst.g_star, inst.optimal, T=T, pbar=inst.pbar)
    ts = np.arange(T + 1)
    assert run.v[0] == pytest.approx(-np.log(inst.p0[inst.optimal].sum()))
    assert np.all(1 - run.success <= bound_failure(run.v[0], inst.gamma, ts) + 1e-9)
    quality = run.p @ inst.R.mean(axis=1)
    assert np.all(1 - quality <= bound_quality(run.v[0], kappa, ts) + 1e-9)
