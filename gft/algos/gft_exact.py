"""Exact-population GFT on an enumerated execution space (paper Section 4).

Executions and feedback messages are indexed 0..n_x-1 and 0..n_f-1.
`F[x, f]` is the teacher channel F(f | x); `g_star[f]` is the ideal-feedback law G*.
"""

from dataclasses import dataclass, field

import numpy as np

from gft.metrics import kl, normalized_entropy_on


def feedback_marginal(p, F):
    """m_p(f) = sum_x p(x) F(f | x)   (Eq. 3)."""
    return p @ F


def gft_step(p, F, g_star):
    """One exact GFT round (Lemma 4.1).

    Stage I's population MLE is the Bayes posterior r(x | f) = p(x) F(f|x) / m(f);
    Stage II marginalizes it over f ~ G*, giving p_{t+1}(x) = p(x) W(x) with
    W(x) = sum_{f: G*(f)>0} F(f|x) G*(f) / m(f)   (Eq. 10).
    """
    m = feedback_marginal(p, F)
    on = g_star > 0
    if np.any(m[on] <= 0):
        raise ValueError("coverage violated: an ideal message has zero current mass")
    ratio = np.zeros_like(g_star, dtype=float)
    ratio[on] = g_star[on] / m[on]
    W = F @ ratio
    return p * W, m, W


@dataclass
class ExactRun:
    p: np.ndarray            # (T+1, n_x) execution distributions p_0..p_T
    phi: np.ndarray          # (T+1,) Phi_t = KL(G* || m_t)
    v: np.ndarray            # (T+1,) V_t = KL(pbar || p_t), nan if no witness given
    success: np.ndarray      # (T+1,) p_t(C)
    opt_entropy: np.ndarray  # (T+1,) normalized entropy of p_t restricted to C
    extra: dict = field(default_factory=dict)


def run_gft_exact(p0, F, g_star, optimal, T, pbar=None):
    """Iterate exact GFT for T rounds and record the paper's potentials."""
    ps = [np.asarray(p0, dtype=float)]
    for _ in range(T):
        ps.append(gft_step(ps[-1], F, g_star)[0])
    ps = np.stack(ps)
    phi = np.array([kl(g_star, feedback_marginal(p, F)) for p in ps])
    v = np.array([kl(pbar, p) for p in ps]) if pbar is not None else np.full(T + 1, np.nan)
    success = ps[:, optimal].sum(axis=1)
    ent = np.array([normalized_entropy_on(p, optimal) for p in ps])
    return ExactRun(ps, phi, v, success, ent)


# Bounds from the paper, as functions of V_0 = KL(pbar || p_0) and the round index T.

def bound_phi(v0, T):
    """Eq. (13): KL(G* || m_T) <= V_0 / (T+1)."""
    return v0 / (np.asarray(T) + 1.0)


def bound_failure(v0, gamma, T):
    """Eq. (18): p_T(C^c) <= min(1, gamma^-1 sqrt(V_0 / (2(T+1))))."""
    return np.minimum(1.0, np.sqrt(v0 / (2.0 * (np.asarray(T) + 1.0))) / gamma)


def bound_quality(v0, kappa, T):
    """Eq. (27): 1 - E_{p_T} R <= kappa^-1 sqrt(V_0 / (2(T+1))), equal inclusion kappa."""
    return np.minimum(1.0, np.sqrt(v0 / (2.0 * (np.asarray(T) + 1.0))) / kappa)


def exact_slot_recursion(a0, kappa_sat, kappa_viol, T):
    """Exact GFT on one requirement of a factorized execution law under the partial-audit teacher.

    With independent audits and a factorized p_t, the update (10) factorizes over requirements.
    For one requirement with success probability a, the ideal messages are "not reported"
    (prob 1 - kappa_sat) and "reported satisfied" (prob kappa_sat), so
        a' = a (1 - kappa_sat)^2 / m(nr) + kappa_sat,   m(nr) = (1 - kappa_sat) a + (1 - kappa_viol)(1 - a).
    Violated reports never appear under G*, so at the population level revealing *which* wrong
    outcome occurred does not change the update; any benefit of descriptive feedback comes from
    generalization across requests. Returns a_0..a_T.
    """
    a = np.empty(T + 1)
    a[0] = a0
    for t in range(T):
        m_nr = (1 - kappa_sat) * a[t] + (1 - kappa_viol) * (1 - a[t])
        unreported = a[t] * (1 - kappa_sat) ** 2 / m_nr if kappa_sat < 1 else 0.0
        a[t + 1] = unreported + kappa_sat
    return a
