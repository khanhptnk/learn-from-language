"""Enumerated one-decision instances for exact-population experiments (Tier 0).

Each instance fixes a single request D; executions are the arms.
"""

from dataclasses import dataclass

import numpy as np

from gft.feedback.audit_teacher import audit_channel, ideal_feedback, subset_probs_independent


@dataclass
class Instance:
    name: str
    F: np.ndarray          # (n_x, n_f) teacher channel
    g_star: np.ndarray     # (n_f,) ideal-feedback law
    optimal: np.ndarray    # (n_x,) bool mask of C
    p0: np.ndarray         # (n_x,) initial execution distribution
    pbar: np.ndarray       # (n_x,) a calibrated optimal witness: g_star == pbar @ F
    s: np.ndarray | None = None      # (n_f,) optimality-separating statistic, if known
    gamma: float | None = None       # separation gap for s
    R: np.ndarray | None = None      # (n_x, m) requirement satisfaction, audit instances only
    kappa: np.ndarray | None = None  # (m,) inclusion probabilities, audit instances only


def section7_nonmonotone():
    """Paper Section 7: success falls from 98.1% to ~24.0% after one round while Phi decreases."""
    F = np.array([
        [0.8, 0.0, 0.2],   # x_A optimal
        [0.0, 0.8, 0.2],   # x_B optimal
        [0.0, 0.7, 0.3],   # x_C suboptimal
    ])
    pbar = np.array([0.0, 1.0, 0.0])
    return Instance(
        name="section7_nonmonotone",
        F=F,
        g_star=pbar @ F,
        optimal=np.array([True, True, False]),
        p0=np.array([0.98, 0.001, 0.019]),
        pbar=pbar,
        s=np.array([1.0, 1.0, 0.0]),
        gamma=0.1,
    )


def mixture_aliasing(p0=(0.1, 0.45, 0.45)):
    """Paper Section 7: a mixture of two bad executions reproduces the ideal feedback exactly.

    x_opt says "A" or "B" with equal probability; x_bad1 always "A"; x_bad2 always "B".
    No optimality-separating statistic exists, so feedback matching need not imply success.
    """
    F = np.array([
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    pbar = np.array([1.0, 0.0, 0.0])
    return Instance(
        name="mixture_aliasing",
        F=F,
        g_star=pbar @ F,
        optimal=np.array([True, False, False]),
        p0=np.asarray(p0, dtype=float),
        pbar=pbar,
    )


def best_witness(p0, optimal):
    """Witness minimizing KL(pbar || p0) over distributions on C: p0 restricted to C.

    Valid whenever all optimal executions share one feedback law (true for truthful audits),
    and then V_0 = -log p0(C).
    """
    w = np.where(optimal, p0, 0.0)
    return w / w.sum()


def audit_instance(R, kappa, p0, name="audit"):
    """One-decision instance with a truthful partial-audit teacher (paper Section 6)."""
    R = np.asarray(R, dtype=int)
    kappa = np.asarray(kappa, dtype=float)
    sp = subset_probs_independent(kappa)
    F, messages, s = audit_channel(R, sp)
    g_star = ideal_feedback(sp, messages)
    optimal = R.all(axis=1)
    m = R.shape[1]
    return Instance(
        name=name,
        F=F,
        g_star=g_star,
        optimal=optimal,
        p0=np.asarray(p0, dtype=float),
        pbar=best_witness(p0, optimal),
        s=s,
        gamma=float(kappa.min() / m),
        R=R,
        kappa=kappa,
    )


def random_audit_instance(rng, m, n_opt, n_bad, kappa, opt_mass=None, concentration=1.0):
    """Random audit instance: n_opt optimal executions and n_bad executions failing >= 1 requirement.

    If `opt_mass` is given, p0 is rescaled so that p0(C) = opt_mass (controls initial coverage).
    """
    kappa = np.broadcast_to(np.asarray(kappa, dtype=float), (m,))
    bad = []
    while len(bad) < n_bad:
        r = rng.integers(0, 2, size=m)
        if not r.all():
            bad.append(r)
    R = np.vstack([np.ones((n_opt, m), dtype=int)] + ([np.array(bad)] if bad else []))
    p0 = rng.dirichlet(np.full(n_opt + n_bad, concentration))
    if opt_mass is not None:
        opt = np.arange(n_opt + n_bad) < n_opt
        p0 = np.where(opt, p0 / p0[opt].sum() * opt_mass, p0 / p0[~opt].sum() * (1 - opt_mass))
    return audit_instance(R, kappa, p0, name=f"random_audit_m{m}")


def random_channel_instance(rng, n_x, n_f, n_opt, concentration=0.5):
    """Random channel with an arbitrary calibrated witness; no separation is guaranteed.

    Used to check the feedback-space results (Lemma 4.1, Theorem 4.2), which do not need it.
    """
    F = rng.dirichlet(np.full(n_f, concentration), size=n_x)
    optimal = np.arange(n_x) < n_opt
    pbar = np.zeros(n_x)
    pbar[:n_opt] = rng.dirichlet(np.ones(n_opt))
    p0 = rng.dirichlet(np.ones(n_x))
    return Instance(
        name="random_channel",
        F=F,
        g_star=pbar @ F,
        optimal=optimal,
        p0=p0,
        pbar=pbar,
    )
