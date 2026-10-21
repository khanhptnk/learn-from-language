"""Divergences and summary statistics over finite distributions."""

import numpy as np


def kl(p, q):
    """KL(p || q) summed over the positive support of p. Returns inf if q misses p's support."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    s = p > 0
    if np.any(q[s] <= 0):
        return np.inf
    return float(np.sum(p[s] * np.log(p[s] / q[s])))


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def entropy(p):
    p = np.asarray(p, dtype=float)
    s = p > 0
    return float(-np.sum(p[s] * np.log(p[s])))


def normalized_entropy_on(p, mask):
    """Entropy of p restricted and renormalized to `mask`, divided by log |mask|.

    1 means p spreads uniformly over the masked set (e.g. all optimal executions),
    0 means it has collapsed onto a single element.
    """
    q = np.asarray(p, dtype=float)[mask]
    n = q.size
    if n <= 1 or q.sum() <= 0:
        return 1.0
    return entropy(q / q.sum()) / np.log(n)
