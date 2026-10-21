"""Exact-population (expected-update) comparators for Tier 0.

All act on a tabular softmax over whole executions, with a per-execution reward R(x),
e.g. binary success 1{x in C} or the expected feedback statistic E[s(F) | x].
"""

import numpy as np


def softmax(theta):
    z = np.exp(theta - theta.max())
    return z / z.sum()


def logits_from(p):
    p = np.asarray(p, dtype=float)
    return np.log(np.clip(p, 1e-300, None))


def run_softmax_pg(p0, R, lr, T):
    """Vanilla policy gradient: theta += lr * p * (R - J). Exact expected gradient."""
    theta = logits_from(p0)
    ps = [softmax(theta)]
    for _ in range(T):
        p = ps[-1]
        theta = theta + lr * p * (R - p @ R)
        ps.append(softmax(theta))
    return np.stack(ps)


def run_natural_pg(p0, R, lr, T):
    """Natural PG for tabular softmax: theta += lr * A, i.e. multiplicative weights p <- p exp(lr R)."""
    theta = logits_from(p0)
    ps = [softmax(theta)]
    for _ in range(T):
        theta = theta + lr * R
        ps.append(softmax(theta))
    return np.stack(ps)


def run_filtered_bc(p0, accept, T):
    """Filtered behavior cloning at population: p <- p * accept / normalizer.

    `accept[x]` is the probability that an execution is kept, e.g. 1{x in C} (full audit)
    or P(every audited requirement reported satisfied | x) (partial audit).
    """
    ps = [np.asarray(p0, dtype=float)]
    for _ in range(T):
        q = ps[-1] * accept
        ps.append(q / q.sum())
    return np.stack(ps)
