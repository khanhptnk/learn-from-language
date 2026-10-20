"""Truthful random partial-audit teacher (paper Section 6), enumerated exactly.

An execution is summarized by its requirement-satisfaction vector r(x) in {0,1}^m.
The teacher draws an audit subset S, then truthfully reports the polarity of every j in S.
A message is the pair (S, r restricted to S); wording/order are irrelevant to the population
analysis, so they are not enumerated here.

Messages are encoded as a length-m tuple over {-1 (not audited), 0 (violated), 1 (satisfied)}.
"""

import itertools

import numpy as np


def subset_probs_independent(kappa):
    """P(S) for independent inclusion with probabilities kappa_j. Returns {S (tuple of 0/1): prob}."""
    kappa = np.asarray(kappa, dtype=float)
    out = {}
    for S in itertools.product((0, 1), repeat=len(kappa)):
        S_arr = np.array(S)
        out[S] = float(np.prod(np.where(S_arr == 1, kappa, 1 - kappa)))
    return out


def all_messages(m):
    return list(itertools.product((-1, 0, 1), repeat=m))


def message_for(S, r):
    return tuple(int(r[j]) if S[j] else -1 for j in range(len(S)))


def audit_channel(R, subset_probs):
    """Build F[x, f] for satisfaction matrix R (n_x, m) and an outcome-independent P(S).

    Returns (F, messages, s) where s[f] is the statistic s_D(f) of Eq. (22):
    number of requirements reported satisfied, divided by m.
    """
    R = np.asarray(R, dtype=int)
    n_x, m = R.shape
    messages = all_messages(m)
    index = {msg: i for i, msg in enumerate(messages)}
    F = np.zeros((n_x, len(messages)))
    for x in range(n_x):
        for S, prob in subset_probs.items():
            if prob > 0:
                F[x, index[message_for(S, R[x])]] += prob
    s = np.array([sum(v == 1 for v in msg) / m for msg in messages])
    return F, messages, s


def ideal_feedback(subset_probs, messages):
    """G*: the same audit-selection kernel applied to an execution satisfying every requirement."""
    index = {msg: i for i, msg in enumerate(messages)}
    g = np.zeros(len(messages))
    for S, prob in subset_probs.items():
        g[index[message_for(S, np.ones(len(S), dtype=int))]] += prob
    return g


def all_positive_accept(R, kappa):
    """P(every audited requirement is reported satisfied | x) under independent inclusion.

    This is the keep-probability of filtered BC that only sees partial-audit feedback.
    """
    R = np.asarray(R, dtype=int)
    kappa = np.asarray(kappa, dtype=float)
    return np.prod(np.where(R == 1, 1.0, 1.0 - kappa), axis=1)
