"""Exact-population check: does GLFT converge under each teacher-intent profile?

Small instance: m = 2 requirements (slots in different VAGUE groups), N = 3 outcomes each,
targets (0, 0), so executions are outcome vectors (9 of them), optionally times a style bit.
Every random choice of the teacher in `glft.l1_intents.teach` is enumerated to get the exact
channel F(f | x); G* is the same teacher on the optimal execution (calibrated by construction).
Then `run_glft_exact` iterates the exact update p <- p * W.

Extra teachers beyond the sweep's profiles:
  chatter            says nothing informative                  -> no separation, should NOT converge
  liar(rho)          swaps the satisfied/violated intent mix with prob rho per mentioned requirement
                     (false complaints and false praise); G* either the same liar (calibrated)
                     or the honest teacher (miscalibrated)
  style(c)           executions also have a style bit; both styles optimal; the teacher adds a
                     process complaint with prob c on style B -> should converge to style A

  uv run python experiments/l1/verify_intents_exact.py
"""

import itertools
from collections import defaultdict

import numpy as np

from glft.algos.glft_exact import run_glft_exact
from glft.l1_intents import NONE, PROFILES, REVEALS_OUTCOME, Profile, desc_share_profile

M, N = 2, 3
TARGETS = (0, 0)
GROUPS = (0, 1)  # VAGUE group of each requirement's slot


def per_req_options(prof, o, w, rho):
    """[(prob, (intent, intent2, revealed))] for one requirement with outcome o, target w."""
    sat = o == w
    opts = [(1 - (prof.kappa_sat if sat else prof.kappa_viol), (NONE, NONE, -1))]
    p_mention = prof.kappa_sat if sat else prof.kappa_viol
    for lie, p_lie in ((False, 1 - rho), (True, rho)):
        if p_lie == 0:
            continue
        mix = prof.sat_mix if (sat != lie) else prof.viol_mix
        for i1, p1 in mix.items():
            seconds = [(1 - prof.p_stack, NONE)] + [(prof.p_stack * p2, i2) for i2, p2 in mix.items()]
            for p2, i2 in seconds:
                if p2 == 0:
                    continue
                reveals = i1 in REVEALS_OUTCOME or i2 in REVEALS_OUTCOME
                # A lie that reveals an outcome reports the outcome the lie implies.
                shown = (w if not sat else (w + 1) % N) if lie else o
                opts.append((p_mention * p_lie * p1 * p2, (i1, i2, shown if reveals else -1)))
    return opts


def message_dist(prof, outcomes, rho=0.0, style=0, proc_c=0.0):
    sat = [o == w for o, w in zip(outcomes, TARGETS)]
    nsat = sum(sat)
    fail_groups = [g for g, s in zip(GROUPS, sat) if not s]
    evals = [(1 - prof.p_eval, -1), (prof.p_eval, nsat)]
    vagues = [(1.0, -1)] if not fail_groups else \
        [(1 - prof.p_vague, -1)] + [(prof.p_vague / len(fail_groups), g) for g in fail_groups]
    chats = [(1 - prof.p_chatter, 0), (prof.p_chatter, 1)]
    procs = [(1 - proc_c, 0), (proc_c, 1)] if style == 1 else [(1.0, 0)]
    dist = defaultdict(float)
    reqs = [per_req_options(prof, o, w, rho) for o, w in zip(outcomes, TARGETS)]
    for combo in itertools.product(*reqs, evals, vagues, chats, procs):
        p = np.prod([c[0] for c in combo])
        if p == 0:
            continue
        req_parts = [c[1] for c in combo[:M]]
        ev = combo[M][1]
        if ev == M:  # "m of m right" reveals every outcome (as in `teach`)
            req_parts = [(i1, i2, o) for (i1, i2, _), o in zip(req_parts, outcomes)]
        key = (tuple(req_parts), ev, combo[M + 1][1], combo[M + 2][1], combo[M + 3][1])
        dist[key] += p
    return dist


def build(prof, q=0.02, systematic=0.5, rho=0.0, styles=1, proc_c=0.0, gstar_rho=None):
    execs = [(outs, s) for outs in itertools.product(range(N), repeat=M) for s in range(styles)]
    dists = [message_dist(prof, outs, rho, s, proc_c) for outs, s in execs]
    star = message_dist(prof, TARGETS, rho if gstar_rho is None else gstar_rho, 0, proc_c)
    keys = sorted(set().union(*dists, star), key=repr)
    idx = {k: i for i, k in enumerate(keys)}
    F = np.zeros((len(execs), len(keys)))
    for i, d in enumerate(dists):
        for k, p in d.items():
            F[i, idx[k]] = p
    g_star = np.zeros(len(keys))
    for k, p in star.items():
        g_star[idx[k]] = p
    # Base policy per requirement: q on target, misconception (w + 1) gets (1 - q) * systematic extra.
    aim = np.full(N, (1 - q) * (1 - systematic) / (N - 1))
    aim[1] += (1 - q) * systematic
    aim[0] = q
    p0 = np.array([np.prod([aim[o] for o in outs]) / styles for outs, _ in execs])
    optimal = np.array([outs == TARGETS for outs, _ in execs])
    style_a = np.array([s == 0 for _, s in execs])
    return F, g_star, p0, optimal, style_a


def report(name, prof, T=3000, **kw):
    F, g, p0, opt, style_a = build(prof, **kw)
    if g.sum() == 0 or (g[F.T @ p0 == 0] > 0).any():
        print(f"{name:34s} coverage fails (ideal message impossible under p0)")
        return
    run = run_glft_exact(p0, F, g, opt, T=T)
    s = run.success
    extra = ""
    if kw.get("styles", 1) > 1:
        extra = f"  P(style A | optimal) at T: {run.p[-1][opt & style_a].sum() / max(s[-1], 1e-300):.3f}"
    print(f"{name:34s} p_t(C) at t=0,10,100,1000,{T}: "
          + " ".join(f"{s[t]:.3f}" for t in (0, 10, 100, 1000, T)) + f"   Phi_T={run.phi[-1]:.2e}" + extra)


def main():
    print(f"Exact population, m={M}, N={N}, q=0.02 (base success {0.02 ** 2:.4f})\n")
    for name, prof in PROFILES.items():
        report(name, prof)
    for s in (0.0, 0.1, 0.5):
        report(f"desc{s}", desc_share_profile(s))
    report("chatter only", Profile("chatter", kappa_sat=0, kappa_viol=0, p_chatter=0.5))
    for rho in (0.1, 0.3):
        report(f"liar rho={rho}, calibrated G*", PROFILES["grader"], rho=rho)
        report(f"liar rho={rho}, honest (miscal.) G*", PROFILES["grader"], rho=rho, gstar_rho=0.0)
    report("style grumble c=0.5", PROFILES["reporter"], styles=2, proc_c=0.5)


if __name__ == "__main__":
    main()
