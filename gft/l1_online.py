"""Incremental (SGD) GFT, the batched form of the paper's Online GFT (Algorithm 2), tabular.

Like policy-gradient methods, each round collects a batch and takes a few gradient steps; there is
no full MLE fit or full distillation in an inner loop, and no replay (fresh data each round):
  Stage I   D ~ q*, X ~ pi_theta, F ~ teacher(X) for a batch; `epochs` passes of minibatch SGD on the
            auxiliary policy phi minimizing -log pi^fb_phi(X | D, F).
  Stage II  D' ~ q*, F* ~ G*_D' (the same teacher on an all-satisfied execution),
            X+ ~ pi^fb_phi(. | D', F*) for a batch; `epochs` passes of minibatch SGD on the clean
            policy theta minimizing -log pi_theta(X+ | D') with X+ held fixed.
batch = minibatch = 1, epochs = 1 recovers Algorithm 2 exactly.
Step sizes: aux beta_t, clean alpha_t; constant, the paper's two-timescale schedule
beta_t = beta (1+t)^-0.6, alpha_t = alpha (1+t)^-0.9, or "linear": the clean step decays linearly
to `final_frac` of its value over the budget (aux constant).

Ideal-feedback calibration (Section 9, Eq. 41): by default G* is the teacher applied to an
all-satisfied execution. `gstar_kappa` changes how often G* mentions each requirement, and
`gstar_error` makes G* describe a wrong (hallucinated) outcome for a mentioned requirement with
that probability -- an approximate generator G~ != G*.

Clean policy theta: per-step softmax logits theta[i][k, w, prefix, d] (same as the RL baselines).
Auxiliary policy phi: log-linear in request and per-requirement feedback features, per step:
    logits = phi_req[k, w] + phi_out[k, o]        if the message reveals requirement j's outcome o
                           + phi_fail[k, w]       if j is known failed without its outcome (CORRECT/HINT)
                           + phi_vague[k]         if j's slot group is flagged by VAGUE
                           + phi_eval[k, c]       if EVAL reports c < m satisfied
(each indexed also by [prefix, d]). phi_out is shared across targets, so pooling "what produces
outcome o" across requests is learned by SGD rather than built in. The unfactored control
conditions the outcome feature on the target too: phi_out[k, w, o].

aux_base = "clean" (default) uses the current clean logits theta in place of phi_req, matching the
exact posterior's form r(x | f) ~ p_theta(x) F(f | x): phi then learns only the teacher's
(policy-independent) likelihood terms, plus phi_unm[k, w] for requirements the message says nothing
about (needed when mention rates differ between satisfied and violated requirements). Gradients
flow into phi only; theta enters Stage I as a fixed input. aux_base = "own" trains phi_req.
"""

from dataclasses import dataclass

import numpy as np

from gft.l1_horizon import HEnv, HWorld, path_steps
from gft.l1_intents import CORRECT, HINT, Profile, teach
from gft.l1_tabular import Resources


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def features(msg, slots, m):
    """Per-requirement feedback features of a message (all (n, m) arrays)."""
    from gft.l1_intents import N_GROUPS
    fail = np.isin(msg.intent, [CORRECT, HINT]) | np.isin(msg.intent2, [CORRECT, HINT])
    return dict(
        out=msg.revealed,                                                    # -1 or outcome
        fail=fail & (msg.revealed < 0),
        vague=(slots % N_GROUPS) == msg.vague_group[:, None],
        eval=np.where(((msg.eval >= 0) & (msg.eval < m))[:, None], msg.eval[:, None], -1)
             * np.ones_like(slots),
    )


@dataclass
class OnlineConfig:
    batch: int = 256           # Stage I episodes per round (one teacher call each)
    stage2_batch: int = 256    # Stage II rollouts per round (one G* sample each)
    minibatch: int = 64        # SGD minibatch; epochs * batch / minibatch steps per stage per round
    epochs: int = 1
    lr_aux: float = 1000.0     # beta: auxiliary step size (per minibatch-average gradient)
    lr_clean: float = 30.0     # alpha: clean step size
    schedule: str = "const"    # const | two_timescale | linear
    final_frac: float = 0.1    # linear schedule: clean step at the end of the budget, relative
    total_rounds: int = 0      # set by run() from the budget (needed by the linear schedule)
    gstar_kappa: float = -1.0  # < 0: same mention rate as the teacher (calibrated)
    gstar_error: float = 0.0   # P(G* reports a wrong outcome for a mentioned requirement)
    unm_lr_frac: float = 1.0   # step-size multiplier for phi_unm ("requirement not mentioned" term)
    stage2_grad: str = "sample"  # sample: onehot(a) - pi_theta (paper); expected: pi^fb - pi_theta at each
                                 # visited step (same expected gradient, lower variance; logit distillation)
    factored: bool = True
    aux_base: str = "clean"    # clean: logits = theta + feedback terms; own: logits = phi_req + feedback terms


class OnlineGFT:
    def __init__(self, world: HWorld, P0, prof: Profile, cfg: OnlineConfig):
        self.w, self.prof, self.cfg = world, prof, cfg
        e, h = world.env, world.h
        K, N, b, k, m = e.K, e.N, h.b, h.k, e.m
        logP0 = [np.log(np.clip(p, 1e-300, None)) for p in P0]
        self.theta = [x.copy() for x in logP0]
        self.req = [x.copy() for x in logP0]
        shape = lambda i, *lead: lead + (b ** i, b)
        self.out = [np.zeros(shape(i, K, N, N) if not cfg.factored else shape(i, K, N)) for i in range(k)]
        self.fail = [np.zeros(shape(i, K, N)) for i in range(k)]
        self.vague = [np.zeros(shape(i, K)) for i in range(k)]
        self.eval = [np.zeros(shape(i, K, m)) for i in range(k)]
        self.unm = [np.zeros(shape(i, K, N)) for i in range(k)]
        self.t = 0
        self.res = Resources()
        from dataclasses import replace
        self._gstar_prof = prof if cfg.gstar_kappa < 0 else replace(prof, kappa_sat=cfg.gstar_kappa)

    @property
    def P(self):
        return [softmax(t) for t in self.theta]

    def _steps(self, base):
        c = self.cfg
        if c.schedule == "const":
            return c.lr_aux, c.lr_clean
        if c.schedule == "linear":
            frac = min(1.0, self.t / max(1, c.total_rounds))
            return c.lr_aux, c.lr_clean * (1 - (1 - c.final_frac) * frac)
        return c.lr_aux * (1 + self.t) ** -0.6, c.lr_clean * (1 + self.t) ** -0.9

    @staticmethod
    def _uninformed(f):
        return (f["out"] < 0) & ~f["fail"] & ~f["vague"] & (f["eval"] < 0)

    def _aux_logits(self, i, slots, targets, pre, f):
        base = self.theta[i] if self.cfg.aux_base == "clean" else self.req[i]
        z = base[slots, targets, pre].copy()
        un = self._uninformed(f)
        if self.cfg.aux_base == "clean":
            z[un] += self.unm[i][slots[un], targets[un], pre[un]]
        o = f["out"]
        sel = o >= 0
        if self.cfg.factored:
            z[sel] += self.out[i][slots[sel], o[sel], pre[sel]]
        else:
            z[sel] += self.out[i][slots[sel], targets[sel], o[sel], pre[sel]]
        z[f["fail"]] += self.fail[i][slots[f["fail"]], targets[f["fail"]], pre[f["fail"]]]
        z[f["vague"]] += self.vague[i][slots[f["vague"]], pre[f["vague"]]]
        ev = f["eval"] >= 0
        z[ev] += self.eval[i][slots[ev], f["eval"][ev], pre[ev]]
        return z

    def _sample_aux(self, slots, targets, f):
        w, h = self.w, self.w.h
        n, m = slots.shape
        path = np.zeros((n, m), dtype=np.int64)
        for i in range(h.k):
            probs = softmax(self._aux_logits(i, slots, targets, path, f))
            u = w.rng.random((n, m, 1))
            d = np.minimum((probs.cumsum(-1) < u).sum(-1), h.b - 1)
            path = path * h.b + d
        return path

    def _aux_step(self, lr, slots, targets, pre, dig, f):
        """One SGD step on phi: minibatch-average gradient of log pi^fb(X | D, F)."""
        c, h = self.cfg, self.w.h
        scale = lr / len(slots)
        for i in range(h.k):
            g = -softmax(self._aux_logits(i, slots, targets, pre[i], f))          # d log-lik / d logits
            np.put_along_axis(g, dig[i][..., None], np.take_along_axis(g, dig[i][..., None], -1) + 1, -1)
            g *= scale
            if c.aux_base == "own":
                np.add.at(self.req[i], (slots, targets, pre[i]), g)
            else:
                un = self._uninformed(f)
                np.add.at(self.unm[i], (slots[un], targets[un], pre[i][un]), c.unm_lr_frac * g[un])
            sel = f["out"] >= 0
            if c.factored:
                np.add.at(self.out[i], (slots[sel], f["out"][sel], pre[i][sel]), g[sel])
            else:
                np.add.at(self.out[i], (slots[sel], targets[sel], f["out"][sel], pre[i][sel]), g[sel])
            fs = f["fail"]
            np.add.at(self.fail[i], (slots[fs], targets[fs], pre[i][fs]), g[fs])
            vs = f["vague"]
            np.add.at(self.vague[i], (slots[vs], pre[i][vs]), g[vs])
            es = f["eval"] >= 0
            np.add.at(self.eval[i], (slots[es], f["eval"][es], pre[i][es]), g[es])

    def _clean_step(self, lr, slots, targets, pre, dig, f=None):
        """One SGD step on theta: minibatch-average gradient of log pi_theta(X+ | D'), X+ fixed.

        With stage2_grad = "expected", the sampled action's one-hot at each visited step is replaced
        by pi^fb's action distribution there (needs the ideal-feedback features f)."""
        P = self.P
        for i in range(self.w.h.k):
            g = -P[i][slots, targets, pre[i]]
            if self.cfg.stage2_grad == "expected":
                g += softmax(self._aux_logits(i, slots, targets, pre[i], f))
            else:
                np.put_along_axis(g, dig[i][..., None], np.take_along_axis(g, dig[i][..., None], -1) + 1, -1)
            np.add.at(self.theta[i], (slots, targets, pre[i]), (lr / len(slots)) * g)

    def _minibatches(self, n):
        c = self.cfg
        for _ in range(c.epochs):
            perm = self.w.rng.permutation(n)
            for j in range(0, n, c.minibatch):
                yield perm[j: j + c.minibatch]

    def round(self):
        w, c, e, h = self.w, self.cfg, self.w.env, self.w.h
        beta, alpha = self._steps(None)
        # Stage I: a batch of on-policy executions with hindsight feedback; a few SGD steps on phi.
        slots, targets = w.requests(c.batch)
        paths = w.act_steps(self.P, slots, targets)
        outc = w.outcome(slots, paths)
        f = features(teach(w.rng, self.prof, slots, targets, outc), slots, e.m)
        self.res.env_episodes += c.batch
        self.res.teacher_calls += c.batch
        pre, dig = path_steps(paths, h.b, h.k)
        for mb in self._minibatches(c.batch):
            self._aux_step(beta, slots[mb], targets[mb], [p[mb] for p in pre], [d[mb] for d in dig],
                           {k_: v[mb] for k_, v in f.items()})
        # Stage II: a batch of ideal-feedback-conditioned rollouts of the updated phi; a few SGD steps on theta.
        s2, t2 = w.requests(c.stage2_batch)
        f2 = features(teach(w.rng, self._gstar_prof, s2, t2, t2), s2, e.m)
        if c.gstar_error > 0:
            bad = (f2["out"] >= 0) & (w.rng.random(f2["out"].shape) < c.gstar_error)
            f2["out"] = np.where(bad, (t2 + w.rng.integers(1, e.N, t2.shape)) % e.N, f2["out"])
        p2 = self._sample_aux(s2, t2, f2)
        self.res.env_episodes += c.stage2_batch
        self.res.gstar_samples += c.stage2_batch
        pre2, dig2 = path_steps(p2, h.b, h.k)
        for mb in self._minibatches(c.stage2_batch):
            self._clean_step(alpha, s2[mb], t2[mb], [p[mb] for p in pre2], [d[mb] for d in dig2],
                             {k_: v[mb] for k_, v in f2.items()})
        self.t += 1


def run(arm, env: HEnv, prof: Profile, seed=0, budget=65536, eval_every=2048, overrides=None):
    """gft-sgd | gft-sgd-unf here; any other arm is delegated to its module."""
    from gft.l1_horizon import conditionals
    if arm not in ("gft-sgd", "gft-sgd-unf"):
        from gft.l1_intents import run as run_intents
        return run_intents(arm, env, prof, seed=seed, budget=budget, eval_every=eval_every, overrides=overrides)
    world = HWorld(env, np.random.default_rng(seed))
    P0 = conditionals(env.flat().base_table(), env.b, env.k)
    cfg = OnlineConfig(factored=(arm == "gft-sgd"), **(overrides or {}))
    cfg.total_rounds = budget // (cfg.batch + cfg.stage2_batch)
    learner = OnlineGFT(world, P0, prof, cfg)
    curve, nxt = [], 0
    while True:
        if learner.res.env_episodes >= nxt:
            s, f = world.success_rate_steps(learner.P)
            curve.append({**learner.res.__dict__, "success": s, "frac": f})
            nxt += eval_every
        if learner.res.env_episodes >= budget:
            return curve
        learner.round()
