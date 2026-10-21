"""L1-H: the L1 tabular bandit with depth. Each requirement is achieved by a chain of k decisions.

A requirement's "argument" is now a path of k decisions, each among b options (N = b^k paths),
and the outcome is sigma[slot, path] (a hidden permutation of whole paths, so a prefix says
nothing about the outcome until the chain is complete). Horizon H = m * k. Policies are
autoregressive per-step tables P[i][k, w, prefix, d] = pi(d_i | slot, target, d_<i); the base
policy has the same path distribution as L1 with N = b^k, so changing (b, k) at fixed N changes
only the sequential structure.

Learners mirror `l1_tabular`: GFT counts paths (grounding pooled over targets for reported
requirements; per-target current-round counts for unreported) and converts them to per-step
conditionals; RL uses per-step softmax logits with the terminal (or per-requirement) advantage
on every step of the chain; filtered BC counts accepted paths. Path index: digits most
significant first, path = sum_i d_i b^(k-1-i).

Distractor steps are not modelled: with tabular per-step parameters they cannot interfere with
the relevant steps. Their effect on credit assignment needs shared parameters (level L2).
"""

from dataclasses import dataclass, replace

import numpy as np

from gft.l1_tabular import Env, Resources, World, smooth


@dataclass(frozen=True)
class HEnv:
    b: int = 4           # options per decision
    k: int = 3           # decisions per requirement (depth)
    K: int = 8
    m: int = 4
    q: float = 0.35
    systematic: float = 0.5
    kappa_sat: float = 0.5
    kappa_viol: float = 0.5
    reveal: bool = True
    world_seed: int = 0

    @property
    def N(self):
        return self.b ** self.k

    def flat(self) -> Env:
        """The equivalent one-shot L1 environment (same sigma, base path law, teacher)."""
        return Env(K=self.K, N=self.N, m=self.m, q=self.q, systematic=self.systematic, kappa_sat=self.kappa_sat,
                   kappa_viol=self.kappa_viol, reveal=self.reveal, world_seed=self.world_seed)


def to_steps(path_table, b, k):
    """(K, N, b^k) path masses -> list over steps i of (K, N, b^i, b) prefix-extension masses."""
    K, N, _ = path_table.shape
    return [path_table.reshape(K, N, b ** i, b, b ** (k - 1 - i)).sum(-1) for i in range(k)]


def normalize(t):
    s = t.sum(-1, keepdims=True)
    return np.where(s > 0, t / np.where(s > 0, s, 1), 1.0 / t.shape[-1])


def conditionals(path_dist, b, k):
    return [normalize(x) for x in to_steps(path_dist, b, k)]


def sample_paths(P, slots, targets, b, rng):
    n, m = slots.shape
    path = np.zeros((n, m), dtype=np.int64)
    for Pi in P:
        probs = Pi[slots, targets, path]                     # (n, m, b)
        u = rng.random((n, m, 1))
        d = np.minimum((probs.cumsum(-1) < u).sum(-1), b - 1)
        path = path * b + d
    return path


def path_steps(paths, b, k):
    """Prefix before each step and the digit taken: lists of (n, m) arrays."""
    pre, dig = [], []
    for i in range(k):
        pre.append(paths // b ** (k - i))
        dig.append((paths // b ** (k - 1 - i)) % b)
    return pre, dig


def path_prob(P, slots, targets, paths, b, k):
    pre, dig = path_steps(paths, b, k)
    p = np.ones(paths.shape)
    for i, Pi in enumerate(P):
        p *= Pi[slots, targets, pre[i], dig[i]]
    return p


class HWorld(World):
    def __init__(self, env: HEnv, rng):
        super().__init__(env.flat(), rng)
        self.h = env

    def act_steps(self, P, slots, targets):
        return sample_paths(P, slots, targets, self.h.b, self.rng)

    def success_rate_steps(self, P, n=4000):
        slots, targets = self._eval_set(n)
        c = path_prob(P, slots, targets, self.correct[slots, targets], self.h.b, self.h.k)
        return float(c.prod(1).mean()), float(c.mean())


def smooth_steps(path_counts, P, beta, b, k):
    return [smooth(c, Pi, beta) for c, Pi in zip(to_steps(path_counts, b, k), P)]


@dataclass
class GFTConfig:
    episodes_per_round: int = 256
    stage2_per_round: int = 256
    factored: bool = True
    beta: float = 0.01
    beta_unrep: float = 100.0
    beta_distill: float = 4.0


class GFT:
    def __init__(self, world: HWorld, P0, cfg: GFTConfig):
        self.w, self.P, self.cfg = world, [p.copy() for p in P0], cfg
        e = world.env
        self.ground = np.zeros((e.K, e.N, e.N))
        self.res = Resources()

    def round(self):
        w, c, e, h = self.w, self.cfg, self.w.env, self.w.h
        slots, targets = w.requests(c.episodes_per_round)
        paths = w.act_steps(self.P, slots, targets)
        outc = w.outcome(slots, paths)
        sat = outc == targets
        rep = w.audit(sat)
        self.res.env_episodes += len(slots)
        self.res.teacher_calls += len(slots)
        key = np.where(rep, outc, -1) if (c.factored and e.reveal) else np.where(rep & sat, targets, -1)
        sel = key >= 0
        np.add.at(self.ground, (slots[sel], key[sel], paths[sel]), 1)
        unrep = np.zeros_like(self.ground)
        np.add.at(unrep, (slots[~rep], targets[~rep], paths[~rep]), 1)
        fb_rep = smooth_steps(self.ground, self.P, c.beta, h.b, h.k)
        fb_unrep = smooth_steps(unrep, self.P, c.beta_unrep, h.b, h.k)
        s2, t2 = w.requests(c.stage2_per_round)
        star = w.rng.random(s2.shape) < e.kappa_sat
        p2 = np.where(star, w.act_steps(fb_rep, s2, t2), w.act_steps(fb_unrep, s2, t2))
        self.res.env_episodes += len(s2)
        self.res.gstar_samples += len(s2)
        D = np.zeros_like(self.ground)
        np.add.at(D, (s2, t2, p2), 1)
        self.P = smooth_steps(D, self.P, c.beta_distill, h.b, h.k)


@dataclass
class RLConfig:
    algo: str = "grpo"
    reward: str = "frac"
    episodes_per_round: int = 256
    group_size: int = 8
    lr: float = 300.0
    schedule: str = "const"    # const | linear (decays to final_frac of lr over total_rounds)
    final_frac: float = 0.1
    total_rounds: int = 0


class PolicyGradient:
    def __init__(self, world: HWorld, P0, cfg: RLConfig):
        self.w, self.cfg = world, cfg
        self.theta = [np.log(np.clip(p, 1e-300, None)) for p in P0]
        self.res = Resources()

    def _reward(self, slots, targets, outc):
        c, e = self.cfg, self.w.env
        sat = outc == targets
        if c.reward == "parse":
            self.res.teacher_calls += len(slots)
            return (self.w.audit(sat) & sat).sum(1) / e.m
        self.res.verifier_calls += len(slots)
        return {"bin": sat.all(1).astype(float), "frac": sat.mean(1), "align": sat.astype(float)}[c.reward]

    @property
    def P(self):
        out = []
        for t in self.theta:
            z = np.exp(t - t.max(-1, keepdims=True))
            out.append(z / z.sum(-1, keepdims=True))
        return out

    def round(self):
        w, c, e, h = self.w, self.cfg, self.w.env, self.w.h
        G = c.group_size
        slots, targets = w.requests(c.episodes_per_round // G)
        slots, targets = np.repeat(slots, G, 0), np.repeat(targets, G, 0)
        P = self.P
        paths = w.act_steps(P, slots, targets)
        outc = w.outcome(slots, paths)
        self.res.env_episodes += len(slots)
        r = self._reward(slots, targets, outc)
        g = r.reshape(-1, G, *r.shape[1:])
        if c.algo == "rloo":
            adv = g - (g.sum(1, keepdims=True) - g) / (G - 1)
        else:
            adv = (g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True, ddof=1) + 1e-6)
        adv = adv.reshape(r.shape)
        if adv.ndim == 1:
            adv = np.repeat(adv[:, None], e.m, 1)
        pre, dig = path_steps(paths, h.b, h.k)
        for i in range(h.k):
            grad = np.zeros_like(self.theta[i])
            np.add.at(grad, (slots, targets, pre[i], dig[i]), adv)
            np.add.at(grad, (slots, targets, pre[i]), -adv[..., None] * P[i][slots, targets, pre[i]])
            self.theta[i] += self._lr() * grad / len(slots)
        self.t = getattr(self, "t", 0) + 1

    def _lr(self):
        c = self.cfg
        if c.schedule == "linear":
            frac = min(1.0, getattr(self, "t", 0) / max(1, c.total_rounds))
            return c.lr * (1 - (1 - c.final_frac) * frac)
        return c.lr


@dataclass
class FBCConfig:
    episodes_per_round: int = 256
    beta: float = 0.003


class FilteredBC:
    def __init__(self, world: HWorld, P0, cfg: FBCConfig):
        self.w, self.P0, self.P, self.cfg = world, P0, [p.copy() for p in P0], cfg
        e = world.env
        self.kept = np.zeros((e.K, e.N, e.N))
        self.res = Resources()

    def round(self):
        w, c, h = self.w, self.cfg, self.w.h
        slots, targets = w.requests(c.episodes_per_round)
        paths = w.act_steps(self.P, slots, targets)
        sat = w.outcome(slots, paths) == targets
        self.res.env_episodes += len(slots)
        self.res.verifier_calls += len(slots)
        keep = sat.all(1)
        np.add.at(self.kept, (slots[keep], targets[keep], paths[keep]), 1)
        self.P = smooth_steps(self.kept, self.P0, c.beta, h.b, h.k)


def make_learner(arm, world, P0, overrides):
    if arm == "gft":
        return GFT(world, P0, GFTConfig(**overrides))
    if arm == "gft-unfactored":
        return GFT(world, P0, GFTConfig(factored=False, **overrides))
    if arm == "fbc-verifier":
        return FilteredBC(world, P0, FBCConfig(**overrides))
    algo, reward = arm.split("-")
    return PolicyGradient(world, P0, RLConfig(algo=algo, reward=reward, **overrides))


def run(arm, env: HEnv, seed=0, budget=65536, eval_every=2048, overrides=None):
    world = HWorld(env, np.random.default_rng(seed))
    P0 = conditionals(env.flat().base_table(), env.b, env.k)
    learner = make_learner(arm, world, P0, overrides or {})
    if hasattr(learner.cfg, "total_rounds"):
        learner.cfg.total_rounds = budget // learner.cfg.episodes_per_round
    curve = []

    def log():
        s, f = world.success_rate_steps(learner.P)
        curve.append({**learner.res.__dict__, "success": s, "frac": f})

    log()
    nxt = eval_every
    while learner.res.env_episodes < budget:
        learner.round()
        if learner.res.env_episodes >= nxt:
            log()
            nxt += eval_every
    return curve
