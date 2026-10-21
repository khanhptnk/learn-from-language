"""Level L1: tabular, count-based learners on the one-shot opaque-mapping bandit (numpy, fast).

Environment (one request = m requirements on distinct slots):
  requirement j asks slot k_j to reach target outcome w_j in [0, N);
  the policy picks an opaque argument a_j in [0, N); the outcome is sigma[k_j, a_j]
  (sigma: hidden per-slot permutation); requirement j holds iff sigma[k_j, a_j] == w_j.
Request-only policies are tables P[k, w, a] = pi(a | slot k, target w), factored over requirements.

Teacher (truthful partial audit): requirement j is reported with prob kappa_sat / kappa_viol
(by its truth value). A report is descriptive ("slot k became o", plus polarity) or, with
reveal=False, polarity only. G* reports each requirement with prob kappa_sat, as satisfied.

Learners:
  GFT (factored)     Stage I fits the exact-form posterior pi^fb:
                        reported cell -> grounding table over (slot, reported outcome), pooled over
                                         all targets and all rounds (sigma is deterministic, so this
                                         part of the posterior is policy-independent);
                        unreported    -> per-(slot, target) counts from the current round.
                      Stage II rolls out pi^fb under F* ~ G* and distills by counts.
  GFT (unfactored)   reported cell conditioned on (slot, target, report): only POS reports of the
                      same target inform it, i.e. no cross-request pooling. Isolates the mechanism.
  RL                  softmax logits theta[k, w, a]; REINFORCE / RLOO / GRPO with terminal rewards
                      bin / frac / parse, or the ORACLE per-requirement reward `align`.
  Filtered BC         counts of accepted executions (verifier, or all-positive partial audit).
Counts are smoothed toward the current table with pseudo-count beta, which keeps full support.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Env:
    K: int = 8          # slots
    N: int = 16         # outcomes = arguments per slot
    m: int = 4          # requirements per request
    q: float = 0.35     # base competence: P(base picks the correct argument)
    systematic: float = 0.5  # share of base errors on the misconception outcome (w + 1) mod N
    kappa_sat: float = 0.5
    kappa_viol: float = 0.5
    reveal: bool = True      # descriptive feedback (reveals the achieved outcome)
    world_seed: int = 0

    def sigma(self):
        g = np.random.default_rng(self.world_seed)
        return np.stack([g.permutation(self.N) for _ in range(self.K)])

    def base_table(self):
        """P0[k, w, a]: the base policy, aimed at outcomes then mapped through sigma^{-1}."""
        K, N = self.K, self.N
        aim = np.full((N, N), (1 - self.q) * (1 - self.systematic) / (N - 1))  # aim[w, o]
        aim[np.arange(N), (np.arange(N) + 1) % N] += (1 - self.q) * self.systematic
        aim[np.arange(N), np.arange(N)] = self.q
        inv = np.argsort(self.sigma(), axis=1)  # inv[k, o] = argument producing o
        P = np.zeros((K, N, N))
        for k in range(K):
            P[k][:, inv[k]] = aim
        return P


@dataclass
class Resources:
    env_episodes: int = 0
    teacher_calls: int = 0
    gstar_samples: int = 0
    verifier_calls: int = 0


class World:
    def __init__(self, env: Env, rng):
        self.env, self.rng = env, rng
        self.sigma = env.sigma()
        self.correct = np.argsort(self.sigma, axis=1)  # correct[k, w]

    def requests(self, n):
        e = self.env
        slots = np.argsort(self.rng.random((n, e.K)), axis=1)[:, : e.m]
        targets = self.rng.integers(e.N, size=(n, e.m))
        return slots, targets

    def act(self, P, slots, targets):
        """Sample one argument per requirement from table P. (n, m)."""
        probs = P[slots, targets]  # (n, m, N)
        u = self.rng.random(probs.shape[:2] + (1,))
        return np.minimum((probs.cumsum(-1) < u).sum(-1), self.env.N - 1)

    def outcome(self, slots, args):
        return self.sigma[slots, args]

    def audit(self, sat):
        e = self.env
        kappa = np.where(sat, e.kappa_sat, e.kappa_viol)
        return self.rng.random(sat.shape) < kappa  # reported mask

    def success_rate(self, P, n=4000):
        slots, targets = self._eval_set(n)
        c = P[slots, targets, self.correct[slots, targets]]
        return float(c.prod(1).mean()), float(c.mean())

    def _eval_set(self, n):
        if not hasattr(self, "_eval"):
            g = np.random.default_rng(12345)
            e = self.env
            self._eval = (np.argsort(g.random((n, e.K)), axis=1)[:, : e.m], g.integers(e.N, size=(n, e.m)))
        return self._eval


def smooth(counts, prior, beta):
    """(counts + beta * prior) / (n + beta) along the last axis."""
    return (counts + beta * prior) / (counts.sum(-1, keepdims=True) + beta)


@dataclass
class GFTConfig:
    episodes_per_round: int = 256
    stage2_per_round: int = 256
    factored: bool = True
    beta: float = 1.0          # Stage I pseudo-count toward the current policy, reported cells
    beta_unrep: float = 1.0    # ... unreported cells (exact posterior = prior when kappa_sat = kappa_viol)
    beta_distill: float = 2.0  # Stage II pseudo-count toward the current policy
    stage2_exact: bool = False  # distill the exact expectation over G* (idealized; no rollouts)


class GFT:
    def __init__(self, world: World, P0, cfg: GFTConfig):
        self.w, self.P, self.cfg = world, P0.copy(), cfg
        K, N = world.env.K, world.env.N
        self.ground = np.zeros((K, N, N))  # [k, outcome-or-target, a] counts from reported requirements
        self.res = Resources()

    def round(self):
        w, c, e = self.w, self.cfg, self.w.env
        slots, targets = w.requests(c.episodes_per_round)
        args = w.act(self.P, slots, targets)
        outc = w.outcome(slots, args)
        sat = outc == targets
        rep = w.audit(sat)
        self.res.env_episodes += len(slots)
        self.res.teacher_calls += len(slots)
        # Stage I. Reported requirements -> grounding counts.
        if c.factored and e.reveal:
            key = np.where(rep, outc, -1)            # every report reveals its achieved outcome
        else:
            key = np.where(rep & sat, targets, -1)   # only "satisfied" reports identify the outcome
        sel = key >= 0
        np.add.at(self.ground, (slots[sel], key[sel], args[sel]), 1)
        unrep = np.zeros_like(self.P)
        np.add.at(unrep, (slots[~rep], targets[~rep], args[~rep]), 1)
        fb_rep = smooth(self.ground, self.P, c.beta)   # pi^fb(a | k, w, reported w satisfied)
        fb_unrep = smooth(unrep, self.P, c.beta_unrep)  # pi^fb(a | k, w, not reported)
        # Stage II.
        if c.stage2_exact:
            self.P = e.kappa_sat * fb_rep + (1 - e.kappa_sat) * fb_unrep
            return
        s2, t2 = w.requests(c.stage2_per_round)
        star = w.rng.random(s2.shape) < e.kappa_sat     # F*: which requirements are reported
        a_rep = w.act(fb_rep, s2, t2)
        a_unrep = w.act(fb_unrep, s2, t2)
        a2 = np.where(star, a_rep, a_unrep)
        self.res.env_episodes += len(s2)
        self.res.gstar_samples += len(s2)
        D = np.zeros_like(self.P)
        np.add.at(D, (s2, t2, a2), 1)
        self.P = smooth(D, self.P, c.beta_distill)


@dataclass
class RLConfig:
    algo: str = "grpo"     # reinforce | rloo | grpo
    reward: str = "frac"   # bin | frac | parse | align
    episodes_per_round: int = 256
    group_size: int = 8
    lr: float = 1.0


class PolicyGradient:
    def __init__(self, world: World, P0, cfg: RLConfig):
        self.w, self.cfg = world, cfg
        self.theta = np.log(P0)
        self.res = Resources()

    @property
    def P(self):
        z = np.exp(self.theta - self.theta.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    def round(self):
        w, c, e = self.w, self.cfg, self.w.env
        G = c.group_size
        slots, targets = w.requests(c.episodes_per_round // G)
        slots, targets = np.repeat(slots, G, 0), np.repeat(targets, G, 0)
        P = self.P
        args = w.act(P, slots, targets)
        sat = w.outcome(slots, args) == targets
        self.res.env_episodes += len(slots)
        if c.reward == "parse":
            self.res.teacher_calls += len(slots)
            r = (w.audit(sat) & sat).sum(1) / e.m
        else:
            self.res.verifier_calls += len(slots)
            r = {"bin": sat.all(1).astype(float), "frac": sat.mean(1), "align": sat.astype(float)}[c.reward]
        if c.algo == "reinforce":
            adv = r - r.mean(0)
        else:
            g = r.reshape(-1, G, *r.shape[1:])
            if c.algo == "rloo":
                adv = g - (g.sum(1, keepdims=True) - g) / (G - 1)
            elif c.algo == "grpo":
                adv = (g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True, ddof=1) + 1e-6)
            adv = adv.reshape(r.shape)
        if adv.ndim == 1:
            adv = np.repeat(adv[:, None], e.m, 1)
        # Softmax policy gradient: d log pi(a)/d theta = onehot(a) - pi.
        grad = np.zeros_like(self.theta)
        np.add.at(grad, (slots, targets, args), adv)
        np.add.at(grad, (slots, targets), -adv[..., None] * P[slots, targets])
        self.theta += c.lr * grad / len(slots)


@dataclass
class FBCConfig:
    accept: str = "verifier"   # verifier | all_positive
    episodes_per_round: int = 256
    beta: float = 2.0


class FilteredBC:
    def __init__(self, world: World, P0, cfg: FBCConfig):
        self.w, self.P0, self.P, self.cfg = world, P0, P0.copy(), cfg
        self.kept = np.zeros_like(P0)
        self.res = Resources()

    def round(self):
        w, c = self.w, self.cfg
        slots, targets = w.requests(c.episodes_per_round)
        args = w.act(self.P, slots, targets)
        sat = w.outcome(slots, args) == targets
        self.res.env_episodes += len(slots)
        if c.accept == "verifier":
            self.res.verifier_calls += len(slots)
            keep = sat.all(1)
        else:
            self.res.teacher_calls += len(slots)
            keep = ~(w.audit(sat) & ~sat).any(1)
        np.add.at(self.kept, (slots[keep], targets[keep], args[keep]), 1)
        self.P = smooth(self.kept, self.P0, c.beta)


ARMS = {
    "gft": lambda w, P0, o: GFT(w, P0, GFTConfig(**o)),
    "gft-unfactored": lambda w, P0, o: GFT(w, P0, GFTConfig(factored=False, **o)),
    "fbc-verifier": lambda w, P0, o: FilteredBC(w, P0, FBCConfig(**o)),
    "fbc-all_positive": lambda w, P0, o: FilteredBC(w, P0, FBCConfig(accept="all_positive", **o)),
}


def make_learner(arm, world, P0, overrides):
    if arm in ARMS:
        return ARMS[arm](world, P0, overrides)
    algo, reward = arm.split("-")
    return PolicyGradient(world, P0, RLConfig(algo=algo, reward=reward, **overrides))


@dataclass
class RunResult:
    arm: str
    env: Env
    overrides: dict
    seed: int
    curve: list = field(default_factory=list)  # dicts: resources + success + frac


def run(arm, env: Env, seed=0, budget=32768, eval_every=1024, overrides=None):
    rng = np.random.default_rng(seed)
    world = World(env, rng)
    learner = make_learner(arm, world, env.base_table(), overrides or {})
    out = RunResult(arm, env, overrides or {}, seed)

    def log():
        s, f = world.success_rate(learner.P)
        out.curve.append({**learner.res.__dict__, "success": s, "frac": f})

    log()
    nxt = eval_every
    while learner.res.env_episodes < budget:
        learner.round()
        if learner.res.env_episodes >= nxt:
            log()
            nxt += eval_every
    return out


def episodes_to(curve, threshold, key="success", x="env_episodes"):
    for p in curve:
        if p[key] >= threshold:
            return p[x]
    return np.inf
