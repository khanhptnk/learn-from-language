"""L1-I: mixed-intent feedback on the L1 tabular bandit.

A teacher message is a set of segments, each with an intent. Per requirement j (slot k_j,
target w_j, achieved outcome o_j), a mentioned requirement gets ONE of:
  DESC      "slot became o_j"                       reveals o_j (satisfied or not)
  CONFIRM   "slot is right"                         satisfied only; reveals o_j = w_j
  CORRECT   "slot should be w_j"                    violated only; failure + restated target
  DIAG      "failed because slot became o_j"        violated only; reveals o_j
  CONTRAST  "o_j rather than w_j"                   violated only; reveals o_j
  HINT      "the right argument is ..."             violated only; prescriptive (about sigma^-1(w_j))
Message-level segments:
  EVAL      "c of m right"                          exact count of satisfied requirements
  VAGUE     "something's off with group g"          one slot group containing a failure
  CHATTER   "thanks!"                               nothing
A single message mixes intents: different requirements get different intents, a mentioned
requirement may stack a second intent (prob p_stack, e.g. CONTRAST + CORRECT: "edit rather than
view; he should only read it"), and message-level segments are added independently.
A profile sets mention rates (kappa_sat / kappa_viol), the intent mix for satisfied and violated
requirements, stacking, and the message-level segment rates. Ideal feedback G* is the same teacher applied
to an all-satisfied execution, so it is calibrated by construction.

Runs on the depth environment L1-H (k decisions per requirement; k = 1 is flat L1), with
per-step tables built from path counts as in `l1_horizon`.

Tabular GLFT keeps Stage I in exact Bayes form:
  * a requirement whose message reveals its achieved outcome (DESC, CONFIRM, DIAG, CONTRAST)
    feeds the grounding table G[k, o, a], pooled over requests and rounds;
  * otherwise pi^fb conditions on the requirement's signature in the message. Only the
    signatures that occur under G* are ever queried: not mentioned, its slot group not flagged by
    VAGUE, and EVAL absent or equal to m. Those cells are fit from current-round counts.
Prescriptive content (CORRECT's target, HINT's argument) describes what *should* have been done;
it never enters GLFT's model except as "this requirement failed".
"""

from dataclasses import dataclass, field

import numpy as np

from glft.l1_horizon import HEnv, HWorld, PolicyGradient, RLConfig, conditionals, smooth_steps


def path_table(P):
    """Per-step tables [(K, N, b^i, b)] -> full path distribution (K, N, b^k)."""
    cur = P[0][:, :, 0, :]
    for Pi in P[1:]:
        cur = (cur[..., None] * Pi).reshape(cur.shape[0], cur.shape[1], -1)
    return cur
from glft.l1_tabular import Resources

NONE, DESC, CONFIRM, CORRECT, DIAG, CONTRAST, HINT = range(7)
INTENT_NAMES = ["none", "desc", "confirm", "correct", "diag", "contrast", "hint"]
REVEALS_OUTCOME = {DESC, CONFIRM, DIAG, CONTRAST}
FAILURE_ONLY = [CORRECT, DIAG, CONTRAST, HINT]
N_GROUPS = 4


@dataclass(frozen=True)
class Profile:
    name: str
    sat_mix: dict = field(default_factory=lambda: {DESC: 1.0})     # intents for satisfied requirements
    viol_mix: dict = field(default_factory=lambda: {DESC: 1.0})    # intents for violated requirements
    kappa_sat: float = 0.5
    kappa_viol: float = 0.5
    p_eval: float = 0.0
    p_vague: float = 0.0
    p_chatter: float = 0.0
    p_stack: float = 0.0       # P(a mentioned requirement gets a second intent from the same mix)

    def __hash__(self):
        return hash(self.name)


PROFILES = {
    "reporter": Profile("reporter"),
    "grader": Profile("grader", {CONFIRM: 1.0}, {CORRECT: 1.0}),
    "explainer": Profile("explainer", {CONFIRM: 1.0}, {DIAG: 0.5, CONTRAST: 0.5}),
    "coach": Profile("coach", {CONFIRM: 1.0}, {CORRECT: 0.5, HINT: 0.5}),
    "scorer": Profile("scorer", kappa_sat=0.0, kappa_viol=0.0, p_eval=1.0, p_chatter=0.5),
    "vague": Profile("vague", kappa_sat=0.0, kappa_viol=0.0, p_vague=1.0, p_chatter=0.5),
    "mixed": Profile("mixed", {DESC: 0.3, CONFIRM: 0.7},
                     {DESC: 0.15, CORRECT: 0.3, DIAG: 0.15, CONTRAST: 0.1, HINT: 0.3},
                     kappa_sat=0.3, kappa_viol=0.7, p_eval=0.3, p_vague=0.3, p_chatter=0.5, p_stack=0.3),
}


def desc_share_profile(s):
    """Per-requirement intents: descriptive with prob s, else pass/fail (CONFIRM / CORRECT)."""
    return Profile(f"desc{s}", {DESC: s, CONFIRM: 1 - s}, {DESC: s, CORRECT: 1 - s})


@dataclass
class Message:
    intent: np.ndarray     # (n, m) per-requirement intent (NONE if not mentioned)
    intent2: np.ndarray    # (n, m) stacked second intent, NONE if absent
    revealed: np.ndarray   # (n, m) achieved outcome if the segment reveals it, else -1
    eval: np.ndarray       # (n,) count of satisfied requirements if EVAL present, else -1
    vague_group: np.ndarray  # (n,) slot group flagged by VAGUE, else -1
    chatter: np.ndarray    # (n,) bool

    def signature_ok(self, slots, m):
        """(n, m) bool: requirement is in an ideal-feedback signature cell that is not outcome-revealing."""
        not_flagged = (slots % N_GROUPS) != self.vague_group[:, None]
        eval_ok = (self.eval == -1) | (self.eval == m)
        return (self.intent == NONE) & not_flagged & eval_ok[:, None]


def _choose(rng, mix, shape):
    keys = np.array(list(mix.keys()))
    p = np.array(list(mix.values()), dtype=float)
    return keys[rng.choice(len(keys), size=shape, p=p / p.sum())]


def teach(rng, prof: Profile, slots, targets, outc):
    sat = outc == targets
    n, m = sat.shape
    mention = rng.random(sat.shape) < np.where(sat, prof.kappa_sat, prof.kappa_viol)
    intent = np.where(sat, _choose(rng, prof.sat_mix, sat.shape), _choose(rng, prof.viol_mix, sat.shape))
    intent = np.where(mention, intent, NONE)
    second = np.where(sat, _choose(rng, prof.sat_mix, sat.shape), _choose(rng, prof.viol_mix, sat.shape))
    intent2 = np.where(mention & (rng.random(sat.shape) < prof.p_stack), second, NONE)
    ev = np.where(rng.random(n) < prof.p_eval, sat.sum(1), -1)
    # "m of m right" entails every requirement reached its target, so it reveals every outcome.
    reveals = (np.isin(intent, list(REVEALS_OUTCOME)) | np.isin(intent2, list(REVEALS_OUTCOME))
               | (ev == m)[:, None])
    revealed = np.where(reveals, outc, -1)
    # VAGUE flags one slot group that contains a violated requirement (if any).
    groups = slots % N_GROUPS
    vague = np.full(n, -1)
    has_fail = ~sat.all(1)
    pick = rng.random(n) < prof.p_vague
    for i in np.nonzero(has_fail & pick)[0]:
        vague[i] = rng.choice(groups[i][~sat[i]])
    return Message(intent, intent2, revealed, ev, vague, rng.random(n) < prof.p_chatter)


@dataclass
class GLFTConfig:
    episodes_per_round: int = 256
    stage2_per_round: int = 256
    beta: float = 0.01
    beta_unrep: float = 100.0
    beta_distill: float = 4.0
    replay_unrep: bool = False  # accumulate non-revealing signature cells over rounds (theory: current round)
    distill: str = "sample"     # "sample": MLE on sampled X+ (the paper's Stage II)
                                # "dist":   PROPOSED VARIANT, distill pi^fb's full distribution (soft counts)


class GLFT:
    def __init__(self, world: HWorld, P0, prof: Profile, cfg: GLFTConfig):
        self.w, self.P, self.prof, self.cfg = world, [p.copy() for p in P0], prof, cfg
        e = world.env
        self.ground = np.zeros((e.K, e.N, e.N))
        self.U = np.zeros((2,) + self.ground.shape)
        self.res = Resources()

    def round(self):
        w, c, e, h, prof = self.w, self.cfg, self.w.env, self.w.h, self.prof
        slots, targets = w.requests(c.episodes_per_round)
        args = w.act_steps(self.P, slots, targets)
        outc = w.outcome(slots, args)
        msg = teach(w.rng, prof, slots, targets, outc)
        self.res.env_episodes += len(slots)
        self.res.teacher_calls += len(slots)
        sel = msg.revealed >= 0
        np.add.at(self.ground, (slots[sel], msg.revealed[sel], args[sel]), 1)
        # Signature cells used under G*: index 0 = EVAL absent, 1 = EVAL == m.
        U = self.U if c.replay_unrep else np.zeros((2,) + self.ground.shape)
        ok = msg.signature_ok(slots, e.m) & (msg.revealed < 0)
        cell = np.broadcast_to((msg.eval == e.m)[:, None], ok.shape).astype(int)
        np.add.at(U, (cell[ok], slots[ok], targets[ok], args[ok]), 1)
        fb_rep = smooth_steps(self.ground, self.P, c.beta, h.b, h.k)
        fb_unrep = [smooth_steps(U[i], self.P, c.beta_unrep, h.b, h.k) for i in range(2)]
        # Stage II: ideal feedback = the teacher on an all-satisfied execution.
        s2, t2 = w.requests(c.stage2_per_round)
        star = teach(w.rng, prof, s2, t2, t2)
        self.res.env_episodes += len(s2)
        self.res.gstar_samples += len(s2)
        if c.distill == "dist":
            # Soft counts: each Stage II requirement contributes pi^fb's whole path distribution under F*.
            tables = [path_table(fb_rep), path_table(fb_unrep[0]), path_table(fb_unrep[1])]
            which = np.where(star.revealed >= 0, 0, np.where((star.eval == e.m)[:, None], 2, 1))
            D = np.zeros_like(self.ground)
            for i, tab in enumerate(tables):
                sel = which == i
                np.add.at(D, (s2[sel], t2[sel]), tab[s2[sel], t2[sel]])
            self.P = smooth_steps(D, self.P, c.beta_distill, h.b, h.k)
            return
        a_rep = w.act_steps(fb_rep, s2, t2)
        a_u0 = w.act_steps(fb_unrep[0], s2, t2)
        a_u1 = w.act_steps(fb_unrep[1], s2, t2)
        a2 = np.where(star.revealed >= 0, a_rep, np.where((star.eval == e.m)[:, None], a_u1, a_u0))
        D = np.zeros_like(self.ground)
        np.add.at(D, (s2, t2, a2), 1)
        self.P = smooth_steps(D, self.P, c.beta_distill, h.b, h.k)


class ParsePG(PolicyGradient):
    """RL whose reward is parsed from the same mixed-intent teacher message (information-matched).

    The exact fraction if EVAL is present; otherwise
      pessimistic: fraction of requirements known satisfied (unmentioned count as failed);
      optimistic:  1 - fraction known or flagged failed (unmentioned count as satisfied; a VAGUE
                   complaint counts as one failure), so silence reads as success.
    """

    def __init__(self, world, P0, prof: Profile, cfg: RLConfig, parser="pessimistic"):
        super().__init__(world, P0, cfg)
        self.prof, self.parser = prof, parser

    def _reward(self, slots, targets, outc):
        m = self.w.env.m
        msg = teach(self.w.rng, self.prof, slots, targets, outc)
        self.res.teacher_calls += len(slots)
        if self.parser == "pessimistic":
            confirmed = (msg.intent == CONFIRM) | (msg.intent2 == CONFIRM) | (msg.revealed == targets)
            r = confirmed.sum(1) / m
        else:
            fail_only = np.isin(msg.intent, FAILURE_ONLY) | np.isin(msg.intent2, FAILURE_ONLY)
            failed = fail_only | ((msg.revealed >= 0) & (msg.revealed != targets))
            r = 1 - np.minimum(m, failed.sum(1) + (msg.vague_group >= 0)) / m
        return np.where(msg.eval >= 0, msg.eval / m, r)


def run(arm, env: HEnv, prof: Profile, seed=0, budget=65536, eval_every=2048, overrides=None):
    world = HWorld(env, np.random.default_rng(seed))
    P0 = conditionals(env.flat().base_table(), env.b, env.k)
    o = dict(overrides or {})
    if arm in ("glft", "glft-dist"):
        learner = GLFT(world, P0, prof, GLFTConfig(**({"distill": "dist"} if arm == "glft-dist" else {}), **o))
    else:
        algo, reward = arm.split("-")
        assert reward == "parse"
        parser = o.pop("parser", "pessimistic")
        learner = ParsePG(world, P0, prof, RLConfig(algo=algo, reward=reward, **o), parser=parser)
    curve = []
    nxt = 0
    while True:
        if learner.res.env_episodes >= nxt:
            s, f = world.success_rate_steps(learner.P)
            curve.append({**learner.res.__dict__, "success": s, "frac": f})
            nxt += eval_every
        if learner.res.env_episodes >= budget:
            return curve
        learner.round()
