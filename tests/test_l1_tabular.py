import numpy as np
import pytest

from gft.algos.gft_exact import exact_slot_recursion
from gft.l1_tabular import GFT, Env, GFTConfig, PolicyGradient, RLConfig, World, run


def test_base_table_competence_and_normalization():
    env = Env(N=16, q=0.2, systematic=0.5)
    P0 = env.base_table()
    np.testing.assert_allclose(P0.sum(-1), 1.0)
    w = World(env, np.random.default_rng(0))
    K, N = env.K, env.N
    correct = P0[np.arange(K)[:, None], np.arange(N)[None, :], w.correct]
    np.testing.assert_allclose(correct, 0.2)
    assert (P0 > 0).all()  # full support (coverage)


@pytest.mark.parametrize("factored", [True, False])
@pytest.mark.parametrize("kappa", [0.25, 0.5, 1.0])
def test_gft_tracks_exact_recursion_with_ample_data(factored, kappa):
    env = Env(N=8, m=2, q=0.3, kappa_sat=kappa, kappa_viol=kappa)
    w = World(env, np.random.default_rng(1))
    learner = GFT(w, env.base_table(), GFTConfig(episodes_per_round=200_000, factored=factored, beta=1e-6,
                                                     beta_unrep=1e9, stage2_exact=True))
    ref = exact_slot_recursion(0.3, kappa, kappa, 3)
    for t in range(1, 4):
        learner.round()
        _, frac = w.success_rate(learner.P)
        assert frac == pytest.approx(ref[t], abs=0.01)


def test_align_reward_gives_per_requirement_advantage():
    env = Env(N=8, m=3)
    w = World(env, np.random.default_rng(2))
    pg = PolicyGradient(w, env.base_table(), RLConfig(algo="grpo", reward="align", lr=10.0))
    before = pg.P.copy()
    pg.round()
    assert not np.allclose(before, pg.P)
    np.testing.assert_allclose(pg.P.sum(-1), 1.0)


def test_descriptive_feedback_beats_pass_fail_when_base_rarely_correct():
    env = dict(N=16, m=4, q=0.02)
    o = dict(beta=0.01, beta_unrep=100, beta_distill=4)
    desc = run("gft", Env(**env), seed=0, budget=16384, overrides=o).curve[-1]["success"]
    pf = run("gft", Env(reveal=False, **env), seed=0, budget=16384, overrides=o).curve[-1]["success"]
    assert desc > 0.8 and pf < 0.2


def test_horizon_k1_matches_flat_and_base_path_law_is_depth_invariant():
    from gft.l1_horizon import HEnv, conditionals, path_prob
    from gft.l1_horizon import run as hrun
    h1 = HEnv(b=16, k=1, m=3, q=0.1)
    flat = run("grpo-frac", h1.flat(), seed=0, budget=4096, eval_every=2048, overrides=dict(lr=300)).curve
    deep = hrun("grpo-frac", h1, seed=0, budget=4096, eval_every=2048, overrides=dict(lr=300))
    np.testing.assert_allclose([p["success"] for p in flat], [p["success"] for p in deep], atol=1e-12)
    # Same N = 64 under different (b, k): the base law over whole paths is identical.
    for b, k in [(64, 1), (8, 2), (4, 3), (2, 6)]:
        h = HEnv(b=b, k=k, q=0.1)
        P = conditionals(h.flat().base_table(), b, k)
        paths = np.arange(64)[None, None, :].repeat(2, 0)
        probs = path_prob(P, np.zeros((2, 64), int) + 3, np.zeros((2, 64), int) + 5, paths[:, 0], b, k)
        np.testing.assert_allclose(probs[0], h.flat().base_table()[3, 5], atol=1e-12)


def test_intent_teacher_semantics_and_ideal_calibration():
    from gft.l1_intents import CONFIRM, CORRECT, HINT, NONE, PROFILES, teach
    rng = np.random.default_rng(0)
    env = Env(N=16, m=4)
    w = World(env, rng)
    slots, targets = w.requests(5000)
    outc = w.outcome(slots, w.act(env.base_table(), slots, targets))
    sat = outc == targets
    msg = teach(rng, PROFILES["mixed"], slots, targets, outc)
    # Truthful: revealed outcomes are the achieved ones; failure-only intents never on satisfied reqs.
    sel = msg.revealed >= 0
    assert (msg.revealed[sel] == outc[sel]).all()
    for it in (CORRECT, HINT):
        assert not ((msg.intent == it) & sat).any() and not ((msg.intent2 == it) & sat).any()
    assert ((msg.intent2 != NONE) & (msg.intent != NONE)).sum() > 0      # stacked intents occur
    assert (msg.eval[msg.eval >= 0] == sat.sum(1)[msg.eval >= 0]).all()
    # Ideal feedback (all satisfied): no failure intents, EVAL == m, no VAGUE.
    star = teach(rng, PROFILES["mixed"], slots, targets, targets)
    assert not np.isin(star.intent, [CORRECT, HINT]).any()
    assert ((star.eval == -1) | (star.eval == 4)).all() and (star.vague_group == -1).all()
    assert ((star.intent == CONFIRM) | (star.intent == NONE) | (star.revealed == targets)).all()


def test_path_table_and_optimistic_parser():
    from gft.l1_horizon import HEnv, conditionals
    from gft.l1_intents import PROFILES, ParsePG, path_table
    from gft.l1_horizon import HWorld, RLConfig
    h = HEnv(b=4, k=3, q=0.1)
    base = h.flat().base_table()
    np.testing.assert_allclose(path_table(conditionals(base, 4, 3)), base, atol=1e-12)
    # Vague teacher: optimistic parser gives 1 exactly on successes, < 1 on failures.
    w = HWorld(h, np.random.default_rng(0))
    pg = ParsePG(w, conditionals(base, 4, 3), PROFILES["vague"], RLConfig(algo="grpo", reward="parse"),
                 parser="optimistic")
    slots, targets = w.requests(2000)
    outc = np.where(w.rng.random(targets.shape) < 0.7, targets, (targets + 1) % h.N)
    r = pg._reward(slots, targets, outc)
    ok = (outc == targets).all(1)
    assert (r[ok] == 1).all() and (r[~ok] < 1).all()
