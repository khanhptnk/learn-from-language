import re
import numpy as np

from gft.minicraft.core import (BRIDGE, CRAFT0, GEM, MiniCraft, N_WORLD_ACTIONS, PASSABLE, RECIPE_FOR, ROCK, STONE, TREE,
                                USE, WOOD, WOOD_PICKAXE, WorldConfig, generate_world, reachable)
from gft.minicraft.expert import expert_action
from gft.minicraft.goals import GOALS, sample_requests, requirement_status
from gft.minicraft.teacher import PROFILES, episode_facts, feedback

CFG = WorldConfig()


def run(req, policy, cfg=CFG):
    env = MiniCraft(cfg, req.seeds)
    log = []
    for _ in range(cfg.max_steps):
        sat, mask = requirement_status(env, req)
        env.done |= (sat | ~mask).all(1)
        if env.done.all():
            break
        log.append(env.step(policy(env)))
    sat, mask = requirement_status(env, req)
    return env, log, sat, mask


def test_generation_is_deterministic_and_reachable():
    for s in range(30):
        g1, p1, f1 = generate_world(CFG, np.random.default_rng(s))
        g2, p2, f2 = generate_world(CFG, np.random.default_rng(s))
        assert np.array_equal(g1, g2) and np.array_equal(p1, p2) and f1 == f2
        reach = reachable(g1, p1)
        assert reach[1:-1, CFG.river_col - 1].any()
        assert (g1 == GEM).sum() == 1 and np.argwhere(g1 == GEM)[0, 1] >= CFG.river_col + 2


def test_mechanics_tool_requirements_and_crafting():
    env = MiniCraft(CFG, [0])
    env.grid[0, 1:-1, 1:CFG.river_col] = 0                 # clear the west side
    env.pos[0] = (5, 2)
    env.grid[0, 5, 3] = ROCK
    env.face[0] = 1
    ev = env.step([USE])                                    # rock without a pickaxe fails
    assert ev["tool"][0] == WOOD_PICKAXE and env.inv[0, STONE] == 0
    env.grid[0, 4, 2] = TREE
    env.step([0]); env.step([USE])                         # face up (blocked by the tree), chop it
    assert env.inv[0, WOOD] == 1
    ev = env.step([CRAFT0 + RECIPE_FOR[1]])                 # plank without a workbench fails
    assert ev["station"][0] >= 0 and env.inv[0, WOOD] == 1


def test_expert_solves_random_requests_deterministically():
    rng = np.random.default_rng(0)
    req = sample_requests(rng, 60, np.arange(60))
    env, _, sat, mask = run(req, lambda e: np.array([expert_action(e, b, req.goals[b]) for b in range(60)]))
    assert (sat | ~mask).all()
    env2, _, _, _ = run(req, lambda e: np.array([expert_action(e, b, req.goals[b]) for b in range(60)]))
    assert np.array_equal(env.t, env2.t)


def test_teacher_feedback_is_truthful():
    rng = np.random.default_rng(1)
    req = sample_requests(rng, 30, np.arange(100, 130))
    pol = lambda e: np.array([expert_action(e, b, req.goals[b]) if rng.random() < 0.6 else int(rng.integers(N_WORLD_ACTIONS))
                              for b in range(30)])
    env, log, sat, mask = run(req, pol)
    for b in range(30):
        f = episode_facts(log, b, env, req.goals[b], sat[b])
        assert sum(f.collected.values()) == sum((ev["type"][b] == 3) for ev in log)
        assert f.crossed == (env.pos[b, 1] > CFG.river_col)
        for p in PROFILES:
            msg = feedback(f, p, rng)
            assert isinstance(msg, str) and msg
            for g, ok in zip(f.goals, f.satisfied):          # confirm statements match the outcome
                assert (f"did not {GOALS[g].phrase}" in msg or f"Not done: {GOALS[g].phrase}" in msg) <= (not ok)


def test_env_speed():
    import time
    env = MiniCraft(CFG, np.arange(512))
    rng = np.random.default_rng(0)
    t = time.time()
    for _ in range(50):
        env.step(rng.integers(N_WORLD_ACTIONS, size=512))
    assert 512 * 50 / (time.time() - t) > 50_000           # >50k steps/s even on a slow CPU


# ------------------------------------------------------------------ feedback accuracy

from gft.minicraft.learn import run_episodes                     # noqa: E402
from gft.minicraft.verify import replay, verify_message          # noqa: E402


def _episodes(n, p_expert, seed):
    rng = np.random.default_rng(seed)
    req = sample_requests(rng, n, rng.integers(0, 2 ** 31, size=n))
    env, log, _, acts, valid, sat, _ = run_episodes(CFG, req, lambda e, t: np.array(
        [expert_action(e, b, req.goals[b]) if (not e.done[b] and rng.random() < p_expert) else int(rng.integers(N_WORLD_ACTIONS))
         for b in range(n)]), record=False)
    out = []
    for b in range(n):
        f = episode_facts(log, b, env, req.goals[b], sat[b])
        out.append((f, replay(CFG, int(req.seeds[b]), req.goals[b], acts[b][valid[b]])))
    return out, rng


def test_every_feedback_sentence_is_consistent_with_the_replayed_trajectory():
    for p in (1.0, 0.6, 0.3, 0.0):
        eps, rng = _episodes(40, p, seed=int(p * 10))
        for f, tr in eps:
            for prof in PROFILES:
                for s, ok, why in verify_message(feedback(f, prof, rng), tr):
                    assert ok, (s, why)


def _swap_item(msg, tr):
    """Replace the item of the first count sentence with an item whose true count differs (always false)."""
    from gft.minicraft.core import ITEM_NAMES
    m = re.search(r"(collected|picked up) (\d+) ([a-z ]+?)\.", msg)
    if not m:
        return msg
    n = int(m.group(2))
    other = next(i for i in range(len(ITEM_NAMES)) if tr.collected[i] != n and ITEM_NAMES[i] in
                 ("wood", "stone", "ore", "iron"))
    return msg[:m.start(3)] + ITEM_NAMES[other] + msg[m.end(3):]


def test_verifier_catches_corrupted_feedback():
    eps, rng = _episodes(40, 0.5, seed=7)
    caught = total = 0
    for f, tr in eps:
        msg = feedback(f, "mixed", rng)
        corrupted = [
            re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1), msg, count=1),         # wrong number
            msg.replace("Not done:", "Done:", 1) if "Not done:" in msg else msg.replace("Done:", "Not done:", 1),
            _swap_item(msg, tr),                                                              # wrong item
            msg + " You collected 99 gems.",                                                  # fabricated claim
            msg + " You flew over the river.",                                                # unparseable
        ]
        for c in corrupted:
            if c != msg:
                total += 1
                caught += not all(ok for _, ok, _ in verify_message(c, tr))
    assert total > 100 and caught == total


# ------------------------------------------------------------------ optimized code == reference code

def test_vectorized_requirements_match_reference():
    from gft.minicraft.goals import requirement_status_reference
    for p, seed in ((1.0, 0), (0.5, 1), (0.0, 2)):
        rng = np.random.default_rng(seed)
        req = sample_requests(rng, 60, rng.integers(0, 2 ** 31, size=60))
        env = MiniCraft(CFG, req.seeds)
        log = []
        for _ in range(CFG.max_steps):
            s1, m1 = requirement_status(env, req)
            s2, m2 = requirement_status_reference(env, req)
            assert np.array_equal(s1, s2) and np.array_equal(m1, m2)
            env.done |= (s1 | ~m1).all(1)
            if env.done.all():
                break
            log.append(env.step(np.array([expert_action(env, b, req.goals[b]) if rng.random() < p
                                          else int(rng.integers(N_WORLD_ACTIONS)) for b in range(60)])))


def test_fast_expert_matches_reference():
    """The list-based BFS expert returns exactly the original expert's action on states from noisy rollouts."""
    from gft.minicraft.expert import expert_action_reference
    n = 0
    for p, seed in ((1.0, 3), (0.7, 4), (0.3, 5)):
        rng = np.random.default_rng(seed)
        req = sample_requests(rng, 80, rng.integers(0, 2 ** 31, size=80))
        env = MiniCraft(CFG, req.seeds)
        for _ in range(CFG.max_steps):
            sat, mask = requirement_status(env, req)
            env.done |= (sat | ~mask).all(1)
            if env.done.all():
                break
            fast = [expert_action(env, b, req.goals[b]) for b in range(80)]
            assert fast == [expert_action_reference(env, b, req.goals[b]) for b in range(80)]
            n += 80
            env.step(np.array([a if rng.random() < p else int(rng.integers(N_WORLD_ACTIONS)) for a in fast]))
    assert n > 20_000


def test_expert_pool_matches_expert_action():
    from gft.minicraft.expert import ExpertPool
    rng = np.random.default_rng(6)
    req = sample_requests(rng, 50, rng.integers(0, 2 ** 31, size=50))
    env = MiniCraft(CFG, req.seeds)
    pool = ExpertPool(workers=3)
    try:
        for _ in range(40):
            live = np.nonzero(~env.done)[0]
            got = pool(env, req.goals, live)
            assert got[live].tolist() == [expert_action(env, b, req.goals[b]) for b in live]
            env.step(np.where(rng.random(50) < 0.7, got, rng.integers(N_WORLD_ACTIONS, size=50)))
    finally:
        pool.close()


def test_done_action_ends_episode_and_success_is_checked_then():
    from gft.minicraft.core import DONE, EV_DONE
    from gft.minicraft.learn import run_episodes
    rng = np.random.default_rng(7)
    req = sample_requests(rng, 30, rng.integers(0, 2 ** 31, size=30))
    env, log, _, acts, valid, sat, mask = run_episodes(
        CFG, req, lambda e, t: np.array([expert_action(e, b, req.goals[b]) for b in range(30)]), record=False)
    assert (sat | ~mask).all(1).all()                                   # the expert completes, then declares done
    last = valid.sum(1) - 1
    assert (acts[np.arange(30), last] == DONE).all()
    assert all(log[last[b]]["type"][b] == EV_DONE for b in range(30))
    # without DONE the episode runs to the step budget even when requirements already hold
    env, _, _, acts, valid, sat, mask = run_episodes(CFG, req, lambda e, t: np.zeros(30, int), record=False)
    assert valid.sum(1).min() == CFG.max_steps and not (acts == DONE).any()
