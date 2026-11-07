import numpy as np

from gft.minicraft.core import (BRICK, CLAY, CRAFT, KILN, LOOM, PICKAXE, PLANK, PUT, RAFT, ROCK, ROPE, SELECT,
                                SITE, STATIONS, STONE, TAKE, TREE, USE, WOOD, WORKBENCH, MiniCraft, N_ITEMS,
                                N_WORLD_ACTIONS, PASSABLE, WorldConfig, craftable, generate_world, reachable)
from gft.minicraft.expert import expert_action
from gft.minicraft.goals import GOALS, GOAL_ID, sample_requests, requirement_status
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
    g1, p1, f1 = generate_world(CFG, np.random.default_rng(7))
    g2, p2, f2 = generate_world(CFG, np.random.default_rng(7))
    assert np.array_equal(g1, g2) and (p1 == p2).all() and f1 == f2
    reach = reachable(g1, p1)
    objects = np.argwhere(~PASSABLE[g1] & (g1 != 1))
    assert len(objects) == len(STATIONS) + CFG.n_trees + CFG.n_reeds + CFG.n_clay + CFG.n_rocks
    assert all(any(reach[o[0] + dr, o[1] + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))) for o in objects)
    # no cell touches two stations, so a craft is unambiguous about where it happens
    nbrs = [{(o[0] + dr, o[1] + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
            for o in np.argwhere(np.isin(g1, STATIONS))]
    assert all(not (a & b) for i, a in enumerate(nbrs) for b in nbrs[i + 1:])


def test_mechanics_tool_requirements_and_crafting():
    env = MiniCraft(CFG, [0])
    env.grid[0, 1:-1, 1:-1] = 0                             # clear the map
    env.pos[0] = (5, 2)
    env.grid[0, 5, 3] = ROCK
    env.face[0] = 1
    ev = env.step([USE])                                    # rock without a pickaxe fails
    assert ev["tool"][0] == PICKAXE and env.inv[0, STONE] == 0
    env.grid[0, 4, 2] = TREE
    env.step([0]); env.step([USE])                         # face up (blocked by the tree), chop it
    assert env.inv[0, WOOD] == 1 and env.cursor[0] == -1
    ev = env.step([PUT])                                    # nothing selected, no station: put fails
    assert ev["type"][0] == 11 and env.inv[0, WOOD] == 1
    ev = env.step([SELECT])                                 # the cursor moves to the only kind held
    assert ev["type"][0] == 9 and env.cursor[0] == WOOD
    ev = env.step([CRAFT])                                  # no station around
    assert ev["type"][0] == 7 and ev["station"][0] == -1 and env.inv[0, WOOD] == 1
    env.grid[0, 6, 2] = KILN
    env.step([PUT])                                         # wood into the kiln's slot; crafting it makes nothing
    assert env.inv[0, WOOD] == 0 and env.slot[0, STATIONS.index(KILN), WOOD] == 1
    ev = env.step([CRAFT])
    assert ev["type"][0] == 7 and ev["station"][0] == KILN and env.slot[0, STATIONS.index(KILN), WOOD] == 1
    ev = env.step([TAKE])                                   # take it back
    assert ev["type"][0] == 12 and ev["count"][0] == 1 and env.inv[0, WOOD] == 1 and env.slot[0].sum() == 0
    env.grid[0, 6, 2] = WORKBENCH
    env.step([PUT]); ev = env.step([CRAFT])                 # wood at a workbench: a plank
    assert ev["type"][0] == 6 and env.inv[0, PLANK] == 1 and env.inv[0, WOOD] == 0 and env.slot[0].sum() == 0
    env.inv[0, PLANK], env.inv[0, ROPE] = 2, 1
    env.cursor[0] = PLANK
    env.step([PUT]); env.step([PUT]); env.step([SELECT])    # two planks in, then the cursor moves on to the rope
    assert env.cursor[0] == ROPE
    env.step([PUT])                                         # planks + planks + rope: no recipe takes exactly that
    assert env.step([CRAFT])["type"][0] == 7
    env.step([TAKE]); env.cursor[0] = PLANK
    env.step([PUT]); env.step([SELECT]); env.step([PUT]); env.step([CRAFT])   # plank + rope: the raft
    assert env.inv[0, RAFT] == 1 and env.inv[0, PLANK] == 1 and env.inv[0, ROPE] == 0
    env.inv[0, PLANK] = 2
    env.cursor[0] = PLANK
    env.step([PUT]); env.step([PUT]); env.step([CRAFT])     # exactly two planks: the pickaxe
    assert env.inv[0, PICKAXE] == 1 and env.inv[0, PLANK] == 0
    env.face[0] = 1
    env.step([USE])                                         # now the rock gives stone
    assert env.inv[0, STONE] == 1 and env.grid[0, 5, 3] == 0
    st = env.agent_state()
    assert st.shape == (1, 3 * N_ITEMS) and st[0, STONE] == 1 and st[0, N_ITEMS + PLANK] == 1 and st[0, 2 * N_ITEMS:].sum() == 0


def test_craftable_needs_exactly_the_recipe_inputs():
    slot = np.zeros(N_ITEMS, int)
    slot[[PLANK, BRICK]] = (1, 1)
    assert craftable(slot, KILN) == 6                       # oven: plank + brick at the kiln
    slot[CLAY] = 1
    assert craftable(slot, KILN) == -1                      # anything extra: nothing
    slot[:] = 0
    slot[CLAY] = 1
    assert craftable(slot, KILN) == 2 and craftable(slot, SITE) == -1 and craftable(slot, LOOM) == -1
    slot[CLAY] = 2
    assert craftable(slot, KILN) == -1                      # counts must match too


def test_expert_solves_random_requests_deterministically():
    rng = np.random.default_rng(0)
    req = sample_requests(rng, 60, np.arange(60))
    env, _, sat, mask = run(req, lambda e: np.array([expert_action(e, b, req.goals[b]) for b in range(60)]))
    assert (sat | ~mask).all()
    env2, _, _, _ = run(req, lambda e: np.array([expert_action(e, b, req.goals[b]) for b in range(60)]))
    assert np.array_equal(env.t, env2.t)


def test_teacher_feedback_is_truthful():
    from gft.minicraft.verify import replay, verify_message
    rng = np.random.default_rng(1)
    n = 40
    req = sample_requests(rng, n, np.arange(n))
    p = np.array([0.0, 0.3, 0.7, 0.9])[rng.integers(4, size=n)]
    acts = []

    def policy(e):
        a = np.array([expert_action(e, b, req.goals[b]) if rng.random() < p[b] else int(rng.integers(N_WORLD_ACTIONS))
                      for b in range(n)])
        acts.append(a)
        return a
    env, log, sat, mask = run(req, policy)
    acts = np.array(acts).T
    for b in range(n):
        f = episode_facts(log, b, env, req.goals[b], sat[b])
        tr = replay(CFG, req.seeds[b], req.goals[b], acts[b][:env.t[b]])
        for prof in PROFILES:
            msg = feedback(f, prof, rng)
            bad = [(s, why) for s, ok, why in verify_message(msg, tr) if not ok]
            assert not bad, (req.text(b), msg, bad)


def test_env_speed():
    import time
    rng = np.random.default_rng(0)
    B = 256
    env = MiniCraft(CFG, np.arange(B))
    t0 = time.time()
    for _ in range(100):
        env.step(rng.integers(0, N_WORLD_ACTIONS, size=B))
    assert (time.time() - t0) / (100 * B) < 2e-5 * 20        # generous: ~100k steps / s


def test_realistic_feedback_reads_true_in_plain_english():
    """The realistic teacher never says "You did not G." of a goal it achieved and used up, never "You should have
    gotten X first." when it got X, never the same sentence twice, and never a plural for a one-item goal."""
    from gft.minicraft.teacher import achieved, compose
    from gft.minicraft.verify import parse, split_sentences
    rng = np.random.default_rng(2)
    n = 96
    req = sample_requests(rng, n, np.arange(100, 100 + n))
    p = np.array([0.1, 0.3, 0.6, 0.9])[rng.integers(4, size=n)]
    env, log, sat, mask = run(req, lambda e: np.array([expert_action(e, b, req.goals[b]) if rng.random() < p[b]
                                                       else int(rng.integers(N_WORLD_ACTIONS)) for b in range(n)]))
    for b in range(n):
        f = episode_facts(log, b, env, req.goals[b], sat[b])
        text = " ".join(x[2] for x in compose(f, "realistic", rng))
        sents = split_sentences(text)
        assert len(set(sents)) == len(sents), text
        for s in sents:
            q = parse(s)
            assert q is not None, s
            k, c = q
            if k == "goal_not_done" and s.startswith("You did not"):
                assert not achieved(f, c[1]), text
            if k == "needed" and s.startswith("You should have gotten"):
                assert f.collected[c[1]] + f.crafted[c[1]] == 0, text


def test_verifier_catches_corrupted_feedback():
    from gft.minicraft.verify import replay, verify_message
    rng = np.random.default_rng(3)
    req = sample_requests(rng, 1, np.array([5]), goal_pool=[GOAL_ID["make planks"]])
    acts = []

    def policy(e):
        a = np.array([expert_action(e, 0, req.goals[0])])
        acts.append(a)
        return a
    env, log, sat, mask = run(req, policy)
    tr = replay(CFG, 5, req.goals[0], np.array(acts)[:, 0])
    assert all(ok for _, ok, _ in verify_message("Great job! You did make planks.", tr))
    assert not all(ok for _, ok, _ in verify_message("Not done: make planks.", tr))
    assert not all(ok for _, ok, _ in verify_message("You collected 7 clay.", tr))
    assert not all(ok for _, ok, _ in verify_message("Something else entirely.", tr))


def test_vectorized_requirements_match_reference():
    from gft.minicraft.goals import requirement_status_reference
    rng = np.random.default_rng(0)
    req = sample_requests(rng, 50, np.arange(50))
    env = MiniCraft(CFG, req.seeds)
    env.inv[:] = rng.integers(0, 3, size=env.inv.shape)
    a, m = requirement_status(env, req)
    b, m2 = requirement_status_reference(env, req)
    assert np.array_equal(a, b) and np.array_equal(m, m2)


def test_fast_expert_matches_reference():
    from gft.minicraft.expert import expert_action_reference
    rng = np.random.default_rng(4)
    n = 32
    req = sample_requests(rng, n, np.arange(200, 200 + n))
    env = MiniCraft(CFG, req.seeds)
    for _ in range(80):
        a = np.array([expert_action(env, b, req.goals[b]) for b in range(n)])
        r = np.array([expert_action_reference(env, b, req.goals[b]) for b in range(n)])
        assert np.array_equal(a, r)
        env.step(a)


def test_expert_pool_matches_expert_action():
    from gft.minicraft.expert import ExpertPool
    rng = np.random.default_rng(5)
    n = 24
    req = sample_requests(rng, n, np.arange(300, 300 + n))
    env = MiniCraft(CFG, req.seeds)
    pool = ExpertPool(workers=2)
    try:
        for _ in range(40):
            a = np.array([expert_action(env, b, req.goals[b]) for b in range(n)])
            assert np.array_equal(pool(env, req.goals), a)
            env.step(a)
    finally:
        pool.close()


def test_done_action_ends_episode_and_success_is_checked_then():
    from gft.minicraft.core import DONE
    from gft.minicraft.learn import run_episodes
    rng = np.random.default_rng(0)
    req = sample_requests(rng, 4, np.arange(4), goal_pool=[GOAL_ID["gather wood"]])
    env, log, _, acts, valid, sat, mask = run_episodes(CFG, req, lambda e, t: np.full(4, DONE))
    assert (env.t == 1).all() and env.done.all() and not sat.any() and valid.shape == (4, 1)


def test_goal_depths_match_the_expert_plan():
    from gft.minicraft.expert import plan
    from gft.minicraft.core import N_ITEMS
    for i, g in enumerate(GOALS):
        assert g.depth == len(plan(np.zeros(N_ITEMS, int), [i])), g.phrase


def test_failed_craft_narration_names_what_the_slot_held():
    """craft_held claims come from the station slot replayed from the events (put adds, take / craft empty it)."""
    from gft.minicraft.core import EV_CRAFT_FAIL
    from gft.minicraft.teacher import event_arrays, slot_steps
    env = MiniCraft(CFG, [3])
    acts = []
    def step(a):
        acts.append(a)
        return env.step([a])
    env.grid[0, 1:-1, 1:-1] = 0
    env.pos[0], env.face[0] = (5, 5), 1
    env.grid[0, 6, 5] = WORKBENCH
    env.inv[0, [PLANK, STONE]] = (1, 1)
    log = [step(SELECT), step(PUT), step(SELECT), step(PUT), step(CRAFT), step(TAKE), step(CRAFT)]
    ev = event_arrays(log)
    slot = slot_steps(ev["type"][:, 0], ev["item"][:, 0], ev["station"][:, 0])
    assert slot[4, PLANK] == 1 and slot[4, STONE] == 1 and slot[6].sum() == 0
    assert ev["type"][4, 0] == EV_CRAFT_FAIL                # plank + stone makes nothing at a workbench
    f = episode_facts(log, 0, env, np.array([-1, -1, -1]), np.array([False, False, False]))
    held = [c for c, _, _ in f.events if c[0] == "craft_held"]
    assert held == [("craft_held", WORKBENCH, ((STONE, 1), (PLANK, 1)))]


def test_expert_takes_back_goal_items_and_tools_left_in_station_slots():
    """Items in a slot count as held for crafting, but a goal's items and the tools must be in hand: the expert takes
    them back before collecting or declaring done (it used to say done early, or hit a rock forever)."""
    from gft.minicraft.core import DONE
    from gft.minicraft.expert import expert_action_reference
    def run_goal(goal, setup, steps=150):
        env = MiniCraft(CFG, [0])
        setup(env)
        for _ in range(steps):
            a = expert_action(env, 0, [GOAL_ID[goal]])
            assert a == expert_action_reference(env, 0, [GOAL_ID[goal]])
            env.step([a])
            if a == DONE:
                break
        return env
    def wood_in_slot(env):
        env.inv[0, WOOD] = 1
        env.slot[0, STATIONS.index(WORKBENCH), WOOD] = 1
    env = run_goal("gather wood", wood_in_slot)
    assert env.done[0] and env.inv[0, WOOD] >= 2
    def pickaxe_in_slot(env):
        env.slot[0, 0, PICKAXE] = 1
    env = run_goal("quarry stone", pickaxe_in_slot)
    assert env.inv[0, STONE] >= 1 and env.inv[0, PICKAXE] == 1


def test_success_needs_done_counts_only_episodes_ended_by_done():
    from gft.minicraft.core import DONE
    from gft.minicraft.learn import run_episodes
    from gft.minicraft.verify import replay
    cfg = WorldConfig(success_needs_done=True, max_steps=40)
    rng = np.random.default_rng(0)
    req = sample_requests(rng, 4, np.arange(4), goal_pool=[GOAL_ID["gather wood"]])
    def expert_no_done(e, t):                                # the expert, but never pressing done
        a = np.array([expert_action(e, b, req.goals[b]) for b in range(4)])
        return np.where(a == DONE, 0, a)
    env, _, _, acts, _, sat, mask = run_episodes(cfg, req, expert_no_done)
    assert (env.inv[:, WOOD] >= 2).all() and not sat[mask].any()          # held, but never said done: not met
    assert not any(replay(cfg, req.seeds[b], req.goals[b], acts[b]).satisfied[0] for b in range(4))
    env, _, _, acts, valid, sat, mask = run_episodes(cfg, req, lambda e, t: np.array(
        [expert_action(e, b, req.goals[b]) for b in range(4)]))
    assert sat[mask].all()
    assert all(replay(cfg, req.seeds[b], req.goals[b], acts[b][valid[b]]).satisfied[0] for b in range(4))


def test_teacher_names_a_missing_done_and_a_short_count_and_both_verify():
    """Success needing DONE: an expert that never presses it is told the goal was ready but not declared done; an
    expert stopped after its first plank (make planks takes 2) is told how many it had. Every sentence verifies."""
    from gft.minicraft.core import DONE, UP
    from gft.minicraft.learn import run_episodes
    from gft.minicraft.teacher import compose, episode_facts_all
    from gft.minicraft.verify import parse, replay, split_sentences, verify_message
    cfg = WorldConfig(success_needs_done=True, max_steps=60)
    rng = np.random.default_rng(0)
    for goal, stop in (("gather wood", None), ("make planks", PLANK)):
        req = sample_requests(rng, 8, np.arange(8), goal_pool=[GOAL_ID[goal]])
        def act(e, t):
            a = np.array([expert_action(e, b, req.goals[b]) for b in range(len(req))])
            a = np.where(a == DONE, UP, a)                                   # never says done
            if stop is not None:                                             # ... and stops after one plank
                a = np.where(e.inv[:, stop] >= 1, UP, a)
            return a
        env, log, _, acts, valid, sat, mask = run_episodes(cfg, req, act)
        facts = episode_facts_all(log, env, req.goals, sat)
        kind = "no_done" if stop is None else "short"
        for b in range(len(req)):
            assert not sat[b, 0]
            msg = " ".join(x[2] for x in compose(facts[b], "realistic", rng))
            kinds = {parse(s)[0] for s in split_sentences(msg)}
            assert kind in kinds, msg
            tr = replay(cfg, req.seeds[b], req.goals[b], acts[b][valid[b]])
            assert all(ok for _, ok, _ in verify_message(msg, tr)), verify_message(msg, tr)
            if stop is not None:
                assert "You had 1 of the 2 planks needed to make planks." in msg or \
                    "You needed 2 planks to make planks but had only 1." in msg


def test_teacher_says_how_a_goal_item_is_made_never_that_it_was_still_needed_and_praise_names_the_goal():
    """Realistic teacher: a failed craft goal is never told "you still needed rope" (that restates the goal) but how
    rope is made; a success always names its (deepest) goal."""
    from gft.minicraft.learn import run_episodes
    from gft.minicraft.teacher import compose, episode_facts_all
    from gft.minicraft.verify import parse, split_sentences
    cfg = WorldConfig(success_needs_done=True, max_steps=60)
    rng = np.random.default_rng(0)
    for goal in ("make rope", "make planks", "make bricks"):
        g, item = GOAL_ID[goal], GOALS[GOAL_ID[goal]].item
        req = sample_requests(rng, 64, np.arange(64), goal_pool=[g])
        p = np.where(np.arange(64) < 16, 1.0, 0.3)                        # experts and noisy experts
        env, log, _, _, _, sat, _ = run_episodes(cfg, req, lambda e, t: np.array(
            [expert_action(e, b, req.goals[b]) if rng.random() < p[b] else int(rng.integers(N_WORLD_ACTIONS))
             for b in range(64)]))
        facts = episode_facts_all(log, env, req.goals, sat)
        hints = 0
        for b in range(64):
            for _ in range(4):
                cl = [parse(s)[1] for s in split_sentences(" ".join(x[2] for x in compose(facts[b], "realistic", rng)))]
                assert ("needed", item) not in cl
                hints += ("hint_recipe", item) in cl
                if sat[b, 0]:
                    assert ("goal_done", g) in cl
        assert sat[:16, 0].all() and hints > 0


def test_verdict_teacher_says_only_praise_and_verdicts_and_success_reads_from_them():
    """The language ablation's teacher: the realistic teacher's outcome sentences only -- a success always has praise
    and a "done" verdict, a failure always a "not done" verdict, and nothing else is said."""
    from gft.minicraft.teacher import VERDICT_KINDS, compose
    from gft.minicraft.verify import parse
    rng = np.random.default_rng(2)
    n = 60
    req = sample_requests(rng, n, np.arange(n))
    p = np.array([0.0, 0.5, 1.0])[rng.integers(3, size=n)]
    env, log, sat, mask = run(req, lambda e: np.array([expert_action(e, b, req.goals[b]) if rng.random() < p[b]
                                                       else int(rng.integers(N_WORLD_ACTIONS)) for b in range(n)]))
    ok = (sat | ~mask).all(1)
    assert ok.any() and not ok.all()
    for b in range(n):
        parts = compose(episode_facts(log, b, env, req.goals[b], sat[b]), "verdict", rng)
        kinds = [parse(s)[0] for _, _, s in parts]
        assert set(kinds) <= set(VERDICT_KINDS), kinds
        assert ("goal_not_done" not in kinds) == ok[b] and ("eval_all" in kinds) == ok[b]
        assert ok[b] or kinds.count("goal_not_done") == int((~sat[b] & mask[b]).sum())
