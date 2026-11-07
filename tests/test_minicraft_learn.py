import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.goals import GOALS, N_GOALS
from gft.minicraft.gft_message import GFT, GFTConfig
from gft.minicraft.learn import (CleanPolicy, Curriculum, CurriculumConfig, TaskPool, build_pool,
                                 claim_type, obs_tensor, snapshot)
from gft.minicraft.core import N_ACTIONS, MiniCraft
from gft.minicraft.teacher import compose, episode_facts

CFG = WorldConfig(max_steps=25)


def _pool(goals):
    goals = np.array(goals)
    return TaskPool(np.arange(len(goals)), goals, [[None]] * len(goals))


def test_curriculum_samples_unlocked_tasks_and_unlocks_on_success():
    pool = _pool([[g, -1, -1] for g in range(N_GOALS)] * 10 + [[5, 2, -1]] * 10)
    cur = Curriculum(pool, CurriculumConfig(start=4, min_count=20, threshold=0.5))
    rng = np.random.default_rng(0)
    idx = cur.sample(rng, 5000)
    deepest = pool.goals[idx].max(1)
    assert deepest.max() == 3
    assert 0.7 < (deepest == 3).mean() < 0.8                  # frontier share plus frontier tasks drawn as replay
    goals = np.array([[3, -1, -1]] * 30)
    mask = goals >= 0
    cur.update(goals, np.zeros_like(mask), mask, 100)          # failures: stays locked
    assert cur.unlocked == 4
    cur.update(goals, mask.copy(), mask, 200)                  # 30/60 successes = threshold
    assert cur.unlocked == 5 and cur.unlock_log == [(200, 4)]
    assert pool.goals[cur.sample(rng, 2000)].max() == 4
    off = Curriculum(pool, CurriculumConfig(enabled=False))
    assert pool.goals[off.sample(rng, 2000)].max() == N_GOALS - 1


def test_encoder_is_size_independent():
    torch.manual_seed(0)
    for size, arch in ((11, {}), (13, {}), (17, {}), (17, dict(rec=12))):
        model = CleanPolicy(None, **arch)
        env = MiniCraft(WorldConfig(size=size), np.arange(4))
        grid, pf, inv = snapshot(env)
        x, iv = obs_tensor(grid, pf, inv, "cpu")
        logits, v = model(x, iv, None)
        assert logits.shape == (4, N_ACTIONS) and v.shape == (4,)


def test_encoder_crop_is_centred_on_the_agent():
    torch.manual_seed(0)
    enc = CleanPolicy(None).enc
    env = MiniCraft(CFG, np.arange(3))
    env.pos[1] = (0, 0)                                        # crop must zero-pad at the map edge
    grid, pf, inv = snapshot(env)
    x, _ = obs_tensor(grid, pf, inv, "cpu")
    f = enc.features(x)
    loc = enc.local(f, x)
    r = enc.crop // 2
    for b in range(3):
        pr, pc = int(pf[b, 0]), int(pf[b, 1])
        for i in range(enc.crop):
            for j in range(enc.crop):
                rr, cc = pr + i - r, pc + j - r
                inside = 0 <= rr < CFG.size and 0 <= cc < CFG.size
                want = f[b, :, rr, cc] if inside else torch.zeros(f.shape[1])
                assert torch.equal(loc[b, i, j], want)


def test_claim_type_drops_counts_only():
    assert claim_type(("collected", 1, 5)) == ("collected", 1)
    assert claim_type(("nothing_fail", 8, 3)) == ("nothing_fail", 8) and claim_type(("station_fail", 4)) == ("station_fail",)
    assert claim_type(("steps", 40)) == ("steps",)
    assert claim_type(("eval_count", 1, 2)) == ("eval_count",)


def test_episode_batches_cover_each_step_once_with_few_episodes():
    pool = build_pool(CFG, 4, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(minibatch=64, episode_chunk=8))
    b = np.repeat(np.arange(40), np.random.default_rng(0).integers(1, 30, size=40))
    batches = list(g._episode_batches(b, epochs=2))
    steps = np.concatenate(batches)
    assert sorted(steps.tolist()) == sorted(list(range(len(b))) * 2)          # each step once per pass
    assert all(len(np.unique(b[m])) <= 64 // 8 for m in batches)               # at most 8 episodes per minibatch


def test_graph_curriculum_opens_goals_after_prerequisites_are_mastered():
    from gft.minicraft.learn import goal_prerequisites
    pre = goal_prerequisites()
    from gft.minicraft.goals import GOAL_ID
    wall, bricks, stone, pick = (GOAL_ID[k] for k in ("build a wall", "make bricks", "quarry stone", "make a pickaxe"))
    assert pre[0] == set() and pre[wall] >= {bricks, stone, pick} and wall not in pre[bricks] | pre[stone]
    pool = _pool([[g, -1, -1] for g in range(N_GOALS)] * 10 + [[wall, 0, -1]] * 5)
    cur = Curriculum(pool, CurriculumConfig(mode="graph", mastery=0.7, min_count=20))
    rng = np.random.default_rng(0)
    raw = {GOAL_ID[k] for k in ("gather wood", "gather fiber", "dig clay")}      # no prerequisites: open at once
    assert set(pool.goals[cur.sample(rng, 500)].ravel()) <= {-1} | raw
    ok = np.ones((30, 3), bool)
    cur.update(np.array([[0, -1, -1]] * 30), ok, np.array([[True, False, False]] * 30), 30, feedbacks=30)
    planks = GOAL_ID["make planks"]                                             # needs wood only: opens next
    assert cur.mastered == {0} and set(pool.goals[cur.sample(rng, 500)].ravel()) <= {-1, planks} | raw


def test_pixel_policy_renders_and_runs_on_any_size():
    from gft.minicraft.render import TILE, render
    torch.manual_seed(0)
    for size in (11, 13, 17):
        env = MiniCraft(WorldConfig(size=size), np.arange(3))
        x, iv = obs_tensor(*snapshot(env), "cpu")
        img = render(x)
        assert img.shape == (3, 3, size * TILE, size * TILE) and 0 <= float(img.min()) and float(img.max()) <= 1
        model = CleanPolicy(None, pixels=True)
        logits, v = model(x, iv, None)
        assert logits.shape == (3, N_ACTIONS) and v.shape == (3,)
    # the agent tile differs from the plain cell under it, and the facing notch moves with the facing
    env = MiniCraft(CFG, np.arange(1))
    a = render(obs_tensor(*snapshot(env), "cpu")[0])
    env.face[:] = (env.face + 1) % 4
    b = render(obs_tensor(*snapshot(env), "cpu")[0])
    assert not torch.equal(a, b)


def test_entropy_controller_moves_alpha_toward_target():
    from gft.minicraft.learn import EntropyController
    c = EntropyController(target=0.8, alpha=0.01, lr=0.5)
    a0 = c.alpha
    c.update(0.2)                      # too deterministic -> stronger bonus
    assert c.alpha > a0
    for _ in range(50):
        c.update(2.0)                  # too random -> bonus shrinks to the floor
    assert abs(c.alpha - 1e-4) < 1e-9
    assert EntropyController(None, 0.01).alpha == 0.01   # fixed coefficient when no target


def test_tokens_keep_legacy_ids_and_exact_vocab_has_no_collisions():
    import zlib
    from gft.minicraft.learn import VOCAB, PromptEncoder, prompt_tokens
    from gft.minicraft.vocab import NUMBERS, WORDS
    tok = prompt_tokens(["request: make planks feedback: you still needed clay 3"], "cpu")
    words = "request make planks feedback you still needed clay 3".split()
    legacy = PromptEncoder(8, 16).ids(tok)[0]
    assert legacy.tolist() == [1 + zlib.crc32(w.encode()) % (VOCAB - 1) for w in words]   # old checkpoints unchanged
    exact = PromptEncoder(8, 16, states=True, exact=True)
    ids = exact.ids(tok)[0].tolist()
    assert ids[:-1] == [WORDS.index(w) + 1 for w in words[:-1]] and ids[-1] == len(WORDS) + 1 + 3
    assert len(set(exact.ids(prompt_tokens(list(WORDS), "cpu"))[:, 0].tolist())) == len(WORDS)
    unk = exact.ids(prompt_tokens(["zzzq"], "cpu"))[0, 0]
    assert len(WORDS) + NUMBERS < unk < exact.tok.num_embeddings


def test_prompt_vocabulary_covers_requests_and_teacher_feedback():
    import re
    from gft.minicraft.goals import ALIASES
    from gft.minicraft.teacher import PROFILES
    from gft.minicraft.learn import run_episodes
    from gft.minicraft.vocab import WORDS
    pool = build_pool(CFG, 12, seed=0)
    n = len(pool.seeds)
    rng = np.random.default_rng(0)
    req = pool.requests(np.arange(n))
    env, log, _, _, _, sat, _ = run_episodes(CFG, req, lambda e, t: rng.integers(0, N_ACTIONS, len(req)))
    texts = ["request: x feedback: y"] + [g.phrase for g in GOALS] + [a.phrase for a in ALIASES]
    for prof in PROFILES:
        texts += [" ".join(s for _, _, s in compose(episode_facts(log, b, env, req.goals[b], sat[b]), prof, rng))
                  for b in range(len(req))]
        texts += [" ".join(s for _, _, s in compose(pool.ideal_facts(i, rng), prof, rng)) for i in range(n)]
    words = {w for t in texts for w in re.findall(r"[a-z]+", t.lower())} - {"x", "y"}
    assert words <= set(WORDS), sorted(words - set(WORDS))


def test_attention_crop_on_the_agent_is_the_agent_crop():
    from gft.minicraft.learn import N_CELLS, attention_crop, crop_at_agent
    env = MiniCraft(WorldConfig(size=13), np.arange(5))
    x, _ = obs_tensor(*snapshot(env), "cpu")
    f = torch.randn(5, 6, 13, 13)
    a = x[:, N_CELLS].flatten(1)                          # one-hot on the agent's cell
    assert torch.allclose(attention_crop(f, a, 7), crop_at_agent(f, x, 7).permute(0, 3, 1, 2).flatten(1), atol=1e-5)


def test_split_sentences():
    from gft.minicraft.learn import split_sentences
    assert split_sentences("You did not get some metal. Great job! 3 of 3 done.") == \
        ["You did not get some metal.", "Great job!", "3 of 3 done."]
    assert split_sentences("Okay") == ["Okay"]


def test_episode_facts_all_matches_the_per_step_version_and_worlds_are_cached():
    from gft.minicraft.core import _WORLD_CACHE
    from gft.minicraft.learn import run_episodes
    from gft.minicraft.teacher import episode_facts_all
    pool = build_pool(CFG, 8, seed=0)
    req = pool.requests(np.arange(len(pool.seeds)))
    rng = np.random.default_rng(3)
    env, log, _, _, _, sat, _ = run_episodes(CFG, req, lambda e, t: rng.integers(0, N_ACTIONS, len(req)))
    fast = episode_facts_all(log, env, req.goals, sat)
    for b in range(len(req)):
        assert fast[b] == episode_facts(log, b, env, req.goals[b], sat[b])
    before = len(_WORLD_CACHE)
    e1 = MiniCraft(CFG, req.seeds)
    assert len(_WORLD_CACHE) == before                 # already cached by the rollout above
    e2 = MiniCraft(CFG, req.seeds)
    e1.grid[0, 1, 1] = 7
    assert np.array_equal(e2.grid, MiniCraft(CFG, req.seeds).grid)   # copies: one world's edits do not leak
