import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.goals import GOALS, N_GOALS
from gft.minicraft.learn import (GFT, PPO, CleanPolicy, Curriculum, CurriculumConfig, GFTConfig, PPOConfig, TaskPool,
                                 build_pool, claim_type, obs_tensor, snapshot)
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
    assert claim_type(("missing_fail", 3, 1)) == ("missing_fail", 3, 1)
    assert claim_type(("steps", 40)) == ("steps",)
    assert claim_type(("eval_count", 1, 2)) == ("eval_count",)


def test_rounds_log_diagnostics():
    torch.set_num_threads(1)
    pool = build_pool(CFG, 12, seed=0, noisy=())
    torch.manual_seed(0)
    gft = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(episodes=32, stage2=8, minibatch=64), seed=0)
    info = gft.round()
    assert 0.0 <= info["coverage"] <= 1.0 and isinstance(info["unseen_claims"], dict)
    assert info["reader_loss_seen"] > 0 and info["curriculum_unlocked"] == 4
    assert len(gft.seen_claims) == 1
    ppo = PPO(CFG, pool, CleanPolicy(CFG.size), "cpu", PPOConfig(episodes=16, minibatch=64), seed=0)
    assert ppo.round()["curriculum_unlocked"] == 4


def test_policy_reads_prompts_and_gft_pi_fb_is_a_prompted_copy():
    from gft.minicraft.goals import Requests
    from gft.minicraft.learn import prompt_text, prompt_tokens, request_prompts
    torch.manual_seed(0)
    model = CleanPolicy(CFG.size)
    env = MiniCraft(CFG, np.arange(4))
    x, iv = obs_tensor(*snapshot(env), "cpu")
    req = Requests(np.array([[9, -1, -1], [3, 1, -1], [0, -1, -1], [8, 4, 2]]), np.arange(4))
    p_req = request_prompts(model, req, "cpu")
    p_fb = request_prompts(model, req, "cpu", feedback=["You collected 2 stone."] * 4)
    with torch.no_grad():                                                        # fresh text pathway: no effect
        assert torch.allclose(model(x, iv, p_req)[0], model(x, iv, p_fb)[0])
        assert torch.allclose(model(x, iv, p_req)[0], model(x, iv, None)[0])
    assert prompt_text("fetch the gem", "Great job!") == "request: fetch the gem feedback: Great job!"
    tok = prompt_tokens([prompt_text(req.text(b), "You collected 2 stone.") for b in range(4)], "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    target = torch.arange(4) % 12
    losses = []
    for _ in range(40):                                                         # text pathway is trainable
        loss = torch.nn.functional.cross_entropy(model(x, iv, model.encode_prompt(tok))[0], target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5
    with torch.no_grad():                                                        # text now matters
        assert not torch.allclose(model(x, iv, model.encode_prompt(tok))[0], model(x, iv, None)[0])
    pool = build_pool(CFG, 6, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(episodes=8, stage2=8, minibatch=64))
    assert all(torch.equal(p, q) for p, q in zip(g.clean.parameters(), g.aux.parameters()))   # pi^fb starts as pi
    info = g.round()
    assert np.isfinite(info["aux_loss"]) and "reader_loss_heldout_no_feedback" in info   # NaN if no held-out episode


def test_gft_replay_buffer_keeps_last_rounds():
    torch.set_num_threads(1)
    pool = build_pool(CFG, 6, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu",
            GFTConfig(episodes=8, stage2=8, minibatch=64, replay_rounds=2, epochs_aux=2, epochs_clean=2))
    for _ in range(3):
        assert np.isfinite(g.round()["aux_loss"])
    assert len(g.replay) == 2
    R = g._replay_tensors("cpu")
    assert len(R["a"]) == len(R["grid"]) == len(R["b"]) and int(R["b"].max()) < len(R["tok"])


def test_episode_batches_cover_each_step_once_with_few_episodes():
    pool = build_pool(CFG, 4, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(minibatch=64, episode_chunk=8))
    b = np.repeat(np.arange(40), np.random.default_rng(0).integers(1, 30, size=40))
    batches = list(g._episode_batches(b, epochs=2))
    steps = np.concatenate(batches)
    assert sorted(steps.tolist()) == sorted(list(range(len(b))) * 2)          # each step once per pass
    assert all(len(np.unique(b[m])) <= 64 // 8 for m in batches)               # at most 8 episodes per minibatch


def test_block_curriculum_follows_feedback_budget():
    pool = _pool([[g, -1, -1] for g in range(N_GOALS)] * 10)
    cur = Curriculum(pool, CurriculumConfig(mode="blocks", block_size=2, block_budget=100))
    rng = np.random.default_rng(0)
    assert pool.goals[cur.sample(rng, 1000)].max() == 1
    goals = np.array([[0, -1, -1]] * 4)
    mask = goals >= 0
    cur.update(goals, mask, mask, 50, feedbacks=99)
    assert cur.unlocked == 2
    cur.update(goals, mask, mask, 60, feedbacks=250)                      # two budgets spent -> blocks 0..2
    assert cur.unlocked == 6 and cur.unlock_log == [(250, 6)]
    deepest = pool.goals[cur.sample(rng, 5000)].max(1)
    assert deepest.max() == 5 and 0.7 < (deepest >= 4).mean() < 0.85    # frontier share plus replayed frontier tasks


def test_alias_requests_pool_and_two_phase_schedule():
    from gft.minicraft.goals import ALIASES, alias_requests
    from gft.minicraft.learn import AliasSchedule, concat_pools
    r = alias_requests(0, [11, 12])
    assert r.text(0) == ALIASES[0].phrase and set(r.goals[0][r.goals[0] >= 0]) == set(ALIASES[0].goals)
    base = build_pool(CFG, 6, seed=0, noisy=())
    aliases = concat_pools(*[build_pool(WorldConfig(), 0, 3 + k, noisy=(), requests=alias_requests(k, [100 + k, 200 + k]))
                             for k in range(len(ALIASES))])
    pool = concat_pools(base, aliases)
    assert (pool.alias[:len(base.seeds)] == -1).all() and set(pool.alias[len(base.seeds):]) <= set(range(len(ALIASES)))
    assert pool.requests([len(pool.seeds) - 1]).text(0) in {a.phrase for a in ALIASES}
    sch = AliasSchedule(pool, budget=100)
    rng = np.random.default_rng(0)
    assert (pool.alias[sch.sample(rng, 500)] == -1).all()                  # phase 1: base tasks only
    sch.start_phase2(1000)
    s = pool.alias[sch.sample(rng, 2000)]
    assert set(s) <= {-1, 0} and 0.6 < (s == 0).mean() < 0.8               # newest alias + replay of base tasks
    sch.update(None, None, None, 0, feedbacks=1250)                         # 2.5 budgets later: 3 aliases
    assert sch.unlocked == 3 and set(pool.alias[sch.sample(rng, 2000)]) <= {-1, 0, 1, 2}
    assert not sch.finished(1399) and sch.finished(1000 + len(ALIASES) * 100)


def test_graph_curriculum_opens_goals_after_prerequisites_are_mastered():
    from gft.minicraft.learn import goal_prerequisites
    pre = goal_prerequisites()
    assert pre[0] == set() and pre[9] >= {7, 8} and 9 not in pre[7] | pre[8]
    pool = _pool([[g, -1, -1] for g in range(N_GOALS)] * 10 + [[9, 0, -1]] * 5)
    cur = Curriculum(pool, CurriculumConfig(mode="graph", mastery=0.7, min_count=20))
    rng = np.random.default_rng(0)
    assert set(pool.goals[cur.sample(rng, 500)].ravel()) <= {-1, 0}            # only firewood is open
    ok = np.ones((30, 3), bool)
    cur.update(np.array([[0, -1, -1]] * 30), ok, np.array([[True, False, False]] * 30), 30, feedbacks=30)
    assert cur.mastered == {0} and set(pool.goals[cur.sample(rng, 500)].ravel()) <= {-1, 0, 1}


def test_masked_claim_examples_cover_only_their_steps():
    from gft.minicraft.teacher import claim_steps
    torch.set_num_threads(1)
    pool = build_pool(CFG, 6, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(episodes=8, stage2=8, minibatch=64, masked_claims=3))
    info = g.round()
    assert np.isfinite(info["aux_loss"])
    R = g.replay[-1]
    assert len(R["tok"]) > 0 and int(R["b"].max()) < len(R["tok"]) and len(R["a"]) == len(R["grid"])
    # toy log for one episode: move, collect wood (t=1), move, craft a plank (t=3)
    from gft.minicraft.core import EV_COLLECT, EV_CRAFT, EV_MOVE, PLANK, WOOD
    mk = lambda typ, item: dict(type=np.array([typ]), item=np.array([item]), tool=np.array([-1]),
                                station=np.array([-1]), missing=np.array([-1]))
    log = [mk(EV_MOVE, -1), mk(EV_COLLECT, WOOD), mk(EV_MOVE, -1), mk(EV_CRAFT, PLANK)]
    assert claim_steps(log, 0, ("collected", WOOD, 1)).tolist() == [0, 1]
    assert claim_steps(log, 0, ("crafted", PLANK, 1)).tolist() == [2, 3]
    assert claim_steps(log, 0, ("steps", 4)) is None


def test_round_logs_per_goal_margins_coverage_and_base_share():
    torch.set_num_threads(1)
    pool = build_pool(CFG, 6, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(episodes=16, stage2=8, minibatch=64, base_share=0.5))
    info = g.round()
    assert set(info["stage1_coverage"]) == {goal.phrase for goal in GOALS}
    assert all(0 <= a2 <= a1 <= 1 for a1, a2 in info["stage1_coverage"].values())
    assert info["stage1_per_goal"] and info["stage2_per_goal"] is not None


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


def test_no_feedback_ablation_prompts_pi_fb_with_the_request_alone():
    torch.set_num_threads(1)
    pool = build_pool(CFG, 6, seed=0, noisy=())
    g = GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", GFTConfig(episodes=16, stage2=8, minibatch=64, no_feedback=True))
    info = g.round()
    assert all(e.get("feedback", None) is None for e in info["examples"])
    assert not g.replay[-1]["fb_pos"].any()          # Stage I prompts hold no feedback words
    h, h0 = info["reader_loss_heldout"], info["reader_loss_heldout_no_feedback"]
    assert h != h or abs(h - h0) < 1e-5              # NaN when no episode was held out


def test_tokens_keep_legacy_ids_and_exact_vocab_has_no_collisions():
    import zlib
    from gft.minicraft.learn import VOCAB, PromptEncoder, prompt_tokens
    from gft.minicraft.vocab import NUMBERS, WORDS
    tok = prompt_tokens(["request: make some planks feedback: you still needed ore 3"], "cpu")
    words = "request make some planks feedback you still needed ore 3".split()
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


def test_soft_crop_matches_the_integer_crop_at_the_agent():
    from gft.minicraft.learn import N_CELLS, cell_coords, crop_at_agent, soft_crop
    env = MiniCraft(WorldConfig(size=13), np.arange(5))
    x, _ = obs_tensor(*snapshot(env), "cpu")
    f = torch.randn(5, 6, 13, 13)
    pos = cell_coords(13, "cpu")[x[:, N_CELLS].flatten(1).argmax(1)]
    assert torch.allclose(soft_crop(f, pos, 7), crop_at_agent(f, x, 7).permute(0, 3, 1, 2), atol=1e-5)


def test_new_architectures_run_on_any_size_and_start_as_their_text_free_version():
    from gft.minicraft.learn import prompt_text, prompt_tokens
    torch.manual_seed(0)
    for kind in ("gvin", "fusion"):
        for pixels in (True, False):
            for size in (11, 13):
                env = MiniCraft(WorldConfig(size=size), np.arange(3))
                x, iv = obs_tensor(*snapshot(env), "cpu")
                m = CleanPolicy(None, kind=kind, pixels=pixels)
                p = m.encode_prompt(prompt_tokens([prompt_text("make some planks", "You still needed ore.")] * 3, "cpu"))
                (l1, v1), (l0, v0) = m(x, iv, p), m(x, iv, None)
                assert l1.shape == (3, N_ACTIONS) and v1.shape == (3,)
                assert torch.allclose(l1, l0, atol=1e-6) and torch.allclose(v1, v0, atol=1e-6)


def test_new_architectures_learn_to_follow_the_text():
    """Same states, two prompts asking for different actions: the text pathway must open up and carry the signal."""
    from gft.minicraft.learn import prompt_tokens
    torch.set_num_threads(2)
    env = MiniCraft(WorldConfig(size=11), np.arange(16))
    x, iv = obs_tensor(*snapshot(env), "cpu")
    x, iv = torch.cat([x, x]), torch.cat([iv, iv])
    y = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    tok = prompt_tokens(["request: go up"] * 16 + ["request: go down"] * 16, "cpu")
    for kind in ("gvin", "fusion"):
        torch.manual_seed(0)
        small = dict(d=64, ch=16, rec=1, trunk=1) if kind == "gvin" else dict(d=64, w=32, layers=1, pre_conv=1)
        m = CleanPolicy(None, kind=kind, pixels=False, dt=32, **small)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(100):
            loss = torch.nn.functional.cross_entropy(m(x, iv, m.encode_prompt(tok))[0], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert float(loss.detach()) < 0.1, (kind, float(loss.detach()))


def test_gft_round_runs_with_the_new_architectures():
    torch.set_num_threads(1)
    pool = build_pool(CFG, 6, seed=0, noisy=())
    for kind in ("gvin", "fusion"):
        g = GFT(CFG, pool, CleanPolicy(CFG.size, kind=kind, pixels=True), "cpu",
                GFTConfig(episodes=8, stage2=8, minibatch=64))
        info = g.round()
        assert np.isfinite(info["aux_loss"]) and np.isfinite(info["distill_loss"])
