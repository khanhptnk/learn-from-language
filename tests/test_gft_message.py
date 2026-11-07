"""GFT (gft/minicraft/gft_message.py), the one implementation."""
import copy

import numpy as np
import pytest
import torch

import gft.minicraft.gft_message as M
from gft.minicraft.core import WorldConfig
from gft.minicraft.learn import CleanPolicy, build_pool
from gft.minicraft.teacher import compose

CFG = WorldConfig()
TINY = dict(episodes=16, stage2=16, rerank_k=4, minibatch=64, scorer_batch=16, scorer_max_steps=3, warmup_rounds=0)


@pytest.fixture(scope="module")
def pool():
    return build_pool(CFG, 6, seed=0, noisy=())


def learner(pool, **kw):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    return M.GFT(CFG, pool, CleanPolicy(CFG.size), "cpu", M.GFTConfig(**{**TINY, **kw}), seed=0)


def test_round_runs_logs_and_resumes_exactly(pool):
    g = learner(pool)
    info = g.round()
    assert g.teacher_calls == 16 and g.episodes == 16 + 4 * info["rerank_tasks"]
    for k in ("stage1_success", "text_success", "aux_loss", "scorer_steps", "rerank_tasks"):
        assert k in info, k
    sd = copy.deepcopy(g.state_dict())                            # as torch.save in train.py: no shared lists
    g2 = learner(pool)
    g2.load_state_dict(sd)
    rs = torch.get_rng_state()                                    # train.py checkpoints torch's RNG next to it
    a = g.round()
    torch.set_rng_state(rs)
    b = g2.round()
    assert a["aux_loss"] == b["aux_loss"] and a.get("distill_loss") == b.get("distill_loss")
    assert all(torch.equal(p, q) for p, q in zip(g.clean.state_dict().values(), g2.clean.state_dict().values()))


def _goal_phrase(i):
    from gft.minicraft.goals import GOALS
    return GOALS[i].phrase


def test_built_success_message_is_praise_and_verdicts_in_received_wordings(pool):
    g = learner(pool)
    req = pool.requests(np.arange(len(pool.goals)))
    assert g._built(req.goals[0]) is None                         # nothing received yet: no wording to use
    g.received.observe("Nice work! Done: gather wood. You got 2 wood from a tree.")
    for i in range(len(req)):
        b = g._built(req.goals[i])
        goals = [int(q) for q in req.goals[i] if q >= 0]
        assert [c for _, c, _ in b] == [("eval_all",)] + [("goal_done", q) for q in goals]
        assert [t for _, _, t in b] == ["Nice work!"] + [f"Done: {_goal_phrase(q)}." for q in goals]


def test_tasks_target_the_improved_message_and_the_ideal_set(pool):
    g = learner(pool, ideal_k=2, stage2=10 ** 6)                  # room for every episode as a task
    rng = np.random.default_rng(0)
    req = pool.requests(np.arange(len(pool.goals)))
    fail = [[("confirm", ("goal_not_done", int(req.goals[i, 0])), f"Not done: {_goal_phrase(int(req.goals[i, 0]))}.")]
            for i in range(len(req))]                             # failures stating no cause: no improved message
    assert len(g._tasks(req, fail)[0]) == 0                       # and nothing received: no built message either
    g.received.observe("That worked. You did gather wood.")
    src, from_bank, imp, ideal = g._tasks(req, fail)
    assert len(src) == len(req) and not from_bank.any() and all(p is None for p in imp)
    assert all(len(ps) == 1 and ps[0][0][2] == "That worked." for ps in ideal)
    for i in range(len(req)):                                     # banks of 4 successes: only the last 2 are targets
        g.succ_parts[req.text(i)] = [compose(pool.ideal_facts(i, rng), "realistic", rng) for _ in range(4)]
    src, from_bank, imp, ideal = g._tasks(req, fail)
    assert len(src) and from_bank.all()
    for e, ps in zip(src, ideal):
        bank = g.succ_parts[req.text(int(e))]
        assert len(ps) == 3 and ps[1] is bank[-2] and ps[2] is bank[-1]


def test_s_averages_the_improved_and_the_ideal_halves(pool, monkeypatch):
    """D(x, m) = 100 x + m (one sentence per message): s = 1/2 (imp gain) + 1/2 (mean ideal gain) = the episode gain;
    with only an ideal set, s is its gain alone; with neither, -inf."""
    g = learner(pool)
    monkeypatch.setattr(M, "message_totals", lambda scorer, X, valid, eps, msgs, tok, dev, mean=False:
                        torch.tensor([100.0 * float(X[e]) + sum(m) for e, m in zip(eps, msgs)]))
    X, X1 = torch.tensor([3.0, 5.0, 7.0, 9.0]), torch.tensor([4.0, 4.0, 4.0])
    task_of = np.array([0, 0, 1, 2])
    targets = [[[[1]], [], []], [[[2], [6]], [[3]], []]]
    s = g._s(X, None, X1, None, task_of, 4, targets, None)
    assert np.allclose(s[:3], [-100.0, 100.0, 300.0]) and s[3] == -np.inf


@pytest.mark.parametrize("src, keep", [(2.5, [3]), (0.5, [2, 3])])
def test_each_task_keeps_its_top_attempts_by_s_and_only_s_above_zero(pool, monkeypatch, src, keep):
    """Stage II source of task k scores 4k + src and attempt i of it 4k + i under any message: s = i - src. With
    src 2.5 only the last of each task's four attempts has s > 0 -- kept; with src 0.5 three do, and keep_frac 0.5
    keeps the top two of them (rounded up)."""
    g = learner(pool)
    rng = np.random.default_rng(1)
    for i in range(len(pool.goals)):                              # every request has received successes: tasks exist
        r = pool.requests(np.array([i])).text(0)
        p = compose(pool.ideal_facts(i, rng), "realistic", rng)
        g.succ_parts[r] = [p]
        g.succ_msgs[r] = [g._claim_sentences(p)]
    seen = {}

    def fake(scorer, X, valid, eps, msgs, tok, dev, mean=False):
        seen.setdefault("n", X.shape[0])                          # the first call scores the Stage II attempts
        return torch.tensor([float(e) if X.shape[0] == seen["n"] else 4 * e + src for e in eps])
    got = []
    real_imitate = g._reader_imitate
    g._reader_imitate = lambda kept, *a: (got.append(kept.copy()), real_imitate(kept, *a))
    real_stage2 = g._stage2

    def stage2(*a):
        seen.clear()
        monkeypatch.setattr(M, "message_totals", fake)
        try:
            return real_stage2(*a)
        finally:
            monkeypatch.undo()
    g._stage2 = stage2
    info = g.round()
    assert info["rerank_tasks"] > 0 and info["ideal_avail"] == 1.0
    kept = got[0]
    for k in range(len(kept) // 4):
        assert list(np.nonzero(kept[4 * k:4 * k + 4])[0]) == keep


def test_bank_mean_is_the_plain_mean_over_each_banks_own_messages(monkeypatch):
    monkeypatch.setattr(M, "message_totals",
                        lambda scorer, X, valid, eps, msgs, tok, dev, mean=False:
                        torch.tensor([10.0 * e + sum(m) for e, m in zip(eps, msgs)]))
    rng = np.random.default_rng(0)
    banks = [[list(rng.integers(0, 50, rng.integers(1, 4))) for _ in range(rng.integers(1, 9))] for _ in range(37)]
    eps = list(rng.integers(0, 20, len(banks)))
    want = np.array([np.mean([10.0 * e + sum(m) for m in bk]) for e, bk in zip(eps, banks)])
    for chunk in (3, 7, 256):
        assert np.allclose(M.bank_mean(None, None, None, eps, banks, None, "cpu", chunk=chunk), want)


def test_stage2_tasks_draw_from_successful_episodes_too(pool):
    g = learner(pool, stage2=10 ** 6)                             # room for every episode as a task
    rng = np.random.default_rng(0)
    for i in range(len(pool.goals)):
        r = pool.requests(np.array([i])).text(0)
        p = compose(pool.ideal_facts(i, rng), "realistic", rng)
        g.succ_parts[r], g.succ_msgs[r] = [p], [g._claim_sentences(p)]
    req = pool.requests(np.arange(len(pool.goals)))
    ok = [compose(pool.ideal_facts(i, rng), "realistic", rng) for i in range(len(req))]   # every episode a success
    src = g._tasks(req, ok)[0]
    assert len(src) == len(req)


def test_reader_without_kept_imitation_still_runs(pool):
    g = learner(pool, reader_imitate_epochs=0)
    info = g.round()
    assert info["rerank_tasks"] >= 0 and np.isfinite(info["aux_loss"])


def test_scorer_pairs_are_symmetric_failures_against_successes_and_successes_against_failures(pool):
    """A failure's message group has a received success message of its request as a negative; a success's has a
    received failure message of it. Both banks are kept, and survive a resume."""
    rng = np.random.default_rng(0)
    r = pool.requests(np.array([0])).text(0)
    meta = [dict(req=r, msg=[1], ok=False, imp=None, ideal=[9]),  # a failure
            dict(req=r, msg=[5], ok=True, imp=None, ideal=[9])]   # a success (its built message is never used)
    seen = []

    def fake_totals(scorer, X, valid, pe, pm, tok, device, mean=True):
        seen.append(list(zip(pe, map(tuple, pm))))
        return torch.zeros(len(pe))
    real = M.message_totals
    M.message_totals = fake_totals
    try:
        M.message_scorer_loss(None, None, None, meta, {r: [[5], [6]]}, {r: [[1], [2]]}, None, rng, "cpu", P=2,
                              n_others=0)
    finally:
        M.message_totals = real
    pairs = set(seen[0])
    assert any(e == 0 and m in ((5,), (6,)) for e, m in pairs)    # the failure against a success message
    assert (0, (9,)) in pairs and (1, (9,)) not in pairs          # ... and against its built success message
    assert any(e == 1 and m in ((1,), (2,)) for e, m in pairs)    # the success against a failure message
    g = learner(pool)
    info = g.round()
    assert np.isfinite(info["aux_loss"])
    n_s, n_f = sum(map(len, g.succ_msgs.values())), sum(map(len, g.fail_msgs.values()))
    assert n_f > 0 and n_s + n_f <= 16                            # every message with a claim joins one bank
    g2 = learner(pool)
    g2.load_state_dict(copy.deepcopy(g.state_dict()))
    assert g2.fail_msgs == g.fail_msgs and g2.succ_msgs == g.succ_msgs


def test_message_writer_learns_messages_per_request_and_samples_valid_messages():
    """Two requests with their own messages (sentence ids into a table): after training, samples follow the request,
    use each sentence at most once, have at least one and at most max_sents sentences."""
    torch.manual_seed(0)
    texts = M.TextTable(["Done: gather wood.", "You got 2 wood from a tree.", "Not done: make planks.",
                         "You made a plank at a workbench.", "Nice work!"])
    tok = texts.tokens("cpu")
    w = M.MessageWriter(d_s=8, d=32, dt=32, max_sents=3)
    opt = torch.optim.Adam(w.parameters(), lr=3e-3)
    from gft.minicraft.learn import prompt_text, prompt_tokens
    reqs = ["gather wood", "make planks"] * 16
    want = {"gather wood": [4, 0, 1], "make planks": [2, 3]}
    rt = prompt_tokens([prompt_text(r) for r in reqs], "cpu")
    h0 = torch.zeros(len(reqs), 8)
    first = None
    for _ in range(150):
        loss = w.loss(rt, h0, [want[r] for r in reqs], tok)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.2 * first
    out = w.sample(rt, h0, tok)
    assert all(1 <= len(m) <= 3 and len(set(m)) == len(m) for m in out)
    hit = np.mean([m == want[r] for m, r in zip(out, reqs)])
    assert hit > 0.8, (hit, out[:4])


def test_stage2_without_a_writer_has_no_written_prompts(pool):
    g0 = learner(pool, writer=False)
    info0 = g0.round()
    assert g0.writer is None and info0["gen_share"] == 0 and "writer_imitated" not in info0


def test_warmup_rounds_train_the_reader_and_scorer_but_leave_pi_and_skip_stage2(pool):
    g = learner(pool, warmup_rounds=2)
    pi0 = copy.deepcopy(g.clean.state_dict())
    sc0, wr0 = copy.deepcopy(g.scorer.state_dict()), copy.deepcopy(g.writer.state_dict())
    for r in range(2):
        info = g.round()
        assert info["warmup"] and info["rerank_tasks"] == 0 and "kept" not in info
    assert all(torch.equal(p, q) for p, q in zip(pi0.values(), g.clean.state_dict().values()))
    assert any(not torch.equal(p, q) for p, q in zip(sc0.values(), g.scorer.state_dict().values()))
    assert any(not torch.equal(p, q) for p, q in zip(wr0.values(), g.writer.state_dict().values()))
    assert g.teacher_calls == 32 and g.episodes == 32
    info = g.round()
    assert not info["warmup"] and info["rerank_tasks"] > 0


def test_stage2_puts_half_under_written_messages_and_half_under_one_target_per_task(pool):
    g = learner(pool, rerank_k=6)
    g.received.observe("Nice work! Done: gather wood.")
    g.round()
    info = g.round()
    assert abs(info["gen_share"] - 0.5) < 1e-9 and abs(info["imp_share"] + info["ideal_share"] - 0.5) < 1e-9
    ex = [e for e in info["examples"] if "kind" in e]             # the first task's attempts: one target, 3 written
    assert len({e["prompt"] for e in ex if e["kind"] != "gen"}) == 1 and sum(e["kind"] == "gen" for e in ex) == 3
    assert np.isfinite(info["writer_loss"]) and info["writer_imitated"] == info["kept"]
    assert len(info["gen_examples"]) and all(e["message"] for e in info["gen_examples"])


def test_a_checkpoint_from_before_the_message_head_loads_with_a_fresh_head(pool):
    g = learner(pool)
    g.round()
    sd = copy.deepcopy(g.state_dict())
    sd["writer"], sd["opt_writer"] = {"other.weight": torch.zeros(1)}, None          # another network's state
    g2 = learner(pool)
    w0 = copy.deepcopy(g2.writer.state_dict())
    g2.load_state_dict(sd)
    assert all(torch.equal(w0[k], v) for k, v in g2.writer.state_dict().items())
    assert all(torch.equal(p, q) for p, q in zip(g.clean.state_dict().values(), g2.clean.state_dict().values()))
