"""Pretraining v2 (gft/minicraft/pretrain.py): corpus leakage rules, narration spans, the grounded policy's interface,
and the reader objective."""
import numpy as np
import pytest
import torch

from gft.minicraft.core import CRAFT, N_ACTIONS, USE, WorldConfig
from gft.minicraft.goals import GOALS, GOAL_ID
from gft.minicraft.learn import prompt_text, prompt_tokens
from gft.minicraft.pretrain import (CONTENT_WORDS, GroundedPolicy, PretrainData, audit, build_corpus,
                                    content_mask_table, mask_words, pick_frames, pretrain_grounded, word_class)
from gft.minicraft.verify import parse, split_sentences

CFG = WorldConfig()
HELD = (GOAL_ID["build a well"], GOAL_ID["make a sling"])
ARCH = dict(d=32, w=16, layers=1, pre_conv=1, rec=1, reader_layers=1)


@pytest.fixture(scope="module")
def corpus():
    c, counts, stats = build_corpus(CFG, 96, HELD, seed=0, chunk=48, workers=1)
    return c, counts, stats


def test_corpus_never_requests_or_mentions_heldout_goals_and_reports_coverage(corpus):
    c, counts, stats = corpus
    assert audit(c, counts, HELD)
    held_phr = {GOALS[h].phrase for h in HELD}
    for r in c["request"]:
        assert not set(r.split(" and ")) & held_phr
    for h in c["hindsight"]:
        assert h is None or not set(h.split(" and ")) & held_phr
    assert set(c["mode"]) <= {"expert", "wander", "free"} and len(set(c["mode"])) == 3
    assert counts["episodes"] == len(c["request"]) == len(c["spans"]) and counts["steps"] == len(c["a"])
    assert stats["drop_rate"] < 0.2 and stats["sentences_per_message"] > 2
    assert set(stats["episodes_per_mode"]) == {"expert", "wander", "free"}
    # free explorers get a hindsight request only for goals they actually reached
    for m, h in zip(c["mode"], c["hindsight"]):
        if m == "free" and h:
            assert all(g in [x.phrase for x in GOALS] for g in h.split(" and "))


def test_narration_spans_point_at_their_events_and_parse(corpus):
    c, _, _ = corpus
    first = np.searchsorted(c["b"], np.arange(len(c["request"])))
    length = np.bincount(c["b"], minlength=len(c["request"]))
    n = 0
    for e, spans in enumerate(c["spans"]):
        sents = split_sentences(c["message"][e])
        for s, t0, t1 in spans:
            assert s in sents and 0 <= t0 <= t1 < length[e]
            q = parse(s)
            assert q is not None, s
            kind, claim = q
            inv = c["inv"][first[e]: first[e] + length[e]]
            if kind == "crafted":                       # one span per craft: the craft happens at t1
                assert c["a"][first[e] + t1] == CRAFT and inv[min(t1 + 1, length[e] - 1), claim[1]] >= 1
                n += 1
            if kind == "collected":                     # one span per pickup: the use happens at t1
                assert c["a"][first[e] + t1] == USE
            if kind == "craft_held":                    # a failed craft at t1 (what its slot held: test_minicraft)
                assert c["a"][first[e] + t1] == CRAFT
            assert kind in ("collected", "crafted", "used_on", "craft_held"), kind   # narration only
    assert n > 0
    # a free explorer's hindsight request holds at the end of its episode
    for e, (m, h) in enumerate(zip(c["mode"], c["hindsight"])):
        if m == "free" and h:
            end = c["inv"][first[e] + length[e] - 1]
            for g in h.split(" and "):
                goal = GOALS[GOAL_ID[g]]
                assert end[goal.item] >= goal.count, (h, end)


def test_grounded_policy_has_the_clean_policy_interface():
    torch.manual_seed(0)
    m = GroundedPolicy(CFG.size, **ARCH)
    from gft.minicraft.core import MiniCraft
    from gft.minicraft.learn import obs_tensor
    env = MiniCraft(CFG, [1, 2, 3])
    x, inv = obs_tensor(env.grid, np.concatenate([env.pos, env.face[:, None]], 1), env.agent_state(), "cpu")
    p = m.encode_prompt(prompt_tokens([prompt_text("gather wood"), prompt_text("make rope", "Not done: make rope."),
                                       prompt_text("dig clay")], "cpu"))
    logits, v = m(x, inv, p)
    assert logits.shape == (3, N_ACTIONS) and v.shape == (3,)
    lg2, _, (h, f, _) = m(x, inv, p, return_h=True)
    assert torch.allclose(lg2, logits) and h.shape == (3, m.d)
    assert m.frames(x, inv).shape == (3, m.d)
    assert m(x, inv, None)[0].shape == (3, N_ACTIONS)              # text-free call (the reader's frame summaries)


def test_word_masking_targets_content_words_and_never_pads():
    m = GroundedPolicy(CFG.size, **ARCH)
    content = content_mask_table(m.prompt)
    tok = prompt_tokens([prompt_text("gather wood", "You made a plank at a workbench from wood.")] * 64, "cpu")
    ids = m.prompt.ids(tok)
    gen = torch.Generator().manual_seed(0)
    mk = mask_words(ids, content, gen)
    assert mk.shape == ids.shape and not (mk & (ids == 0)).any() and mk.any(1).all()
    words = prompt_text("gather wood", "You made a plank at a workbench from wood.").lower().replace(".", "").split()
    is_content = torch.tensor([w in CONTENT_WORDS for w in words])
    rate = mk[:, :len(words)].float().mean(0)
    assert rate[is_content].mean() > rate[~is_content].mean()
    assert word_class("workbench") == "station" and word_class("planks") == "item" and word_class("3") == "number"


def test_pick_frames_keeps_both_ends_and_the_event_step():
    rng = np.random.default_rng(0)
    ts = pick_frames(10, 40, 100, 8, rng)
    assert len(ts) == 8 and ts[0] == 10 and ts[-1] == 41 and 40 in ts and np.all(np.diff(ts) > 0)
    assert pick_frames(5, 7, 8, 8, rng).tolist() == [5, 6, 7]       # t1 + 1 clipped to the episode
    assert pick_frames(0, 0, 1, 4, rng).tolist() == [0]


def test_pretraining_runs_and_reports_the_grounding_gap(corpus):
    c, _, _ = corpus
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model, hist = pretrain_grounded(CFG, "cpu", c, steps=6, batch=32, reader_batch=8, k_frames=6, arch=ARCH,
                                    log_every=3, heldout_frac=0.1, max_eval=4)
    assert len(hist) == 2
    row = hist[-1]
    for k in ("bc_request", "bc_message", "bc_span", "text_loss", "act_loss", "heldout_request", "heldout_message",
              "info_gain", "with_frames_item", "without_frames_item", "gap_item"):
        assert k in row and np.isfinite(row[k]), k
    D = PretrainData(c, "cpu", np.random.default_rng(0), heldout_frac=0.1)
    assert len(D.span_rows) > 0 and D.held.sum() > 0
    steps, tok, times = D.reader_batch("span", D.train_spans[:3], 6, np.random.default_rng(0))
    assert steps.shape == (3, 6) and tok.shape[0] == 3 and (times[steps >= 0] >= 0).all()


def test_reader_fits_masked_words_on_text_alone():
    """The masked-word head must tell masked positions apart (positions survive masking) and fit a fixed set of
    messages quickly on text alone; the old reader predicted one word per sentence."""
    from gft.minicraft.learn import prompt_text, tokenize
    torch.manual_seed(0)
    texts = [prompt_text("build a raft", "Great job! You made a raft at a workbench from a plank and rope."),
             prompt_text("make rope", "Not done: make rope. You still needed fiber."),
             prompt_text("build a wall", "Done: build a wall. At a site you turned a brick and stone into a wall."),
             prompt_text("dig clay", "Great job! You picked up 2 clay at a clay pit.")] * 8
    tok = tokenize(texts)
    tok = tok[:, : int((tok > 0).sum(1).max())]
    m = GroundedPolicy(CFG.size, **ARCH)
    ids = m.prompt.ids(tok)
    gen = torch.Generator().manual_seed(0)
    mask = mask_words(ids, content_mask_table(m.prompt), gen)
    B, K = len(texts), 2
    zeros = torch.zeros(B, K)
    args = (torch.zeros(B, K, m.d), torch.zeros(B, K, 14), zeros.long(), zeros.bool(), zeros.long(), zeros.bool())
    with pytest.raises(AssertionError):
        GroundedPolicy(CFG.size, reader_layer=1)                  # misspelled arch keys are rejected
    opt = torch.optim.AdamW(list(m.reader.parameters()) + list(m.prompt.parameters()), lr=3e-3)
    for _ in range(150):
        words, _ = m.reader(m.prompt, tok, mask, *args, frames_on=False)
        loss = torch.nn.functional.cross_entropy(words[mask], ids[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred = words.argmax(-1)
    assert (pred[mask] == ids[mask]).float().mean() > 0.9, float((pred[mask] == ids[mask]).float().mean())
    row = int(mask.sum(1).argmax())                     # several masked words in one sentence get different answers
    assert len(set(pred[row][mask[row]].tolist())) > 1
