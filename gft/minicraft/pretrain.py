"""Pretraining v2 (plan: ~/.claude/plans/this-is-a-new-glowing-candle.md): a grounded language of the world, built
from seen tasks only.

Corpus: episodes of an explorer mixture, each with a request of 1-3 *seen* goals, the teacher's message about that
request (verdicts, narration of what happened and where, correctives), the narration sentences' time spans, and a
hindsight request for request-mode BC. Held-out goals are never requested; episodes that make a held-out item by
accident are dropped; the audit checks that no held-out word or item is left anywhere.

Explorers (per episode):
  expert   noisy expert on the request, skill p in EXPERT_LEVELS (expert action w.p. p, else a random world action)
  wander   the same, interrupted by wander segments (random walks that pick things up and press craft where they stand)
  free     ignores the request: pursues random seen goals one after another, with wander segments, and declares
           done at a random time; its hindsight request is a subset of the goals it reached
"""

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from gft.minicraft.core import (AGENT_FEATS, COLLECT, CRAFT, DIRS, DONE, EV_CRAFT, EV_CRAFT_FAIL, ITEM_NAMES,
                                N_ACTIONS, N_ITEMS, N_WORLD_ACTIONS, USE)
from gft.minicraft.expert import ExpertPool
from gft.minicraft.goals import GOALS, MAX_GOALS, N_GOALS, sample_requests
from gft.minicraft.learn import (FusionEncoder, PromptEncoder, attention_crop, obs_tensor, on_device, prompt_text,
                                 run_episodes, stack_steps, tokenize)
from gft.minicraft.teacher import claim_steps, compose, episode_facts_all, event_arrays, slot_steps
from gft.minicraft.verify import split_sentences
from gft.minicraft.vocab import NUMBERS, WORDS

EXPERT_LEVELS = (0.7, 0.8, 0.9, 1.0)
MODES = ("expert", "wander", "free")
MODE_P = (0.5, 0.3, 0.2)
WANDER_START = 0.04            # per-step chance that a wander segment begins (wander / free explorers)
WANDER_LEN = (5, 20)
FREE_DONE = (60, 200)          # a free explorer declares done at a step drawn from this range


class Explorer:
    """Batch action function for one chunk of episodes: mode, skill and private goals per episode."""

    def __init__(self, env, req, modes, rng, expert, seen, max_steps):
        n = len(req)
        self.req, self.modes, self.rng, self.expert, self.seen = req, modes, rng, expert, seen
        self.p = np.asarray(EXPERT_LEVELS)[rng.integers(len(EXPERT_LEVELS), size=n)]
        self.p[modes == 2] = 1.0                                       # free explorers follow their own goals exactly
        self.goals = req.goals.copy()                                  # goals pursued now (free: private, changing)
        for b in np.nonzero(modes == 2)[0]:
            self.goals[b] = self._pick(b)
        self.wander = np.zeros(n, int)                                 # steps left in the current wander segment
        self.heading = rng.integers(4, size=n)
        lo = min(FREE_DONE[0], max_steps)
        self.stop = np.where(modes == 2, rng.integers(lo, min(FREE_DONE[1], max_steps) + 1, size=n), max_steps + 1)
        self.reached = [set() for _ in range(n)]                       # free: goals reached (hindsight request)

    def _pick(self, b):
        row = np.full(MAX_GOALS, -1)
        row[0] = self.rng.choice(self.seen)
        return row

    def __call__(self, env, t):
        n = env.B
        rng = self.rng
        live = ~env.done
        # free explorers: next goal once the current one holds; done at their stop time
        for b in np.nonzero(live & (self.modes == 2))[0]:
            g = self.goals[b, 0]
            if env.inv[b, GOALS[g].item] >= GOALS[g].count:
                self.reached[b].add(int(g))
                self.goals[b] = self._pick(b)
        ex = self.expert(env, self.goals, np.nonzero(live)[0])
        a = np.where(rng.random(n) < self.p, ex, rng.integers(N_WORLD_ACTIONS, size=n))
        # wander segments (wander and free explorers): a persistent random walk that uses what it faces and presses
        # craft now and then
        can = self.modes >= 1
        start = can & (self.wander == 0) & (rng.random(n) < WANDER_START)
        self.wander[start] = rng.integers(WANDER_LEN[0], WANDER_LEN[1] + 1, size=int(start.sum()))
        w = np.nonzero(live & (self.wander > 0))[0]
        if len(w):
            turn = rng.random(len(w)) < 0.3
            self.heading[w[turn]] = rng.integers(4, size=int(turn.sum()))
            faced = env.grid[w, env.pos[w, 0] + DIRS[env.face[w], 0], env.pos[w, 1] + DIRS[env.face[w], 1]]
            u = rng.random(len(w))
            act = self.heading[w].copy()
            act[(u < 0.5) & np.isin(faced, list(COLLECT))] = USE
            act[(u >= 0.5) & (u < 0.65)] = CRAFT
            a[w] = act
            self.wander[w] -= 1
        idle = live & (self.modes == 2) & (ex == DONE) & (env.t < self.stop)   # goal already held: walk on
        a[idle] = self.heading[idle]
        a[live & (env.t >= self.stop)] = DONE
        return a


def seen_goals(heldout):
    return np.array([g for g in range(N_GOALS) if g not in set(heldout)])


def build_corpus(cfg, n_episodes, heldout, seed=0, chunk=512, workers=None, profile="realistic", mode_p=MODE_P):
    """The pretraining corpus. Returns per-step arrays (grid, pf, inv, a, b), per-episode request / message /
    hindsight / mode and `spans` (per episode: list of (sentence, t0, t1) for the narration sentences of its message),
    and the coverage / audit counts."""
    rng = np.random.default_rng(seed)
    held = sorted(set(int(h) for h in heldout))
    seen = seen_goals(held)
    held_items = [GOALS[h].item for h in held]
    held_words = {w for h in held for w in GOALS[h].phrase.split()} - {"a", "an", "the", "make", "build", "gather"}
    expert = ExpertPool(workers)
    out = dict(grid=[], pf=[], inv=[], a=[], b=[], request=[], message=[], hindsight=[], mode=[], spans=[])
    counts = Counter()
    cov = dict(collected=Counter(), crafted=Counter(), craft_attempts=set(), end_inv=np.zeros(N_ITEMS), kinds=Counter(),
               success=Counter(), episodes_mode=Counter(), length=0, sentences=0, distinct=set())
    n_ep = 0
    while counts["generated"] < n_episodes:
        n = min(chunk, n_episodes - counts["generated"])
        req = sample_requests(rng, n, rng.integers(0, 2 ** 31, size=n), goal_pool=seen)
        modes = rng.choice(len(MODES), size=n, p=mode_p)
        ex = Explorer(None, req, modes, rng, expert, seen, cfg.max_steps)
        env, log, snaps, acts, valid, sat, mask = run_episodes(cfg, req, ex)
        facts = episode_facts_all(log, env, req.goals, sat)
        ev = event_arrays(log)
        done_by_agent = valid.sum(1) < cfg.max_steps
        grid, pf, inv, a, b, _ = stack_steps(snaps, acts, valid)
        new_id = np.full(n, -1)
        for e, f in enumerate(facts):
            counts["generated"] += 1
            if any(f.collected[i] + f.crafted[i] > 0 for i in held_items):
                counts["dropped_heldout_made"] += 1
                continue
            parts = compose(f, profile, rng)
            text = " ".join(x[2] for x in parts) or "Okay."
            words = set(re.findall(r"[a-z]+", text.lower()))
            if words & held_words:
                counts["dropped_heldout_word"] += 1            # should never happen: nothing about them is ever true
                continue
            new_id[e] = n_ep
            n_ep += 1
            out["request"].append(req.text(e))
            out["message"].append(text)
            out["mode"].append(MODES[modes[e]])
            narrated = {c for c, _, _ in f.events}            # narration claims only, one span per event segment
            spans = []
            for _, claim, sentence in parts:
                if claim not in narrated:
                    continue
                steps = claim_steps(ev, e, claim)
                steps = steps[steps < env.t[e]]
                for seg in np.split(steps, np.nonzero(np.diff(steps) > 1)[0] + 1):   # contiguous runs = events
                    if len(seg):
                        spans.append((sentence, int(seg[0]), int(seg[-1])))
            out["spans"].append(spans)
            ok = bool((sat[e] | ~mask[e]).all()) and done_by_agent[e]
            if modes[e] == 2:                                   # free: a subset of what it reached and still holds
                reached = sorted(g for g in ex.reached[e] if env.inv[e, GOALS[g].item] >= GOALS[g].count)
                if reached:
                    k = int(rng.integers(1, min(MAX_GOALS, len(reached)) + 1))
                    gs = sorted(rng.choice(reached, size=k, replace=False).tolist(), reverse=True)
                    out["hindsight"].append(" and ".join(GOALS[g].phrase for g in gs))
                else:
                    out["hindsight"].append(None)
            else:
                out["hindsight"].append(req.text(e) if ok else None)
            counts["hindsight"] += out["hindsight"][-1] is not None
            # coverage
            cov["episodes_mode"][MODES[modes[e]]] += 1
            cov["success"][MODES[modes[e]]] += ok
            cov["collected"].update({ITEM_NAMES[i]: c for i, c in f.collected.items()})
            cov["crafted"].update({ITEM_NAMES[i]: c for i, c in f.crafted.items()})
            cov["end_inv"] += np.array(f.final_inv)
            cov["length"] += f.steps
            cov["kinds"].update(c[0] for _, c, _ in parts)
            cov["sentences"] += len(parts)
            cov["distinct"].update(split_sentences(text))
            typ, st = ev["type"][:, e], ev["station"][:, e]
            slot = slot_steps(typ, ev["item"][:, e], st)
            for t in np.nonzero(np.isin(typ, (EV_CRAFT_FAIL, EV_CRAFT)) & (st >= 0))[0]:
                cov["craft_attempts"].add((int(st[t]), tuple(np.nonzero(slot[t] > 0)[0].tolist())))
        sel = new_id[b] >= 0
        for k, v in (("grid", grid), ("pf", pf), ("inv", inv), ("a", a)):
            out[k].append(v[sel])
        out["b"].append(new_id[b[sel]])
    expert.close()
    for k in ("grid", "pf", "inv", "a", "b"):
        out[k] = np.concatenate(out[k])
    counts["episodes"] = n_ep
    counts["steps"] = len(out["a"])
    counts["heldout_items_in_inventories"] = int(out["inv"][:, held_items].sum()) if held_items else 0
    words = Counter(w for m in out["message"] + out["request"] for w in re.findall(r"[a-z]+", m.lower()))
    counts["word_types"] = len(words)
    counts["heldout_words_in_text"] = sum(words[w] for w in held_words)
    stats = dict(
        episodes_per_mode=dict(cov["episodes_mode"]),
        success_per_mode={m: round(cov["success"][m] / max(1, cov["episodes_mode"][m]), 3) for m in MODES},
        mean_length=round(cov["length"] / max(1, n_ep), 1),
        collected_per_episode={k: round(v / n_ep, 2) for k, v in sorted(cov["collected"].items())},
        crafted_per_episode={k: round(v / n_ep, 3) for k, v in sorted(cov["crafted"].items())},
        end_inventory_mean={ITEM_NAMES[i]: round(float(x) / n_ep, 2) for i, x in enumerate(cov["end_inv"]) if x},
        distinct_craft_situations=len(cov["craft_attempts"]),
        sentence_kinds=dict(cov["kinds"].most_common()),
        sentences_per_message=round(cov["sentences"] / max(1, n_ep), 2),
        distinct_sentences=len(cov["distinct"]),
        drop_rate=round(counts["dropped_heldout_made"] / max(1, counts["generated"]), 4),
    )
    return out, dict(counts), stats


def audit(corpus, counts, heldout):
    """Assert the leakage rules: no held-out item in any inventory, no held-out word in any text, no held-out goal in
    any request or hindsight request."""
    held_phr = {GOALS[h].phrase for h in heldout}
    assert counts["heldout_items_in_inventories"] == 0, counts
    assert counts["heldout_words_in_text"] == 0, counts
    for r in corpus["request"] + [h for h in corpus["hindsight"] if h]:
        assert not (set(r.split(" and ")) & held_phr), r
    return True


# ---------------------------------------------------------------- model

STATION_WORDS = {"workbench", "loom", "kiln", "site"}
PLACE_WORDS = STATION_WORDS | {"tree", "reeds", "clay", "pit", "rock"}
ITEM_WORDS = set(ITEM_NAMES) | {n + "s" for n in ITEM_NAMES}
CONTENT_WORDS = PLACE_WORDS | ITEM_WORDS | {w for g in GOALS for w in g.phrase.split()} - {"a", "an"}
GROUNDED_ARCH = dict(d=256, w=128, layers=4, pre_conv=2, rec=6, reader_layers=3, reader_frames="cheap")


def word_class(word):
    """'station' / 'item' / 'number' / 'other': the class a word is reported under in the grounding diagnostics."""
    if word in STATION_WORDS:
        return "station"
    if word in ITEM_WORDS:
        return "item"
    if word.isdigit():
        return "number"
    return "other"


_CONTENT_IDS = None


def content_mask_table(prompt):
    """(num_embeddings,) bool: exact-vocab rows that are content words (items, places, goal words, numbers)."""
    global _CONTENT_IDS
    if _CONTENT_IDS is None or len(_CONTENT_IDS) != prompt.tok.num_embeddings:
        words = list(WORDS) + [str(i) for i in range(NUMBERS)]
        m = torch.zeros(prompt.tok.num_embeddings, dtype=torch.bool)
        for i, w in enumerate(words):
            if w in CONTENT_WORDS or w.isdigit():
                m[i + 1] = True
        _CONTENT_IDS = m
    return _CONTENT_IDS.to(prompt.tok.weight.device)


class TrajectoryReader(nn.Module):
    """Bidirectional transformer over [words ; frame tokens ; action tokens] that fills masked words from the
    trajectory (and masked actions from the words and the frames around them). Word embeddings and the output head
    are tied to the policy's prompt encoder, frame tokens come from the policy's frame encoder."""

    def __init__(self, w, d, vocab, layers=3, heads=4, max_t=256, max_k=32):
        super().__init__()
        self.w = w
        self.frame_in = nn.Linear(d, w)
        self.inv_in = nn.Linear(AGENT_FEATS, w)
        self.act_emb = nn.Embedding(N_ACTIONS + 1, w)            # +1: masked action
        self.time = nn.Embedding(max_t + 2, w)
        self.kind = nn.Embedding(3, w)                           # word / frame / action
        self.mask_word = nn.Parameter(torch.zeros(w))
        layer = nn.TransformerEncoderLayer(w, heads, 4 * w, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(w)
        self.act_out = nn.Linear(w, N_ACTIONS)
        self.word_bias = nn.Parameter(torch.zeros(vocab))      # per-word output bias of the tied head

    def forward(self, prompt: PromptEncoder, tokens, masked, h_frames, inv, acts, act_masked, times, fvalid,
                frames_on=True):
        """tokens (B, L) crc tokens, masked (B, L) bool; h_frames (B, K, d), inv (B, K, AGENT_FEATS), acts (B, K) long,
        act_masked (B, K) bool, times (B, K) long, fvalid (B, K) bool. Returns word logits (B, L, V) and action
        logits (B, K, N_ACTIONS). frames_on=False drops the frame and action tokens (text-only baseline)."""
        B, L = tokens.shape
        ids = prompt.ids(tokens)
        e = prompt.tok(ids)
        e = torch.where(masked[..., None], self.mask_word.expand_as(e), e)      # mask the word, keep its position
        e = e + prompt.pos.weight[:L] + self.kind.weight[0]
        wvalid = tokens > 0
        if frames_on:
            tt = self.time(times.clamp(min=0))
            f = self.frame_in(h_frames) + self.inv_in(torch.log1p(inv.float())) + tt + self.kind.weight[1]
            a = self.act_emb(torch.where(act_masked, torch.full_like(acts, N_ACTIONS), acts)) + tt + self.kind.weight[2]
            x = torch.cat([e, f, a], 1)
            valid = torch.cat([wvalid, fvalid, fvalid], 1)
        else:
            x, valid = e, wvalid
        x = self.ln(self.blocks(x, src_key_padding_mask=~valid))
        words = x[:, :L] @ prompt.tok.weight.T / self.w ** 0.5 + self.word_bias   # tied head, scaled
        acts_out = self.act_out(x[:, L:L + h_frames.shape[1]]) if frames_on else None
        return words, acts_out


class GroundedPolicy(nn.Module):
    """pi(a | pixels, prompt) with token-level text fusion (FusionEncoder: cell tokens cross-attend to the words), a
    value head, and the TrajectoryReader for pretraining. Same calling convention as learn.CleanPolicy: prompts are
    encoded once per episode (encode_prompt -> packed word states) and indexed per step; forward(x, inv, p) ->
    (action logits, value)."""

    def __init__(self, size, d=256, w=128, layers=4, pre_conv=2, rec=6, reader_layers=3, reader_frames="cheap",
                 crop=3, sep=False, **unknown):
        """reader_frames: what the reader sees of a frame -- "cheap": the encoder's conv features (agent-centred crop
        around the located agent + global mean / max pooling; a few convs per frame, so the reader batch can be
        large), or "full": the text-free [ACT] token of the whole fusion encoder (one transformer pass per frame)."""
        super().__init__()
        assert not unknown, f"unknown GroundedPolicy arch keys: {sorted(unknown)}"
        self.d, self.w, self.kind, self.text_model = d, w, "grounded", None
        self.reader_frames, self.crop = reader_frames, crop
        self.prompt = PromptEncoder(d, w, states=True, exact=True, sep=sep)
        self.enc = FusionEncoder(size, d, w, dt=w, layers=layers, pre_conv=pre_conv, pixels=True, rec=rec)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)
        if reader_frames == "cheap":
            self.frame_fc = nn.Linear(w * crop * crop + 2 * w, d)
        self.reader = TrajectoryReader(w, d, self.prompt.tok.num_embeddings, layers=reader_layers)

    def encode_prompt(self, tokens):
        return self.prompt(tokens)

    def forward(self, x, inv, p, return_h=False):
        h, f = self.enc(x, inv, p, return_f=True)
        out = (self.pi(h), self.v(h).squeeze(-1))
        return out + ((h, f, x),) if return_h else out

    def conv_features(self, x):
        """The fusion encoder's cell features before the transformer blocks: (B, w, S, S)."""
        e = self.enc
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        from gft.minicraft.render import render
        c = F_.relu(e.patch(render(x)))
        for conv in e.convs:
            c = F_.relu(conv(c) + c)
        if e.rec:
            u = c
            for _ in range(e.rec):
                c = F_.relu(e.rconv(c) + u)
        return c

    def frames(self, x, inv):
        """Text-free frame summaries (B, d) for the reader (see reader_frames)."""
        if self.reader_frames != "cheap":
            return self.enc(x, inv, None)
        c = self.conv_features(x)
        a = F_.softmax(self.enc.locate(c).flatten(1).float(), -1).to(c.dtype)      # where the agent is
        local = attention_crop(c, a, self.crop)                                     # (B, w * crop * crop)
        return self.frame_fc(torch.cat([local, c.mean((2, 3)), c.amax((2, 3))], 1))


# ---------------------------------------------------------------- pretraining

def pick_frames(t0, t1, length, k, rng):
    """Up to k step indices in [t0, min(t1 + 1, length - 1)]: both ends, the event step t1, uniform in between."""
    hi = min(t1 + 1, length - 1)
    lo = max(0, min(t0, hi))
    n = hi - lo + 1
    if n <= k:
        return np.arange(lo, hi + 1)
    must = {lo, hi, min(t1, hi)}
    pool = [t for t in range(lo, hi + 1) if t not in must]
    extra = rng.choice(pool, size=k - len(must), replace=False)
    return np.array(sorted(must | set(extra.tolist())))


class PretrainData:
    """The corpus on the device, indexed for the three BC modes and the reader."""

    def __init__(self, corpus, device, rng, heldout_frac=0.05):
        self.device = device
        E = len(corpus["request"])
        self.E = E
        b = corpus["b"]
        self.first = np.searchsorted(b, np.arange(E))
        self.length = np.bincount(b, minlength=E)
        self.held = np.zeros(E, bool)
        self.held[rng.choice(E, size=int(heldout_frac * E), replace=False)] = True
        has_h = np.array([h is not None for h in corpus["hindsight"]])
        self.has_h = has_h
        tok_h = tokenize([prompt_text(h or "") for h in corpus["hindsight"]])
        tok_m = tokenize([prompt_text(r, m) for r, m in zip(corpus["request"], corpus["message"])])
        # swapped messages for the held-out episodes (same request where possible): the information-gain baseline
        swap = np.arange(E)
        by_req = {}
        for e in np.nonzero(self.held)[0]:
            by_req.setdefault(corpus["request"][e], []).append(e)
        for e in np.nonzero(self.held)[0]:
            same = [o for o in by_req[corpus["request"][e]] if o != e]
            swap[e] = rng.choice(same) if same else rng.choice([o for o in np.nonzero(self.held)[0] if o != e] or [e])
        tok_s = tokenize([prompt_text(r, corpus["message"][swap[e]]) for e, r in enumerate(corpus["request"])])
        # narration spans: (episode, t0, t1) with the prompt request + that sentence
        sp = [(e, t0, t1, s) for e, spans in enumerate(corpus["spans"]) for s, t0, t1 in spans if t1 >= t0]
        self.span_ep = np.array([x[0] for x in sp], dtype=np.int64)
        self.span_t0 = np.array([x[1] for x in sp], dtype=np.int64)
        self.span_t1 = np.array([x[2] for x in sp], dtype=np.int64)
        tok_sp = tokenize([prompt_text(corpus["request"][e], s) for e, _, _, s in sp]) if sp else torch.zeros((0, 1), dtype=torch.long)
        L = max(1, int(max((t > 0).sum(1).max() if t.numel() else 0 for t in (tok_h, tok_m, tok_s, tok_sp))))
        pad = lambda t: F_.pad(t[:, :L], (0, L - min(L, t.shape[1])))
        self.tok = torch.cat([pad(tok_h), pad(tok_m), pad(tok_s)]).to(device)      # e | E+e | 2E+e
        self.tok_sp = pad(tok_sp).to(device)
        self.a = torch.as_tensor(corpus["a"], device=device)
        self.b = torch.as_tensor(b, device=device)
        self.grid, self.pf, self.inv = on_device(device, corpus["grid"], corpus["pf"].astype(np.int16),
                                                 corpus["inv"].astype(np.int16))
        train_ep = ~self.held
        self.steps_h = torch.as_tensor(np.nonzero(has_h[b] & train_ep[b])[0], device=device)
        self.steps_m = torch.as_tensor(np.nonzero(train_ep[b])[0], device=device)
        # span steps: every (span, step) pair inside a training span, flattened
        sp_train = np.nonzero(train_ep[self.span_ep])[0] if len(sp) else np.zeros(0, int)
        rows = [np.stack([np.full(self.span_t1[i] - self.span_t0[i] + 1, i),
                          self.first[self.span_ep[i]] + np.arange(self.span_t0[i], self.span_t1[i] + 1)], 1)
                for i in sp_train]
        rows = np.concatenate(rows) if rows else np.zeros((0, 2), int)
        rows = rows[rows[:, 1] < self.first[self.span_ep[rows[:, 0]]] + self.length[self.span_ep[rows[:, 0]]]] if len(rows) else rows
        self.span_rows = torch.as_tensor(rows, device=device)
        self.train_eps = np.nonzero(train_ep)[0]
        self.held_eps = np.nonzero(self.held)[0]
        self.train_spans = sp_train
        self.held_spans = np.nonzero(self.held[self.span_ep])[0] if len(sp) else np.zeros(0, int)

    def obs(self, idx):
        return obs_tensor(self.grid[idx], self.pf[idx], self.inv[idx], self.device)

    def reader_batch(self, kind, idx, k, rng):
        """Frames / actions / tokens of reader samples: kind 'episode' (idx = episodes, prompt = request + message,
        frames over the whole episode) or 'span' (idx = spans, prompt = request + sentence, frames over the span).
        Returns step indices (B, K) with -1 padding, tokens (B, L), times (B, K)."""
        B = len(idx)
        steps = np.full((B, k), -1, dtype=np.int64)
        times = np.zeros((B, k), dtype=np.int64)
        for j, i in enumerate(idx):
            if kind == "episode":
                e, t0, t1 = int(i), 0, int(self.length[i]) - 1
            else:
                e, t0, t1 = int(self.span_ep[i]), int(self.span_t0[i]), int(self.span_t1[i])
            ts = pick_frames(t0, t1, int(self.length[e]), k, rng)
            steps[j, :len(ts)] = self.first[e] + ts
            times[j, :len(ts)] = ts
        tok = self.tok[self.E + torch.as_tensor(idx, device=self.device)] if kind == "episode" \
            else self.tok_sp[torch.as_tensor(idx, device=self.device)]
        return torch.as_tensor(steps, device=self.device), tok, torch.as_tensor(times, device=self.device)


def mask_words(tokens, content, gen, p_content=0.5, p_other=0.1):
    """Random word mask: content words w.p. p_content, others p_other, at least one word per row."""
    is_content = content[tokens.clamp(max=len(content) - 1)] & (tokens > 0)   # tokens: exact-vocab ids
    p = torch.where(is_content, p_content, p_other)
    m = (torch.rand(tokens.shape, generator=gen, device=tokens.device) < p) & (tokens > 0)
    none = ~m.any(1)
    if none.any():
        valid = (tokens > 0).float()
        pick = torch.multinomial(valid[none] + 1e-9, 1).squeeze(1)
        m[torch.nonzero(none).squeeze(1), pick] = True
    return m


def pretrain_grounded(cfg, device, corpus, steps=10_000, batch=512, reader_batch=256, k_frames=12, lr=3e-4, seed=0,
                      arch=None, log_every=0, heldout_frac=0.05, w_text=1.0, w_act=0.5, span_share=0.3,
                      request_share=0.5, p_act_mask=0.2, max_eval=64, bf16=True, compile=True):
    """Pretraining v2 on a build_corpus corpus. Each step: a BC batch (request mode on episodes with a hindsight
    label, message mode, and span mode: request + one narration sentence, loss on that sentence's steps) plus a
    reader batch (masked words / masked actions given the frames, half whole episodes, half narration spans).
    Diagnostics on held-out episodes: CE per BC mode, message information gain, masked-word accuracy per word class
    with the frames vs. without them (the grounding gap), masked-action accuracy.

    Speed (CUDA only; profiled 2026-10-29 on an RTX PRO 4000: 581 -> 163 ms per step, same losses): TF32 matmuls /
    convs, bf16 autocast for the forward passes (losses in fp32) and torch.compile of the policy and frame encoders."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arch = {**GROUNDED_ARCH, **(arch or {})}
    model = GroundedPolicy(cfg.size, **arch).to(device)
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 and cuda else \
        (lambda: torch.autocast("cpu", enabled=False))
    policy_fwd = torch.compile(model.forward) if compile and cuda else model.forward
    frames_fwd = torch.compile(model.frames) if compile and cuda else model.frames
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    D = PretrainData(corpus, device, rng, heldout_frac)
    content = content_mask_table(model.prompt)
    gen = torch.Generator(device=device).manual_seed(seed)
    E = D.E
    n_req = int(batch * request_share)
    n_span = int(batch * span_share) if len(D.span_rows) else 0
    n_msg = batch - n_req - n_span

    def bc_batch():
        i_h = D.steps_h[torch.randint(len(D.steps_h), (n_req,), device=device, generator=gen)]
        i_m = D.steps_m[torch.randint(len(D.steps_m), (n_msg,), device=device, generator=gen)]
        pid = torch.cat([D.b[i_h], D.b[i_m] + E])
        tok = D.tok
        idx = torch.cat([i_h, i_m])
        if n_span:
            r = D.span_rows[torch.randint(len(D.span_rows), (n_span,), device=device, generator=gen)]
            idx = torch.cat([idx, r[:, 1]])
            # span prompts live in a separate table: encode both and concatenate
            eps, inverse = torch.unique(pid, return_inverse=True)
            sps, inv_s = torch.unique(r[:, 0], return_inverse=True)
            p = torch.cat([model.encode_prompt(tok[eps])[inverse], model.encode_prompt(D.tok_sp[sps])[inv_s]])
        else:
            eps, inverse = torch.unique(pid, return_inverse=True)
            p = model.encode_prompt(tok[eps])[inverse]
        x, iv = D.obs(idx)
        logits, _ = policy_fwd(x, iv, p)
        return F_.cross_entropy(logits.float(), D.a[idx], reduction="none")

    def reader_draw(kind, idx, rng_, gen_):
        """One random draw of a reader batch: frames, tokens, times, word and action masks."""
        steps_, tok, times = D.reader_batch(kind, idx, k_frames, rng_)
        fvalid = steps_ >= 0
        ids = model.prompt.ids(tok)
        wmask = mask_words(ids, content, gen_)
        amask = (torch.rand(fvalid.shape, generator=gen_, device=device) < p_act_mask) & fvalid
        return steps_, tok, times, fvalid, ids, wmask, amask

    def reader_forward(kind, idx, frames_on=True, draw=None):
        """kind: 'episode', 'span', or 'both' (idx = (episodes, spans): one forward for the two sample kinds)."""
        if kind == "both":
            d1, d2 = reader_draw("episode", idx[0], rng, gen), reader_draw("span", idx[1], rng, gen)
            draw = tuple(torch.cat([a, b]) for a, b in zip(d1, d2))
            n = len(idx[0]) + len(idx[1])
        else:
            n = len(idx)
        steps_, tok, times, fvalid, ids, wmask, amask = draw or reader_draw(kind, idx, rng, gen)
        flat = steps_.clamp(min=0).flatten()
        x, iv = D.obs(flat)
        h = frames_fwd(x, iv).view(n, k_frames, -1)
        inv_k = D.inv[flat].view(n, k_frames, -1)
        acts = D.a[flat].view(n, k_frames)
        words, aout = model.reader(model.prompt, tok, wmask, h, inv_k, acts, amask, times, fvalid, frames_on=frames_on)
        l_text = F_.cross_entropy(words[wmask].float(), ids[wmask])
        l_act = F_.cross_entropy(aout[amask].float(), acts[amask]) if frames_on and amask.any() else \
            torch.zeros((), device=device)
        return l_text, l_act, (words, ids, wmask, aout, acts, amask)

    @torch.no_grad()
    def ce_on(eps, offset):
        """Mean per-step CE of BC on all steps of episodes `eps` with prompt table offset (0 request, E message,
        2E swapped)."""
        eps = [e for e in eps if offset or D.has_h[e]][:max_eval]
        if not eps:
            return float("nan")
        idx = torch.cat([torch.arange(D.first[e], D.first[e] + D.length[e], device=device) for e in eps])
        pid = D.b[idx] + offset
        u, inverse = torch.unique(pid, return_inverse=True)
        tot, n = 0.0, 0
        for j in range(0, len(idx), 2048):
            sl = slice(j, j + 2048)
            x, iv = D.obs(idx[sl])
            with amp():
                logits, _ = policy_fwd(x, iv, model.encode_prompt(D.tok[u])[inverse[sl]])
            tot += float(F_.cross_entropy(logits.float(), D.a[idx[sl]], reduction="sum"))
            n += len(idx[sl])
        return tot / n

    @torch.no_grad()
    def grounding(eps):
        """Masked-word accuracy per word class with and without the frames on held-out episodes: the same frames
        and masks for both (own diagnostic RNGs, so logging does not change the training stream)."""
        out = {}
        d_rng, d_gen = np.random.default_rng(seed + 1), torch.Generator(device=device).manual_seed(seed + 1)
        draws = [(eps[j:j + 16], reader_draw("episode", eps[j:j + 16], d_rng, d_gen)) for j in range(0, len(eps), 16)]
        for on in (True, False):
            hits, tot = Counter(), Counter()
            for idx, draw in draws:
                with amp():
                    _, _, (words, ids, wmask, aout, acts, amask) = reader_forward("episode", idx, frames_on=on, draw=draw)
                pred = words.argmax(-1)
                for i, t in zip(ids[wmask].tolist(), pred[wmask].tolist()):
                    c = word_class(_word_of(i))
                    tot[c] += 1
                    hits[c] += t == i
                if on and amask.any():
                    tot["action"] += int(amask.sum())
                    hits["action"] += int((aout.argmax(-1)[amask] == acts[amask]).sum())
            for c in tot:
                out[f"{'with' if on else 'without'}_frames_{c}"] = round(hits[c] / tot[c], 3)
        for c in ("station", "item", "number", "other"):
            if f"with_frames_{c}" in out:
                out[f"gap_{c}"] = round(out[f"with_frames_{c}"] - out.get(f"without_frames_{c}", 0.0), 3)
        return out

    hist = []
    for s in range(steps):
        model.train()
        n_ep = reader_batch // 2
        ep_idx = D.train_eps[rng.integers(len(D.train_eps), size=n_ep)]
        with amp():
            ce = bc_batch()
            if len(D.train_spans):
                sp_idx = D.train_spans[rng.integers(len(D.train_spans), size=reader_batch - n_ep)]
                lt, la, _ = reader_forward("both", (ep_idx, sp_idx))
            else:
                lt, la, _ = reader_forward("episode", ep_idx)
        lt_e = lt_s = lt
        la_e = la_s = la
        loss = ce.mean() + w_text * lt + w_act * la
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (s + 1) % log_every == 0:
            model.eval()
            held = D.held_eps[:max_eval]
            own, swapped = ce_on(held, E), ce_on(held, 2 * E)
            ce = ce.detach()
            row = dict(step=s + 1, bc_request=float(ce[:n_req].mean()), bc_message=float(ce[n_req:n_req + n_msg].mean()),
                       bc_span=float(ce[n_req + n_msg:].mean()) if n_span else float("nan"),
                       text_loss=float((lt_e + lt_s).detach() / 2), act_loss=float((la_e + la_s).detach() / 2),
                       heldout_request=ce_on(held, 0), heldout_message=own, info_gain=swapped - own)
            row.update(grounding(held))
            hist.append(row)
            print(f"  pretrain {s + 1}/{steps}: BC req {row['bc_request']:.3f} msg {row['bc_message']:.3f} span "
                  f"{row['bc_span']:.3f} | text {row['text_loss']:.3f} act {row['act_loss']:.3f} | held-out req "
                  f"{row['heldout_request']:.3f} msg {own:.3f} gain {row['info_gain']:+.4f} | masked acc station "
                  f"{row.get('with_frames_station', float('nan')):.2f}/{row.get('without_frames_station', float('nan')):.2f} "
                  f"item {row.get('with_frames_item', float('nan')):.2f}/{row.get('without_frames_item', float('nan')):.2f} "
                  f"action {row.get('with_frames_action', float('nan')):.2f}", flush=True)
    model.eval()
    return model, hist


_WORDS_BY_ID = None


def _word_of(exact_id):
    global _WORDS_BY_ID
    if _WORDS_BY_ID is None:
        _WORDS_BY_ID = [""] + list(WORDS) + [str(i) for i in range(NUMBERS)]
    return _WORDS_BY_ID[exact_id] if exact_id < len(_WORDS_BY_ID) else "<unk>"
