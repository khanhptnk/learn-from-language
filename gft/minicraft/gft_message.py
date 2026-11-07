"""GFT, the one implementation (docs/STATUS.md 2026-11-06): the old curriculum's run 17g ported to this game, with the
pass ratio R3 (pi, reader Stage I, reader kept, scorer) = (4, 1, 2, 1), Stage II tasks from every Stage I episode,
symmetric scorer pairs, a built success message, and a reader that writes its own messages (2026-11-07).

The built success message of a request: the teacher's praise and a "done" verdict for each requested goal, in
wordings received before (for any goal). It needs no recipe knowledge -- only the request and how the teacher says
"done" -- and exists before the request's first success.

The first warmup_rounds rounds are Stage I only: the reader's heads and the scorer learn from pi's episodes and the
teacher's messages before Stage II acts on what they say.

One round:
  Stage I   pi plays `episodes` curriculum tasks under the request; the teacher answers each. Success is read from
            the message (no "not done" verdict) and its message joins the request's success bank (a failure's joins
            its failure bank). The reader pi^fb (a copy of pi) imitates the episodes under request + message
            (hindsight; whole sentences dropped with probability text_sentence_dropout), on a replay of the last
            replay_rounds rounds. The reader's message head (MessageWriter) learns to write the teacher's message
            of each episode from its request and first frame, on the same replay. The scorer D(x, m) learns which
            message goes with which episode
            (message_scorer_loss: the episode's own message against its improved message, if it failed the built
            success message and a received success message of its request, if it succeeded a received failure message
            of it, and other episodes' messages; and the message against other episodes), on a replay of the last
            disc_replay rounds.
  Stage II  stage2 // rerank_k Stage I episodes become tasks (their world and request). Each task has two targets:
            the source's improved message (improve.py: the stated cause fixed; none for a success or a message stating
            no cause) and the ideal set = the built success message + the request's last ideal_k received success
            messages. The reader makes rerank_k attempts at stage2_temperature: half under messages it writes itself
            (one per attempt, from the task's request and first frame), half under the task's m* (the improved
            message with probability 1/2, else a member of the ideal set). Each task keeps the top keep_frac (rounded
            up) of its attempts with s > 0 -- whatever prompted them -- where
              s = 1/2 [D(x', imp) - D(x_src, imp)] + 1/2 [D(x', ideal) - D(x_src, ideal)]
            (D(., ideal) = mean D over the ideal set; a missing target drops its half). pi distils the kept attempts
            under the request (damped: stay near pi_old on Stage II states and, with stay_stage1, on its own Stage I
            states; entropy controller), and the reader imitates them under the message that prompted each; its
            message head learns to write that prompt (a written message or m* alike).

Nothing here reads the environment's success or the teacher's facts: the curriculum does (shared with PPO), and
diagnostics (logged only) do.
"""

import copy
import math
import re
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

import gft.minicraft.learn as learn_mod
from gft.minicraft.core import N_ACTIONS, N_ITEMS
from gft.minicraft.goals import GOALS, N_GOALS
from gft.minicraft.improve import Received, _say_like, _say_received, improve_feedback
from gft.minicraft.learn import (MAX_TOKENS, Curriculum, CurriculumConfig, EntropyController, PromptEncoder,
                                 mean_stat, obs_tensor, on_device, policy_actions, prompt_text, prompt_tokens,
                                 request_prompts, run_episodes, split_prompt, stack_steps)
from gft.minicraft.teacher import compose, episode_facts_all
from gft.minicraft.verify import parse, split_sentences

STEP_FEATS = 2 * N_ITEMS + N_ACTIONS     # besides the frozen trunk vector: log1p inventory, its change, the action
SUCCESS_BANK = 32                        # received success (and failure) messages kept per request


def _lap(tm, key, t0, device):
    """Adds the seconds since t0 to tm[key], GPU work included (synchronised), and returns the new start."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t = time.time()
    tm[key] = tm.get(key, 0.0) + t - t0
    return t


# ---------------------------------------------------------------- text helpers

def _n_words(s):
    return len(re.findall(r"[a-z]+|\d+", s.lower()))


def sentence_prompts(requests, sentence_lists, device):
    """Prompts request + feedback sentences joined, as prompt tokens (B, L), and the feedback sentence index of every
    token (B, L) long (-1 for request words and padding), for whole-sentence dropout; tokens past MAX_TOKENS are cut.
    sentence_lists[i] = None -> the request alone."""
    texts, ws = [], []
    for r, ss in zip(requests, sentence_lists):
        if ss is None:
            texts.append(prompt_text(r))
            ws.append([-1] * _n_words(prompt_text(r)))
            continue
        texts.append(prompt_text(r, " ".join(ss)))
        w = [-1] * _n_words(f"request: {r} feedback:")
        for j, s in enumerate(ss):
            w += [j] * _n_words(s)
        ws.append(w[:MAX_TOKENS])
    tok = prompt_tokens(texts, device)
    out = torch.full(tok.shape, -1, dtype=torch.long)
    for i, w in enumerate(ws):
        out[i, :min(len(w), tok.shape[1])] = torch.tensor(w[:tok.shape[1]], dtype=torch.long)
    return tok, out.to(device)


class TextTable:
    """Distinct sentences seen by the scorer, with stable ids; tokens() tokenizes all of them (exact vocabulary)."""

    def __init__(self, strings=()):
        self.strings, self.index = [], {}
        for s in strings:
            self.id(s)

    def id(self, s):
        if s not in self.index:
            self.index[s] = len(self.strings)
            self.strings.append(s)
        return self.index[s]

    def tokens(self, device):
        return prompt_tokens([f"feedback: {s}" for s in self.strings] or ["feedback:"], device)


# ---------------------------------------------------------------- the scorer

def step_features(feat, grid, pf, state, act, b, n_ep, final_inv, device, T):
    """Per-step scorer inputs of whole episodes (steps grouped by episode in time order, as stack_steps gives them;
    grid/pf/state on the device; state = agent states, whose first N_ITEMS columns are the inventory):
    (n_ep, T, d + STEP_FEATS) float16 -- the frozen encoder's trunk vector of the state, log1p inventory, signed
    log1p inventory change caused by the step's action (the last step's from the episode's final inventory), the
    action one-hot -- and validity (n_ep, T)."""
    act = torch.as_tensor(np.asarray(act), device=device)
    with torch.no_grad():
        h = torch.cat([feat(*obs_tensor(grid[j:j + 4096], pf[j:j + 4096], state[j:j + 4096], device), None,
                            return_h=True)[2][0] for j in range(0, len(b), 4096)]).float()
    first = np.searchsorted(b, np.arange(n_ep + 1))
    inv_f = state[:, :N_ITEMS].float()
    nxt = torch.empty_like(inv_f)
    nxt[:-1] = inv_f[1:]
    nxt[torch.as_tensor(first[1:] - 1, device=device)] = torch.as_tensor(np.asarray(final_inv), device=device).float()
    d = nxt - inv_f
    x = torch.cat([h, torch.log1p(inv_f), torch.sign(d) * torch.log1p(d.abs()), F_.one_hot(act, N_ACTIONS).float()], -1)
    bb = torch.as_tensor(b, device=device)
    t = torch.arange(len(b), device=device) - torch.as_tensor(first[b], device=device)
    keep = t < T
    out = torch.zeros(n_ep, T, x.shape[1], device=device, dtype=torch.float16)
    valid = torch.zeros(n_ep, T, dtype=torch.bool, device=device)
    out[bb[keep], t[keep]] = x[keep].half()
    valid[bb[keep], t[keep]] = True
    return out, valid


class SentenceScorer(nn.Module):
    """d(x, sentence): how well one feedback sentence describes a trajectory. Steps -> tokens (frozen base trunk
    vector, inventory, inventory change, action, time) -> one transformer layer over the episode -> keys and values.
    The sentence (exact-vocabulary words, mean pooled) gives a query; its attention over the steps pools the values;
    the score is the cosine of the pooled vector and the sentence vector times a learned scale."""

    def __init__(self, d_s=256, d=128, dt=128, heads=4, max_steps=256, emb_std=0.02):
        super().__init__()
        self.d, self.dt = d, dt
        self.text = PromptEncoder(d, dt, states=True, exact=True)
        self.text_proj = nn.Sequential(nn.Linear(dt, d), nn.LayerNorm(d))
        self.step_in = nn.Sequential(nn.Linear(d_s + STEP_FEATS, d), nn.LayerNorm(d))
        self.time = nn.Embedding(max_steps, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.traj = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.k, self.v, self.q = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.sent_out = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
        for emb in (self.time, self.text.tok, self.text.pos):
            nn.init.normal_(emb.weight, std=emb_std)
        with torch.no_grad():
            self.text.tok.weight[0].zero_()

    def encode_text(self, tokens):
        """Sentence tokens (m, L) -> sentence vectors (m, d)."""
        _, _, mean = split_prompt(self.text(tokens), self.dt)
        return self.text_proj(mean)

    def encode_traj(self, X, valid):
        """Step features (n, T, F) and validity (n, T) -> keys, values (n, T, d)."""
        h = self.step_in(X.float()) + self.time.weight[:X.shape[1]]
        h = self.ln(self.traj(h, src_key_padding_mask=~valid))
        return self.k(h), self.v(h)

    def scale(self):
        """The learned score scale (inverse temperature), capped at 100 as in CLIP."""
        return self.logit_scale.clamp(max=math.log(100.0)).exp()


def message_totals(scorer, X, valid, eps, msgs, tok, device, mean=False):
    """D(x, f) for pairs (episode eps[i], message msgs[i] = sentence ids): the sum (mean: the mean) over the message's
    sentences of d(x, sentence), each with the scorer's own attention over the whole episode, with gradients.
    Computed as a (unique episode x unique sentence) matrix, then summed per pair. Returns (len(msgs),)."""
    ue, e_inv = np.unique(np.asarray(eps), return_inverse=True)
    us, s_inv = np.unique(np.array([s for m in msgs for s in m]), return_inverse=True)
    sel = torch.as_tensor(ue, device=device)
    K, V = scorer.encode_traj(X[sel].to(device), valid[sel].to(device))
    S = scorer.encode_text(tok[torch.as_tensor(us, device=device)])
    att = torch.einsum("etd,sd->est", K, scorer.q(S)) / math.sqrt(scorer.d)
    a = F_.softmax(att.masked_fill(~valid[sel].to(device)[:, None], float("-inf")), -1)
    z = F_.normalize(scorer.out(torch.einsum("est,etd->esd", a, V)), dim=-1)
    d = ((z * F_.normalize(scorer.sent_out(S), dim=-1)[None]).sum(-1) * scorer.scale()).float()    # (ue, us)
    lens = torch.as_tensor([len(m) for m in msgs], device=device)
    pe = torch.as_tensor(np.repeat(e_inv, [len(m) for m in msgs]), device=device)
    tot = torch.zeros(len(msgs), device=device).index_add(
        0, torch.repeat_interleave(torch.arange(len(msgs), device=device), lens), d[pe, torch.as_tensor(s_inv, device=device)])
    return tot / lens.clamp(min=1) if mean else tot


def bank_mean(scorer, X, valid, eps, banks, tok, device, chunk=256):
    """Mean over each bank's own messages of D(x, m) (message_totals, mean over sentences): row i -> episode eps[i],
    bank banks[i] (a list of messages, each a list of sentence ids, at least one). No gradients. Returns (len(eps),)."""
    pairs = [(i, m) for i, bk in enumerate(banks) for m in bk]
    out = np.zeros(len(eps))
    with torch.no_grad():
        for j in range(0, len(pairs), chunk):
            pp = pairs[j:j + chunk]
            d = message_totals(scorer, X, valid, [eps[i] for i, _ in pp], [m for _, m in pp], tok, device,
                               mean=True).cpu().numpy()
            np.add.at(out, [i for i, _ in pp], d)
    return out / np.array([len(bk) for bk in banks], float)


def message_scorer_loss(scorer, X, valid, meta, succ, fails, tok, rng, device, P=64, n_others=4):
    """The message-level scorer objective, from received feedback only: for an episode e with its message f_e
    (meta[e]["msg"], sentence ids), D = the mean of the sentence scores (a longer message must not score higher or
    lower for its length).
      message direction:    D(x_e, f_e) against D(x_e, f') for f' = its improved message (a step x_e did not take),
                            if e failed its built success message (meta[e]["ideal"]) and a received success message
                            of the same request (the ideal against a non-ideal episode), if e succeeded a received
                            failure message of it (a non-ideal message against the ideal episode), and other
                            episodes' messages;
      trajectory direction: D(x_e, f_e) against D(x_j, f_e) for other episodes j (messages differ).
    Cross-entropy with the positive first. succ / fails: request -> received success / failure messages. Returns
    (loss, diagnostics) or None."""
    cand = [e for e, m in enumerate(meta) if m["msg"]]
    if len(cand) < 2:
        return None
    eps = rng.choice(cand, size=min(P, len(cand)), replace=False)
    pe, pm, gm, gt = [], [], [], []

    def add(e, m):
        pe.append(int(e))
        pm.append(m)
        return len(pe) - 1
    for e in eps.tolist():
        m = meta[e]
        key = tuple(m["msg"])
        g = [add(e, m["msg"])]
        if m["imp"] and tuple(m["imp"]) != key:
            g.append(add(e, m["imp"]))
        if not m["ok"] and m.get("ideal") and tuple(m["ideal"]) not in (key, tuple(m["imp"] or ())):
            g.append(add(e, m["ideal"]))
        pool = [f for f in (fails if m["ok"] else succ).get(m["req"], ()) if tuple(f) != key]
        if pool:
            g.append(add(e, pool[int(rng.integers(len(pool)))]))
        others = [j for j in rng.choice(cand, size=min(len(cand), 4 * n_others), replace=False).tolist()
                  if j != e and tuple(meta[j]["msg"]) != key][:n_others]
        g += [add(e, meta[j]["msg"]) for j in others]
        gm.append(g)
        gt.append([g[0]] + [add(j, m["msg"]) for j in others])
    D = message_totals(scorer, X, valid, pe, pm, tok, device, mean=True)    # length must not decide

    def ce(groups):
        groups = [x for x in groups if len(x) > 1]
        if not groups:
            return torch.zeros((), device=device), torch.full((), float("nan"), device=device)
        L = max(len(x) for x in groups)
        idx = np.full((len(groups), L), -1)                      # one transfer, not one per group
        for i, x in enumerate(groups):
            idx[i, :len(x)] = x
        idx = torch.as_tensor(idx, device=device)
        M = D[idx.clamp(min=0)].masked_fill(idx < 0, float("-inf"))
        return -F_.log_softmax(M, 1)[:, 0].mean(), (M.argmax(1) == 0).float().mean()
    lm, am = ce(gm)
    lt, at = ce(gt)
    return lm + lt, dict(loss_msg=lm.detach(), loss_traj=lt.detach(), acc_msg=am, acc_traj=at)


# ---------------------------------------------------------------- the reader's message head

class MessageWriter(nn.Module):
    """p(f | request, first frame): writes a message one sentence at a time, each chosen among the sentences the
    teacher has said (the learner's TextTable), until STOP or max_sents. Request and sentences by one exact-vocabulary
    text encoder (mean word state); the first frame by the frozen base trunk vector (the scorer's view of a state); a
    GRU over the sentences written so far; a sentence's logit is the dot product of the state and its embedding."""

    def __init__(self, d_s, d=128, dt=128, max_sents=8):
        super().__init__()
        self.d, self.dt, self.max_sents = d, dt, max_sents
        self.text = PromptEncoder(d, dt, states=True, exact=True)
        self.text_proj = nn.Sequential(nn.Linear(dt, d), nn.LayerNorm(d))
        self.ctx = nn.Sequential(nn.Linear(d_s + d, d), nn.GELU(), nn.Linear(d, d))
        self.start = nn.Parameter(torch.zeros(d))
        self.gru = nn.GRUCell(d, d)
        self.q = nn.Linear(d, d)
        self.stop = nn.Linear(d, 1)

    def embed(self, tokens):
        return self.text_proj(split_prompt(self.text(tokens), self.dt)[2])

    def _logits(self, h, E):
        """(B, S + 1): every sentence of the table, then STOP."""
        return torch.cat([self.q(h) @ E.T / math.sqrt(self.d), self.stop(h)], -1)

    def loss(self, req_tok, h0, seqs, sent_tok):
        """Teacher-forced cross-entropy of messages seqs (lists of sentence ids; a message longer than max_sents is cut
        and has no STOP target), mean over predicted positions."""
        E = self.embed(sent_tok)
        S, B = E.shape[0], len(seqs)
        h = torch.tanh(self.ctx(torch.cat([h0, self.embed(req_tok)], -1)))
        inp = self.start.expand(B, -1)
        L = min(self.max_sents, max(len(q) for q in seqs))
        tgt = torch.full((B, L + 1), -100, dtype=torch.long, device=h0.device)
        for i, q in enumerate(seqs):
            q = q[:self.max_sents]
            tgt[i, :len(q)] = torch.as_tensor(q, device=h0.device)
            if len(q) <= L and len(seqs[i]) <= self.max_sents:
                tgt[i, len(q)] = S
        out = []
        for t in range(L + 1):
            h = self.gru(inp, h)
            out.append(self._logits(h, E))
            inp = E[tgt[:, t].clamp(0, S - 1)]
        return F_.cross_entropy(torch.stack(out, 1).flatten(0, 1), tgt.flatten(), ignore_index=-100)

    @torch.no_grad()
    def sample(self, req_tok, h0, sent_tok, temperature=1.0):
        """One message per row (at least one sentence, no sentence twice), as lists of sentence ids."""
        E = self.embed(sent_tok)
        S, B = E.shape[0], h0.shape[0]
        h = torch.tanh(self.ctx(torch.cat([h0, self.embed(req_tok)], -1)))
        inp = self.start.expand(B, -1)
        used = torch.zeros(B, S + 1, dtype=torch.bool, device=h0.device)
        alive = torch.ones(B, dtype=torch.bool, device=h0.device)
        out = [[] for _ in range(B)]
        for t in range(self.max_sents):
            h = self.gru(inp, h)
            lg = (self._logits(h, E) / temperature).masked_fill(used, float("-inf"))
            if t == 0:
                lg[:, S] = float("-inf")
            pick = torch.multinomial(F_.softmax(lg, -1), 1)[:, 0]
            alive &= pick != S
            for i in torch.nonzero(alive)[:, 0].tolist():
                out[i].append(int(pick[i]))
            used[alive, pick[alive]] = True
            if not alive.any():
                break
            inp = E[pick.clamp(max=S - 1)]
        return out


# ---------------------------------------------------------------- the learner

@dataclass
class GFTConfig:
    profile: str = "realistic"    # the teacher
    episodes: int = 256           # Stage I episodes (= teacher messages) per round
    stage2: int = 512             # Stage II reader attempts per round (stage2 // rerank_k tasks)
    rerank_k: int = 16            # attempts per task
    keep_frac: float = 0.5        # each task keeps the top keep_frac (rounded up) of its attempts with s > 0
    stage2_temperature: float = 1.0
    ideal_k: int = 8              # received success messages per request: the prompt is drawn from, and the ideal
                                  # averages over, the last ideal_k
    minibatch: int = 1024
    epochs_aux: int = 1           # reader passes over its Stage I replay per round
    epochs_clean: int = 4         # pi's distillation passes over the kept steps
    replay_rounds: int = 4        # Stage I rounds the reader trains on
    episode_chunk: int = 16       # reader minibatches draw steps in chunks of this many from few episodes
    text_sentence_dropout: float = 0.1   # whole feedback sentences dropped from the reader's prompt (Stage I)
    reader_imitate_epochs: int = 2       # reader passes over the kept attempts
    damp: float = 0.9             # pi: (1 - damp) * stay near pi_old + damp * push toward the kept actions
    stay_stage1: bool = True      # half of each stay minibatch from pi's own Stage I states
    lr_clean: float = 3e-4
    lr_aux: float = 3e-4
    aux_weight_decay: float = 1e-4
    entropy_target: float = 0.8   # pi's entropy controller (nats per step)
    entropy_alpha0: float = 0.1
    entropy_lr: float = 0.1
    disc_replay: int = 6          # Stage I rounds the scorer trains on
    disc_lr: float = 1e-3
    disc_epochs: float = 1.0      # scorer steps per round = disc_epochs * records / scorer_batch (capped)
    scorer_batch: int = 64
    scorer_max_steps: int = 200
    scorer_others: int = 4
    disc_emb_std: float = 0.02
    writer: bool = True           # the reader's message head; its messages are one of the Stage II prompt kinds
    writer_max_sents: int = 8
    writer_lr: float = 1e-3
    writer_epochs: int = 1        # message-head passes over its Stage I replay per round
    writer_imitate_epochs: int = 2   # message-head passes over the kept attempts' prompts that describe them
    writer_batch: int = 256       # messages per message-head update
    warmup_rounds: int = 10       # first rounds: Stage I only (the reader's heads and the scorer train on pi's episodes
                                  # and the teacher's messages; no Stage II, so pi is not updated); they count toward
                                  # the feedback budget like any round


class GFT:
    """GFT (module doc). pi (`clean`) is prompted with the request; the reader (`aux`) is a copy of the same network
    prompted with the request plus a message."""

    def __init__(self, cfg, pool, base, device, g: GFTConfig, seed=0, curriculum: CurriculumConfig = None):
        self.cfg, self.pool, self.device, self.g = cfg, pool, device, g
        self.rng = np.random.default_rng(seed)
        self.cur = curriculum if hasattr(curriculum, "sample") else Curriculum(pool, curriculum or CurriculumConfig())
        self.clean = base
        self.aux = copy.deepcopy(base)
        self.rounds = 0
        self.feat = copy.deepcopy(base).eval()        # frozen: the scorer's view of the states
        for q in self.feat.parameters():
            q.requires_grad_(False)
        self.texts = TextTable()
        self._new_scorer()
        self.opt_aux = torch.optim.AdamW(self.aux.parameters(), lr=g.lr_aux, weight_decay=g.aux_weight_decay)
        self.opt_clean = torch.optim.Adam(self.clean.parameters(), lr=g.lr_clean)
        self.ent_ctl = EntropyController(g.entropy_target, g.entropy_alpha0, lr=g.entropy_lr)
        self.episodes = self.teacher_calls = self.gstar = 0
        self.replay = []                              # reader's Stage I data of recent rounds (dicts of CPU tensors)
        self.scorer_replay = []                       # per round: (step features, validity on CPU, records)
        self.received = Received()
        self.succ_msgs, self.succ_parts = {}, {}      # request -> received success messages (sentence ids / parts)
        self.fail_msgs = {}                           # request -> received failure messages (sentence ids)
        self.writer = MessageWriter(base.d, max_sents=g.writer_max_sents).to(device) if g.writer else None
        self.opt_writer = torch.optim.Adam(self.writer.parameters(), lr=g.writer_lr) if self.writer else None
        self.writer_replay = []                       # per round: (request texts, first-frame vectors on CPU, messages)

    def _new_scorer(self):
        self.scorer = SentenceScorer(d_s=self.clean.d, max_steps=self.cfg.max_steps + 1,
                                     emb_std=self.g.disc_emb_std).to(self.device)
        self.opt_scorer = torch.optim.AdamW(self.scorer.parameters(), lr=self.g.disc_lr, weight_decay=1e-4)

    # ------------------------------------------------------------ messages
    def _claim_sentences(self, parts):
        """Sentence ids of the parts that assert something about the episode, read from the text (chatter, step
        counts and "k of m done" are dropped, as in the old curriculum's claim_set)."""
        return [self.texts.id(s) for _, _, s in parts
                if (p := parse(s)) is not None and p[0] not in ("chatter", "steps", "eval_count")]

    def _improved(self, parts, which):
        """One-step-better messages for the episodes `which` of this round (None: no cause stated), judged by the
        feedback received so far, this round's included."""
        for p in parts:
            self.received.observe(" ".join(x[2] for x in p))
        return [improve_feedback(parts[i], self.rng, self.received) for i in which]

    def _bank(self, r):
        """Indices into the request's success bank of the messages Stage II targets: its last ideal_k received
        success messages."""
        return list(range(len(self.succ_parts.get(r, []))))[-self.g.ideal_k:]

    def _built(self, goals):
        """The built success message of a request (module doc) as (intent, claim, sentence) parts: praise, then a
        "done" verdict per goal, each in a wording received before (a verdict in a wording received for another goal
        if this goal's was never received); None if some piece was never received."""
        out = [_say_received("evaluative", ("eval_all",), self.rng, self.received)]
        out += [_say_received("confirm", ("goal_done", int(q)), self.rng, self.received)
                or _say_like(("goal_done", int(q)), self.rng, self.received) for q in goals if q >= 0]
        return None if any(x is None for x in out) or len(out) < 2 else out

    @staticmethod
    def _failed(p):
        """The message says a goal is not done (success is read from the text, not the environment)."""
        return any((q := parse(x[2])) is not None and q[0] == "goal_not_done" for x in p)

    # ------------------------------------------------------------ batching
    def _episode_batches(self, b, epochs):
        """Minibatches that each pass cover every step exactly once but draw from few episodes: each episode's steps
        are shuffled into chunks of `episode_chunk`, chunks are shuffled, and minibatch // episode_chunk chunks form
        a minibatch -- so the text encoder runs on at most that many messages per update."""
        S = self.g.episode_chunk
        units_per_mb = max(1, self.g.minibatch // S)
        n = len(b)
        for _ in range(epochs):
            order = np.lexsort((self.rng.random(n), b))
            bs = b[order]
            rank = np.arange(n) - np.searchsorted(bs, bs, side="left")
            _, unit = np.unique(bs * (n + 1) + rank // S, return_inverse=True)
            unit_pos = self.rng.permutation(unit.max() + 1)[unit] // units_per_mb
            grouped = np.argsort(unit_pos, kind="stable")
            for chunk in np.split(order[grouped], np.flatnonzero(np.diff(unit_pos[grouped])) + 1):
                yield chunk

    # ------------------------------------------------------------ Stage I
    def _stage1_reader(self, req, texts, grid, pf, state, a_t, b_t, aux_losses):
        """The reader imitates this round's and recent rounds' Stage I episodes under request + message, with whole
        sentences dropped at random."""
        g, dev = self.g, self.device
        n = len(req)
        tok, sw = sentence_prompts([req.text(e) for e in range(n)], [split_sentences(t) or None for t in texts], dev)
        self.replay = (self.replay + [dict(grid=grid.cpu(), pf=pf.cpu(), state=state.cpu(), a=a_t.cpu(),
                                           b=b_t.cpu(), tok=tok.cpu(), sw=sw.cpu())])[-g.replay_rounds:]
        L = max(r["tok"].shape[1] for r in self.replay)
        R = {k: torch.cat([r[k] for r in self.replay]).to(dev) for k in ("grid", "pf", "state", "a")}
        offs = np.cumsum([0] + [len(r["tok"]) for r in self.replay[:-1]])
        R["b"] = torch.cat([r["b"] + o for r, o in zip(self.replay, offs)]).to(dev)
        R["tok"] = torch.cat([F_.pad(r["tok"], (0, L - r["tok"].shape[1])) for r in self.replay]).to(dev)
        R["sw"] = torch.cat([F_.pad(r["sw"], (0, L - r["sw"].shape[1]), value=-1) for r in self.replay]).to(dev)
        self.aux.train()
        for mb in self._episode_batches(R["b"].cpu().numpy(), g.epochs_aux):
            mb = torch.as_tensor(mb, device=dev)
            x, iv = obs_tensor(R["grid"][mb], R["pf"][mb], R["state"][mb], dev)
            eps, inverse = torch.unique(R["b"][mb], return_inverse=True)
            tk, sw_ = R["tok"][eps], R["sw"][eps]
            if g.text_sentence_dropout > 0:           # whole sentences out, as in an improved message
                gone = torch.rand(len(eps), int(sw_.max()) + 2, device=dev) < g.text_sentence_dropout
                tk = torch.where((sw_ >= 0) & gone.gather(1, sw_.clamp(min=0)), torch.zeros_like(tk), tk)
            loss = F_.cross_entropy(self.aux(x, iv, self.aux.encode_prompt(tk)[inverse])[0], R["a"][mb])
            self.opt_aux.zero_grad()
            loss.backward()
            self.opt_aux.step()
            aux_losses.append(loss.detach())

    def _writer_train(self, reqs, h0, seqs, epochs, losses):
        """epochs passes of the message head over (request text, first-frame vector, message) triples."""
        keep = [i for i, q in enumerate(seqs) if q]
        if not keep or epochs == 0:
            return
        g, dev = self.g, self.device
        tok_s = self.texts.tokens(dev)
        self.writer.train()
        for _ in range(epochs):
            order = self.rng.permutation(keep)
            for j in range(0, len(order), g.writer_batch):
                mb = order[j:j + g.writer_batch]
                rt = prompt_tokens([prompt_text(reqs[i]) for i in mb], dev)
                loss = self.writer.loss(rt, h0[torch.as_tensor(mb)].to(dev).float(), [seqs[i] for i in mb], tok_s)
                self.opt_writer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.writer.parameters(), 1.0)
                self.opt_writer.step()
                losses.append(loss.detach())
        self.writer.eval()

    def _stage1_writer(self, req, meta, X):
        """The message head learns to write each Stage I episode's message (its claim sentences) from the request and
        the first frame, on a replay of the last replay_rounds rounds. Returns the mean loss."""
        d_s = self.clean.d
        self.writer_replay = (self.writer_replay + [([m["req"] for m in meta], X[:, 0, :d_s].float().cpu(),
                                                     [m["msg"] for m in meta])])[-self.g.replay_rounds:]
        reqs = sum((r for r, _, _ in self.writer_replay), [])
        h0 = torch.cat([h for _, h, _ in self.writer_replay])
        seqs = sum((q for _, _, q in self.writer_replay), [])
        losses = []
        self._writer_train(reqs, h0, seqs, self.g.writer_epochs, losses)
        return mean_stat(losses)

    def _stage1_scorer(self, req, parts, X, valid, tm):
        """This round's episodes become scorer records (their message, their improved message, success read from the
        text); successes join their request's bank; the scorer is measured on them first (they are new), then
        trained on the replay. Returns diagnostics."""
        g, dev = self.g, self.device
        t0 = time.time()
        n = len(req)
        imp = self._improved(parts, range(n))
        meta = []
        for e in range(n):
            msg = self._claim_sentences(parts[e])
            ok_e = not self._failed(parts[e])
            r = req.text(e)
            built = None if ok_e else self._built(req.goals[e])
            meta.append(dict(req=r, msg=msg, ok=ok_e, imp=self._claim_sentences(imp[e]) if imp[e] is not None else None,
                             ideal=self._claim_sentences(built) if built is not None else None))
            if not msg:
                continue
            if ok_e:
                self.succ_msgs[r] = (self.succ_msgs.get(r, []) + [msg])[-SUCCESS_BANK:]
                self.succ_parts[r] = (self.succ_parts.get(r, []) + [parts[e]])[-SUCCESS_BANK:]
            else:
                self.fail_msgs[r] = (self.fail_msgs.get(r, []) + [msg])[-SUCCESS_BANK:]
        tok = self.texts.tokens(dev)
        diag = {}
        t0 = _lap(tm, "s1_scorer_meta", t0, dev)
        self.scorer.eval()
        with torch.no_grad():
            r = message_scorer_loss(self.scorer, X, valid, meta, self.succ_msgs, self.fail_msgs, tok, self.rng, dev,
                                    P=min(256, n), n_others=g.scorer_others)
            if r is not None:
                diag.update({f"scorer_fresh_{k}": float(v) for k, v in r[1].items()})
        t0 = _lap(tm, "s1_scorer_fresh", t0, dev)
        self.scorer_replay = (self.scorer_replay + [(X.cpu(), valid.cpu(), meta)])[-g.disc_replay:]
        Xr = torch.cat([x for x, _, _ in self.scorer_replay]).to(dev)
        vr = torch.cat([v for _, v, _ in self.scorer_replay]).to(dev)
        mr = sum((m for _, _, m in self.scorer_replay), [])
        steps = min(g.scorer_max_steps, max(1, int(g.disc_epochs * len(mr) / g.scorer_batch)))
        self.scorer.train()
        hist = []
        for _ in range(steps):
            r = message_scorer_loss(self.scorer, Xr, vr, mr, self.succ_msgs, self.fail_msgs, tok, self.rng, dev,
                                    P=g.scorer_batch, n_others=g.scorer_others)
            if r is None:
                break
            self.opt_scorer.zero_grad()
            r[0].backward()
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
            self.opt_scorer.step()
            hist.append(r[1])
        _lap(tm, "s1_scorer_train", t0, dev)
        tail = hist[-max(1, len(hist) // 4):]
        for k in (tail[0] if tail else {}):
            diag[f"scorer_{k}"] = float(torch.stack([h[k].float() for h in tail]).nanmean())
        diag["scorer_steps"] = len(hist)
        if self.writer is not None:
            t0 = time.time()
            diag["writer_loss"] = self._stage1_writer(req, meta, X)
            _lap(tm, "s1_writer", t0, dev)
        return diag

    # ------------------------------------------------------------ Stage II
    def _tasks(self, req, parts):
        """Stage II tasks: Stage I episodes with their targets (module doc). Returns (source episodes, whether the
        request has received successes, improved message parts or None, ideal set as a list of parts); episodes
        with neither target are left out."""
        g = self.g
        src = np.sort(self.rng.choice(len(req), size=min(max(1, g.stage2 // g.rerank_k), len(req)), replace=False))
        imp = self._improved(parts, src)
        out = []
        for e, p in zip(src, imp):
            r_e = req.text(int(e))
            bk = [self.succ_parts[r_e][i] for i in self._bank(r_e)]
            built = self._built(req.goals[int(e)])
            ideal = ([built] if built is not None else []) + bk
            if p is None and not ideal:
                continue                                  # no cause stated and nothing to aim at
            out.append((int(e), bool(bk), p, ideal))
        if not out:
            return np.zeros(0, int), np.zeros(0, bool), [], []
        e, fb, imp_p, ideal = zip(*out)
        return np.array(e), np.array(fb, bool), list(imp_p), list(ideal)

    def _stage2(self, req, parts, s1, aux_losses, tm):
        g, cfg, dev = self.g, self.cfg, self.device
        src, from_bank, imp_parts, ideal_parts = self._tasks(req, parts)
        if len(src) == 0:
            return dict(rerank_tasks=0)
        n_tasks, K = len(src), g.rerank_k
        task_of = np.repeat(np.arange(n_tasks), K)
        tasks = np.asarray(s1["idx"])[src]
        reqc = self.pool.requests(tasks[task_of])
        n = len(reqc)
        # the source episodes' features (Stage I): s's baseline D(x_src, .), and the first frame the message head reads
        first1 = np.searchsorted(s1["b"], np.arange(len(s1["idx"]) + 1))
        rows1 = np.concatenate([np.arange(first1[e], first1[e + 1]) for e in src])
        b1 = np.repeat(np.arange(n_tasks), first1[src + 1] - first1[src])
        r1 = torch.as_tensor(rows1, device=dev)
        X1, v1 = step_features(self.feat, s1["grid"][r1], s1["pf"][r1], s1["state"][r1], np.asarray(s1["a"])[rows1],
                               b1, n_tasks, np.asarray(s1["final_inv"])[src], dev, cfg.max_steps)
        draw = lambda bank: bank[int(self.rng.integers(len(bank)))]
        n_gen = K // 2 if self.writer is not None else 0
        star = []
        for t in range(n_tasks):                      # one m* per task: improved w.p. 1/2 (if any), else ideal
            use_imp = imp_parts[t] is not None and (not ideal_parts[t] or self.rng.random() < 0.5)
            star.append(("imp", imp_parts[t]) if use_imp else ("ideal", draw(ideal_parts[t])))
        kind = np.array(sum(([star[t][0]] * (K - n_gen) + ["gen"] * n_gen for t in range(n_tasks)), []))
        prompt_parts = [star[task_of[i]][1] if kind[i] != "gen" else None for i in range(n)]
        prompt_ids = [self._claim_sentences(p) if p is not None else [] for p in prompt_parts]
        texts = [" ".join(x[2] for x in p) if p is not None else "" for p in prompt_parts]
        gen = kind == "gen"
        if gen.any():
            gi = np.nonzero(gen)[0]
            self.writer.eval()
            msgs = self.writer.sample(prompt_tokens([prompt_text(reqc.text(int(i))) for i in gi], dev),
                                      X1[torch.as_tensor(task_of[gi], device=dev), 0, :self.clean.d].float(),
                                      self.texts.tokens(dev))
            for i, m in zip(gi, msgs):
                prompt_ids[i] = m
                texts[i] = " ".join(self.texts.strings[j] for j in m)
        ftok = prompt_tokens([prompt_text(reqc.text(i), texts[i]) for i in range(n)], dev)
        rtok = prompt_tokens([prompt_text(reqc.text(i)) for i in range(n)], dev)
        self.aux.eval()
        with torch.no_grad():
            p_fb = self.aux.encode_prompt(ftok)
        t0 = time.time()
        env2, _, snaps2, acts2, valid2, sat2, mask2 = run_episodes(
            cfg, reqc, lambda e, t: policy_actions(self.aux, e, reqc, dev, self.rng, prompt=p_fb,
                                                   temperature=g.stage2_temperature))
        tm["stage2_rollout"] = time.time() - t0
        t0 = time.time()
        grid, pf, state, a, b, _ = stack_steps(snaps2, acts2, valid2)
        grid, pf, state = on_device(dev, grid, pf, state)
        a_t, b_t = torch.as_tensor(a, device=dev), torch.as_tensor(b, device=dev)
        X, vmask = step_features(self.feat, grid, pf, state, a, b, n, env2.inv, dev, cfg.max_steps)
        t1 = _lap(tm, "s2_feat", t0, dev)
        cs = lambda ps: [m for m in (self._claim_sentences(p) for p in ps) if m]      # empty messages say nothing
        targets = [[cs([p]) if p is not None else [] for p in imp_parts], [cs(ps) for ps in ideal_parts]]
        tok = self.texts.tokens(dev)
        self.scorer.eval()
        with torch.no_grad():
            s_att = self._s(X, vmask, X1, v1, task_of, n, targets, tok)
        self.scorer.train()
        kept = np.zeros(n, bool)
        for k in range(n_tasks):                      # the top keep_frac by s, among s > 0 (ties at random)
            sel = np.nonzero(task_of == k)[0]
            ok = sel[s_att[sel] > 0]
            top = math.ceil(g.keep_frac * len(ok))
            kept[ok[np.lexsort((self.rng.random(len(ok)), -s_att[ok]))[:top]]] = True
        t1 = _lap(tm, "s2_score", t1, dev)
        self.last_stage2 = dict(src=src, task_of=task_of, s=s_att, kept=kept, from_bank=from_bank)   # diagnostics
        self._reader_imitate(kept, b, a_t, grid, pf, state, ftok, aux_losses)
        t1 = _lap(tm, "s2_imitate", t1, dev)
        wd = {}
        if self.writer is not None:
            ki = np.nonzero(kept)[0]
            wl = []
            self._writer_train([reqc.text(int(i)) for i in ki], X1[torch.as_tensor(task_of[ki], device=dev), 0,
                                                                    :self.clean.d].float(),
                               [prompt_ids[i] for i in ki], g.writer_imitate_epochs, wl)
            mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
            wd = dict(writer_imitate_loss=mean_stat(wl),
                      writer_imitated=int(len(ki)), gen_examples=[dict(request=reqc.text(int(i)), message=texts[i])
                                                                  for i in np.nonzero(gen)[0][:3]])
            t1 = _lap(tm, "s2_writer", t1, dev)
        with torch.no_grad():                         # pi_old under the request, on every Stage II state
            p_req = self.clean.encode_prompt(rtok)
            C = learn_mod.FWD_CHUNK                   # states per no-grad forward (train.py lowers it for pixels)
            old_probs = torch.cat([F_.softmax(self.clean(*obs_tensor(grid[j:j + C], pf[j:j + C], state[j:j + C], dev),
                                                          p_req[b_t[j:j + C]])[0], -1)
                                   for j in range(0, len(b), C)])
        kept_steps = torch.nonzero(torch.as_tensor(kept, device=dev)[b_t])[:, 0]
        t1 = _lap(tm, "s2_old_probs", t1, dev)
        distill_losses, ent_clean = [], []
        self._distil_kept(kept_steps, grid, pf, state, a_t, b_t, old_probs, rtok, req, s1, distill_losses, ent_clean)
        _lap(tm, "s2_distil", t1, dev)
        tm["stage2_update"] = time.time() - t0
        self.episodes += n
        self.gstar += n_tasks
        ok2 = (sat2 | ~mask2).all(1)                  # logged only
        per_goal = lambda gl, st: {GOALS[q].phrase: float(st[gl == q].mean()) for q in range(N_GOALS) if (gl == q).any()}
        bank_rows = from_bank[task_of]
        mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
        for k in ("imp", "ideal", "gen"):                # per prompt kind: share of attempts, kept rate, env success
            wd.update({f"{k}_share": mean(kind == k), f"{k}_kept": mean(kept[kind == k]),
                       f"{k}_success": mean(ok2[kind == k])})
        return dict(rerank_tasks=n_tasks, ideal_avail=float(from_bank.mean()),
                    imp_avail=float(np.mean([bool(m) for m in targets[0]])),
                    s_pass=mean(s_att > 0), s_ideal_pass=mean(s_att[bank_rows] > 0),
                    kept=int(kept.sum()), kept_success=mean(ok2[kept]), stage2_success=float(ok2.mean()),
                    stage2_per_goal=per_goal(reqc.goals, sat2), kept_step_share=float(len(kept_steps) / max(1, len(b))),
                    distill_loss=mean_stat(distill_losses), entropy_clean=mean_stat(ent_clean),
                    kept_nll_first=self.kept_nll[0], kept_nll_last=self.kept_nll[-1],
                    **wd,
                    stage2_examples=[dict(request=reqc.text(int(i)), kind=str(kind[i]), prompt=texts[int(i)],
                                          s=float(s_att[i]), kept=bool(kept[i]), success=bool(ok2[i]))
                                     for i in range(min(K, n))])

    def _s(self, X, vmask, X1, v1, task_of, n, targets, tok):
        """s of every Stage II attempt (module doc): for each target kind (improved message, ideal set) with messages,
        mean D over its messages of the attempt minus the same of the task's source, averaged over the task's kinds;
        -inf for a task with none. targets: per kind, per task, a list of messages (sentence ids; empty: none)."""
        n_tasks = len(targets[0])
        s_att, w = np.zeros(n), np.zeros(n_tasks)
        for banks in targets:
            tk = [k for k in range(n_tasks) if banks[k]]
            if not tk:
                continue
            rows = np.nonzero(np.isin(task_of, tk))[0]
            Da = bank_mean(self.scorer, X, vmask, rows.tolist(), [banks[task_of[i]] for i in rows], tok, self.device)
            Ds = dict(zip(tk, bank_mean(self.scorer, X1, v1, tk, [banks[k] for k in tk], tok, self.device)))
            s_att[rows] += Da - np.array([Ds[task_of[i]] for i in rows])
            w[tk] += 1
        return np.where(w[task_of] > 0, s_att / np.maximum(w[task_of], 1), -np.inf)

    def _reader_imitate(self, kept, b, a_t, grid, pf, state, ftok, aux_losses):
        """reader_imitate_epochs passes of the reader over the kept attempts' steps, under the message each attempt
        was made with."""
        g, dev = self.g, self.device
        steps = np.nonzero(kept[b])[0]
        if len(steps) == 0 or g.reader_imitate_epochs == 0:
            return
        self.aux.train()
        perm = torch.cat([torch.as_tensor(self.rng.permutation(steps), device=dev)
                          for _ in range(g.reader_imitate_epochs)])
        bt = torch.as_tensor(b, device=dev)
        for mb in perm.split(g.minibatch):
            x, iv = obs_tensor(grid[mb], pf[mb], state[mb], dev)
            rows, inverse = torch.unique(bt[mb], return_inverse=True)
            loss = F_.cross_entropy(self.aux(x, iv, self.aux.encode_prompt(ftok[rows])[inverse])[0], a_t[mb])
            self.opt_aux.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.aux.parameters(), 1.0)
            self.opt_aux.step()
            aux_losses.append(loss.detach())
        self.aux.eval()

    def _distil_kept(self, kept_steps, grid, pf, state, a_t, b_t, old_probs, tok_req, req, s1, distill_losses,
                     ent_clean):
        """pi's damped distillation in proportion to the kept steps: epochs_clean passes over them, each minibatch
        paired with a stay minibatch of the same size -- Stage II states, and with stay_stage1 half of it pi's own
        Stage I states (pi_old's probabilities under the request)."""
        g, dev = self.g, self.device
        n_push = len(kept_steps)
        self.kept_nll = (float("nan"), float("nan"))
        if n_push == 0:
            return
        stay1 = None
        if g.stay_stage1 and len(s1["b"]):
            n1 = len(s1["b"])
            take = torch.as_tensor(self.rng.choice(n1, size=min(n1, n_push * g.epochs_clean), replace=False),
                                   device=dev)
            b1 = torch.as_tensor(np.asarray(s1["b"]), device=dev)[take]
            tok1 = prompt_tokens([prompt_text(req.text(e)) for e in range(len(req))], dev)
            g1, p1, i1 = s1["grid"][take], s1["pf"][take], s1["state"][take]
            with torch.no_grad():
                pr1 = self.clean.encode_prompt(tok1)
                C = learn_mod.FWD_CHUNK
                old1 = torch.cat([F_.softmax(self.clean(*obs_tensor(g1[j:j + C], p1[j:j + C], i1[j:j + C], dev),
                                                         pr1[b1[j:j + C]])[0], -1)
                                  for j in range(0, len(take), C)])
            stay1 = (g1, p1, i1, b1, old1, tok1)
        n_all = len(b_t)
        nll = [[] for _ in range(g.epochs_clean)]   # pi's NLL of the kept actions per pass (absorption diagnostic)
        for ep in range(g.epochs_clean):
            perm = torch.as_tensor(self.rng.permutation(n_push), device=dev)
            for jj in perm.split(g.minibatch):
                km = kept_steps[jj]
                xk, ivk = obs_tensor(grid[km], pf[km], state[km], dev)
                ek, ik = torch.unique(b_t[km], return_inverse=True)
                lpk = F_.log_softmax(self.clean(xk, ivk, self.clean.encode_prompt(tok_req[ek])[ik])[0], -1)
                push = -lpk.gather(1, a_t[km, None])[:, 0].mean()
                n_s = len(jj)
                n2 = n_s - (n_s // 2 if stay1 is not None else 0)
                sm = torch.as_tensor(self.rng.choice(n_all, size=n2, replace=n2 > n_all), device=dev)
                xs, ivs = obs_tensor(grid[sm], pf[sm], state[sm], dev)
                es, is_ = torch.unique(b_t[sm], return_inverse=True)
                lp_all = F_.log_softmax(self.clean(xs, ivs, self.clean.encode_prompt(tok_req[es])[is_])[0], -1)
                p_old = old_probs[sm]
                if stay1 is not None and n_s // 2:
                    g1, p1, i1, b1, old1, tok1 = stay1
                    t1 = torch.as_tensor(self.rng.choice(len(b1), size=n_s // 2, replace=n_s // 2 > len(b1)),
                                         device=dev)
                    x1, iv1 = obs_tensor(g1[t1], p1[t1], i1[t1], dev)
                    e1, in1 = torch.unique(b1[t1], return_inverse=True)
                    lp1 = F_.log_softmax(self.clean(x1, iv1, self.clean.encode_prompt(tok1[e1])[in1])[0], -1)
                    p_old, lp_all = torch.cat([p_old, old1[t1]]), torch.cat([lp_all, lp1])
                with torch.no_grad():
                    ent_clean.append(-(p_old * p_old.clamp_min(1e-12).log()).sum(-1).mean())
                stay = -(p_old * lp_all).sum(-1).mean()
                loss = (1 - g.damp) * stay + g.damp * push
                ent_pi = -(lp_all.exp() * lp_all).sum(-1).mean()
                if self.ent_ctl.alpha:
                    loss = loss - self.ent_ctl.alpha * ent_pi
                self.opt_clean.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.clean.parameters(), 1.0)
                self.opt_clean.step()
                self.ent_ctl.update(ent_pi.detach())
                distill_losses.append(loss.detach())
                nll[ep].append(push.detach())
        self.kept_nll = (mean_stat(nll[0]), mean_stat(nll[-1]))

    # ------------------------------------------------------------ one round
    def round(self):
        """One round (module doc). Returns diagnostics; environment success appears only as logged statistics."""
        g, cfg, dev = self.g, self.cfg, self.device
        tm = {}
        t0 = time.time()
        self.rounds += 1
        idx = self.cur.sample(self.rng, g.episodes)
        req = self.pool.requests(idx)
        prompt = request_prompts(self.clean, req, dev)
        env, log, snaps, acts, valid, sat, mask = run_episodes(
            cfg, req, lambda e, t: policy_actions(self.clean, e, req, dev, self.rng, prompt=prompt))
        tm["stage1_rollout"] = time.time() - t0
        t0 = time.time()
        self.cur.update(req.goals, sat, mask, self.episodes + g.episodes, feedbacks=self.teacher_calls + g.episodes)
        facts = episode_facts_all(log, env, req.goals, sat)
        parts = [compose(f, g.profile, self.rng) for f in facts]
        texts = [" ".join(x[2] for x in p) or "Okay." for p in parts]
        tm["teacher"] = time.time() - t0
        t0 = time.time()
        grid, pf, state, a, b, _ = stack_steps(snaps, acts, valid)
        grid, pf, state = on_device(dev, grid, pf, state)
        a_t, b_t = torch.as_tensor(a, device=dev), torch.as_tensor(b, device=dev)
        aux_losses = []
        self._stage1_reader(req, texts, grid, pf, state, a_t, b_t, aux_losses)
        t1 = _lap(tm, "s1_reader", t0, dev)
        X, valid_x = step_features(self.feat, grid, pf, state, a, b, len(req), env.inv, dev, cfg.max_steps)
        _lap(tm, "s1_feat", t1, dev)
        diag = self._stage1_scorer(req, parts, X, valid_x, tm)
        tm["stage1_update"] = time.time() - t0
        self.episodes += g.episodes
        self.teacher_calls += g.episodes
        s1 = dict(idx=idx, grid=grid, pf=pf, state=state, a=a, b=b, final_inv=env.inv)
        ok1 = (sat | ~mask).all(1)                    # logged only
        warm = self.rounds <= g.warmup_rounds
        info = dict(rerank_tasks=0) if warm else self._stage2(req, parts, s1, aux_losses, tm)
        per_goal = lambda gl, st: {GOALS[q].phrase: float(st[gl == q].mean()) for q in range(N_GOALS) if (gl == q).any()}
        examples = [dict(request=req.text(i), feedback=texts[i]) for i in range(min(2, len(req)))]
        examples += info.pop("stage2_examples", [])  # one key, which train.py keeps only at evaluation rounds
        return dict(stage1_success=float(ok1.mean()), stage1_per_goal=per_goal(req.goals, sat),
                    text_success=float(np.mean([not self._failed(p) for p in parts])),
                    aux_loss=mean_stat(aux_losses), entropy_alpha=self.ent_ctl.alpha,
                    curriculum_unlocked=self.cur.unlocked, warmup=warm, **diag, **info, time=tm, examples=examples)

    # ------------------------------------------------------------ resume
    def state_dict(self):
        return dict(clean=self.clean.state_dict(), aux=self.aux.state_dict(), opt_aux=self.opt_aux.state_dict(),
                    opt_clean=self.opt_clean.state_dict(), scorer=self.scorer.state_dict(),
                    opt_scorer=self.opt_scorer.state_dict(), rng=self.rng.bit_generator.state,
                    counters=(self.episodes, self.teacher_calls, self.gstar), curriculum=self.cur.state_dict(),
                    replay=self.replay, scorer_replay=self.scorer_replay, texts=list(self.texts.strings),
                    ent_ctl=self.ent_ctl.state_dict(), received=self.received.state_dict(), succ_msgs=self.succ_msgs,
                    fail_msgs=self.fail_msgs, succ_parts=self.succ_parts, rounds=self.rounds,
                    writer=self.writer.state_dict() if self.writer else None,
                    opt_writer=self.opt_writer.state_dict() if self.writer else None, writer_replay=self.writer_replay)

    def load_state_dict(self, sd):
        self.clean.load_state_dict(sd["clean"])
        self.aux.load_state_dict(sd["aux"])
        self.opt_aux.load_state_dict(sd["opt_aux"])
        self.opt_clean.load_state_dict(sd["opt_clean"])
        self.scorer.load_state_dict(sd["scorer"])
        self.opt_scorer.load_state_dict(sd["opt_scorer"])
        self.rng.bit_generator.state = sd["rng"]
        self.episodes, self.teacher_calls, self.gstar = sd["counters"]
        self.cur.load_state_dict(sd["curriculum"])
        cpu = lambda t: t.cpu() if torch.is_tensor(t) else t     # both replays live on CPU (a GPU map_location
        self.replay = [{k: cpu(v) for k, v in r.items()} for r in sd["replay"]]      # would mix devices in torch.cat)
        self.scorer_replay = [(x.cpu(), v.cpu(), m) for x, v, m in sd["scorer_replay"]]
        self.texts = TextTable(sd["texts"])
        self.ent_ctl.load_state_dict(sd["ent_ctl"])
        self.received.load_state_dict(sd["received"])
        self.succ_msgs = {k: list(v) for k, v in sd["succ_msgs"].items()}     # copies: banks grow in place
        self.fail_msgs = {k: list(v) for k, v in sd.get("fail_msgs", {}).items()}
        self.succ_parts = {k: list(v) for k, v in sd["succ_parts"].items()}
        self.rounds = sd.get("rounds", 0)
        w = sd.get("writer")         # a run from before the message head has none (R3's "writer" was another network)
        if self.writer is not None and w is not None and set(w) == set(self.writer.state_dict()):
            self.writer.load_state_dict(sd["writer"])
            self.opt_writer.load_state_dict(sd["opt_writer"])
            self.writer_replay = [(list(r), h.cpu(), [list(q) for q in m]) for r, h, m in sd["writer_replay"]]
