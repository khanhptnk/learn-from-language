"""Neural learners on MiniCraft: base-policy BC, Batch GFT (incremental), PPO, filtered BC.

All learners share the clean policy architecture; GFT adds an auxiliary feedback-conditioned policy whose logits are
the (detached) clean logits plus a learned feedback term -- the prior-times-likelihood form of the exact posterior.
"""

import copy
from collections import Counter
import hashlib
import math
import pickle
import re
import time
import zlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F_
from torch import nn

from gft.minicraft.core import (BRIDGE_ITEM, N_ACTIONS, N_CELLS, N_ITEMS, N_WORLD_ACTIONS, RECIPES, MiniCraft,
                                WorldConfig)
from gft.minicraft.expert import ExpertPool, expert_action
from gft.minicraft.goals import (ALIASES, CROSS, GOALS, MAX_GOALS, N_GOALS, Requests, alias_requests,
                                 requirement_status, sample_requests)
from gft.minicraft.teacher import (achieved, claim_steps, compose, episode_facts, event_arrays, feedback,
                                   mentions)
from gft.minicraft.vocab import NUMBERS, UNK_BUCKETS, WORDS

VOCAB = 4096
MAX_TOKENS = 128
COVERAGE_ROUNDS = 5          # window for the coverage diagnostic
HELDOUT_FRAC = 0.1           # Stage I episodes held out each round to measure reader generalization
FWD_CHUNK = 8192             # states per no-grad forward pass (train.py lowers it for pixel observations: GPU memory)


def tokenize(texts):
    """Word tokens = 1 + crc32(word) (stable across runs), padded with 0, (B, MAX_TOKENS) long. Each PromptEncoder maps
    them to its embedding rows: the legacy one folds them into VOCAB hashed buckets, the exact one looks words up in
    vocab.WORDS."""
    out = np.zeros((len(texts), MAX_TOKENS), dtype=np.int64)
    for i, t in enumerate(texts):
        words = re.findall(r"[a-z]+|\d+", t.lower())[:MAX_TOKENS]
        out[i, :len(words)] = [1 + zlib.crc32(w.encode()) for w in words]
    return torch.from_numpy(out)


def _exact_vocab_table():
    """(sorted word tokens, their exact ids) for vocab.WORDS and the numbers 0..NUMBERS-1."""
    words = list(WORDS) + [str(i) for i in range(NUMBERS)]
    tok = torch.tensor([1 + zlib.crc32(w.encode()) for w in words], dtype=torch.long)
    order = torch.argsort(tok)
    return tok[order], (torch.arange(len(words)) + 1)[order]


def prompt_text(request, feedback=None):
    """The policy's text input: the request, plus feedback for pi^fb (one prompt format for every model)."""
    return f"request: {request}" + (f" feedback: {feedback}" if feedback is not None else "")


def prompt_tokens(texts, device):
    """Tokens of prompt texts on the device, trimmed to the longest one."""
    tok = tokenize(texts).to(device)
    return tok[:, : int((tok > 0).sum(1).max()) or 1]


# ---------------------------------------------------------------- observations

def snapshot(env):
    """Compact per-step state: grid (B,S,S) int8, pos/face (B,3), inventory (B,N_ITEMS)."""
    return env.grid.copy(), np.concatenate([env.pos, env.face[:, None]], 1), env.inv.copy()


def obs_tensor(grid, pf, inv, device):
    """One-hot cells + agent position + facing channels, (B, N_CELLS + 5, S, S); inventory (B, N_ITEMS)."""
    g = torch.as_tensor(grid, device=device).long()
    B, S, _ = g.shape
    x = F_.one_hot(g, N_CELLS).permute(0, 3, 1, 2).float()
    agent = torch.zeros(B, 5, S, S, device=device)
    pf = torch.as_tensor(pf, device=device).long()
    rows = torch.arange(B, device=device)
    agent[rows, 0, pf[:, 0], pf[:, 1]] = 1.0
    agent[rows, 1 + pf[:, 2], pf[:, 0], pf[:, 1]] = 1.0
    inv = torch.log1p(torch.as_tensor(inv, device=device).float())
    return torch.cat([x, agent], 1), inv


def crop_at_agent(f, x, crop):
    """(B, crop, crop, ch) feature-map crop centred on the agent (zero outside the map); agent from obs channel."""
    B, _, S, _ = f.shape
    r = crop // 2
    agent = x[:, N_CELLS].flatten(1).argmax(1)              # agent-position channel
    ar, ac = agent // S, agent % S
    fp = F_.pad(f, (r, r, r, r))
    idx = torch.arange(crop, device=x.device)
    rows = (ar[:, None] + idx)[:, :, None]
    cols = (ac[:, None] + idx)[:, None, :]
    return fp[torch.arange(B, device=x.device)[:, None, None], :, rows, cols]


class Encoder(nn.Module):
    """Size-independent state encoder: conv features, then (i) a crop of them centred on the agent and (ii) global
    mean/max pooling, plus inventory and request-goal embeddings. Works on any map size.

    Conv features: a stack of `layers` 3x3 convs, or with rec=K a stem conv followed by one shared 3x3 conv applied K
    times with the stem output re-injected each time (f <- relu(conv(f) + stem)), so information travels K cells --
    enough to find paths across the map, with few parameters (value-iteration-network style)."""

    def __init__(self, size=None, d=128, ch=32, crop=5, layers=3, trunk=1, rec=0, text_attn=False, dt=128,
                 pixels=False, looks=4):
        """pixels: the network sees only an RGB rendering of the state (render.render, a fixed first layer); a patch
        conv reads each cell's tile, and since the agent's position is no longer given, the agent-centred crop is
        replaced by `looks` learned attention maps over the map."""
        super().__init__()
        self.crop, self.rec, self.ch, self.d, self.text_attn, self.pixels = crop, rec, ch, d, text_attn, pixels
        in_ch = N_CELLS + 5
        if pixels:
            from gft.minicraft.render import TILE
            self.patch = nn.Conv2d(3, ch, TILE, stride=TILE)
            self.look = nn.Conv2d(ch, looks, 1)
            in_ch = ch
        if text_attn:
            # the state queries the prompt's word states (cross-attention), added to the pooled summary; both
            # projections start at zero, so a fresh text pathway leaves the policy unchanged
            self.q = nn.Linear(3 * d, dt)
            self.heads, self.dt = 4, dt
            self.p_pool = nn.Linear(dt, d)
            self.p_attn = nn.Linear(dt, d)
            for m in (self.p_pool, self.p_attn):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        if rec:
            self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)
            self.rconv = nn.Conv2d(ch, ch, 3, padding=1)
        else:
            convs = []
            for i in range(layers):
                convs += [nn.Conv2d(in_ch if i == 0 else ch, ch, 3, padding=1), nn.ReLU()]
            self.conv = nn.Sequential(*convs)
        self.local_fc = nn.Linear(ch * looks if pixels else ch * crop * crop, d)
        self.global_fc = nn.Linear(2 * ch, d)
        self.inv_fc = nn.Linear(N_ITEMS, d)
        mlp = [nn.Linear(4 * d, d), nn.ReLU()]
        for _ in range(trunk - 1):
            mlp += [nn.Linear(d, d), nn.ReLU()]
        self.trunk = nn.Sequential(*mlp)

    def features(self, x):
        if self.pixels:
            from gft.minicraft.render import render
            x = F_.relu(self.patch(render(x)))               # the network sees pixels only
        if not self.rec:
            return self.conv(x)
        u = self.stem(x)
        f = F_.relu(u)
        for _ in range(self.rec):
            f = F_.relu(self.rconv(f) + u)
        return f

    def local(self, f, x):
        if self.pixels:                                      # learned attention maps instead of the agent crop
            w = F_.softmax(self.look(f).flatten(2), -1)      # (B, looks, S*S)
            return torch.einsum("bkn,bcn->bkc", w, f.flatten(2))
        return crop_at_agent(f, x, self.crop)

    def forward(self, x, inv, p, return_f=False):
        """p: per-step prompt encoding (PromptEncoder output indexed per step): (B, d) vectors, or with text_attn
        (B, L, dt + 1) word states with a validity channel; None = no text."""
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)   # faster cuDNN convolutions; same math
        f = self.features(x)                                # (B, ch, S, S)
        h_local = self.local_fc(self.local(f, x).flatten(1))
        h_global = self.global_fc(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        state = torch.cat([F_.relu(h_local), F_.relu(h_global), F_.relu(self.inv_fc(inv))], 1)
        if p is None:
            p = state.new_zeros(len(state), self.d)
        elif self.text_attn:
            # p = [keys, values, validity x 8] per word, projected once per prompt by PromptEncoder
            B, L, H, dt = p.shape[0], p.shape[1], self.heads, self.dt
            k, v, valid = p[..., :dt], p[..., dt:2 * dt], p[..., -1] > 0.5
            pooled = v.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
            q = self.q(state).view(B, H, 1, dt // H)
            k = k.view(B, L, H, dt // H).transpose(1, 2)
            v = v.view(B, L, H, dt // H).transpose(1, 2)
            ctx = F_.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])   # fused kernel
            p = self.p_pool(pooled) + self.p_attn(ctx.reshape(B, dt))
        # the prompt vector enters linearly: a ReLU at exactly 0 would pass no gradient to a fresh (zero) text pathway
        h = self.trunk(torch.cat([state, p], 1))
        return (h, f) if return_f else h


def cell_coords(S, device):
    """(S*S, 2) normalized (x = column, y = row) coordinates of cell centres in [-1, 1] (grid_sample convention)."""
    c = (2 * torch.arange(S, device=device, dtype=torch.float32) + 1) / S - 1
    y, x = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], 1)


def soft_crop(f, pos, crop):
    """(B, ch, crop, crop) bilinear crop of feature map f (B, ch, S, S) centred on normalized positions pos (B, 2);
    zero outside the map. With pos at a cell centre this equals the integer crop around that cell."""
    S, r = f.shape[-1], crop // 2
    off = 2 * torch.arange(-r, r + 1, device=f.device, dtype=f.dtype) / S
    dy, dx = torch.meshgrid(off, off, indexing="ij")
    grid = pos[:, None, None, :] + torch.stack([dx, dy], -1)[None]
    return F_.grid_sample(f, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class TextAttention(nn.Module):
    """Multi-head attention of queries (B, N, dq) over word states (B, L, dt) with a validity mask (B, L); output
    projection starts at zero, so a fresh text pathway adds nothing."""

    def __init__(self, dq, dt, d_out, heads=4):
        super().__init__()
        self.heads, self.dt = heads, dt
        self.q, self.k, self.v = nn.Linear(dq, dt), nn.Linear(dt, dt), nn.Linear(dt, dt)
        self.o = nn.Linear(dt, d_out)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, q, words, valid):
        B, N, L, H = q.shape[0], q.shape[1], words.shape[1], self.heads
        q = self.q(q).view(B, N, H, -1).transpose(1, 2)
        k = self.k(words).view(B, L, H, -1).transpose(1, 2)
        v = self.v(words).view(B, L, H, -1).transpose(1, 2)
        out = F_.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.o(out.transpose(1, 2).reshape(B, N, self.dt))


def split_prompt(p, dt):
    """Word states, validity mask and mean word state from a PromptEncoder(states=True) output."""
    words, valid = p[..., :dt], p[..., -1] > 0.5
    mean = words.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
    return words, valid, mean


class GroundedEncoder(nn.Module):
    """Candidate A (docs/architecture_design.md): text acts early and spatially, planning keeps the value-iteration
    prior, and the agent's surroundings are read egocentrically from pixels.

    1. cells: a patch conv reads each cell's tile (or a 1x1 conv on symbolic one-hot cells).
    2. grounding: every cell attends over the prompt's words (cell -> text) and is FiLM-modulated by the text summary
       and the inventory -- the text marks which cells matter before planning.
    3. planning: a stem conv over [cells, coordinates], then one shared 3x3 conv applied `rec` times (stem re-injected).
    4. locate: a text-free attention map finds the agent (it is drawn distinctly); a crop x crop window of the planning
       features is read around it (bilinear) plus the features at the agent.
    5. decide: the state attends over the words again (what does the message say about now), then an MLP trunk.
    Every text projection starts at zero: with a fresh text pathway the network equals its text-free version."""

    def __init__(self, size=None, d=256, ch=64, dt=128, rec=8, trunk=3, crop=7, heads=4, pixels=True):
        super().__init__()
        self.d, self.ch, self.dt, self.rec, self.crop, self.pixels = d, ch, dt, rec, crop, pixels
        if pixels:
            from gft.minicraft.render import TILE
            self.patch = nn.Conv2d(3, ch, TILE, stride=TILE)
        else:
            self.patch = nn.Conv2d(N_CELLS + 5, ch, 1)
        self.cell = nn.Conv2d(ch, ch, 1)
        self.inv_fc = nn.Linear(N_ITEMS, d)
        self.ground = TextAttention(ch, dt, ch, heads)
        self.film = nn.Linear(dt + d, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.stem = nn.Conv2d(ch + 2, ch, 3, padding=1)
        self.rconv = nn.Conv2d(ch, ch, 3, padding=1)
        self.locate = nn.Conv2d(ch, 1, 1)
        self.local_fc = nn.Linear(ch * crop * crop + ch, d)
        self.global_fc = nn.Linear(2 * ch, d)
        self.read = TextAttention(3 * d, dt, d, heads)
        self.p_pool = nn.Linear(dt, d)
        nn.init.zeros_(self.p_pool.weight)
        nn.init.zeros_(self.p_pool.bias)
        mlp = [nn.Linear(4 * d, d), nn.ReLU()]
        for _ in range(trunk - 1):
            mlp += [nn.Linear(d, d), nn.ReLU()]
        self.trunk = nn.Sequential(*mlp)

    def forward(self, x, inv, p, return_f=False):
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.pixels:
            from gft.minicraft.render import render
            x = render(x)
        c0 = F_.relu(self.cell(F_.relu(self.patch(x))))      # (B, ch, S, S)
        B, _, S, _ = c0.shape
        coords = cell_coords(S, c0.device)                   # (S*S, 2)
        iv = F_.relu(self.inv_fc(inv))
        if p is None:
            words, mean, c = None, iv.new_zeros(B, self.dt), c0
        else:
            words, valid, mean = split_prompt(p, self.dt)
            g = self.ground(c0.flatten(2).transpose(1, 2), words, valid)          # (B, S*S, ch)
            c = c0 + g.transpose(1, 2).reshape(B, self.ch, S, S)
        gamma, beta = self.film(torch.cat([mean, iv], 1)).chunk(2, 1)
        c = c * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        grid = coords.t().reshape(1, 2, S, S).expand(B, -1, -1, -1).to(c.dtype)
        u = self.stem(torch.cat([c, grid], 1))
        f = F_.relu(u)
        for _ in range(self.rec):
            f = F_.relu(self.rconv(f) + u)
        a = F_.softmax(self.locate(c0).flatten(1).float(), -1).to(f.dtype)    # (B, S*S) where the agent is
        pos = a @ coords.to(f.dtype)                                         # (B, 2)
        at = torch.einsum("bn,bcn->bc", a, f.flatten(2))
        h_local = self.local_fc(torch.cat([soft_crop(f, pos, self.crop).flatten(1), at], 1))
        h_global = self.global_fc(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        state = torch.cat([F_.relu(h_local), F_.relu(h_global), iv], 1)
        if words is None:
            pv = state.new_zeros(B, self.d)
        else:
            pv = self.p_pool(mean) + self.read(state[:, None], words, valid)[:, 0]
        h = self.trunk(torch.cat([state, pv], 1))
        return (h, f) if return_f else h


class FusionBlock(nn.Module):
    """Pre-LN transformer block over state tokens with a gated cross-attention to the words first (Flamingo-style:
    the gate tanh(alpha) starts at 0, so a fresh text pathway leaves the block unchanged)."""

    def __init__(self, w, dt, heads=4):
        super().__init__()
        self.heads = heads
        self.ln_x, self.ln_a, self.ln_m = nn.LayerNorm(w), nn.LayerNorm(w), nn.LayerNorm(w)
        self.xattn = TextAttention(w, dt, w, heads)
        nn.init.normal_(self.xattn.o.weight, std=0.02)       # the gate, not the projection, makes it start as a no-op
        self.gate = nn.Parameter(torch.zeros(1))
        self.qkv = nn.Linear(w, 3 * w)
        self.proj = nn.Linear(w, w)
        self.mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))

    def forward(self, t, words, valid):
        if words is not None:
            t = t + torch.tanh(self.gate) * self.xattn(self.ln_x(t), words, valid)
        B, N, w = t.shape
        q, k, v = self.qkv(self.ln_a(t)).view(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        t = t + self.proj(F_.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, w))
        return t + self.mlp(self.ln_m(t))


class FusionEncoder(nn.Module):
    """Candidate B (docs/architecture_design.md): a small vision-language transformer. Tokens: [ACT], inventory, and
    one token per cell (patch conv + `pre_conv` 3x3 convs for local structure; positions as Fourier features of the
    absolute coordinates and of the coordinates relative to the located agent). Each block cross-attends to the words
    through a zero-initialized gate, then self-attends. The policy reads the [ACT] token."""

    def __init__(self, size=None, d=256, w=128, dt=128, layers=4, heads=4, pre_conv=2, freqs=4, pixels=True):
        super().__init__()
        self.d, self.w, self.dt, self.pixels = d, w, dt, pixels
        if pixels:
            from gft.minicraft.render import TILE
            self.patch = nn.Conv2d(3, w, TILE, stride=TILE)
        else:
            self.patch = nn.Conv2d(N_CELLS + 5, w, 1)
        self.convs = nn.ModuleList(nn.Conv2d(w, w, 3, padding=1) for _ in range(pre_conv))
        self.locate = nn.Conv2d(w, 1, 1)
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(freqs), persistent=False)
        self.pos = nn.Linear(8 * freqs, w)
        self.inv_tok = nn.Linear(N_ITEMS, w)
        self.act_tok = nn.Parameter(torch.zeros(1, 1, w))
        self.blocks = nn.ModuleList(FusionBlock(w, dt, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(w)
        self.out = nn.Linear(w, d)

    def forward(self, x, inv, p, return_f=False):
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        if self.pixels:
            from gft.minicraft.render import render
            x = render(x)
        c = F_.relu(self.patch(x))
        for conv in self.convs:
            c = F_.relu(conv(c) + c)
        B, _, S, _ = c.shape
        coords = cell_coords(S, c.device).to(c.dtype)                        # (S*S, 2)
        a = F_.softmax(self.locate(c).flatten(1).float(), -1).to(c.dtype)
        rel = coords[None] - (a @ coords)[:, None]                           # (B, S*S, 2) relative to the agent
        xy = torch.cat([coords[None].expand(B, -1, -1), rel], -1)[..., None] * self.freqs.to(c.dtype)
        cells = c.flatten(2).transpose(1, 2) + self.pos(torch.cat([xy.sin(), xy.cos()], -1).flatten(2))
        t = torch.cat([self.act_tok.expand(B, -1, -1).to(c.dtype), self.inv_tok(inv)[:, None], cells], 1)
        words, valid = (None, None) if p is None else split_prompt(p, self.dt)[:2]
        for blk in self.blocks:
            t = blk(t, words, valid)
        h = self.out(self.ln(t[:, 0]))
        return (h, t[:, 2:].transpose(1, 2).reshape(B, self.w, S, S)) if return_f else h


class PromptEncoder(nn.Module):
    """Text prompt (request, or request + feedback) -> one vector: word embeddings, a small transformer,
    mean pooling, projection to the policy width. The projection starts at zero, so a fresh text pathway leaves the
    policy unchanged (every prompt maps to 0, as in text-free pretraining) and learns from the first update.

    packed: return per-word keys/values + validity for the legacy encoder's attention. states: return the word states
    themselves + validity, (B, L, dt + 8), for encoders with their own text projections. exact: embed words by their
    vocab.WORDS id (no collisions) instead of VOCAB hashed buckets (legacy; kept so old checkpoints load unchanged)."""

    def __init__(self, d, dt=128, layers=2, packed=False, states=False, exact=False):
        super().__init__()
        self.packed, self.states, self.exact = packed, states, exact
        if exact:
            known, ids = _exact_vocab_table()
            self.register_buffer("known", known, persistent=False)
            self.register_buffer("known_ids", ids, persistent=False)
            self.tok = nn.Embedding(len(known) + 1 + UNK_BUCKETS, dt, padding_idx=0)
        else:
            self.tok = nn.Embedding(VOCAB, dt, padding_idx=0)
        if packed:
            self.kv = nn.Linear(dt, 2 * dt)
        self.pos = nn.Embedding(MAX_TOKENS, dt)
        layer = nn.TransformerEncoderLayer(dt, 4, 2 * dt, dropout=0.0, batch_first=True, norm_first=True)
        self.text = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dt, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def ids(self, tokens):
        """Embedding rows of word tokens (see tokenize); 0 stays padding."""
        if not self.exact:
            return torch.where(tokens > 0, 1 + (tokens - 1) % (VOCAB - 1), tokens)
        tokens = tokens.contiguous()
        i = torch.searchsorted(self.known, tokens).clamp(max=len(self.known) - 1)
        hit = self.known[i] == tokens
        unk = len(self.known) + 1 + tokens % UNK_BUCKETS
        return torch.where(tokens > 0, torch.where(hit, self.known_ids[i], unk), tokens)

    def forward(self, tokens):
        pad = tokens == 0
        h = self.text(self.tok(self.ids(tokens)) + self.pos.weight[: tokens.shape[1]], src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        if self.states:
            return torch.cat([h * valid, valid.expand(-1, -1, 8)], -1)            # 8 validity channels: aligned strides
        if self.packed:
            return torch.cat([self.kv(h) * valid, valid.expand(-1, -1, 8)], -1)   # 8 validity channels keep strides aligned
        return self.out((h * valid).sum(1) / valid.sum(1).clamp(min=1))


# Policy network chosen by the architecture study (check_bc.py / check_dagger.py, 2026-10-22): 8 shared conv passes,
# width 256, 3-layer trunk. Best sample efficiency under DAgger at about equal compute.
# text_attn: the policy attends over the prompt's words (added 2026-10-22; requests and feedback are read per state).
POLICY_ARCH = dict(d=256, rec=8, trunk=3, text_attn=True)
# Candidates of docs/architecture_design.md, selected with arch=dict(kind=...): "gvin" = grounded VIN (A), "fusion" =
# vision-language transformer (B). Both use the exact-vocabulary text encoder.
ARCHS = {"vin": POLICY_ARCH,
         "gvin": dict(d=256, ch=64, rec=8, trunk=3, crop=7),
         "fusion": dict(d=256, w=128, layers=4, pre_conv=2)}


class CleanPolicy(nn.Module):
    """Policy pi(a | state, prompt) with a value head (used by PPO). The prompt is text: the request for the deployed
    policy and PPO; the request plus feedback for GFT's pi^fb, which is a copy of the same network (LLM-style).

    Prompts are encoded once per episode (encode_prompt) and the vectors indexed per step."""

    def __init__(self, size, dt=128, **arch):
        super().__init__()
        kind = arch.pop("kind", "vin")
        arch = {**ARCHS[kind], **arch}
        d = arch.pop("d")
        self.d, self.kind = d, kind
        if kind == "vin":
            self.prompt = PromptEncoder(d, dt, packed=arch.get("text_attn", False))
            self.enc = Encoder(size, d, dt=dt, **arch)
        else:
            self.prompt = PromptEncoder(d, dt, states=True, exact=True)
            self.enc = {"gvin": GroundedEncoder, "fusion": FusionEncoder}[kind](size, d, dt=dt, **arch)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)

    def encode_prompt(self, tokens):
        return self.prompt(tokens)

    def forward(self, x, inv, p, return_h=False):
        h, f = self.enc(x, inv, p, return_f=True)
        out = (self.pi(h), self.v(h).squeeze(-1))
        return out + ((h, f, x),) if return_h else out


def rng_state():
    """Global torch RNG state (CPU and, if present, all CUDA devices) -- policy sampling draws from it."""
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(st):
    """Restore rng_state(); the states must be CPU byte tensors (torch.load(map_location="cuda") moves them)."""
    torch.set_rng_state(st["cpu"].cpu())
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])


# ---------------------------------------------------------------- curriculum

@dataclass
class CurriculumConfig:
    enabled: bool = True
    start: int = 4               # goals 0..start-1 (the easy, depth-ordered ones) are unlocked at the start
    threshold: float = 0.5       # unlock the next goal when recent success on the newest goal reaches this
    window: int = 512            # recent requirement outcomes kept per goal
    min_count: int = 128         # outcomes needed before a goal can count as learned
    frontier_frac: float = 0.7   # share of requests that contain the newest unlocked goal(s); the rest replay older ones
    mode: str = "gate"           # gate: unlock the next goal on success (above); blocks: fixed schedule below;
                                 # graph: a goal opens once all its prerequisite goals (GOAL_PREREQS) are mastered
    mastery: float = 0.7         # graph: recent success a goal needs to count as mastered
    block_size: int = 2          # blocks: goals introduced together, in depth order
    block_budget: int = 10_000   # blocks: feedbacks (GFT teacher messages / PPO rewards) per block before the next one


def goal_prerequisites():
    """{goal: set of goals whose item the expert's plan for it produces (or crossing, for the river goal)} --
    the goals that must be achievable before it."""
    from gft.minicraft.expert import RAW, plan
    out = {}
    for g in range(N_GOALS):
        items, crossed = set(), False
        for op in plan(np.zeros(N_ITEMS, int), [g], False, False):
            if op[0] == "craft":
                items.add(RECIPES[op[1]].output)
            elif op[0] == "collect":
                items.add(next(i for i, (c, _) in RAW.items() if c == op[1]))
            elif op[0] == "bridge":
                items.add(BRIDGE_ITEM)
            crossed |= op[0] == "cross"
        out[g] = {h for h, o in enumerate(GOALS)
                  if h != g and ((o.item == CROSS and crossed) or (o.item != CROSS and o.item in items))}
    return out


class Curriculum:
    """Adaptive, depth-ordered task curriculum shared by all learners (goal ids are ordered by depth).

    Requests are drawn from the training pool restricted to tasks whose goals are all unlocked; a share
    `frontier_frac` must include the newest goal. The next goal unlocks when the newest goal's recent success rate
    (ground truth, from the learner's own on-policy episodes) reaches `threshold`. Disabled = all goals unlocked."""

    def __init__(self, pool, cfg: CurriculumConfig):
        self.c = cfg
        self.blocks = cfg.enabled and cfg.mode == "blocks"
        self.graph = cfg.enabled and cfg.mode == "graph"
        self.unlocked = (cfg.block_size if self.blocks else cfg.start) if cfg.enabled else N_GOALS
        self.pool_goals = pool.goals
        if self.graph:
            self.prereqs = goal_prerequisites()
            self.mastered = set()
            self.unlocked = len(self._trainable())
        self.hist = {g: [] for g in range(N_GOALS)}
        self.deepest = pool.goals.max(1)                    # deepest goal of each pool task
        self.unlock_log = []                                # (episodes, newly unlocked goal)

    def _open(self):
        return {g for g in range(N_GOALS) if g not in self.mastered and self.prereqs[g] <= self.mastered}

    def _trainable(self):
        return self.mastered | self._open()

    def _graph_sample(self, rng, n):
        trainable, open_ = self._trainable(), self._open()
        g = self.pool_goals
        ok_mask = np.all((g < 0) | np.isin(g, list(trainable)), axis=1)
        frontier = np.nonzero(ok_mask & np.isin(g, list(open_)).any(1))[0]
        ok = np.nonzero(ok_mask)[0]
        if len(frontier) == 0:
            return rng.choice(ok, size=n)
        use = rng.random(n) < self.c.frontier_frac
        return np.where(use, rng.choice(frontier, size=n), rng.choice(ok, size=n))

    def sample(self, rng, n):
        if self.graph:
            return self._graph_sample(rng, n)
        ok = np.nonzero(self.deepest < self.unlocked)[0]
        width = self.c.block_size if self.blocks else 1          # newest goal(s)
        frontier = np.nonzero((self.deepest >= self.unlocked - width) & (self.deepest < self.unlocked))[0]
        if not self.c.enabled or len(frontier) == 0:
            return rng.choice(ok, size=n)
        use_frontier = rng.random(n) < self.c.frontier_frac
        return np.where(use_frontier, rng.choice(frontier, size=n), rng.choice(ok, size=n))

    def update(self, goals, sat, mask, episodes, feedbacks=None):
        """Record requirement outcomes; advance the curriculum. `feedbacks` (teacher messages for GFT, rewards for
        PPO) drives the block schedule, so both methods get the same feedback budget per block."""
        for g, ok in zip(goals[mask], sat[mask]):
            h = self.hist[int(g)]
            h.append(bool(ok))
            del h[: max(0, len(h) - self.c.window)]
        if self.graph:
            for g in sorted(self._open()):
                h = self.hist[g]
                if len(h) >= self.c.min_count and np.mean(h) >= self.c.mastery:
                    self.mastered.add(g)
                    self.unlock_log.append((feedbacks if feedbacks is not None else episodes, g))
            self.unlocked = len(self._trainable())
            return
        if self.blocks:
            target = min(N_GOALS, self.c.block_size * (1 + feedbacks // self.c.block_budget))
            if target > self.unlocked:
                self.unlocked = target
                self.unlock_log.append((feedbacks, self.unlocked))
            return
        if self.c.enabled and self.unlocked < N_GOALS:
            h = self.hist[self.unlocked - 1]
            if len(h) >= self.c.min_count and np.mean(h) >= self.c.threshold:
                self.unlock_log.append((episodes, self.unlocked))
                self.unlocked += 1

    def state_dict(self):
        return dict(unlocked=self.unlocked, hist=self.hist, unlock_log=self.unlock_log,
                    mastered=sorted(getattr(self, "mastered", ())))

    def load_state_dict(self, sd):
        self.unlocked, self.hist, self.unlock_log = sd["unlocked"], sd["hist"], sd["unlock_log"]
        if self.graph:
            self.mastered = set(sd["mastered"])


def claim_type(c):
    """A claim with episode-specific numbers removed, e.g. ('collected', wood, 5) -> ('collected', wood)."""
    kind = c[0]
    if kind in ("collected", "crafted", "station_fail", "use_fail", "goal_done", "goal_not_done", "subgoal", "needed",
                "hint_recipe", "hint_raw"):
        return c[:2]
    if kind == "missing_fail":
        return c[:3]
    return (kind,)


# ---------------------------------------------------------------- tasks and rollouts

@dataclass
class TaskPool:
    """Fixed training tasks (world seed, request) and a bank of verified successful runs per task.

    Ideal feedback G* for a task = the teacher applied to one of that task's successful runs, drawn uniformly. The bank
    holds successes of the expert and of noisy experts, so G* does not demand expert-like (mistake-free) execution.
    """
    seeds: np.ndarray
    goals: np.ndarray
    success_facts: list          # per task: list of Facts of successful runs
    alias: np.ndarray = None     # per task: alias id of its request (-1 = plain goal phrases), or None

    def requests(self, idx):
        return Requests(self.goals[idx], self.seeds[idx], None if self.alias is None else self.alias[idx])

    def ideal_facts(self, i, rng):
        bank = self.success_facts[i]
        return bank[int(rng.integers(len(bank)))]


def concat_pools(*pools):
    """One pool from several (alias ids -1 where a pool has none)."""
    alias = [p.alias if p.alias is not None else np.full(len(p.seeds), -1) for p in pools]
    return TaskPool(np.concatenate([p.seeds for p in pools]), np.concatenate([p.goals for p in pools]),
                    sum((p.success_facts for p in pools), []), np.concatenate(alias))


def run_episodes(cfg, req, act_fn, record=True):
    """Roll out B worlds until all requirements hold or the step budget ends.
    act_fn(env, t) -> actions. Returns env, event log, per-step snapshots, actions, valid mask (B,T), sat."""
    env = MiniCraft(cfg, req.seeds)
    log, snaps, acts, valid = [], [], [], []
    for t in range(cfg.max_steps):
        if not cfg.explicit_done:
            sat, mask = requirement_status(env, req)
            env.done |= (sat | ~mask).all(1)
        if env.done.all():
            break
        live = ~env.done
        if record:
            snaps.append(snapshot(env))
        a = act_fn(env, t)
        acts.append(a)
        valid.append(live.copy())
        log.append(env.step(a))
    sat, mask = requirement_status(env, req)
    return env, log, snaps, np.array(acts).T if acts else np.zeros((len(req), 0), int), \
        np.array(valid).T if valid else np.zeros((len(req), 0), bool), sat, mask


def cached_pool(cfg, n, seed, tag="", cache_dir="outputs/minicraft/pools", **kw):
    """build_pool, cached on disk. Key: cfg, n, seed, `tag` (names any filter) and non-callable kwargs -- function
    objects are excluded because their repr differs between processes."""
    stable = sorted((k, repr(v)) for k, v in kw.items() if not callable(v))
    key = hashlib.sha1(repr((cfg, n, seed, tag, stable)).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"pool_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    pool = build_pool(cfg, n, seed, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(pool))
    tmp.rename(path)
    return pool


def build_pool(cfg, n, seed, goal_pool=None, combo_filter=None, noisy=((0.8, 2), (0.6, 2)), requests=None,
               max_goals=MAX_GOALS):
    """n training tasks with a success bank: one expert run plus `repeats` runs per (expert-action probability).
    `requests`: use these requests instead of sampling n."""
    rng = np.random.default_rng(seed)
    req = requests if requests is not None else sample_requests(
        rng, n, rng.integers(0, 2 ** 31, size=n), goal_pool=goal_pool, max_goals=max_goals, combo_filter=combo_filter)
    n = len(req)
    bank = [[] for _ in range(n)]
    for p_expert, repeats in ((1.0, 1),) + tuple(noisy):
        for _ in range(repeats):
            env, log, _, _, _, sat, mask = run_episodes(cfg, req, lambda e, t: np.array(
                [expert_action(e, b, req.goals[b]) if (not e.done[b] and rng.random() < p_expert)
                 else int(rng.integers(N_WORLD_ACTIONS)) for b in range(len(req))]), record=False)
            for b in np.nonzero((sat | ~mask).all(1))[0]:
                bank[b].append(episode_facts(log, b, env, req.goals[b], sat[b]))
    keep = [b for b in range(n) if bank[b]]
    return TaskPool(req.seeds[keep], req.goals[keep], [bank[b] for b in keep],
                    None if req.alias is None else req.alias[keep])


def cached_alias_pool(cfg, n_per_alias, seed, cache_dir="outputs/minicraft/pools"):
    """Training tasks for every alias request (n_per_alias each, fresh training-range world seeds), cached."""
    key = hashlib.sha1(repr((cfg, n_per_alias, seed, "aliases", ALIASES)).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"aliaspool_{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    rng = np.random.default_rng(seed)
    pools = [build_pool(cfg, 0, seed + 1 + k, requests=alias_requests(k, rng.integers(0, 2 ** 31, size=n_per_alias)))
             for k in range(len(ALIASES))]
    pool = concat_pools(*pools)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(pool))
    tmp.rename(path)
    return pool


class AliasSchedule:
    """Two-phase task schedule (same interface as Curriculum). Phase 1: the base tasks (alias -1). Phase 2, started
    by the trainer once phase 1 has converged: the alias requests, introduced one at a time every `budget` feedbacks
    (teacher messages for GFT, rewards for PPO); a share `frontier_frac` of tasks use the newest alias, the rest replay
    base tasks and earlier aliases."""

    def __init__(self, pool, budget, frontier_frac=0.7):
        self.alias, self.budget, self.frontier_frac = pool.alias, budget, frontier_frac
        self.n_alias = int(self.alias.max()) + 1
        self.phase2_at = None
        self.unlocked = 0                         # aliases introduced so far
        self.unlock_log = []                      # (feedbacks, aliases introduced)

    def start_phase2(self, feedbacks):
        self.phase2_at, self.unlocked = feedbacks, 1
        self.unlock_log.append((feedbacks, 1))

    def finished(self, feedbacks):
        return self.phase2_at is not None and feedbacks >= self.phase2_at + self.n_alias * self.budget

    def sample(self, rng, n):
        if self.phase2_at is None:
            return rng.choice(np.nonzero(self.alias < 0)[0], size=n)
        newest = np.nonzero(self.alias == self.unlocked - 1)[0]
        replay = np.nonzero(self.alias < self.unlocked - 1)[0]
        use = rng.random(n) < self.frontier_frac
        return np.where(use, rng.choice(newest, size=n), rng.choice(replay, size=n))

    def update(self, goals, sat, mask, episodes, feedbacks=None):
        if self.phase2_at is None:
            return
        target = min(self.n_alias, 1 + (feedbacks - self.phase2_at) // self.budget)
        if target > self.unlocked:
            self.unlocked = target
            self.unlock_log.append((feedbacks, target))

    def state_dict(self):
        return dict(phase2_at=self.phase2_at, unlocked=self.unlocked, unlock_log=self.unlock_log)

    def load_state_dict(self, sd):
        self.phase2_at, self.unlocked, self.unlock_log = sd["phase2_at"], sd["unlocked"], sd["unlock_log"]


class GoalPrompts:
    """Request prompts for per-step goal arrays (N, MAX_GOALS): distinct requests are tokenized once; `vectors(model,
    idx)` encodes only the distinct requests among steps `idx` (with gradients) and indexes them per step."""

    def __init__(self, step_goals, device):
        rows, inverse = np.unique(np.asarray(step_goals), axis=0, return_inverse=True)
        texts = [prompt_text(" and ".join(GOALS[g].phrase for g in row if g >= 0)) for row in rows]
        self.tok = prompt_tokens(texts, device)
        self.row = torch.as_tensor(inverse.reshape(-1), device=device)

    def vectors(self, model, idx):
        r, inverse = torch.unique(self.row[idx], return_inverse=True)
        return model.encode_prompt(self.tok[r])[inverse]


def request_prompts(model, req, device, feedback=None):
    """Prompt vectors (B, d) for a batch of requests (optionally with one feedback text each), without gradients."""
    texts = [prompt_text(req.text(b), None if feedback is None else feedback[b]) for b in range(len(req))]
    with torch.no_grad():
        return model.encode_prompt(prompt_tokens(texts, device))


def policy_actions(model, env, req, device, rng, greedy=False, prompt=None, temperature=1.0):
    """Actions for all worlds; `prompt` = per-episode prompt vectors (default: the requests alone)."""
    x, inv = obs_tensor(env.grid, np.concatenate([env.pos, env.face[:, None]], 1), env.inv, device)
    prompt = request_prompts(model, req, device) if prompt is None else prompt
    with torch.inference_mode():
        logits, _ = model(x, inv, prompt)
        if greedy:
            return logits.argmax(-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits / temperature).sample().cpu().numpy()


def on_device(device, *arrays):
    """Move per-step arrays to the device once per round, so minibatches are sliced there (no per-step copies)."""
    return [torch.as_tensor(x, device=device) for x in arrays]


def mean_stat(xs):
    """Mean of per-minibatch scalars kept as tensors (one device sync per round instead of one per minibatch)."""
    return float(torch.stack(xs).float().mean()) if xs else float("nan")


def stack_steps(snaps, acts, valid):
    """Flatten valid (episode, step) pairs: returns arrays and the episode index of each."""
    grids = np.stack([s[0] for s in snaps], 1)       # (B, T, S, S)
    pfs = np.stack([s[1] for s in snaps], 1)
    invs = np.stack([s[2] for s in snaps], 1)
    b, t = np.nonzero(valid)
    return grids[b, t], pfs[b, t], invs[b, t], acts[b, t], b, t


# ---------------------------------------------------------------- evaluation

def evaluate(model, cfg, req, device, greedy=False, seed=0, detail=False):
    """Success rate (all requirements) and fraction of requirements met; with detail=True also per-episode success,
    per-goal success rates and mean episode length."""
    rng = np.random.default_rng(seed)
    model.eval()
    prompt = request_prompts(model, req, device)
    env, _, _, _, _, sat, mask = run_episodes(
        cfg, req, lambda e, t: policy_actions(model, e, req, device, rng, greedy, prompt=prompt), record=False)
    model.train()
    ok = (sat | ~mask).all(1)
    succ, frac = float(ok.mean()), float((sat & mask).sum() / mask.sum())
    if not detail:
        return succ, frac
    per_goal = {}
    for gid, goal in enumerate(GOALS):
        m = (req.goals == gid)
        if m.any():
            per_goal[goal.phrase] = [float(sat[m].mean()), int(m.sum())]
    return dict(success=succ, frac=frac, episode_success=ok.astype(int).tolist(), per_goal=per_goal,
                mean_steps=float(env.t.mean()))


# ---------------------------------------------------------------- base policy

# Fixed scrambled meaning of each goal phrase for the "permuted" base (a derangement: no phrase keeps its meaning).
_perm_rng = np.random.default_rng(12345)
while True:
    PERMUTED = _perm_rng.permutation(N_GOALS)
    if not (PERMUTED == np.arange(N_GOALS)).any():
        break


def behaviour_goals(goals, behaviour, rng, q=0.3, combo_filter=None):
    """Goals whose plan the base's demonstrator executes for requests `goals` (B, MAX_GOALS):
      request   the requests themselves
      mixture   the request with probability q, else an independently sampled request (random subgoals)
      permuted  PERMUTED applied to each goal (coherent behaviour, systematically misread request)"""
    if behaviour == "request":
        return goals.copy()
    if behaviour == "permuted":
        return np.where(goals >= 0, PERMUTED[np.maximum(goals, 0)], -1)
    if behaviour == "mixture":
        other = sample_requests(rng, len(goals), np.zeros(len(goals), int), combo_filter=combo_filter).goals
        return np.where((rng.random(len(goals)) < q)[:, None], goals, other)
    raise ValueError(behaviour)


def pretrain_base(cfg, device, n_episodes=2000, goal_pool=(0, 1, 2, 3), noise=0.1, steps=2000, batch=512, seed=0,
                  size=None, behaviour="request", q=0.3, combo_filter=None, workers=None, text=False, feedback_frac=0.5,
                  arch=None,
                  profiles=("mixed", "reporter", "diagnoser", "coach", "grader")):
    """BC on (slightly noisy) expert runs. The demonstrator executes the plan for behaviour_goals(request); the
    episode ends when that plan's goals are met.

    text=False (default): the prompt vector is zero throughout pretraining, so the text pathway stays untrained (and,
    with PromptEncoder's zero-initialized projection, maps every prompt to 0 until training starts) -- skills without
    any text meaning, and a text pathway that can still learn. text=True: the prompt is the request, plus for a share
    `feedback_frac` of episodes a teacher message about a *different* episode; the network then learns to ignore
    text altogether (measured: action distributions independent of the prompt), which blocks later learning."""
    rng = np.random.default_rng(seed)
    req = sample_requests(rng, n_episodes, rng.integers(0, 2 ** 31, size=n_episodes),
                          goal_pool=None if goal_pool is None else list(goal_pool), combo_filter=combo_filter)
    beh = Requests(behaviour_goals(req.goals, behaviour, rng, q, combo_filter), req.seeds)
    expert = ExpertPool(workers)

    def act_fn(e, t):                       # same random stream as labelling one world at a time
        ex = expert(e, beh.goals, np.nonzero(~e.done)[0])
        return np.array([ex[b] if rng.random() > noise else int(rng.integers(N_WORLD_ACTIONS)) for b in range(len(req))])

    env, log, snaps, acts, valid, sat, mask = run_episodes(cfg, beh, act_fn)
    expert.close()
    messages = [feedback(episode_facts(log, b, env, beh.goals[b], sat[b]), profiles[int(rng.integers(len(profiles)))],
                         rng) for b in range(len(req))]
    shift = int(rng.integers(1, len(req))) if len(req) > 1 else 0        # message of another episode
    texts = [prompt_text(req.text(b), messages[(b + shift) % len(req)] if rng.random() < feedback_frac else None)
             for b in range(len(req))]
    tok = prompt_tokens(texts, device)
    grid, pf, inv, a, b, _ = stack_steps(snaps, acts, valid)
    grid, pf, inv = on_device(device, grid, pf, inv)
    model = CleanPolicy(cfg.size, **(arch or {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    b_t = torch.as_tensor(b, device=device)
    a = torch.as_tensor(a, device=device)
    for _ in range(steps):
        i = rng.integers(len(a), size=batch)
        x, iv = obs_tensor(grid[i], pf[i], inv[i], device)
        if text:
            eps, inverse = torch.unique(b_t[i], return_inverse=True)
            p = model.encode_prompt(tok[eps])[inverse]
        else:
            p = None                                     # no text: the prompt pathway stays untrained
        logits, _ = model(x, iv, p)
        loss = F_.cross_entropy(logits, a[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------- GFT (Batch GFT, incremental)

class EntropyController:
    """Automatic entropy coefficient (as in SAC): after each update log(alpha) moves by lr * (target - entropy), so
    alpha grows while the policy is more deterministic than `target` (nats per step) and shrinks toward 0 when it is
    more random. target=None: a fixed coefficient `alpha` (no adaptation)."""

    def __init__(self, target=None, alpha=0.0, lr=0.01, lo=1e-4, hi=1.0):
        self.target, self.lr, self.lo, self.hi = target, lr, lo, hi
        self.log_alpha = float(np.log(max(alpha, lo))) if target is not None else None
        self.fixed = alpha

    @property
    def alpha(self):
        return self.fixed if self.target is None else float(np.exp(self.log_alpha))

    def update(self, entropy):
        if self.target is not None:
            self.log_alpha = float(np.clip(self.log_alpha + self.lr * (self.target - float(entropy)),
                                           np.log(self.lo), np.log(self.hi)))

    def state_dict(self):
        return dict(log_alpha=self.log_alpha)

    def load_state_dict(self, sd):
        self.log_alpha = sd["log_alpha"]


@dataclass
class GFTConfig:
    episodes: int = 256          # Stage I on-policy episodes per round (teacher calls)
    stage2: int = 256            # Stage II ideal-feedback rollouts per round (G* samples)
    profile: str = "realistic"   # teacher (gft.minicraft.teacher): structured like human feedback
    lr_aux: float = 3e-4
    lr_clean: float = 3e-4
    epochs_aux: int = 1          # Stage I passes over pi^fb's training data (current round + replay) per round
    epochs_clean: int = 1        # Stage II passes over the distillation data per round
    replay_rounds: int = 1       # pi^fb trains on Stage I data of the last `replay_rounds` rounds (1 = this round only)
    minibatch: int = 1024
    aux_entropy: float = 0.0     # entropy bonus on pi^fb's Stage I fit (flatter reader)
    stage2_temperature: float = 1.0  # sampling temperature of pi^fb's Stage II rollouts (targets stay at T=1)
    clean_entropy: float = 0.0   # entropy bonus on pi's distillation loss (keeps pi from collapsing)
    entropy_target: float = None  # nats per step: adapt the distillation entropy bonus to hold pi's entropy here
    base_share: float = 0.0      # diagnostic: share of Stage I episodes acted by the frozen base policy instead of pi
                                 # (keeps events pi has stopped producing in pi^fb's data; curriculum sees pi's only)
    masked_claims: int = 0       # Stage I: per episode, up to this many extra examples whose prompt keeps ONE spanned
                                 # sentence (rest masked) with loss only on the steps it is about (teacher.claim_steps)
    episode_chunk: int = 16      # Stage I minibatches = minibatch // episode_chunk episodes x episode_chunk steps each
                                 # (0 = uniform random steps); fewer distinct feedback messages to encode per update
    damp: float = 0.3            # Stage II target = (1 - damp) * pi_old + damp * pi^fb (damped operator, Prop. C.1)
    aux_weight_decay: float = 1e-4
    fb_word_dropout: float = 0.1  # Stage I: drop feedback words at random so pi^fb cannot memorize episodes
    no_feedback: bool = False    # ablation: pi^fb's prompt is the request alone in both stages (teacher still called)


def _n_request_tokens(req):
    """Token count of each episode's request-only prompt (feedback tokens come after it)."""
    return np.array([len(re.findall(r"[a-z]+|\d+", prompt_text(req.text(b)).lower())) for b in range(len(req))])


class GFT:
    """Batch GFT with incremental updates. pi (`clean`) is prompted with the request; pi^fb (`aux`) is a copy of the
    same network prompted with the request plus feedback (LLM-style), both starting from the shared base."""

    def __init__(self, cfg, pool, base, device, gcfg: GFTConfig, seed=0, curriculum: CurriculumConfig = None,
                 ppo: "PPOConfig" = None):
        """ppo: also give pi a PPO update on each round's Stage I episodes (reward = fraction of requirements met,
        read from the same episodes), before the distillation update -- GFT+PPO."""
        self.cfg, self.pool, self.device, self.g = cfg, pool, device, gcfg
        self.rng = np.random.default_rng(seed)
        self.cur = curriculum if hasattr(curriculum, "sample") else Curriculum(pool, curriculum or CurriculumConfig())
        self.seen_claims = []                     # claim types voiced in actual feedback, last COVERAGE_ROUNDS rounds
        self.clean = base
        self.aux = copy.deepcopy(base)
        self.opt_aux = torch.optim.AdamW(self.aux.parameters(), lr=gcfg.lr_aux, weight_decay=gcfg.aux_weight_decay)
        self.opt_clean = torch.optim.Adam(self.clean.parameters(), lr=gcfg.lr_clean)
        self.episodes = self.teacher_calls = self.gstar = 0
        self.ppo = PPO(cfg, pool, self.clean, device, ppo, seed=seed + 1, curriculum=self.cur) if ppo else None
        self.ent_ctl = EntropyController(gcfg.entropy_target, gcfg.clean_entropy or 0.01 if gcfg.entropy_target
                                         else gcfg.clean_entropy)
        self.base = copy.deepcopy(base).eval() if gcfg.base_share > 0 else None     # frozen copy for the diagnostic
        self.replay = []                          # Stage I training data of recent rounds (dicts of CPU tensors)

    def state_dict(self):
        """Everything needed to resume exactly: networks, optimizers, counters, RNG state, replay buffer."""
        return dict(clean=self.clean.state_dict(), aux=self.aux.state_dict(), opt_aux=self.opt_aux.state_dict(),
                    opt_clean=self.opt_clean.state_dict(), rng=self.rng.bit_generator.state,
                    counters=(self.episodes, self.teacher_calls, self.gstar), curriculum=self.cur.state_dict(),
                    seen_claims=self.seen_claims, replay=self.replay,
                    ppo=self.ppo.state_dict() if self.ppo else None, ent_ctl=self.ent_ctl.state_dict())

    def load_state_dict(self, sd):
        self.cur.load_state_dict(sd["curriculum"])
        self.seen_claims = sd["seen_claims"]
        self.replay = sd["replay"]
        if self.ppo:
            self.ppo.load_state_dict(sd["ppo"])
        if "ent_ctl" in sd:                       # checkpoints from before the entropy controller
            self.ent_ctl.load_state_dict(sd["ent_ctl"])
        self.clean.load_state_dict(sd["clean"])
        self.aux.load_state_dict(sd["aux"])
        self.opt_aux.load_state_dict(sd["opt_aux"])
        self.opt_clean.load_state_dict(sd["opt_clean"])
        self.rng.bit_generator.state = sd["rng"]
        self.episodes, self.teacher_calls, self.gstar = sd["counters"]

    @torch.no_grad()
    def _reader_losses(self, grid, pf, inv, a_t, b_t, tok_fb, tok_req, step_heldout, cap=4096):
        """pi^fb's action log-loss with feedback on training vs held-out Stage I episodes (memorization check), and
        on held-out episodes without the feedback (the gap = how much pi^fb uses the feedback)."""
        out = {}
        for name, sel, tok in (("seen", ~step_heldout, tok_fb), ("heldout", step_heldout, tok_fb),
                               ("heldout_no_feedback", step_heldout, tok_req)):
            sel = np.nonzero(sel)[0]
            if len(sel) == 0:
                out[name] = float("nan")
                continue
            sel = self.rng.choice(sel, size=min(cap, len(sel)), replace=False)
            eps, inverse = torch.unique(b_t[sel], return_inverse=True)
            x, iv = obs_tensor(grid[sel], pf[sel], inv[sel], self.device)
            out[name] = float(F_.cross_entropy(self.aux(x, iv, self.aux.encode_prompt(tok[eps])[inverse])[0], a_t[sel]))
        return out

    def _episode_batches(self, b, epochs):
        """Minibatches that each pass cover every step exactly once, but draw from few episodes: each episode's steps
        are shuffled into chunks of `episode_chunk`, chunks are shuffled, and minibatch // episode_chunk chunks form a
        minibatch -- so the text encoder runs on at most that many feedback messages per update."""
        S = self.g.episode_chunk
        units_per_mb = max(1, self.g.minibatch // S)
        n = len(b)
        for _ in range(epochs):
            order = np.lexsort((self.rng.random(n), b))      # steps grouped by episode, random order within
            bs = b[order]
            rank = np.arange(n) - np.searchsorted(bs, bs, side="left")
            _, unit = np.unique(bs * (n + 1) + rank // S, return_inverse=True)
            unit_pos = self.rng.permutation(unit.max() + 1)[unit] // units_per_mb   # minibatch of each step
            grouped = np.argsort(unit_pos, kind="stable")
            for chunk in np.split(order[grouped], np.flatnonzero(np.diff(unit_pos[grouped])) + 1):
                yield chunk

    def _batches(self, n, epochs):
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for j in range(0, n, self.g.minibatch):
                yield perm[j: j + self.g.minibatch]

    def _add_masked_claims(self, entry, log, parts, req, b, t, ep_keep, grid, pf, inv, a_t, dev):
        """Append pseudo-episodes to a round's Stage I data: for each kept episode, up to `masked_claims` voiced
        sentences that have a step span; prompt = request + that sentence alone, steps = the episode's steps inside
        the span (so the loss falls only on the steps the sentence is about)."""
        texts, rows, n_req = [], [], []
        n_req_ep = _n_request_tokens(req)
        ev = event_arrays(log)                               # once per round
        first = np.searchsorted(b, np.arange(len(req)))     # steps are grouped by episode, t = 0, 1, ... within
        length = np.bincount(b, minlength=len(req))
        for e in ep_keep:
            cands = parts[e]
            n = 0
            for i in self.rng.permutation(len(cands)):
                if n >= self.g.masked_claims:
                    break
                _, claim, sentence = cands[i]
                span = claim_steps(ev, e, claim)
                if span is None:
                    continue
                r = first[e] + span[span < length[e]]
                if len(r) == 0:
                    continue
                texts.append(prompt_text(req.text(e), sentence))
                rows.append(r)
                n_req.append(n_req_ep[e])
                n += 1
        if not texts:
            return entry
        tok_m = prompt_tokens(texts, dev).cpu()
        fb_m = torch.arange(tok_m.shape[1])[None, :] >= torch.as_tensor(n_req)[:, None]
        r = torch.as_tensor(np.concatenate(rows), device=dev)
        pid = torch.as_tensor(np.repeat(np.arange(len(rows)), [len(x) for x in rows])) + len(entry["tok"])
        L = max(entry["tok"].shape[1], tok_m.shape[1])
        pad = lambda x, v=0: F_.pad(x, (0, L - x.shape[1]), value=v)
        return dict(grid=torch.cat([entry["grid"], grid[r].cpu()]), pf=torch.cat([entry["pf"], pf[r].cpu()]),
                    inv=torch.cat([entry["inv"], inv[r].cpu()]), a=torch.cat([entry["a"], a_t[r].cpu()]),
                    b=torch.cat([entry["b"], pid]), tok=torch.cat([pad(entry["tok"]), pad(tok_m)]),
                    fb_pos=torch.cat([pad(entry["fb_pos"], True), pad(fb_m, True)]))

    def _replay_tensors(self, dev):
        """Concatenate the replay buffer on the device: steps, actions, per-episode prompt tokens (padded to a common
        length) and feedback-word positions, with episode ids made unique across rounds."""
        L = max(r["tok"].shape[1] for r in self.replay)
        pad = lambda t: F_.pad(t, (0, L - t.shape[1]))
        out = {k: torch.cat([r[k] for r in self.replay]).to(dev) for k in ("grid", "pf", "inv", "a")}
        offs = np.cumsum([0] + [len(r["tok"]) for r in self.replay[:-1]])
        out["b"] = torch.cat([r["b"] + o for r, o in zip(self.replay, offs)]).to(dev)
        out["tok"] = torch.cat([pad(r["tok"]) for r in self.replay]).to(dev)
        out["fb_pos"] = torch.cat([F_.pad(r["fb_pos"], (0, L - r["fb_pos"].shape[1]), value=True)
                                   for r in self.replay]).to(dev)
        return out

    def round(self):
        """One Batch GFT round. Returns diagnostics: success of on-policy (Stage I) and ideal-feedback-conditioned
        (Stage II) rollouts, mean losses, timings, and a few example (request, feedback) pairs."""
        g, cfg, dev = self.g, self.cfg, self.device
        tm = {}
        t0 = time.time()
        # Stage I: on-policy episodes of pi, teacher feedback, SGD on pi^fb (prompt = request + feedback).
        idx = self.cur.sample(self.rng, g.episodes)
        req = self.pool.requests(idx)
        prompt = request_prompts(self.clean, req, dev)
        if self.base is None:
            env, log, snaps, acts, valid, sat, mask = run_episodes(
                cfg, req, lambda e, t: policy_actions(self.clean, e, req, dev, self.rng, prompt=prompt))
            from_pi = np.ones(len(req), bool)
        else:                                         # diagnostic: some episodes acted by the frozen base policy
            from_pi = self.rng.random(len(req)) >= g.base_share
            p_base = request_prompts(self.base, req, dev)
            env, log, snaps, acts, valid, sat, mask = run_episodes(cfg, req, lambda e, t: np.where(
                from_pi, policy_actions(self.clean, e, req, dev, self.rng, prompt=prompt),
                policy_actions(self.base, e, req, dev, self.rng, prompt=p_base)))
        tm["stage1_rollout"] = time.time() - t0
        t0 = time.time()
        self.cur.update(req.goals[from_pi], sat[from_pi], mask[from_pi], self.episodes + g.episodes,
                        feedbacks=self.teacher_calls + g.episodes)
        ppo_stats = {}
        if self.ppo:                                  # GFT+PPO: reward update of pi on its own Stage I episodes
            t1 = time.time()
            tok_r = prompt_tokens([prompt_text(req.text(i)) for i in range(len(req))], dev)
            ppo_stats = {f"ppo_{k}": v for k, v in self.ppo.update(req, tok_r, prompt, snaps, acts, valid, sat,
                                                                   mask).items()}
            tm["ppo_update"] = time.time() - t1
        facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
        parts = [compose(f, g.profile, self.rng) for f in facts]
        # diagnostics: per goal, share of Stage I episodes in which it was achieved / achieved and mentioned
        coverage = {GOALS[k].phrase: [float(np.mean([achieved(f, k) for f in facts])),
                                      float(np.mean([achieved(f, k) and any(mentions(x[1], k) for x in pt)
                                                     for f, pt in zip(facts, parts)]))] for k in range(N_GOALS)}
        texts = [None if g.no_feedback else " ".join(x[2] for x in p) or "Okay." for p in parts]
        self.seen_claims = (self.seen_claims + [Counter(claim_type(x[1]) for p in parts for x in p)])[-COVERAGE_ROUNDS:]
        tm["teacher"] = time.time() - t0
        t0 = time.time()
        tok = prompt_tokens([prompt_text(req.text(b), texts[b]) for b in range(len(req))], dev)
        tok_req = prompt_tokens([prompt_text(req.text(b)) for b in range(len(req))], dev)
        fb_pos = (torch.arange(tok.shape[1], device=dev)[None, :] >=
                  torch.as_tensor(_n_request_tokens(req), device=dev)[:, None])       # feedback-word positions
        grid, pf, inv, a, b, t_steps = stack_steps(snaps, acts, valid)
        grid, pf, inv = on_device(dev, grid, pf, inv)
        a_t = torch.as_tensor(a, device=dev)
        b_t = torch.as_tensor(b, device=dev)
        aux_losses, distill_losses = [], []
        heldout_ep = self.rng.random(len(req)) < HELDOUT_FRAC     # episodes kept out of Stage I to test the reader
        keep = torch.as_tensor(~heldout_ep[b], device=dev)
        ep_keep = np.nonzero(~heldout_ep)[0]
        remap = torch.full((len(req),), -1, dtype=torch.long, device=dev)
        remap[torch.as_tensor(ep_keep, device=dev)] = torch.arange(len(ep_keep), device=dev)
        entry = dict(grid=grid[keep].cpu(), pf=pf[keep].cpu(), inv=inv[keep].cpu(), a=a_t[keep].cpu(),
                     b=remap[b_t[keep]].cpu(), tok=tok[torch.as_tensor(ep_keep, device=dev)].cpu(),
                     fb_pos=fb_pos[torch.as_tensor(ep_keep, device=dev)].cpu())
        if g.masked_claims:
            entry = self._add_masked_claims(entry, log, parts, req, b, t_steps, ep_keep, grid, pf, inv, a_t, dev)
        self.replay = (self.replay + [entry])[-g.replay_rounds:]
        R = self._replay_tensors(dev)
        batches = (self._episode_batches(R["b"].cpu().numpy(), g.epochs_aux) if g.episode_chunk
                   else self._batches(len(R["a"]), g.epochs_aux))
        for mb in batches:
            mb = torch.as_tensor(mb, device=dev)
            x, iv = obs_tensor(R["grid"][mb], R["pf"][mb], R["inv"][mb], dev)
            eps, inverse = torch.unique(R["b"][mb], return_inverse=True)   # encode each episode's prompt once
            tk = R["tok"][eps]
            if g.fb_word_dropout > 0:
                drop = R["fb_pos"][eps] & (torch.rand(tk.shape, device=dev) < g.fb_word_dropout)
                tk = torch.where(drop, torch.zeros_like(tk), tk)
            lg_aux = self.aux(x, iv, self.aux.encode_prompt(tk)[inverse])[0]
            loss = F_.cross_entropy(lg_aux, R["a"][mb])
            if g.aux_entropy:
                lp = F_.log_softmax(lg_aux, -1)
                loss = loss + g.aux_entropy * (lp.exp() * lp).sum(-1).mean()        # minus entropy
            self.opt_aux.zero_grad()
            loss.backward()
            self.opt_aux.step()
            aux_losses.append(loss.detach())
        reader = self._reader_losses(grid, pf, inv, a_t, b_t, tok, tok_req, heldout_ep[b])
        tm["stage1_update"] = time.time() - t0
        self.episodes += g.episodes
        self.teacher_calls += g.episodes
        t0 = time.time()
        # Stage II: ideal feedback from the success bank; roll out pi^fb; distill its distribution into pi.
        idx2 = self.cur.sample(self.rng, g.stage2)
        req2 = self.pool.requests(idx2)
        star_parts = [compose(self.pool.ideal_facts(i, self.rng), g.profile, self.rng) for i in idx2]
        star = [" ".join(x[2] for x in p) or "Okay." for p in star_parts]
        star_prompt = [None] * len(star) if g.no_feedback else star
        seen = sum(self.seen_claims, Counter())
        star_types = [{claim_type(x[1]) for x in p} for p in star_parts]
        unseen = Counter(c for ts in star_types for c in ts if c not in seen)
        prompt_fb2 = request_prompts(self.aux, req2, dev, feedback=star_prompt)
        env2, _, snaps2, acts2, valid2, sat2, mask2 = run_episodes(
            cfg, req2, lambda e, t: policy_actions(self.aux, e, req2, dev, self.rng, prompt=prompt_fb2,
                                                   temperature=g.stage2_temperature))
        tm["stage2_rollout"] = time.time() - t0
        t0 = time.time()
        grid, pf, inv, _, b, _ = stack_steps(snaps2, acts2, valid2)
        grid, pf, inv = on_device(dev, grid, pf, inv)
        b_t = torch.as_tensor(b, device=dev)
        tok_req2 = prompt_tokens([prompt_text(req2.text(i)) for i in range(len(req2))], dev)
        with torch.no_grad():                                   # pi_old and pi^fb on all Stage II states
            p_req2 = self.clean.encode_prompt(tok_req2)
            old_probs, fb_probs = [], []
            for j in range(0, len(b), FWD_CHUNK):
                x, iv = obs_tensor(grid[j:j + FWD_CHUNK], pf[j:j + FWD_CHUNK], inv[j:j + FWD_CHUNK], dev)
                old_probs.append(F_.softmax(self.clean(x, iv, p_req2[b_t[j:j + FWD_CHUNK]])[0], -1))
                fb_probs.append(F_.softmax(self.aux(x, iv, prompt_fb2[b_t[j:j + FWD_CHUNK]])[0], -1))
            old_probs, fb_probs = torch.cat(old_probs), torch.cat(fb_probs)
        ent_fb, ent_clean = [], []
        for mb in self._batches(len(b), g.epochs_clean):
            x, iv = obs_tensor(grid[mb], pf[mb], inv[mb], dev)
            eps, inverse = torch.unique(b_t[mb], return_inverse=True)
            logits, _ = self.clean(x, iv, self.clean.encode_prompt(tok_req2[eps])[inverse])
            with torch.no_grad():
                p_fb, p_old = fb_probs[mb], old_probs[mb]
                target = (1 - g.damp) * p_old + g.damp * p_fb
                ent_fb.append(-(p_fb * p_fb.clamp_min(1e-12).log()).sum(-1).mean())
                ent_clean.append(-(p_old * p_old.clamp_min(1e-12).log()).sum(-1).mean())
            lp = F_.log_softmax(logits, -1)
            loss = -(target * lp).sum(-1).mean()                                # logit (expected-gradient) distillation
            ent_pi = -(lp.exp() * lp).sum(-1).mean()
            if self.ent_ctl.alpha:
                loss = loss - self.ent_ctl.alpha * ent_pi                       # entropy bonus on pi
            self.opt_clean.zero_grad()
            loss.backward()
            self.opt_clean.step()
            self.ent_ctl.update(ent_pi.detach())
            distill_losses.append(loss.detach())
        tm["stage2_update"] = time.time() - t0
        self.episodes += g.stage2
        self.gstar += g.stage2
        per_goal = lambda gl, st: {GOALS[k].phrase: float(st[gl == k].mean()) for k in range(N_GOALS) if (gl == k).any()}
        return dict(stage1_success=float((sat | ~mask).all(1).mean()),
                    # per goal: requirement success of pi (Stage I) and of pi^fb under ideal feedback (Stage II)
                    stage1_per_goal=per_goal(req.goals, sat), stage2_per_goal=per_goal(req2.goals, sat2),
                    stage1_coverage=coverage,
                    stage1_frac=float((sat & mask).sum() / mask.sum()),
                    stage2_success=float((sat2 | ~mask2).all(1).mean()),        # pi^fb under ideal feedback
                    aux_loss=mean_stat(aux_losses), distill_loss=mean_stat(distill_losses),
                    entropy_clean=mean_stat(ent_clean), entropy_fb=mean_stat(ent_fb),
                    # coverage: share of ideal messages whose every claim type was voiced in actual feedback in the
                    # last COVERAGE_ROUNDS rounds; unseen: ideal claim types never voiced there (with counts)
                    coverage=float(np.mean([ts <= seen.keys() for ts in star_types])),
                    unseen_claims={repr(k): v for k, v in unseen.most_common()},
                    # pi^fb log-loss: training vs held-out episodes (memorization), held-out without feedback (use)
                    reader_loss_seen=reader["seen"], reader_loss_heldout=reader["heldout"],
                    reader_loss_heldout_no_feedback=reader["heldout_no_feedback"],
                    curriculum_unlocked=self.cur.unlocked, **ppo_stats, entropy_alpha=self.ent_ctl.alpha,
                    time=tm,
                    examples=[dict(request=req.text(i), feedback=texts[i], success=bool((sat[i] | ~mask[i]).all()))
                              for i in range(3)] +
                             [dict(request=req2.text(i), ideal_feedback=star[i]) for i in range(2)])


# ---------------------------------------------------------------- PPO baseline

@dataclass
class PPOConfig:
    episodes: int = 512
    reward: str = "frac"         # bin (all requirements) | frac (fraction satisfied), terminal
    lr: float = 3e-4
    epochs: int = 2
    minibatch: int = 2048
    clip: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    ent: float = 0.01
    entropy_target: float = None  # nats per step: adapt the entropy coefficient to hold the policy's entropy here


class PPO:
    def __init__(self, cfg, pool, base, device, pcfg: PPOConfig, seed=0, curriculum: CurriculumConfig = None):
        self.cfg, self.pool, self.device, self.p = cfg, pool, device, pcfg
        self.rng = np.random.default_rng(seed)
        self.cur = curriculum if hasattr(curriculum, "sample") else Curriculum(pool, curriculum or CurriculumConfig())
        self.model = base
        self.opt = torch.optim.Adam(self.model.parameters(), lr=pcfg.lr)
        self.ent_ctl = EntropyController(pcfg.entropy_target, pcfg.ent)
        self.episodes = self.verifier_calls = 0

    def state_dict(self):
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(), rng=self.rng.bit_generator.state,
                    counters=(self.episodes, self.verifier_calls), curriculum=self.cur.state_dict(),
                    ent_ctl=self.ent_ctl.state_dict())

    def load_state_dict(self, sd):
        self.cur.load_state_dict(sd["curriculum"])
        if "ent_ctl" in sd:                       # checkpoints from before the entropy controller
            self.ent_ctl.load_state_dict(sd["ent_ctl"])
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.rng.bit_generator.state = sd["rng"]
        self.episodes, self.verifier_calls = sd["counters"]

    def round(self):
        """One PPO round. Returns diagnostics: training success, losses, entropy, approx KL, clip fraction, timings."""
        p, cfg, dev = self.p, self.cfg, self.device
        tm = {}
        t0 = time.time()
        idx = self.cur.sample(self.rng, p.episodes)
        req = self.pool.requests(idx)
        tok = prompt_tokens([prompt_text(req.text(i)) for i in range(len(req))], dev)
        with torch.no_grad():
            prompt = self.model.encode_prompt(tok)
        env, _, snaps, acts, valid, sat, mask = run_episodes(
            cfg, req, lambda e, t: policy_actions(self.model, e, req, dev, self.rng, prompt=prompt))
        self.cur.update(req.goals, sat, mask, self.episodes + p.episodes, feedbacks=self.episodes + p.episodes)
        tm["rollout"] = time.time() - t0
        t0 = time.time()
        stats = self.update(req, tok, prompt, snaps, acts, valid, sat, mask)
        tm["update"] = time.time() - t0
        self.episodes += p.episodes
        self.verifier_calls += p.episodes
        return dict(train_success=float((sat | ~mask).all(1).mean()), train_frac=float((sat & mask).sum() / mask.sum()),
                    **stats, time=tm, curriculum_unlocked=self.cur.unlocked)

    def update(self, req, tok, prompt, snaps, acts, valid, sat, mask):
        """PPO update of self.model on one batch of its own episodes (terminal reward per episode, GAE)."""
        p, dev = self.p, self.device
        r_ep = (sat | ~mask).all(1).astype(float) if p.reward == "bin" else (sat & mask).sum(1) / mask.sum(1)
        grid, pf, inv, a, b, t = stack_steps(snaps, acts, valid)
        grid, pf, inv = on_device(dev, grid, pf, inv)
        b_t = torch.as_tensor(b, device=dev)
        a_t = torch.as_tensor(a, device=dev)
        with torch.no_grad():
            lp_old, v_old = [], []
            for j in range(0, len(a), FWD_CHUNK):
                x, iv = obs_tensor(grid[j:j + FWD_CHUNK], pf[j:j + FWD_CHUNK], inv[j:j + FWD_CHUNK], dev)
                lg, v = self.model(x, iv, prompt[b_t[j:j + FWD_CHUNK]])
                lp_old.append(torch.log_softmax(lg, -1).gather(1, a_t[j:j + FWD_CHUNK, None])[:, 0])
                v_old.append(v)
            lp_old, v_old = torch.cat(lp_old), torch.cat(v_old)
        # GAE per episode, terminal reward at each episode's last valid step.
        v_np = v_old.cpu().numpy()
        V = np.zeros(valid.shape)
        V[b, t] = v_np
        lengths = valid.sum(1)
        R = np.zeros(valid.shape)
        R[np.arange(len(req)), lengths - 1] = r_ep
        adv = np.zeros(valid.shape)
        run = np.zeros(len(req))
        for s in reversed(range(valid.shape[1])):
            nxt = V[:, s + 1] if s + 1 < valid.shape[1] else 0
            nxt = np.where(s + 1 < lengths, nxt, 0.0)
            delta = R[:, s] + p.gamma * nxt - V[:, s]
            run = delta + p.gamma * p.lam * run * (s + 1 < lengths)
            adv[:, s] = run * valid[:, s]
        A = torch.as_tensor(adv[b, t], device=dev, dtype=torch.float32)
        ret = A + v_old
        A = (A - A.mean()) / (A.std() + 1e-8)
        stats = {k: [] for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac")}
        for _ in range(p.epochs):
            perm = self.rng.permutation(len(a))
            for j in range(0, len(a), p.minibatch):
                mb = perm[j: j + p.minibatch]
                x, iv = obs_tensor(grid[mb], pf[mb], inv[mb], dev)
                eps, inverse = torch.unique(b_t[mb], return_inverse=True)
                lg, v = self.model(x, iv, self.model.encode_prompt(tok[eps])[inverse])
                logp = torch.log_softmax(lg, -1)
                ratio = (logp.gather(1, a_t[mb, None])[:, 0] - lp_old[mb]).exp()
                pg = -torch.minimum(ratio * A[mb], ratio.clamp(1 - p.clip, 1 + p.clip) * A[mb]).mean()
                v_loss = ((v - ret[mb]) ** 2).mean()
                ent = (-(logp.exp() * logp).sum(-1)).mean()
                loss = pg + 0.5 * v_loss - self.ent_ctl.alpha * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    stats["pg_loss"].append(pg.detach())
                    stats["v_loss"].append(v_loss.detach())
                    stats["entropy"].append(ent.detach())
                    stats["approx_kl"].append((lp_old[mb] - logp.gather(1, a_t[mb, None])[:, 0]).mean())
                    stats["clip_frac"].append(((ratio - 1).abs() > p.clip).float().mean())
                self.ent_ctl.update(ent.detach())
        return {k: mean_stat(v) for k, v in stats.items()}
