"""Shared MiniCraft learning code: the policy network, prompts, task pools and rollouts, the curriculum, evaluation,
base-policy pretraining, the entropy controller and the PPO / GRPO baseline. GFT itself is gft/minicraft/gft_message.py.
"""

from collections import Counter, OrderedDict
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

from gft.minicraft.core import (AGENT_FEATS, DONE, N_ACTIONS, N_CELLS, N_ITEMS, N_WORLD_ACTIONS, RECIPES,
                                MiniCraft)
from gft.minicraft.expert import ExpertPool, expert_action
from gft.minicraft.goals import (ALIASES, GOALS, MAX_GOALS, N_GOALS, Requests, alias_requests,
                                 requirement_status, sample_requests)
from gft.minicraft.teacher import (achieved, compose, episode_facts, episode_facts_all,
                                   feedback)
from gft.minicraft.verify import reported_fraction
from gft.minicraft.vocab import NUMBERS, UNK_BUCKETS, WORDS

VOCAB = 4096
MAX_TOKENS = 128
SEP = "[SEP]"                # separates the request from the feedback in pi^fb's prompt (fb_sep)
SEP_TOKEN = 2 ** 32 + 1      # its token: word tokens are 1 + crc32 <= 2**32, so it never collides
FWD_CHUNK = 8192             # states per no-grad forward pass (train.py lowers it for pixel observations: GPU memory)


def tokenize(texts):
    """Word tokens = 1 + crc32(word) (stable across runs), padded with 0, (B, MAX_TOKENS) long. Each PromptEncoder maps
    them to its embedding rows: the legacy one folds them into VOCAB hashed buckets, the exact one looks words up in
    vocab.WORDS."""
    out = np.zeros((len(texts), MAX_TOKENS), dtype=np.int64)
    for i, t in enumerate(texts):
        tok = []
        for k, piece in enumerate(t.split(SEP)):
            if k:
                tok.append(SEP_TOKEN)
            tok += [1 + zlib.crc32(w.encode()) for w in re.findall(r"[a-z]+|\d+", piece.lower())]
        tok = tok[:MAX_TOKENS]
        out[i, :len(tok)] = tok
    return torch.from_numpy(out)


def _exact_vocab_table():
    """(sorted word tokens, their exact ids) for vocab.WORDS and the numbers 0..NUMBERS-1."""
    words = list(WORDS) + [str(i) for i in range(NUMBERS)]
    tok = torch.tensor([1 + zlib.crc32(w.encode()) for w in words], dtype=torch.long)
    order = torch.argsort(tok)
    return tok[order], (torch.arange(len(words)) + 1)[order]


def prompt_text(request, feedback=None, sep=False):
    """The policy's text input: the request, plus feedback for pi^fb (one prompt format for every model); sep: the
    feedback follows a dedicated separator token instead of the word "feedback"."""
    if feedback is None:
        return f"request: {request}"
    return f"request: {request} {SEP} {feedback}" if sep else f"request: {request} feedback: {feedback}"


def prompt_tokens(texts, device):
    """Tokens of prompt texts on the device, trimmed to the longest one."""
    tok = tokenize(texts).to(device)
    return tok[:, : int((tok > 0).sum(1).max()) or 1]


# ---------------------------------------------------------------- observations

def snapshot(env):
    """Compact per-step state: grid (B,S,S) int8, pos/face (B,3), agent state (B,AGENT_FEATS): the inventory (its
    first N_ITEMS columns), the cursor and the adjacent station's slot (MiniCraft.agent_state)."""
    return env.grid.copy(), np.concatenate([env.pos, env.face[:, None]], 1), env.agent_state()


def obs_tensor(grid, pf, inv, device):
    """One-hot cells + agent position + facing channels, (B, N_CELLS + 5, S, S); log1p agent state (B, AGENT_FEATS),
    whose first N_ITEMS columns are the inventory."""
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
                 pixels=False, looks=4, flat=False):
        """pixels: the network sees only an RGB rendering of the state (render.render, a fixed first layer); a patch
        conv reads each cell's tile, and since the agent's position is no longer given, the agent-centred crop is
        replaced by `looks` learned attention maps over the map -- or, with flat=True, by the whole feature map
        flattened (position explicit from the first step; needs the map size)."""
        super().__init__()
        self.crop, self.rec, self.ch, self.d, self.text_attn, self.pixels = crop, rec, ch, d, text_attn, pixels
        self.flat = flat and pixels
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
        if self.flat:
            assert size is not None, "flat pixel features need the map size"
            self.local_fc = nn.Linear(ch * size * size, d)
        else:
            self.local_fc = nn.Linear(ch * looks if pixels else ch * crop * crop, d)
        self.global_fc = nn.Linear(2 * ch, d)
        self.inv_fc = nn.Linear(AGENT_FEATS, d)
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
        if self.flat:                                        # the whole map, position explicit
            return f.flatten(1)
        if self.pixels:                                      # learned attention maps instead of the agent crop
            w = F_.softmax(self.look(f).flatten(2), -1)      # (B, looks, S*S)
            return torch.einsum("bkn,bcn->bkc", w, f.flatten(2))
        return crop_at_agent(f, x, self.crop)

    def forward(self, x, inv, p, return_f=False):
        """p: per-step prompt encoding (PromptEncoder output indexed per step): (B, d) vectors, or with text_attn
        (B, L, dt + 1) word states with a validity channel; None = no text."""
        state, f = self.encode_state(x, inv)
        h = self.mix(state, p)
        return (h, f) if return_f else h

    def encode_state(self, x, inv):
        """The text-independent part (convolutions, pooling, inventory): (state (B, 3d), feature map f). Scoring one
        state under several prompts reuses it and reruns only mix()."""
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)   # faster cuDNN convolutions; same math
        f = self.features(x)                                # (B, ch, S, S)
        h_local = self.local_fc(self.local(f, x).flatten(1))
        h_global = self.global_fc(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))
        return torch.cat([F_.relu(h_local), F_.relu(h_global), F_.relu(self.inv_fc(inv))], 1), f

    def mix(self, state, p):
        """The text-dependent head: state (B, 3d) and per-step prompt encodings -> trunk vector h (B, d)."""
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
        return self.trunk(torch.cat([state, p], 1))


def cell_coords(S, device):
    """(S*S, 2) normalized (x = column, y = row) coordinates of cell centres in [-1, 1] (grid_sample convention)."""
    c = (2 * torch.arange(S, device=device, dtype=torch.float32) + 1) / S - 1
    y, x = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], 1)


def attention_crop(f, a, crop):
    """(B, ch * crop * crop) mixture of the crop x crop windows of f (B, ch, S, S) around every cell, weighted by an
    attention a (B, S*S) over cells; zero outside the map. With a one-hot on the agent's cell this is the agent crop
    (crop_at_agent); unlike cropping at the attention's mean position, every cell gets its own gradient, so a fresh
    locator can learn where the agent is."""
    windows = F_.unfold(f, crop, padding=crop // 2)        # (B, ch * crop * crop, S*S)
    return torch.einsum("bn,bkn->bk", a, windows)


class TextAttention(nn.Module):
    """Multi-head attention of queries (B, N, dq) over word states (B, L, dt) with a validity mask (B, L); output
    projection starts at zero, so a fresh text pathway adds nothing."""

    def __init__(self, dq, dt, d_out, heads=4, dk=None, zero_out=True):
        super().__init__()
        self.heads, self.dt = heads, dt
        self.q, self.k, self.v = nn.Linear(dq, dt), nn.Linear(dk or dt, dt), nn.Linear(dk or dt, dt)
        self.o = nn.Linear(dt, d_out)
        if zero_out:
            nn.init.zeros_(self.o.weight)
            nn.init.zeros_(self.o.bias)

    def forward(self, q, words, valid=None):
        B, N, L, H = q.shape[0], q.shape[1], words.shape[1], self.heads
        q = self.q(q).view(B, N, H, -1).transpose(1, 2)
        k = self.k(words).view(B, L, H, -1).transpose(1, 2)
        v = self.v(words).view(B, L, H, -1).transpose(1, 2)
        out = F_.scaled_dot_product_attention(q, k, v, attn_mask=None if valid is None else valid[:, None, None, :])
        return self.o(out.transpose(1, 2).reshape(B, N, self.dt))


def split_prompt(p, dt):
    """Word states, validity mask and mean word state from a PromptEncoder(states=True) output."""
    words, valid = p[..., :dt], p[..., -1] > 0.5
    mean = words.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
    return words, valid, mean


class FilmEncoder(nn.Module):
    """The vin pixel encoder (same modules and names, so a text-free vin base loads into it) with the text conditioning
    every layer instead of entering once at the end: the text summary scales and shifts (FiLM) the channels of the stem
    and of every recurrent conv pass, and each trunk layer is FiLM-modulated and cross-attends to the words. All
    conditioning starts at zero, so with a vin base loaded it starts as that base."""

    def __init__(self, size=None, d=256, ch=32, rec=8, trunk=3, looks=4, dt=128, heads=4, pixels=True):
        super().__init__()
        assert pixels, "FilmEncoder is for pixel observations"
        from gft.minicraft.render import TILE
        self.d, self.ch, self.rec, self.dt = d, ch, rec, dt
        self.patch = nn.Conv2d(3, ch, TILE, stride=TILE)
        self.look = nn.Conv2d(ch, looks, 1)
        self.stem = nn.Conv2d(ch, ch, 3, padding=1)
        self.rconv = nn.Conv2d(ch, ch, 3, padding=1)
        self.local_fc = nn.Linear(ch * looks, d)
        self.global_fc = nn.Linear(2 * ch, d)
        self.inv_fc = nn.Linear(AGENT_FEATS, d)
        mlp = [nn.Linear(4 * d, d), nn.ReLU()]
        for _ in range(trunk - 1):
            mlp += [nn.Linear(d, d), nn.ReLU()]
        self.trunk = nn.Sequential(*mlp)
        self.p_pool = nn.Linear(dt, d)                                # text summary at the trunk input (as vin)
        self.conv_film = nn.Linear(dt, (rec + 1) * 2 * ch)            # stem + every recurrent pass
        self.trunk_film = nn.ModuleList(nn.Linear(dt, 2 * d) for _ in range(trunk))
        self.trunk_attn = nn.ModuleList(TextAttention(d, dt, d, heads) for _ in range(trunk))
        for m in [self.p_pool, self.conv_film, *self.trunk_film]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, inv, p, return_f=False):
        from gft.minicraft.render import render
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        B = x.shape[0]
        if p is None:
            words, valid, mean = None, None, x.new_zeros(B, self.dt)
        else:
            words, valid, mean = split_prompt(p, self.dt)
        film = self.conv_film(mean).view(B, self.rec + 1, 2, self.ch, 1, 1)
        u = self.stem(F_.relu(self.patch(render(x))))
        u = u * (1 + film[:, 0, 0]) + film[:, 0, 1]
        f = F_.relu(u)
        for k in range(self.rec):
            f = F_.relu((self.rconv(f) + u) * (1 + film[:, k + 1, 0]) + film[:, k + 1, 1])
        w = F_.softmax(self.look(f).flatten(2), -1)
        local = torch.einsum("bkn,bcn->bkc", w, f.flatten(2)).flatten(1)
        state = torch.cat([F_.relu(self.local_fc(local)), F_.relu(self.global_fc(torch.cat([f.mean((2, 3)),
                                                                                          f.amax((2, 3))], 1))),
                           F_.relu(self.inv_fc(inv))], 1)
        h = torch.cat([state, self.p_pool(mean)], 1)
        layers = [m for m in self.trunk if isinstance(m, nn.Linear)]
        for k, lin in enumerate(layers):
            h = lin(h)
            g, b = self.trunk_film[k](mean).chunk(2, 1)
            h = h * (1 + g) + b
            if words is not None:
                h = h + self.trunk_attn[k](h[:, None], words, valid)[:, 0]
            h = F_.relu(h)
        return (h, f) if return_f else h


class GroundedEncoder(nn.Module):
    """Candidate A (docs/architecture_design.md): text acts early and spatially, planning keeps the value-iteration
    prior, and the agent's surroundings are read egocentrically from pixels.

    1. cells: a patch conv reads each cell's tile (or a 1x1 conv on symbolic one-hot cells).
    2. grounding: every cell attends over the prompt's words (cell -> text) and is FiLM-modulated by the text summary
       and the inventory -- the text marks which cells matter before planning.
    3. planning: a stem conv over [cells, coordinates], then one shared 3x3 conv applied `rec` times (stem re-injected).
    4. locate: a text-free attention map over cells finds the agent (it is drawn distinctly); the crop x crop windows
       of (channel-reduced) planning features around each cell are mixed by that attention, plus the features at it.
    5. decide: the state attends over the words again (what does the message say about now), then an MLP trunk.
    Every text projection starts at zero: with a fresh text pathway the network equals its text-free version."""

    def __init__(self, size=None, d=256, ch=64, dt=128, rec=8, trunk=3, crop=7, heads=4, crop_ch=16, pixels=True):
        super().__init__()
        self.d, self.ch, self.dt, self.rec, self.crop, self.pixels = d, ch, dt, rec, crop, pixels
        if pixels:
            from gft.minicraft.render import TILE
            self.patch = nn.Conv2d(3, ch, TILE, stride=TILE)
        else:
            self.patch = nn.Conv2d(N_CELLS + 5, ch, 1)
        self.cell = nn.Conv2d(ch, ch, 1)
        self.inv_fc = nn.Linear(AGENT_FEATS, d)
        self.ground = TextAttention(ch, dt, ch, heads)
        self.film = nn.Linear(dt + d, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.stem = nn.Conv2d(ch + 2, ch, 3, padding=1)
        self.rconv = nn.Conv2d(ch, ch, 3, padding=1)
        self.locate = nn.Conv2d(ch, 1, 1)
        self.crop_proj = nn.Conv2d(ch, crop_ch, 1)          # fewer channels: every cell's window is materialized
        self.local_fc = nn.Linear(crop_ch * crop * crop + ch, d)
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
        at = torch.einsum("bn,bcn->bc", a, f.flatten(2))
        h_local = self.local_fc(torch.cat([attention_crop(self.crop_proj(f), a, self.crop), at], 1))
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

    def __init__(self, size=None, d=256, w=128, dt=128, layers=4, heads=4, pre_conv=2, freqs=4, pixels=True, rec=0):
        """rec=K: after the pre-convs, one shared 3x3 conv applied K times with the input re-injected (the VIN-style
        recurrence of Encoder), so path information travels K cells before the tokens meet the words."""
        super().__init__()
        self.d, self.w, self.dt, self.pixels, self.rec = d, w, dt, pixels, rec
        if pixels:
            from gft.minicraft.render import TILE
            self.patch = nn.Conv2d(3, w, TILE, stride=TILE)
        else:
            self.patch = nn.Conv2d(N_CELLS + 5, w, 1)
        self.convs = nn.ModuleList(nn.Conv2d(w, w, 3, padding=1) for _ in range(pre_conv))
        if rec:
            self.rconv = nn.Conv2d(w, w, 3, padding=1)
        self.locate = nn.Conv2d(w, 1, 1)
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(freqs), persistent=False)
        self.pos = nn.Linear(8 * freqs, w)
        self.inv_tok = nn.Linear(AGENT_FEATS, w)
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
        if self.rec:
            u = c
            for _ in range(self.rec):
                c = F_.relu(self.rconv(c) + u)
        B, _, S, _ = c.shape
        coords = cell_coords(S, c.device).to(c.dtype)                        # (S*S, 2)
        a = F_.softmax(self.locate(c).flatten(1).float(), -1).to(c.dtype)    # (B, S*S) where the agent is
        w = coords[..., None] * self.freqs.to(c.dtype)                       # (S*S, 2, freqs)
        sn, cs = w.sin().flatten(1), w.cos().flatten(1)                      # absolute Fourier features
        # relative to the agent, averaged over the locator's attention: sin(u - v) = sin u cos v - cos u sin v, so the
        # expectation needs only the attention-weighted absolute features (exact when the attention is one-hot)
        ea_s, ea_c = a @ sn, a @ cs                                          # (B, 2 * freqs)
        rel_s = sn[None] * ea_c[:, None] - cs[None] * ea_s[:, None]
        rel_c = cs[None] * ea_c[:, None] + sn[None] * ea_s[:, None]
        feats = torch.cat([sn[None].expand(B, -1, -1), cs[None].expand(B, -1, -1), rel_s, rel_c], -1)
        cells = c.flatten(2).transpose(1, 2) + self.pos(feats)
        t = torch.cat([self.act_tok.expand(B, -1, -1).to(c.dtype), self.inv_tok(inv)[:, None], cells], 1)
        words, valid = (None, None) if p is None else split_prompt(p, self.dt)[:2]
        for blk in self.blocks:
            t = blk(t, words, valid)
        h = self.out(self.ln(t[:, 0]))
        return (h, t[:, 2:].transpose(1, 2).reshape(B, self.w, S, S)) if return_f else h


def split_sentences(text):
    """Feedback text -> its sentences ("You did not get some metal. Great job!" -> 2)."""
    return [x.strip() for x in re.findall(r"[^.!?]+[.!?]?", text) if x.strip()] or [text]


SENT_TOKENS = 32                  # subword tokens kept per sentence by the pretrained text encoder


class HFTextEncoder:
    """Frozen pretrained text encoder (Hugging Face; default BAAI/bge-small-en-v1.5): texts -> per-token contextual
    states (n, SENT_TOKENS, D) and a validity mask (n, SENT_TOKENS). Weights load from safetensors only."""

    def __init__(self, name, device, max_length=SENT_TOKENS):
        from transformers import AutoModel, AutoTokenizer
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, use_safetensors=True).to(device).eval()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.dim, self.device = self.model.config.hidden_size, device

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(texts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        return self.model(**enc).last_hidden_state.float(), enc["attention_mask"].bool()


class HashTextEncoder:
    """Stand-in for tests (no download): each word -> a fixed random vector seeded by its crc32, plus a leading
    sentence token; the same interface as HFTextEncoder."""

    def __init__(self, dim=16, device="cpu", max_length=SENT_TOKENS):
        self.dim, self.device, self.max_length = dim, device, max_length

    def __call__(self, texts):
        out = torch.zeros(len(texts), self.max_length, self.dim)
        mask = torch.zeros(len(texts), self.max_length, dtype=torch.bool)
        for i, t in enumerate(texts):
            words = ["<s>"] + re.findall(r"[a-z]+|\d+", t.lower())[: self.max_length - 1]
            for j, w in enumerate(words):
                out[i, j] = torch.randn(self.dim, generator=torch.Generator().manual_seed(zlib.crc32(w.encode())))
            mask[i, :len(words)] = True
        return out.to(self.device), mask.to(self.device)


def make_text_encoder(name, device, max_length=SENT_TOKENS):
    return HashTextEncoder(device=device, max_length=max_length) if name == "hash" else \
        HFTextEncoder(name, device, max_length)


class PromptEncoder(nn.Module):
    """Text prompt (request, or request + feedback) -> one vector: word embeddings, a small transformer,
    mean pooling, projection to the policy width. The projection starts at zero, so a fresh text pathway leaves the
    policy unchanged (every prompt maps to 0, as in text-free pretraining) and learns from the first update.

    packed: return per-word keys/values + validity for the legacy encoder's attention. states: return the word states
    themselves + validity, (B, L, dt + 8), for encoders with their own text projections. exact: embed words by their
    vocab.WORDS id (no collisions) instead of VOCAB hashed buckets (legacy; kept so old checkpoints load unchanged)."""

    def __init__(self, d, dt=128, layers=2, packed=False, states=False, exact=False, dropout=0.0, sep=False):
        super().__init__()
        self.packed, self.states, self.exact, self.sep = packed, states, exact, sep
        if exact:
            known, ids = _exact_vocab_table()
            self.register_buffer("known", known, persistent=False)
            self.register_buffer("known_ids", ids, persistent=False)
            self.tok = nn.Embedding(len(known) + 1 + UNK_BUCKETS + sep, dt, padding_idx=0)
        else:
            self.tok = nn.Embedding(VOCAB + sep, dt, padding_idx=0)   # sep: one extra row for the separator
        if packed:
            self.kv = nn.Linear(dt, 2 * dt)
        self.pos = nn.Embedding(MAX_TOKENS, dt)
        layer = nn.TransformerEncoderLayer(dt, 4, 2 * dt, dropout=dropout, batch_first=True, norm_first=True)
        self.text = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dt, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Exact checkpoints from before words were appended to vocab.WORDS: their rows for the old words, the numbers,
        the unknown buckets and the separator are kept, and the new words get this model's fresh rows in between
        (rows are [padding, words, numbers, buckets, separator], so appending words shifts everything after them)."""
        key = prefix + "tok.weight"
        if self.exact and key in state_dict and state_dict[key].shape[0] < self.tok.weight.shape[0]:
            old, new = state_dict[key], self.tok.weight.detach()
            n_new = new.shape[0] - old.shape[0]                  # words appended since
            split = 1 + len(WORDS) - n_new                       # padding + the old words
            state_dict[key] = torch.cat([old[:split], new[split:split + n_new].to(old), old[split:]])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ids(self, tokens):
        """Embedding rows of word tokens (see tokenize); 0 stays padding."""
        if not self.exact:
            out = torch.where(tokens > 0, 1 + (tokens - 1) % (VOCAB - 1), tokens)
        else:
            tokens = tokens.contiguous()
            i = torch.searchsorted(self.known, tokens).clamp(max=len(self.known) - 1)
            hit = self.known[i] == tokens
            unk = len(self.known) + 1 + tokens % UNK_BUCKETS
            out = torch.where(tokens > 0, torch.where(hit, self.known_ids[i], unk), tokens)
        if self.sep:                                  # the separator has its own (last) row
            out = torch.where(tokens == SEP_TOKEN, self.tok.num_embeddings - 1, out)
        return out

    def forward(self, tokens):
        pad = tokens == 0
        h = self.text(self.tok(self.ids(tokens)) + self.pos.weight[: tokens.shape[1]], src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        if self.states:
            return torch.cat([h * valid, valid.expand(-1, -1, 8)], -1)            # 8 validity channels: aligned strides
        if self.packed:
            return torch.cat([self.kv(h) * valid, valid.expand(-1, -1, 8)], -1)   # 8 validity channels keep strides aligned
        return self.out((h * valid).sum(1) / valid.sum(1).clamp(min=1))


PROMPT_TOKENS = 128               # subword tokens kept per prompt (request [SEP] feedback) by a pretrained encoder


class PretrainedPromptEncoder(nn.Module):
    """PromptEncoder's interface on top of a frozen pretrained text encoder (e.g. bge-small): the whole prompt text
    (request, or request [SEP] feedback) -> contextual subword states -> a trainable adapter (LayerNorm + projection +
    one transformer layer) -> per-token keys/values for the policy's cross-attention (packed), or a pooled vector.
    The pretrained states carry word meaning and context ("not done" vs "done") from the start; only the adapter and
    the policy learn. The frozen encoder is not a submodule (not saved, not trained); its outputs are cached per text
    (LRU), since the same prompts recur across acting and training."""

    def __init__(self, d, dt=128, packed=False, text_model="BAAI/bge-small-en-v1.5", dropout=0.0, cache=4096):
        super().__init__()
        self.packed, self.text_model, self.cache_size = packed, text_model, cache
        dim = 16 if text_model == "hash" else 384
        self.adapt = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dt))
        layer = nn.TransformerEncoderLayer(dt, 4, 2 * dt, dropout=dropout, batch_first=True, norm_first=True)
        self.text = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        if packed:
            self.kv = nn.Linear(dt, 2 * dt)
        self.out = nn.Linear(dt, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.__dict__["_enc"], self.__dict__["_cache"] = None, OrderedDict()

    def frozen_states(self, texts):
        """Cached frozen states (n, L, dim) fp16 and masks (n, L) for a list of texts."""
        dev = self.out.weight.device
        if self._enc is None or self._enc.device != dev:
            self.__dict__["_enc"] = make_text_encoder(self.text_model, dev, PROMPT_TOKENS)
            self._cache.clear()
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for j in range(0, len(new), 256):
            st, m = self._enc(new[j:j + 256])
            L = int(m.sum(1).max()) or 1
            for t, a, b in zip(new[j:j + 256], st[:, :L].to(dev, torch.float16), m[:, :L].to(dev)):
                self._cache[t] = (a, b)
        for t in texts:
            self._cache.move_to_end(t)
        rows = [self._cache[t] for t in texts]
        while len(self._cache) > max(self.cache_size, len(set(texts))):
            self._cache.popitem(last=False)
        L = max(len(a) for a, _ in rows)
        states = torch.zeros(len(texts), L, rows[0][0].shape[-1], dtype=torch.float16, device=dev)
        mask = torch.zeros(len(texts), L, dtype=torch.bool, device=dev)
        for i, (a, b) in enumerate(rows):
            states[i, :len(a)], mask[i, :len(b)] = a, b
        return states, mask

    def forward(self, texts):
        states, valid_b = self.frozen_states(list(texts))
        h = self.text(self.adapt(states.float()), src_key_padding_mask=~valid_b)
        valid = valid_b.float().unsqueeze(-1)
        if self.packed:
            return torch.cat([self.kv(h) * valid, valid.expand(-1, -1, 8)], -1)
        return self.out((h * valid).sum(1) / valid.sum(1).clamp(min=1))


# Policy network chosen by the architecture study (check_bc.py / check_dagger.py, 2026-10-22): 8 shared conv passes,
# width 256, 3-layer trunk. Best sample efficiency under DAgger at about equal compute.
# text_attn: the policy attends over the prompt's words (added 2026-10-22; requests and feedback are read per state).
POLICY_ARCH = dict(d=256, rec=8, trunk=3, text_attn=True)
# Candidates of docs/architecture_design.md, selected with arch=dict(kind=...): "gvin" = grounded VIN (A), "fusion" =
# vision-language transformer (B). Both use the exact-vocabulary text encoder.
ARCHS = {"vin": POLICY_ARCH,
         "gvin": dict(d=256, ch=64, rec=8, trunk=3, crop=7),
         "fusion": dict(d=256, w=128, layers=4, pre_conv=2),
         "film": dict(d=256, ch=32, rec=8, trunk=3)}


class CleanPolicy(nn.Module):
    """Policy pi(a | state, prompt) with a value head (used by PPO). The prompt is text: the request for the deployed
    policy and PPO; the request plus feedback for GFT's pi^fb, which is a copy of the same network (LLM-style).

    Prompts are encoded once per episode (encode_prompt) and the vectors indexed per step."""

    def __init__(self, size, dt=128, sep=False, text_model=None, **arch):
        """text_model: read prompts with a frozen pretrained encoder (PretrainedPromptEncoder; encode_prompt then takes
        the prompt texts) instead of word embeddings trained from scratch (token ids)."""
        super().__init__()
        kind = arch.pop("kind", "vin")
        exact_text = arch.pop("exact_text", False)    # words by exact vocab id (no hashed-bucket collisions)
        arch = {**ARCHS[kind], **arch}
        d = arch.pop("d")
        self.d, self.kind, self.text_model = d, kind, text_model
        if kind == "vin" and text_model:
            self.prompt = PretrainedPromptEncoder(d, dt, packed=arch.get("text_attn", False), text_model=text_model)
            self.enc = Encoder(size, d, dt=dt, **arch)
        elif kind == "vin":
            self.prompt = PromptEncoder(d, dt, packed=arch.get("text_attn", False), sep=sep, exact=exact_text)
            self.enc = Encoder(size, d, dt=dt, **arch)
        else:
            self.prompt = PromptEncoder(d, dt, states=True, exact=True, sep=sep)
            self.enc = {"gvin": GroundedEncoder, "fusion": FusionEncoder, "film": FilmEncoder}[kind](size, d, dt=dt,
                                                                                                     **arch)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)

    def encode_prompt(self, tokens):
        return self.prompt(tokens)

    def state(self, x, inv):
        """Text-independent state encoding (default VIN encoder only), reused across prompts by head()."""
        return self.enc.encode_state(x, inv)[0]

    def head(self, state, p):
        """Action logits of state encodings under per-step prompt encodings p."""
        return self.pi(self.enc.mix(state, p))

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
    review: float = 0.0          # graph: a mastered goal whose recent success falls below this is sampled as a
                                 # frontier goal again (it stays mastered: nothing it unlocked is locked); 0 = off
    confirm: bool = False        # graph: mastered needs `mastery` on two disjoint recent windows (the last two `window`s
                                 # of outcomes, or two halves of what there is), not one window
    block_size: int = 2          # blocks: goals introduced together, in depth order
    block_budget: int = 10_000   # blocks: feedbacks (GFT teacher messages / PPO rewards) per block before the next one


def goal_prerequisites():
    """{goal: set of goals whose item the expert's plan for it produces} -- the goals that must be achievable before
    it."""
    from gft.minicraft.expert import RAW, plan
    out = {}
    for g in range(N_GOALS):
        items = set()
        for op in plan(np.zeros(N_ITEMS, int), [g]):
            items.add(RECIPES[op[1]].output if op[0] == "craft" else next(i for i, (c, _) in RAW.items() if c == op[1]))
        out[g] = {h for h, o in enumerate(GOALS) if h != g and o.item in items}
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

    def _regressed(self):
        return {g for g in self.mastered if len(self.hist[g]) >= self.c.min_count     # the recent window only
                and np.mean(self.hist[g][-self.c.window:]) < self.c.review}

    def _graph_sample(self, rng, n):
        trainable, open_ = self._trainable(), self._open() | self._regressed()
        g = self.pool_goals
        ok_mask = np.all((g < 0) | np.isin(g, list(trainable)), axis=1)
        frontier = np.nonzero(ok_mask & np.isin(g, list(open_)).any(1))[0]
        ok = np.nonzero(ok_mask)[0]
        if len(ok) == 0:                      # a small pool may have no task made only of trainable goals
            ok = np.nonzero(np.isin(g, list(trainable)).any(1))[0]
            ok = ok if len(ok) else np.arange(len(g))
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
        keep_n = self.c.window * (2 if self.c.confirm else 1)
        for g, ok in zip(goals[mask], sat[mask]):
            h = self.hist[int(g)]
            h.append(bool(ok))
            del h[: max(0, len(h) - keep_n)]
        if self.graph:
            for g in sorted(self._open()):
                h = self.hist[g]
                if self.c.confirm:                    # two disjoint windows, both at mastery
                    half = min(self.c.window, len(h) // 2)
                    learned = half >= self.c.min_count and np.mean(h[-half:]) >= self.c.mastery and \
                        np.mean(h[-2 * half:-half]) >= self.c.mastery
                else:
                    learned = len(h) >= self.c.min_count and np.mean(h) >= self.c.mastery
                if learned:
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
    if kind in ("collected", "crafted", "nothing_fail", "use_fail", "goal_done", "goal_not_done", "subgoal", "needed",
                "hint_recipe", "hint_raw", "short", "no_done", "pace_quick", "pace_usual", "pace_slow"):
        return c[:2]
    return (kind,)                                  # station_fail, steps, eval_count ...: only their numbers differ


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
    if cfg.success_needs_done and acts:               # met = holds at the end and the agent said done
        A, V = np.array(acts).T, np.array(valid).T
        last = np.where(V.any(1), V.shape[1] - 1 - np.argmax(V[:, ::-1], 1), 0)
        sat = sat & (V.any(1) & (A[np.arange(len(A)), last] == DONE))[:, None]
    return env, log, snaps, np.array(acts).T if acts else np.zeros((len(req), 0), int), \
        np.array(valid).T if valid else np.zeros((len(req), 0), bool), sat, mask


def cached_pool(cfg, n, seed, tag="", cache_dir="outputs/minicraft/pools", **kw):
    """build_pool, cached on disk. Key: cfg, n, seed, `tag` (names any filter) and non-callable kwargs -- function
    objects are excluded because their repr differs between processes."""
    stable = sorted((k, repr(v)) for k, v in kw.items() if not callable(v))
    goal_list = tuple(g.phrase for g in GOALS)                   # a changed goal list changes every request
    key = hashlib.sha1(repr((cfg, n, seed, tag, stable, goal_list)).encode()).hexdigest()[:16]
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


def request_prompts(model, req, device, feedback=None, sep=False):
    """Prompt vectors (B, d) for a batch of requests (optionally with one feedback text each), without gradients."""
    texts = [prompt_text(req.text(b), None if feedback is None else feedback[b], sep) for b in range(len(req))]
    with torch.no_grad():
        return model.encode_prompt(texts if getattr(model, "text_model", None) else prompt_tokens(texts, device))


def policy_actions(model, env, req, device, rng, greedy=False, prompt=None, temperature=1.0, text_free=False):
    """Actions for all worlds; `prompt` = per-episode prompt vectors (default: the requests alone); text_free: no text
    input at all (the text-free base)."""
    x, inv = obs_tensor(env.grid, np.concatenate([env.pos, env.face[:, None]], 1), env.agent_state(), device)
    if not text_free:
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

def evaluate(model, cfg, req, device, greedy=False, seed=0, detail=False, feedback=None, sep=False):
    """Success rate (all requirements) and fraction of requirements met; with detail=True also per-episode success,
    per-goal success rates and mean episode length. feedback: one text per request added to the prompt (a policy
    conditioned on feedback, e.g. the ideal feedback for single-model GFT)."""
    rng = np.random.default_rng(seed)
    model.eval()
    prompt = request_prompts(model, req, device, feedback=feedback, sep=sep)
    saved = rng_state()                       # sampling draws from the torch RNG: seed it, and leave training's as is
    torch.manual_seed(seed)
    env, _, _, _, _, sat, mask = run_episodes(
        cfg, req, lambda e, t: policy_actions(model, e, req, device, rng, greedy, prompt=prompt), record=False)
    set_rng_state(saved)
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
                  profiles=("mixed", "reporter", "diagnoser", "coach", "grader"), on_loss=None):
    """BC on (slightly noisy) expert runs. The demonstrator executes the plan for behaviour_goals(request); the
    episode ends when that plan's goals are met.

    text=False (default): the prompt vector is zero throughout pretraining, so the text pathway stays untrained (and,
    with PromptEncoder's zero-initialized projection, maps every prompt to 0 until training starts) -- skills without
    any text meaning, and a text pathway that can still learn. text=True: the prompt is the request, plus for a share
    `feedback_frac` of episodes a teacher message about a *different* episode; the network then learns to ignore
    text altogether (measured: action distributions independent of the prompt), which blocks later learning.
    on_loss: called as on_loss(step, mean loss of the last 500 steps) every 500 steps."""
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
    recent = []
    for step in range(steps):
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
        if on_loss is not None:
            recent.append(loss.detach())
            if (step + 1) % 500 == 0:
                on_loss(step + 1, float(torch.stack(recent).mean()))
                recent = []
    return model


LANG_LEVELS = (0.5, 0.7, 0.9)   # demonstrator skill: competent but noisy; mostly-random episodes drown the BC signal


def lang_corpus(cfg, n_episodes, heldout, seed=0, levels=LANG_LEVELS, q=0.5, chunk=512, workers=None,
                profile="realistic"):
    """Pretraining corpus with grounded text: noisy-expert episodes, each with the request R it was given and the
    teacher's message about R, plus a hindsight request (the goals its demonstrator was executing, when it met them all
    and pressed DONE). Requests may contain the held-out goals; behaviour never does, and an episode that reaches a
    held-out goal anyway (random actions) is dropped -- their words are seen, the tasks are never done.

    Behaviour: with probability q the request with each held-out goal replaced by a random other goal, else an
    independent request over the other goals. Skill: expert action w.p. p ~ levels per episode, else a random world
    action. Returns per-step arrays (grid, pf, inv, action, episode index), per-episode request / message /
    hindsight-request texts (None: no hindsight label), and counts."""
    rng = np.random.default_rng(seed)
    held = sorted(set(heldout))
    others = [g for g in range(N_GOALS) if g not in held]
    keep_goals = lambda gs: not (gs & frozenset(held))
    expert = ExpertPool(workers)
    out = dict(grid=[], pf=[], inv=[], a=[], b=[], request=[], message=[], hindsight=[])
    counts = Counter()
    n_ep = 0
    while counts["generated"] < n_episodes:
        n = min(chunk, n_episodes - counts["generated"])
        req = sample_requests(rng, n, rng.integers(0, 2 ** 31, size=n))
        own = req.goals.copy()                           # the request with held-out goals replaced
        for b in range(n):
            row = [g for g in own[b] if g >= 0]
            row = sorted({g if g not in held else int(rng.choice([o for o in others if o not in row])) for g in row},
                         key=lambda g: -g)
            own[b] = -1
            own[b, :len(row)] = row
        indep = sample_requests(rng, n, req.seeds, combo_filter=keep_goals).goals
        beh = Requests(np.where((rng.random(n) < q)[:, None], own, indep), req.seeds)
        p = np.asarray(levels)[rng.integers(len(levels), size=n)]

        def act_fn(e, t):
            ex = expert(e, beh.goals, np.nonzero(~e.done)[0])
            return np.where(rng.random(n) < p, ex, rng.integers(N_WORLD_ACTIONS, size=n))

        env, log, snaps, acts, valid, sat_b, mask_b = run_episodes(cfg, beh, act_fn)
        sat_r, _ = requirement_status(env, req)
        facts = episode_facts_all(log, env, req.goals, sat_r)
        done_by_agent = valid.sum(1) < cfg.max_steps
        grid, pf, inv, a, b, _ = stack_steps(snaps, acts, valid)
        new_id = np.full(n, -1)
        for e, f in enumerate(facts):
            counts["generated"] += 1
            if any(achieved(f, h) for h in held):
                counts["dropped_heldout_achieved"] += 1
                continue
            new_id[e] = n_ep
            n_ep += 1
            out["request"].append(req.text(e))
            text = " ".join(x[2] for x in compose(f, profile, rng)) or "Okay."
            out["message"].append(text)
            counts["messages_naming_heldout"] += any(GOALS[h].phrase in text for h in held)
            ok_b = bool((sat_b[e] | ~mask_b[e]).all()) and done_by_agent[e]
            out["hindsight"].append(beh.text(e) if ok_b else None)
            counts["hindsight"] += ok_b
        sel = new_id[b] >= 0
        for k, v in (("grid", grid), ("pf", pf), ("inv", inv), ("a", a)):
            out[k].append(v[sel])
        out["b"].append(new_id[b[sel]])
    expert.close()
    for k in ("grid", "pf", "inv", "a", "b"):
        out[k] = np.concatenate(out[k])
    counts["episodes"] = n_ep
    counts["steps"] = len(out["a"])
    return out, dict(counts)


class PerceptionHeads(nn.Module):
    """Auxiliary heads for pixel pretraining, read off the encoder's feature map: the cell type at every location, the
    agent's position, and its facing. The targets are the symbolic observation the renderer draws from, so this is
    self-supervised perception pretraining (a stand-in for the visual pretraining an agent brings); the heads are
    dropped afterwards. Without it the pixel pathway can sit on the action-marginal plateau for a long time, because
    every action in this world is gated by where the agent is and what it faces."""

    def __init__(self, ch):
        super().__init__()
        self.cell = nn.Conv2d(ch, N_CELLS, 1)
        self.agent = nn.Conv2d(ch, 1, 1)
        self.face = nn.Conv2d(ch, 4, 1)

    def loss(self, f, x):
        """Mean of the three cross-entropies for feature map f (B, ch, S, S) and symbolic observation x."""
        B = len(x)
        cells = x[:, :N_CELLS].argmax(1)                                   # (B, S, S)
        pos = x[:, N_CELLS].flatten(1).argmax(1)                           # (B,)
        face = x[:, N_CELLS + 1:N_CELLS + 5].flatten(2).sum(2).argmax(1)   # (B,)
        l_cell = F_.cross_entropy(self.cell(f), cells)
        l_pos = F_.cross_entropy(self.agent(f).flatten(1), pos)
        l_face = F_.cross_entropy(self.face(f).flatten(2)[torch.arange(B, device=x.device), :, pos], face)
        return l_cell + l_pos + l_face


def pretrain_lang(cfg, device, corpus, steps=10_000, batch=512, lr=1e-3, seed=0, arch=None, log_every=0,
                  heldout_frac=0.05, max_eval_steps=100_000, perception=0.0):
    """BC of one network in two modes on a lang_corpus: request mode, prompt_text(hindsight request), on episodes with
    a hindsight label; message mode, prompt_text(request, message) -- GFT's pi^fb prompt -- on all episodes. Each
    batch is half of each. A share heldout_frac of the episodes is never trained on; every log_every steps the held-out
    loss per mode is measured, and the message's information gain: held-out message-mode loss under another
    episode's message (same request where possible) minus under the episode's own -- positive iff the actions
    depend on what the message says. Returns the model and that history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CleanPolicy(cfg.size, **(arch or {})).to(device)
    heads = PerceptionHeads(model.enc.ch).to(device) if perception and (arch or {}).get("pixels") else None
    opt = torch.optim.Adam(list(model.parameters()) + (list(heads.parameters()) if heads else []), lr=lr)
    n_ep = len(corpus["request"])
    has_h = np.array([h is not None for h in corpus["hindsight"]])
    held = np.zeros(n_ep, bool)
    held[rng.choice(n_ep, size=int(heldout_frac * n_ep), replace=False)] = True
    # swapped messages for the held-out episodes: another held-out episode's, of the same request where possible
    swap = np.arange(n_ep)
    by_req = {}
    for e in np.nonzero(held)[0]:
        by_req.setdefault(corpus["request"][e], []).append(e)
    for e in np.nonzero(held)[0]:
        same = [o for o in by_req[corpus["request"][e]] if o != e]
        swap[e] = rng.choice(same) if same else rng.choice([o for o in np.nonzero(held)[0] if o != e] or [e])
    tok_h = prompt_tokens([prompt_text(h or "") for h in corpus["hindsight"]], device)
    tok_m = prompt_tokens([prompt_text(r, m) for r, m in zip(corpus["request"], corpus["message"])], device)
    tok_s = prompt_tokens([prompt_text(r, corpus["message"][swap[e]]) for e, r in enumerate(corpus["request"])], device)
    L = max(t.shape[1] for t in (tok_h, tok_m, tok_s))
    tok = torch.cat([F_.pad(t, (0, L - t.shape[1])) for t in (tok_h, tok_m, tok_s)])   # prompt id: e | n_ep+e | 2n_ep+e
    b_all = corpus["b"]
    train_step = ~held[b_all]
    steps_h = torch.as_tensor(np.nonzero(has_h[b_all] & train_step)[0], device=device)
    steps_m = torch.as_tensor(np.nonzero(train_step)[0], device=device)
    a_t = torch.as_tensor(corpus["a"], device=device)
    b_t = torch.as_tensor(b_all, device=device)
    grid, pf, inv = on_device(device, corpus["grid"], corpus["pf"].astype(np.int16), corpus["inv"].astype(np.int16))
    offset = torch.cat([torch.zeros(batch // 2, dtype=torch.long), torch.full((batch - batch // 2,), n_ep)]).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    def subsample(sel):
        idx = np.nonzero(sel)[0]
        return idx[:: max(1, len(idx) // max_eval_steps)]
    ev_h, ev_m = subsample(held[b_all] & has_h[b_all]), subsample(held[b_all])

    @torch.no_grad()
    def loss_on(idx, prompt_offset):
        """Mean per-step CE on the steps idx, prompts of their episodes shifted by prompt_offset (mode)."""
        if len(idx) == 0:
            return float("nan")
        tot = 0.0
        for j in range(0, len(idx), FWD_CHUNK):
            i = torch.as_tensor(idx[j:j + FWD_CHUNK], device=device)
            x, iv = obs_tensor(grid[i], pf[i], inv[i], device)
            eps, inverse = torch.unique(b_t[i] + prompt_offset, return_inverse=True)
            logits, _ = model(x, iv, model.encode_prompt(tok[eps])[inverse])
            tot += float(F_.cross_entropy(logits, a_t[i], reduction="sum"))
        return tot / len(idx)

    l_perc = torch.zeros(())

    def diagnostics(step, ce):
        model.eval()
        own, swapped = loss_on(ev_m, n_ep), loss_on(ev_m, 2 * n_ep)
        row = dict(step=step, request=float(ce[:batch // 2].mean()), message=float(ce[batch // 2:].mean()),
                   heldout_request=loss_on(ev_h, 0), heldout_message=own, heldout_swapped=swapped,
                   info_gain=swapped - own)
        model.train()
        print(f"  pretrain {step}/{steps}: train CE request {row['request']:.3f} message {row['message']:.3f} | "
              f"held-out CE request {row['heldout_request']:.3f} message {own:.3f} | message info gain "
              f"{row['info_gain']:+.4f} nats/step" + (f" | perception {float(l_perc):.3f}" if heads is not None else ""),
              flush=True)
        return row

    hist = []
    for s in range(steps):                                           # batches drawn on the device: no host copies
        i = torch.cat([steps_h[torch.randint(len(steps_h), (batch // 2,), device=device, generator=gen)],
                       steps_m[torch.randint(len(steps_m), (batch - batch // 2,), device=device, generator=gen)]])
        pid = b_t[i] + offset
        x, iv = obs_tensor(grid[i], pf[i], inv[i], device)
        eps, inverse = torch.unique(pid, return_inverse=True)
        logits, _, (_, f, _) = model(x, iv, model.encode_prompt(tok[eps])[inverse], return_h=True)
        ce = F_.cross_entropy(logits, a_t[i], reduction="none")
        loss = ce.mean()
        if heads is not None:
            l_perc = heads.loss(f, x)
            loss = loss + perception * l_perc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (s + 1) % log_every == 0:
            row = diagnostics(s + 1, ce.detach())
            if heads is not None:
                row["perception"] = float(l_perc.detach())
            hist.append(row)
    return model, hist


# ---------------------------------------------------------------- entropy control (GFT and PPO)

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
    advantage: str = "gae"        # gae: critic + GAE (per-step credit) | group: GRPO, the episode's reward standardized
                                  # within its request's group of episodes, the same for every step, no critic
    profile: str = "realistic"    # teacher for reward=parse
    value_warmup: int = 0         # first rounds fit only the critic (the policy is not updated; episodes still count)


class PPO:
    def __init__(self, cfg, pool, base, device, pcfg: PPOConfig, seed=0, curriculum: CurriculumConfig = None):
        self.cfg, self.pool, self.device, self.p = cfg, pool, device, pcfg
        self.rng = np.random.default_rng(seed)
        self.cur = curriculum if hasattr(curriculum, "sample") else Curriculum(pool, curriculum or CurriculumConfig())
        self.model = base
        self.opt = torch.optim.Adam(self.model.parameters(), lr=pcfg.lr)
        self.ent_ctl = EntropyController(pcfg.entropy_target, pcfg.ent)
        self.episodes = self.verifier_calls = self.teacher_calls = self.rounds = 0

    def state_dict(self):
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(), rng=self.rng.bit_generator.state,
                    counters=(self.episodes, self.verifier_calls, self.teacher_calls, self.rounds),
                    curriculum=self.cur.state_dict(),
                    ent_ctl=self.ent_ctl.state_dict())

    def load_state_dict(self, sd):
        self.cur.load_state_dict(sd["curriculum"])
        if "ent_ctl" in sd:                       # checkpoints from before the entropy controller
            self.ent_ctl.load_state_dict(sd["ent_ctl"])
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.rng.bit_generator.state = sd["rng"]
        c = tuple(sd["counters"]) + (0, 0)                # checkpoints from before teacher_calls / rounds
        self.episodes, self.verifier_calls, self.teacher_calls, self.rounds = c[:4]

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
        env, log, snaps, acts, valid, sat, mask = run_episodes(
            cfg, req, lambda e, t: policy_actions(self.model, e, req, dev, self.rng, prompt=prompt))
        self.cur.update(req.goals, sat, mask, self.episodes + p.episodes, feedbacks=self.episodes + p.episodes)
        tm["rollout"] = time.time() - t0
        t0 = time.time()
        r_ep, extra = None, {}
        if p.reward == "parse":                       # information-matched: the scalar read off the teacher's message
            facts = episode_facts_all(log, env, req.goals, sat)
            texts = [" ".join(x[2] for x in compose(f, p.profile, self.rng)) or "Okay." for f in facts]
            r_ep = np.array([reported_fraction(t, req.goals[i]) for i, t in enumerate(texts)])
            true = (sat & mask).sum(1) / mask.sum(1)
            extra = dict(parse_matches_frac=float(np.isclose(r_ep, true).mean()))
            self.teacher_calls += p.episodes
        else:
            self.verifier_calls += p.episodes
        stats = self.update(req, tok, prompt, snaps, acts, valid, sat, mask, r_ep=r_ep,
                            value_only=self.rounds < p.value_warmup)
        tm["update"] = time.time() - t0
        self.episodes += p.episodes
        self.rounds += 1
        return dict(train_success=float((sat | ~mask).all(1).mean()), train_frac=float((sat & mask).sum() / mask.sum()),
                    **stats, **extra, time=tm, curriculum_unlocked=self.cur.unlocked)

    def update(self, req, tok, prompt, snaps, acts, valid, sat, mask, r_ep=None, value_only=False):
        """PPO update of self.model on one batch of its own episodes (terminal reward per episode, GAE). r_ep: the
        episodes' rewards (default: from the requirement check, `reward` = bin | frac). value_only: fit the critic
        alone (critic warm-up)."""
        p, dev = self.p, self.device
        if r_ep is None:
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
        if p.advantage == "group":                    # GRPO: sequence-level credit, standardized within the request
            r_np = np.asarray(r_ep, dtype=np.float64).copy()
            groups = {}
            for i in range(len(req)):
                groups.setdefault(req.text(i), []).append(i)
            for sel in groups.values():
                sel = np.array(sel)
                r_np[sel] = (r_np[sel] - r_np[sel].mean()) / (r_np[sel].std() + 1e-6)
            A = torch.as_tensor(r_np[b], device=dev, dtype=torch.float32)
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
                loss = (0.5 * v_loss if value_only else
                        pg + (0.0 if p.advantage == "group" else 0.5) * v_loss - self.ent_ctl.alpha * ent)
                self.opt.zero_grad()
                loss.backward()
                if value_only:                        # warm-up: only the value head moves (the trunk is shared)
                    head = {id(q) for q in self.model.v.parameters()}
                    for q in self.model.parameters():
                        if id(q) not in head:
                            q.grad = None
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
