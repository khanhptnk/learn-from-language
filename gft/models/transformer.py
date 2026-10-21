"""Shared small transformer policy for ReqWorld-C.

One network serves both GFT policies: with a feedback message it is pi^fb(a | h, D, F);
with `msg=None` it is the request-only policy pi(a | h, D). RL baselines use the latter.

Sequence layout (length L = m + 1 + m + 2H):
  m request tokens        slot + value embeddings (an unordered set: no position embedding)
  1 feedback marker       FB_ON / FB_OFF
  m feedback tokens       slot + reported value + polarity, padded where not reported
  2H history tokens       BOS, s_0, v_0, s_1, v_1, ..., s_{H-1}   (LLM-style: one token per decision)
An action is a slot (incl. STOP) followed by a value. The slot s_k is predicted at the token
before it (BOS or v_{k-1}); the value v_k is predicted at the s_k token, so attention can look
up the request/feedback entry for that slot. Prefix (request + feedback) attends
bidirectionally; history attends causally to itself and fully to the prefix.
"""

import torch
from torch import nn

from gft.envs.reqworld import ReqWorldConfig, Requests
from gft.feedback.logical_form import Message

T_REQ, T_FB_ON, T_FB_OFF, T_FB, T_BOS, T_SLOT, T_VAL = range(7)


class Policy(nn.Module):
    def __init__(self, cfg: ReqWorldConfig, d_model=64, n_layers=2, n_heads=4, value_head=False):
        super().__init__()
        self.cfg = cfg
        K, V, H = cfg.n_slots, cfg.n_values, cfg.H
        self.type_emb = nn.Embedding(7, d_model)
        self.slot_emb = nn.Embedding(K + 2, d_model, padding_idx=0)   # 0 none, 1..K slots, K+1 STOP
        self.value_emb = nn.Embedding(V + 3, d_model, padding_idx=0)  # 0 none, 1..V, V+1 unset, V+2 hidden
        self.pol_emb = nn.Embedding(3, d_model, padding_idx=0)        # 0 none, POS, NEG
        self.step_emb = nn.Embedding(H, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.slot_head = nn.Linear(d_model, K + 1)
        self.value_head = nn.Linear(d_model, V)
        self.v_head = nn.Linear(d_model, 1) if value_head else None
        P = 2 * cfg.m + 1
        L = P + 2 * H
        i = torch.arange(L)[:, None]
        j = torch.arange(L)[None, :]
        allowed = (j < P) | ((i >= P) & (j <= i))
        self.register_buffer("attn_mask", ~allowed, persistent=False)
        self.P = P
        hist_types = torch.full((2 * H,), T_VAL)
        hist_types[0] = T_BOS
        hist_types[1::2] = T_SLOT
        self.register_buffer("hist_types", hist_types, persistent=False)
        steps = torch.zeros(2 * H, dtype=torch.long)
        steps[1::2] = torch.arange(H)
        steps[2::2] = torch.arange(H - 1)
        self.register_buffer("hist_steps", steps, persistent=False)

    def _embed(self, req: Requests, msg: Message | None, act_slots, act_values):
        B, m = req.slots.shape
        H = self.cfg.H
        dev = req.slots.device
        req_tok = self.type_emb.weight[T_REQ] + self.slot_emb(req.slots + 1) + self.value_emb(req.values + 1)
        if msg is None:
            marker = self.type_emb.weight[T_FB_OFF].expand(B, 1, -1)
            fb_tok = torch.zeros(B, m, req_tok.shape[-1], device=dev)
            fb_pad = torch.ones(B, m, dtype=torch.bool, device=dev)
        else:
            marker = self.type_emb.weight[T_FB_ON].expand(B, 1, -1)
            fb_tok = (self.type_emb.weight[T_FB] + self.slot_emb(msg.slots + 1) + self.value_emb(msg.values + 1)
                      + self.pol_emb(msg.pol))
            fb_pad = ~msg.mask
        # History content: BOS, then interleaved slot / value tokens.
        hist = self.type_emb(self.hist_types) + self.step_emb(self.hist_steps)
        hist = hist.expand(B, -1, -1).clone()
        hist[:, 1::2] = hist[:, 1::2] + self.slot_emb(act_slots + 1)
        hist[:, 2::2] = hist[:, 2::2] + self.value_emb(act_values[:, : H - 1] + 1)
        x = torch.cat([req_tok, marker, fb_tok, hist], 1)
        pad = torch.cat([torch.zeros(B, m + 1, dtype=torch.bool, device=dev), fb_pad,
                         torch.zeros(B, 2 * H, dtype=torch.bool, device=dev)], 1)
        return x, pad

    def hidden(self, req, msg, act_slots, act_values):
        """Hidden states at slot-decision positions (B, H, d) and value-decision positions (B, H, d)."""
        x, pad = self._embed(req, msg, act_slots, act_values)
        h = self.norm(self.encoder(x, mask=self.attn_mask, src_key_padding_mask=pad)[:, self.P:])
        return h[:, 0::2], h[:, 1::2]

    def log_probs(self, req, msg, act_slots, act_values, lengths, with_value=False):
        """Per-step log pi(a_h | ...) of the given actions, zero beyond `lengths`. (B, H)."""
        hs, hv = self.hidden(req, msg, act_slots, act_values)
        lp = self.slot_head(hs).log_softmax(-1).gather(-1, act_slots[..., None])[..., 0]
        lpv = self.value_head(hv).log_softmax(-1).gather(-1, act_values[..., None])[..., 0]
        lp = lp + torch.where(act_slots == self.cfg.stop, 0.0, lpv)
        valid = torch.arange(self.cfg.H, device=lp.device)[None] < lengths[:, None]
        lp = lp * valid
        if with_value:
            return lp, valid, self.v_head(hs)[..., 0]
        return lp, valid

    @torch.no_grad()
    def sample(self, req, msg, gen=None, greedy=False):
        B = len(req)
        H, stop = self.cfg.H, self.cfg.stop
        act_slots = torch.full((B, H), stop, dtype=torch.long)
        act_values = torch.zeros((B, H), dtype=torch.long)
        lengths = torch.full((B,), H, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)

        def pick(logits):
            if greedy:
                return logits.argmax(-1)
            return torch.multinomial(logits.softmax(-1), 1, generator=gen)[:, 0]

        for t in range(H):
            hs, _ = self.hidden(req, msg, act_slots, act_values)
            s = pick(self.slot_head(hs[:, t]))
            act_slots[~done, t] = s[~done]
            newly = (~done) & (s == stop)
            lengths[newly] = t + 1
            done |= newly
            if done.all():
                break
            _, hv = self.hidden(req, msg, act_slots, act_values)
            v = pick(self.value_head(hv[:, t]))
            act_values[~done, t] = v[~done]
        return act_slots, act_values, lengths
