"""Fast one-shot policy for ReqWorld-C (level L1).

The policy emits a whole execution in one forward pass: one argument per requested slot,
written in request order (so the environment is a contextual combinatorial bandit with
V^m executions per request). Tokens (length 2m + 1):
  m request tokens    slot + target outcome          -> argument logits are read here
  1 feedback marker   FB_ON / FB_OFF
  m feedback tokens   slot + reported outcome + polarity, padded where not reported
Request targets and reported outcomes share one outcome embedding table, as the same words
would in a language model; the argument head is separate, so the argument -> outcome map
(sigma) must be learned. With `msg=None` the marker is FB_OFF: the request-only policy pi.

Implements the same interface as `gft.models.transformer.Policy` (log_probs / sample over
(act_slots, act_values, lengths)), so every Phase 1 learner runs unchanged.
"""

import torch
from torch import nn

from gft.envs.reqworld import ReqWorldConfig, Requests, noisy_args
from gft.feedback.logical_form import Message

T_REQ, T_FB_ON, T_FB_OFF, T_FB = range(4)


class SetPolicy(nn.Module):
    one_shot = True

    def __init__(self, cfg: ReqWorldConfig, d_model=32, n_layers=1, n_heads=2, value_head=False):
        super().__init__()
        self.cfg = cfg
        K, V = cfg.n_slots, cfg.n_values
        self.type_emb = nn.Embedding(4, d_model)
        self.slot_emb = nn.Embedding(K, d_model)
        self.outcome_emb = nn.Embedding(V + 3, d_model, padding_idx=0)  # 0 none, 1..V, V+1 unset, V+2 hidden
        self.pol_emb = nn.Embedding(3, d_model, padding_idx=0)          # 0 none, POS, NEG
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.arg_head = nn.Linear(d_model, V)
        self.v_head = nn.Linear(d_model, 1) if value_head else None

    def hidden(self, req: Requests, msg: Message | None):
        B, m = req.slots.shape
        req_tok = self.type_emb.weight[T_REQ] + self.slot_emb(req.slots) + self.outcome_emb(req.values + 1)
        if msg is None:
            marker = self.type_emb.weight[T_FB_OFF].expand(B, 1, -1)
            fb_tok = torch.zeros_like(req_tok)
            fb_pad = torch.ones(B, m, dtype=torch.bool)
        else:
            marker = self.type_emb.weight[T_FB_ON].expand(B, 1, -1)
            fb_tok = (self.type_emb.weight[T_FB] + self.slot_emb(msg.slots) + self.outcome_emb(msg.values + 1)
                      + self.pol_emb(msg.pol))
            fb_pad = ~msg.mask
        x = torch.cat([req_tok, marker, fb_tok], 1)
        pad = torch.cat([torch.zeros(B, m + 1, dtype=torch.bool), fb_pad], 1)
        return self.norm(self.encoder(x, src_key_padding_mask=pad)[:, :m])

    def log_probs(self, req, msg, act_slots, act_values, lengths=None, with_value=False):
        h = self.hidden(req, msg)
        lp = self.arg_head(h).log_softmax(-1).gather(-1, act_values[..., None])[..., 0]
        valid = torch.ones_like(lp, dtype=torch.bool)
        if with_value:
            return lp, valid, self.v_head(h)[..., 0]
        return lp, valid

    @torch.no_grad()
    def sample(self, req, msg, gen=None, greedy=False):
        logits = self.arg_head(self.hidden(req, msg))
        if greedy:
            args = logits.argmax(-1)
        else:
            B, m, V = logits.shape
            args = torch.multinomial(logits.softmax(-1).view(-1, V), 1, generator=gen).view(B, m)
        lengths = torch.full((len(req),), req.slots.shape[1], dtype=torch.long)
        return req.slots.clone(), args, lengths

    @staticmethod
    def demo_actions(cfg, req, q, gen, systematic=1.0):
        """Noisy one-shot demonstrations (see `reqworld.noisy_args`)."""
        args = noisy_args(cfg, req, q, gen, systematic)
        return req.slots.clone(), args, torch.full((len(req),), req.slots.shape[1], dtype=torch.long)
