"""Feedback logical forms and the symbolic teacher (Phase 1).

A message is a set of reports, one per audited requirement:
  (slot, value, polarity)
where `value` is the slot's actual final value (if the teacher reveals it) and polarity says
whether the requirement was satisfied. Phase 1 feeds these logical forms to the policy
directly; later phases realize them as English.

Batched representation (B, m), aligned with the request's requirement order:
  mask     bool   requirement was reported
  slots    long   slot id
  values   long   reported value, VALUE_UNSET if the slot was never written,
                  VALUE_HIDDEN if the teacher does not reveal values
  pol      long   POS or NEG
"""

from dataclasses import dataclass

import torch

from glft.envs.reqworld import UNSET, Requests

POS, NEG = 1, 2


@dataclass(frozen=True)
class TeacherConfig:
    kappa_sat: float = 0.5     # P(report | requirement satisfied)
    kappa_viol: float = 0.5    # P(report | requirement violated); > kappa_sat models mention bias
    reveal_value: bool = True  # "Bob got edit" vs. only "Bob's permission was wrong"
    lie_rate: float = 0.0      # P(polarity flipped) per reported requirement


@dataclass
class Message:
    mask: torch.Tensor
    slots: torch.Tensor
    values: torch.Tensor
    pol: torch.Tensor

    def __getitem__(self, idx):
        return Message(self.mask[idx], self.slots[idx], self.values[idx], self.pol[idx])

    @staticmethod
    def cat(msgs):
        return Message(*(torch.cat([getattr(m, f) for m in msgs]) for f in ("mask", "slots", "values", "pol")))


def value_tokens(n_values):
    """Indices for the two non-value reports."""
    return n_values, n_values + 1  # VALUE_UNSET, VALUE_HIDDEN


def audit(req: Requests, sat, actual, n_values, cfg: TeacherConfig, gen: torch.Generator) -> Message:
    """Truthful (up to lie_rate) partial audit of executions with requirement status `sat`."""
    unset_tok, hidden_tok = value_tokens(n_values)
    kappa = torch.where(sat, torch.tensor(cfg.kappa_sat), torch.tensor(cfg.kappa_viol))
    mask = torch.rand(sat.shape, generator=gen) < kappa
    truth = torch.where(sat, POS, NEG)
    lie = torch.rand(sat.shape, generator=gen) < cfg.lie_rate
    pol = torch.where(lie, 3 - truth, truth)
    if cfg.reveal_value:
        values = torch.where(actual == UNSET, unset_tok, actual)
    else:
        values = torch.full_like(actual, hidden_tok)
    return Message(mask, req.slots.clone(), values, pol)


def ideal(req: Requests, n_values, cfg: TeacherConfig, gen: torch.Generator) -> Message:
    """G*: the teacher's audit of an execution that satisfies every requirement.

    Calibrated by construction for any truthful teacher, including mention-biased ones,
    because an optimal execution's feedback law depends only on kappa_sat.
    """
    sat = torch.ones_like(req.slots, dtype=torch.bool)
    return audit(req, sat, req.values, n_values, TeacherConfig(cfg.kappa_sat, cfg.kappa_viol, cfg.reveal_value, 0.0),
                 gen)


def parse_reward(msg: Message, m: int):
    """R-parse: fraction of requirements reported satisfied, s_D(F) of paper Eq. (22)."""
    return ((msg.pol == POS) & msg.mask).sum(dim=1).float() / m
