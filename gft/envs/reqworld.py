"""ReqWorld-C: a deterministic, compositional tool-use environment (Phase 1).

A request asks for m of K attributes ("slots", e.g. Bob's permission, attached file, recipient)
to end up at particular outcomes, each one of V. Each action is a tool call `set(slot, arg)`
with an opaque argument (think contact or file ID) or STOP. The argument's effect is a hidden
per-slot permutation sigma: writing `arg` to `slot` makes the slot's outcome sigma[slot, arg].
The final state keeps the last outcome written to each slot. Requirement j is satisfied iff the
requested slot ends at the requested outcome. Requirements may be met in any order, and
unrequested slots may be touched freely, so many executions are optimal.

sigma stands in for the user-specific mapping from IDs to things the user talks about; a
learner must discover it (characteristic K3 in the plan).

Everything is batched in torch:
  request: slots (B, m) long, values (B, m) long  -- target outcomes
  actions: slots (B, H) long in [0, K] (K = STOP), values (B, H) long in [0, V)  -- arguments
  lengths: (B,) number of actions taken before (and including) STOP, or H
"""

import functools
from dataclasses import dataclass

import torch

UNSET = -1


@dataclass(frozen=True)
class ReqWorldConfig:
    n_slots: int = 8      # K
    n_values: int = 4     # V: outcomes per slot (= arguments per slot)
    m: int = 4            # requirements per request
    horizon: int = 0      # H; 0 means m + 2
    scramble: bool = True  # hidden argument -> outcome permutation; False = identity
    world_seed: int = 0

    @property
    def H(self):
        return self.horizon or self.m + 2

    @property
    def stop(self):
        return self.n_slots


@dataclass
class Requests:
    slots: torch.Tensor   # (B, m)
    values: torch.Tensor  # (B, m)

    def __len__(self):
        return self.slots.shape[0]

    def repeat_interleave(self, k):
        return Requests(self.slots.repeat_interleave(k, 0), self.values.repeat_interleave(k, 0))

    def __getitem__(self, idx):
        return Requests(self.slots[idx], self.values[idx])


def sample_requests(cfg: ReqWorldConfig, n: int, gen: torch.Generator) -> Requests:
    slots = torch.rand(n, cfg.n_slots, generator=gen).argsort(dim=1)[:, : cfg.m].sort(dim=1).values
    values = torch.randint(cfg.n_values, (n, cfg.m), generator=gen)
    return Requests(slots, values)


@functools.lru_cache(maxsize=None)
def outcome_perm(cfg: ReqWorldConfig):
    """sigma: (K + 1, V) long, sigma[slot, arg] = outcome. The extra row (STOP) is never read."""
    if not cfg.scramble:
        return torch.arange(cfg.n_values).repeat(cfg.n_slots + 1, 1)
    g = torch.Generator().manual_seed(cfg.world_seed)
    return torch.stack([torch.randperm(cfg.n_values, generator=g) for _ in range(cfg.n_slots + 1)])


@functools.lru_cache(maxsize=None)
def arg_for(cfg: ReqWorldConfig):
    """sigma^{-1}: (K + 1, V) long, the argument producing each outcome."""
    return outcome_perm(cfg).argsort(dim=1)


def final_state(cfg: ReqWorldConfig, act_slots, act_values, lengths):
    """(B, K) final slot outcomes, UNSET where never written."""
    B, H = act_slots.shape
    state = torch.full((B, cfg.n_slots + 1), UNSET, dtype=torch.long)  # extra column absorbs STOP
    valid = torch.arange(H)[None, :] < lengths[:, None]
    for h in range(H):
        rows = valid[:, h]
        state[rows, act_slots[rows, h]] = outcome_perm(cfg)[act_slots[rows, h], act_values[rows, h]]
    return state[:, : cfg.n_slots]


def requirement_status(req: Requests, state):
    """(B, m) bool: requirement j satisfied, and (B, m) long: the slot's actual final value."""
    actual = state.gather(1, req.slots)
    return actual == req.values, actual


def noisy_args(cfg: ReqWorldConfig, req: Requests, q: float, gen: torch.Generator, systematic=1.0):
    """Arguments of a noisy demonstrator, aligned with the request's requirements. (B, m).

    The demonstrator aims at the target outcome with prob q. Otherwise it aims at a wrong outcome:
    with prob `systematic` the "misconception" outcome (w + 1) mod V, else uniform among wrong ones.
    With q < 1/2 and systematic errors even greedy decoding of the cloned policy fails.
    """
    B, m = req.slots.shape
    shift = torch.where(torch.rand(B, m, generator=gen) < systematic, 1,
                        torch.randint(1, cfg.n_values, (B, m), generator=gen))
    aimed = torch.where(torch.rand(B, m, generator=gen) < q, req.values, (req.values + shift) % cfg.n_values)
    return arg_for(cfg)[req.slots, aimed]


def demo_actions(cfg: ReqWorldConfig, req: Requests, q: float, gen: torch.Generator, systematic=1.0):
    """Noisy demonstrations: set each requested slot in random order (see `noisy_args`), then STOP.

    Used to pretrain a base policy whose per-requirement competence is about q, so that
    the initial success rate is about q^m (a stand-in for a pretrained model's prior).
    """
    B, m = req.slots.shape
    order = torch.rand(B, m, generator=gen).argsort(dim=1)
    slots = req.slots.gather(1, order)
    values = noisy_args(cfg, req, q, gen, systematic).gather(1, order)
    H = cfg.H
    act_slots = torch.full((B, H), cfg.stop, dtype=torch.long)
    act_values = torch.zeros((B, H), dtype=torch.long)
    act_slots[:, :m] = slots
    act_values[:, :m] = values
    lengths = torch.full((B,), m + 1, dtype=torch.long).clamp(max=H)
    return act_slots, act_values, lengths
