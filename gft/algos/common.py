"""Shared pieces for Phase 1 learners: episodes, rollouts, BC updates, evaluation, accounting."""

from dataclasses import dataclass, field

import torch

from gft.envs.reqworld import ReqWorldConfig, Requests, demo_actions, final_state, requirement_status, sample_requests
from gft.feedback.logical_form import Message


@dataclass
class Episodes:
    req: Requests
    act_slots: torch.Tensor
    act_values: torch.Tensor
    lengths: torch.Tensor
    sat: torch.Tensor      # (B, m) requirement satisfied
    actual: torch.Tensor   # (B, m) actual final value of each requested slot

    def __len__(self):
        return len(self.req)

    def __getitem__(self, idx):
        return Episodes(self.req[idx], self.act_slots[idx], self.act_values[idx], self.lengths[idx],
                        self.sat[idx], self.actual[idx])

    @staticmethod
    def cat(eps):
        return Episodes(
            Requests(torch.cat([e.req.slots for e in eps]), torch.cat([e.req.values for e in eps])),
            *(torch.cat([getattr(e, f) for e in eps]) for f in ("act_slots", "act_values", "lengths", "sat", "actual")),
        )


@dataclass
class Resources:
    """Interaction budget, logged separately per resource (plan: fair comparison protocol)."""
    env_episodes: int = 0
    teacher_calls: int = 0
    gstar_samples: int = 0
    verifier_calls: int = 0
    grad_steps: int = 0

    def as_dict(self):
        return dict(self.__dict__)


def execute(cfg: ReqWorldConfig, req: Requests, act_slots, act_values, lengths) -> Episodes:
    state = final_state(cfg, act_slots, act_values, lengths)
    sat, actual = requirement_status(req, state)
    return Episodes(req, act_slots, act_values, lengths, sat, actual)


def rollout(policy, cfg, req, msg: Message | None, gen, greedy=False) -> Episodes:
    policy.eval()
    out = policy.sample(req, msg, gen=gen, greedy=greedy)
    policy.train()
    return execute(cfg, req, *out)


def bc_loss(policy, ep: Episodes, msg: Message | None):
    """Negative log-likelihood of whole trajectories, averaged over episodes."""
    lp, _ = policy.log_probs(ep.req, msg, ep.act_slots, ep.act_values, ep.lengths)
    return -lp.sum(1).mean()


def train_bc(policy, opt, ep: Episodes, msg: Message | None, steps, batch_size, gen, res: Resources | None = None):
    n = len(ep)
    for _ in range(steps):
        idx = torch.randint(n, (min(batch_size, n),), generator=gen)
        loss = bc_loss(policy, ep[idx], msg[idx] if msg is not None else None)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if res is not None:
            res.grad_steps += 1
    return loss.item()


def pretrain_base(policy, cfg, q, steps, gen, lr=1e-3, batch_size=256, systematic=1.0):
    """Behavior-clone noisy demonstrations so the base policy has per-requirement competence ~ q."""
    demos = getattr(policy, "demo_actions", demo_actions)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        req = sample_requests(cfg, batch_size, gen)
        ep = execute(cfg, req, *demos(cfg, req, q, gen, systematic))
        loss = bc_loss(policy, ep, None)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(policy, cfg, n, gen):
    req = sample_requests(cfg, n, gen)
    out = {}
    for mode, greedy in (("sample", False), ("greedy", True)):
        ep = rollout(policy, cfg, req, None, gen, greedy=greedy)
        out[f"success_{mode}"] = float(ep.sat.all(1).float().mean())
        out[f"frac_{mode}"] = float(ep.sat.float().mean())
    return out


@dataclass
class Curve:
    points: list = field(default_factory=list)

    def log(self, res: Resources, metrics: dict, **extra):
        self.points.append({**res.as_dict(), **metrics, **extra})
