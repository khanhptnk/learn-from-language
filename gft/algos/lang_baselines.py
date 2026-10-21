"""Non-RL baselines. Phase 1: filtered behavior cloning (RFT / RAFT / STaR-style).

`accept`:
  verifier       keep executions that satisfy every requirement (oracle verifier)
  all_positive   keep executions whose partial-audit feedback reports no violation
"""

from dataclasses import dataclass

import torch

from gft.algos.common import Episodes, Resources, rollout, train_bc
from gft.envs.reqworld import sample_requests
from gft.feedback.logical_form import NEG, TeacherConfig, audit


@dataclass
class FilteredBCConfig:
    accept: str = "verifier"
    episodes_per_round: int = 256
    steps: int = 50
    batch_size: int = 256
    lr: float = 1e-3


class FilteredBC:
    def __init__(self, policy, wcfg, tcfg: TeacherConfig, cfg: FilteredBCConfig, gen):
        self.pi, self.wcfg, self.tcfg, self.cfg, self.gen = policy, wcfg, tcfg, cfg, gen
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.kept: list[Episodes] = []
        self.res = Resources()

    def round(self):
        c, w = self.cfg, self.wcfg
        req = sample_requests(w, c.episodes_per_round, self.gen)
        ep = rollout(self.pi, w, req, None, self.gen)
        self.res.env_episodes += len(ep)
        if c.accept == "verifier":
            self.res.verifier_calls += len(ep)
            keep = ep.sat.all(1)
        else:
            self.res.teacher_calls += len(ep)
            msg = audit(req, ep.sat, ep.actual, w.n_values, self.tcfg, self.gen)
            keep = ~((msg.pol == NEG) & msg.mask).any(1)
        if keep.any():
            self.kept.append(ep[keep])
        if self.kept:
            train_bc(self.pi, self.opt, Episodes.cat(self.kept), None, c.steps, c.batch_size, self.gen, self.res)
        return {"kept": int(keep.sum())}
