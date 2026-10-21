"""GFT with a neural policy (paper Section 3), finite-sample.

Each round:
  Stage I   run pi on fresh requests, get teacher feedback F, fit pi^fb(X | D, F) by BC
            (on the last `replay_rounds` rounds of data; the theory uses only the current round).
  Stage II  sample F* ~ G*_D on fresh requests, roll out pi^fb(. | D, F*), distill into pi(. | D).
pi and pi^fb are the same network (feedback on/off) unless `shared=False`.
"""

import copy
from dataclasses import dataclass

import torch

from gft.algos.common import Episodes, Resources, rollout, train_bc
from gft.envs.reqworld import sample_requests
from gft.feedback.logical_form import Message, TeacherConfig, audit, ideal


@dataclass
class GFTConfig:
    episodes_per_round: int = 256      # Stage I on-policy episodes (each gets one teacher call)
    stage2_per_round: int = 256        # Stage II rollouts of pi^fb (each uses one G* sample)
    stage1_steps: int = 50
    stage2_steps: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    replay_rounds: int = 0             # 0 = all past rounds
    distill_rounds: int = 1
    shared: bool = True


class GFT:
    def __init__(self, policy, wcfg, tcfg: TeacherConfig, cfg: GFTConfig, gen):
        self.pi = policy
        self.pi_fb = policy if cfg.shared else copy.deepcopy(policy)
        self.wcfg, self.tcfg, self.cfg, self.gen = wcfg, tcfg, cfg, gen
        params = list({id(p): p for p in [*self.pi.parameters(), *self.pi_fb.parameters()]}.values())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.stage1: list[tuple[Episodes, Message]] = []
        self.stage2: list[tuple[Episodes]] = []
        self.res = Resources()

    def round(self):
        c, w = self.cfg, self.wcfg
        # Stage I: on-policy executions with hindsight feedback.
        req = sample_requests(w, c.episodes_per_round, self.gen)
        ep = rollout(self.pi, w, req, None, self.gen)
        msg = audit(req, ep.sat, ep.actual, w.n_values, self.tcfg, self.gen)
        self.res.env_episodes += len(ep)
        self.res.teacher_calls += len(ep)
        self.stage1.append((ep, msg))
        window = self.stage1[-c.replay_rounds:] if c.replay_rounds else self.stage1
        data = Episodes.cat([e for e, _ in window])
        msgs = Message.cat([m for _, m in window])
        train_bc(self.pi_fb, self.opt, data, msgs, c.stage1_steps, c.batch_size, self.gen, self.res)
        # Stage II: ideal-feedback-conditioned executions, distilled into the request-only policy.
        req2 = sample_requests(w, c.stage2_per_round, self.gen)
        f_star = ideal(req2, w.n_values, self.tcfg, self.gen)
        ep2 = rollout(self.pi_fb, w, req2, f_star, self.gen)
        self.res.env_episodes += len(ep2)
        self.res.gstar_samples += len(ep2)
        self.stage2.append(ep2)
        data2 = Episodes.cat(self.stage2[-c.distill_rounds:])
        train_bc(self.pi, self.opt, data2, None, c.stage2_steps, c.batch_size, self.gen, self.res)
        return {"stage2_success": float(ep2.sat.all(1).float().mean()),
                "stage1_success": float(ep.sat.all(1).float().mean()),
                "stage2_frac": float(ep2.sat.float().mean()),
                "stage1_frac": float(ep.sat.float().mean())}
