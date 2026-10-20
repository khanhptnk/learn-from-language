"""Policy-gradient baselines on the request-only policy: REINFORCE, RLOO, GRPO, PPO.

Rewards (plan: information-matched arms):
  bin    1{all requirements satisfied}     (oracle verifier)
  frac   fraction of requirements satisfied (full audit)
  parse  s_D(F) from the same partial-audit teacher GLFT sees
  align  ORACLE dense reference: requirement j's indicator credited to the action that set it.
         Needs the action-requirement alignment GLFT never gets; one-shot policies only.
All rewards except `align` are terminal: one scalar per episode.
"""

from dataclasses import dataclass

import torch

from glft.algos.common import Resources, rollout
from glft.envs.reqworld import sample_requests
from glft.feedback.logical_form import TeacherConfig, audit, parse_reward


@dataclass
class RLConfig:
    algo: str = "grpo"            # reinforce | rloo | grpo | ppo
    reward: str = "bin"           # bin | frac | parse | align
    episodes_per_round: int = 256
    group_size: int = 8           # samples per request (rloo, grpo; ppo/reinforce use it for batching only)
    epochs: int = 1               # optimization passes over each round's data
    minibatch: int = 256
    lr: float = 3e-4
    clip: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5


class PolicyGradient:
    def __init__(self, policy, wcfg, tcfg: TeacherConfig, cfg: RLConfig, gen):
        self.pi, self.wcfg, self.tcfg, self.cfg, self.gen = policy, wcfg, tcfg, cfg, gen
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.res = Resources()

    def _reward(self, ep):
        c = self.cfg
        if c.reward == "bin":
            self.res.verifier_calls += len(ep)
            return ep.sat.all(1).float()
        if c.reward == "frac":
            self.res.verifier_calls += len(ep)
            return ep.sat.float().mean(1)
        if c.reward == "parse":
            self.res.teacher_calls += len(ep)
            msg = audit(ep.req, ep.sat, ep.actual, self.wcfg.n_values, self.tcfg, self.gen)
            return parse_reward(msg, self.wcfg.m)
        if c.reward == "align":
            if not getattr(self.pi, "one_shot", False):
                raise ValueError("align reward needs a one-shot policy (action j <-> requirement j)")
            self.res.verifier_calls += len(ep)
            return ep.sat.float()  # (B, m)
        raise ValueError(c.reward)

    def _advantage(self, r):
        """Episode-level (B,) or per-requirement (B, m) rewards -> advantages of the same shape."""
        c = self.cfg
        G = c.group_size
        if c.algo == "reinforce":
            return r - r.mean(0)
        g = r.view(-1, G, *r.shape[1:])
        if c.algo == "rloo":
            return (g - (g.sum(1, keepdim=True) - g) / (G - 1)).view(r.shape)
        if c.algo == "grpo":
            return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)).view(r.shape)
        raise ValueError(c.algo)

    def _gae(self, r, values, valid):
        """Terminal-reward GAE with gamma = 1 over each episode's valid steps."""
        B, H = valid.shape
        last = valid.sum(1).long() - 1
        rewards = torch.zeros(B, H)
        rewards[torch.arange(B), last] = r
        values = values * valid
        adv = torch.zeros(B, H)
        running = torch.zeros(B)
        for t in reversed(range(H)):
            nxt = values[:, t + 1] if t + 1 < H else torch.zeros(B)
            nxt = torch.where(t + 1 < valid.sum(1), nxt, torch.zeros(B))
            delta = rewards[:, t] + nxt - values[:, t]
            running = delta + self.cfg.gae_lambda * running * (t + 1 < valid.sum(1)).float()
            adv[:, t] = running * valid[:, t]
        return adv, adv + values

    def round(self):
        c, w = self.cfg, self.wcfg
        req = sample_requests(w, c.episodes_per_round // c.group_size, self.gen).repeat_interleave(c.group_size)
        ep = rollout(self.pi, w, req, None, self.gen)
        self.res.env_episodes += len(ep)
        r = self._reward(ep)
        with torch.no_grad():
            if c.algo == "ppo":
                old_lp, valid, values = self.pi.log_probs(req, None, ep.act_slots, ep.act_values, ep.lengths,
                                                          with_value=True)
                adv, returns = self._gae(r, values, valid.float())
                adv = (adv - adv[valid].mean()) / (adv[valid].std() + 1e-6) * valid
            else:
                old_lp, valid = self.pi.log_probs(req, None, ep.act_slots, ep.act_values, ep.lengths)
                adv = self._advantage(r)
                adv = (adv if adv.dim() == 2 else adv[:, None]).expand_as(old_lp) * valid
        n = len(ep)
        for _ in range(c.epochs):
            perm = torch.randperm(n, generator=self.gen)
            for i in range(0, n, c.minibatch):
                idx = perm[i: i + c.minibatch]
                sub = ep[idx]
                if c.algo == "ppo":
                    lp, v, values = self.pi.log_probs(sub.req, None, sub.act_slots, sub.act_values, sub.lengths,
                                                      with_value=True)
                else:
                    lp, v = self.pi.log_probs(sub.req, None, sub.act_slots, sub.act_values, sub.lengths)
                ratio = (lp - old_lp[idx]).exp()
                a = adv[idx]
                obj = torch.minimum(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a)
                loss = -(obj * v).sum() / v.sum()
                if c.algo == "ppo":
                    loss = loss + c.value_coef * (((values - returns[idx]) ** 2) * v).sum() / v.sum()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
                self.opt.step()
                self.res.grad_steps += 1
        return {"train_reward": float(r.float().mean()), "train_success": float(ep.sat.all(1).float().mean())}
