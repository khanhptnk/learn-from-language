import io

import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.learn import (GFT, PPO, CleanPolicy, GFTConfig, PPOConfig, build_pool, rng_state,
                                 set_rng_state)

CFG = WorldConfig(max_steps=25)


def _learner(kind, pool, seed=0):
    torch.manual_seed(123)
    model = CleanPolicy(CFG.size)
    if kind in ("gft", "gftppo"):
        return GFT(CFG, pool, model, "cpu", GFTConfig(episodes=8, stage2=8, minibatch=64, replay_rounds=2), seed=seed,
                   ppo=PPOConfig(episodes=8, minibatch=64) if kind == "gftppo" else None)
    return PPO(CFG, pool, model, "cpu", PPOConfig(episodes=16, minibatch=64), seed=seed)


def _params(learner):
    m = learner.clean if hasattr(learner, "clean") else learner.model
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def test_resume_is_exact():
    torch.set_num_threads(1)
    pool = build_pool(CFG, 6, seed=0, noisy=())
    for kind in ("gft", "ppo", "gftppo"):
        torch.manual_seed(7)
        straight = _learner(kind, pool)
        straight.round()
        straight.round()
        torch.manual_seed(7)
        first = _learner(kind, pool)
        first.round()
        buf = io.BytesIO()
        torch.save(dict(learner=first.state_dict(), torch_rng=rng_state()), buf)
        torch.manual_seed(999)                                   # scramble global state before resuming
        np.random.seed(999)
        buf.seek(0)
        st = torch.load(buf, weights_only=False)
        resumed = _learner(kind, pool, seed=42)                  # different construction seed: state must override it
        resumed.load_state_dict(st["learner"])
        set_rng_state(st["torch_rng"])
        resumed.round()
        assert torch.equal(_params(straight), _params(resumed)), kind
        assert straight.episodes == resumed.episodes
        assert straight.cur.state_dict() == resumed.cur.state_dict()
