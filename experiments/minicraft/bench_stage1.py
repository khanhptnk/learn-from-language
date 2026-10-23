"""Micro-benchmark of one Stage I update on the GPU: text encoding of N distinct prompts vs. the policy network on 1024
states, forward + backward (to find what dominates pi^fb training)."""
import time

import numpy as np
import torch
import torch.nn.functional as F_

from gft.minicraft.core import MiniCraft, WorldConfig
from gft.minicraft.learn import CleanPolicy, obs_tensor, snapshot


def timed(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


if __name__ == "__main__":
    dev = "cuda"
    m = CleanPolicy(13).to(dev)
    env = MiniCraft(WorldConfig(), np.arange(1024))
    x, iv = obs_tensor(*snapshot(env), dev)
    act = torch.randint(12, (1024,), device=dev)
    for n_prompts in (1024, 256, 64):
        tok = torch.randint(1, 4096, (n_prompts, 70), device=dev)
        idx = torch.randint(n_prompts, (1024,), device=dev)

        def step():
            p = m.encode_prompt(tok)[idx]
            F_.cross_entropy(m(x, iv, p)[0], act).backward()

        def text_only():
            m.encode_prompt(tok).sum().backward()

        print(f"{n_prompts:5d} prompts/minibatch: full update {timed(step):6.1f} ms, text encoder alone {timed(text_only):6.1f} ms",
              flush=True)
