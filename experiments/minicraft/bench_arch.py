"""Speed of the policy architectures (docs/architecture_design.md) on the GPU, pixel observations: no-grad forward at
a rollout batch, forward + backward at a training batch, with and without torch.compile and bf16 autocast.

  .venv/bin/python experiments/minicraft/bench_arch.py
"""

import argparse
import contextlib
import time

import numpy as np
import torch
import torch.nn.functional as F_

from gft.minicraft.core import N_ACTIONS, MiniCraft, WorldConfig
from gft.minicraft.learn import CleanPolicy, obs_tensor, prompt_text, prompt_tokens, snapshot


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="vin,gvin,fusion")
    ap.add_argument("--rollout", type=int, default=512)
    ap.add_argument("--train", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda"
    env = MiniCraft(WorldConfig(), np.arange(a.train))
    x, iv = obs_tensor(*snapshot(env), dev)
    texts = [prompt_text("get some metal and make some planks",
                         "You did not get some metal. You still needed ore. You managed to make some planks.")]
    y = torch.randint(N_ACTIONS, (a.train,), device=dev)
    for kind in a.kinds.split(","):
        for compiled in (False, True):
            for bf16 in (False, True):
                torch.manual_seed(0)
                m = CleanPolicy(None, kind=kind, pixels=True).to(dev)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
                tok = prompt_tokens(texts * 16, dev)
                if compiled:
                    m.forward = torch.compile(m.forward)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
                idx = torch.randint(16, (a.train,), device=dev)

                @torch.no_grad()
                def roll():
                    with amp:
                        m(x[:a.rollout], iv[:a.rollout], m.encode_prompt(tok)[idx[:a.rollout]])

                def step():
                    with amp:
                        loss = F_.cross_entropy(m(x, iv, m.encode_prompt(tok)[idx])[0].float(), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                n = sum(p.numel() for p in m.parameters())
                print(f"{kind:7s} compile {compiled!s:5s} bf16 {bf16!s:5s} params {n / 1e6:.2f}M  rollout fwd "
                      f"{timed(roll, a.reps):7.1f} ms / {a.rollout}  train step {timed(step, a.reps):7.1f} ms / "
                      f"{a.train}", flush=True)


if __name__ == "__main__":
    main()
