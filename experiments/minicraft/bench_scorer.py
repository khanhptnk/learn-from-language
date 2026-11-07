"""Micro-benchmark of one message-scorer training step (gft_message.message_scorer_loss + backward + Adam) at the 17g
run's shapes: a 6-round replay of 256 episodes each, 200 steps, ~5 sentences per message. Prints ms per step and the
top ops by device time (torch.profiler), with TF32 matmuls off and on.
  .venv/bin/python experiments/minicraft/bench_scorer.py [--device cuda] [--steps 20]"""
import argparse
import time

import numpy as np
import torch

import gft.minicraft.gft_message as M
from gft.minicraft.learn import prompt_tokens


def fake_replay(n, T, n_sent, rng, device):
    X = torch.randn(n, T, 256 + M.STEP_FEATS, device=device).half()
    lens = rng.integers(40, T + 1, n)
    valid = torch.arange(T, device=device)[None] < torch.as_tensor(lens, device=device)[:, None]
    msg = lambda: sorted(rng.choice(n_sent, size=int(rng.integers(2, 7)), replace=False).tolist())
    reqs = [f"r{i}" for i in range(40)]
    meta = [dict(req=reqs[int(rng.integers(40))], msg=msg(), ok=bool(rng.random() < 0.2),
                 imp=msg() if rng.random() < 0.3 else None) for _ in range(n)]
    succ = {r: [msg() for _ in range(8)] for r in reqs}
    words = ["you", "made", "a", "plank", "at", "workbench", "still", "needed", "rope", "not", "done", "build", "raft"]
    sents = [" ".join(rng.choice(words, size=int(rng.integers(4, 10)))) for _ in range(n_sent)]
    return X, valid, meta, succ, prompt_tokens(sents, device)


def run(device, steps, tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    rng = np.random.default_rng(0)
    X, valid, meta, succ, tok = fake_replay(6 * 256, 200, 600, rng, device)
    torch.manual_seed(0)
    scorer = M.SentenceScorer(max_steps=200).to(device)
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-3)

    def step():
        loss, _ = M.message_scorer_loss(scorer, X, valid, meta, succ, tok, rng, device, P=64, n_others=4)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    for _ in range(3):
        step()
    sync()
    t = time.time()
    for _ in range(steps):
        step()
    sync()
    print(f"tf32={tf32}: {(time.time() - t) / steps * 1000:.1f} ms per scorer step", flush=True)
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if device.startswith("cuda")
                                                     else [])
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(3):
            step()
        sync()
    key = "self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=15), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    for tf32 in (False, True):
        run(a.device, a.steps, tf32)
