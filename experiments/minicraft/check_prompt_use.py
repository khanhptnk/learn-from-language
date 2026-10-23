"""How much does a policy's output depend on its text prompt? Compares action distributions on the same states under
different prompts (KL to the mean prompt), and the spread of prompt vectors. A dead text pathway gives ~0 for both.

  .venv/bin/python experiments/minicraft/check_prompt_use.py --base random
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, heldout_splits  # noqa: E402

from gft.minicraft.core import MiniCraft, WorldConfig  # noqa: E402
from gft.minicraft.learn import obs_tensor, prompt_text, prompt_tokens, snapshot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="random")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg, dev = WorldConfig(), a.device
    model = base_policy(cfg, dev, seed=0, kind=a.base).eval()
    req = heldout_splits(64, tune=True)["val_worlds"]
    texts = sorted({req.text(b) for b in range(len(req))})[:32]
    prompts = [prompt_text(t) for t in texts] + [prompt_text(texts[0], fb) for fb in (
        "You collected 3 ore. You made an iron pickaxe.", "You built a bridge over the river.",
        "You did not collect or make anything.", "You collected 2 wood. You made 2 planks.")]
    env = MiniCraft(cfg, req.seeds[:64])
    x, iv = obs_tensor(*snapshot(env), dev)
    with torch.no_grad():
        p = model.encode_prompt(prompt_tokens(prompts, dev))                       # (P, d)
        logp = torch.stack([F_.log_softmax(model(x, iv, p[k].expand(len(x), -1))[0], -1) for k in range(len(p))])
        mean = logp.exp().mean(0).log()
        kl = (logp.exp() * (logp - mean)).sum(-1).mean(1)                          # per prompt, avg over states
    print(f"prompt vectors: norm {p.norm(dim=1).mean():.3f}, spread (std across prompts) {p.std(0).mean():.5f}")
    print(f"KL(action dist | prompt || mean over prompts): requests {kl[:len(texts)].mean():.5f}, "
          f"request+feedback {kl[len(texts):].mean():.5f}")


if __name__ == "__main__":
    main()
