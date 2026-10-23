"""Correctness check on a GFT checkpoint: does pi^fb actually condition on the feedback?

  .venv/bin/python experiments/minicraft/check_conditioning.py --ckpt outputs/minicraft/graph_attn/gft_s0.ckpt --goal 4

1. pi and pi^fb are distinct networks (parameter distance from each other and from the base).
2. The Stage II prompt contains the feedback (token counts with vs without feedback).
3. On states from pi^fb's own rollouts under ideal feedback: KL between pi^fb's action distribution with the ideal
   feedback and with the request alone, and how often the most likely action differs. ~0 = feedback has no effect.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, has_heldout  # noqa: E402

from gft.minicraft.core import WorldConfig  # noqa: E402
from gft.minicraft.learn import (CleanPolicy, cached_pool, obs_tensor, on_device, policy_actions,  # noqa: E402
                                 prompt_text, prompt_tokens, request_prompts, run_episodes, stack_steps)
from gft.minicraft.teacher import compose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg, dev = WorldConfig(), a.device
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)["learner"]
    if "enc.q.weight" in st["clean"]:              # pixel policies have a patch conv; symbolic ones do not
        base = base_policy(cfg, dev, seed=0, kind="random", pixels="enc.patch.weight" in st["clean"])
    else:
        base = CleanPolicy(cfg.size, text_attn=False).to(dev)
        base.load_state_dict(torch.load("outputs/minicraft/base/base_v6_random_s0_size13.pt", map_location=dev))
    clean, aux = copy.deepcopy(base), copy.deepcopy(base)
    clean.load_state_dict(st["clean"])
    aux.load_state_dict(st["aux"])
    flat = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()])
    pb, pc, pa = flat(base), flat(clean), flat(aux)
    print(f"1. parameter distance: |pi - base| {float((pc - pb).norm()):.2f}, |pi^fb - base| {float((pa - pb).norm()):.2f}, "
          f"|pi^fb - pi| {float((pa - pc).norm()):.2f}  (base norm {float(pb.norm()):.1f})")
    # text pathway parameters specifically
    names = [n for n, _ in aux.named_parameters()]
    text = [i for i, n in enumerate(names) if n.startswith("prompt.") or n.startswith("enc.q") or n.startswith("enc.p_")]
    pa_l, pb_l = list(aux.parameters()), list(base.parameters())
    dtext = float(torch.sqrt(sum(((pa_l[i] - pb_l[i]) ** 2).sum() for i in text)))
    print(f"   pi^fb text-pathway change from base: {dtext:.3f} over {len(text)} tensors")

    rng = np.random.default_rng(0)
    idx = rng.choice(np.nonzero((pool.goals[:, 0] == a.goal) & (pool.goals[:, 1] < 0))[0], size=a.n)
    req = pool.requests(idx)
    star = [" ".join(s for _, _, s in compose(pool.ideal_facts(i, rng), "mixed", rng)) or "Okay." for i in idx]
    tok_fb = prompt_tokens([prompt_text(req.text(b), star[b]) for b in range(a.n)], dev)
    tok_rq = prompt_tokens([prompt_text(req.text(b)) for b in range(a.n)], dev)
    print(f"2. prompt tokens per episode: with feedback {float((tok_fb > 0).sum(1).float().mean()):.1f}, "
          f"request only {float((tok_rq > 0).sum(1).float().mean()):.1f}; example: {prompt_text(req.text(0), star[0])!r}")

    p_fb = request_prompts(aux, req, dev, feedback=star)
    r = np.random.default_rng(1)
    _, _, snaps, acts, valid, _, _ = run_episodes(cfg, req, lambda e, t: policy_actions(aux, e, req, dev, r, prompt=p_fb))
    grid, pf, inv, _, b, _ = stack_steps(snaps, acts, valid)
    grid, pf, inv = on_device(dev, grid, pf, inv)
    b_t = torch.as_tensor(b, device=dev)
    with torch.no_grad():
        p_rq = aux.encode_prompt(tok_rq)
        kls, flips, ent = [], [], []
        for j in range(0, len(b), 4096):
            s = slice(j, j + 4096)
            x, iv = obs_tensor(grid[s], pf[s], inv[s], dev)
            lf = F_.log_softmax(aux(x, iv, p_fb[b_t[s]])[0], -1)
            lr = F_.log_softmax(aux(x, iv, p_rq[b_t[s]])[0], -1)
            kls.append((lf.exp() * (lf - lr)).sum(-1))
            flips.append((lf.argmax(-1) != lr.argmax(-1)).float())
            ent.append(-(lf.exp() * lf).sum(-1))
        kl, flip, en = torch.cat(kls), torch.cat(flips), torch.cat(ent)
    print(f"3. on {len(b)} states of pi^fb's ideal-feedback rollouts: KL(pi^fb|ideal || pi^fb|request) mean "
          f"{float(kl.mean()):.4f}, 90th pct {float(kl.quantile(0.9)):.4f}, max {float(kl.max()):.3f}; most likely "
          f"action differs on {float(flip.mean()):.3f} of states; pi^fb entropy {float(en.mean()):.3f}")


if __name__ == "__main__":
    main()
