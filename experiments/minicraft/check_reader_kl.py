"""Does the message change what the reader does? On states from the policy's own single-task episodes, the reader's
action distribution under different messages, compared by KL (nats) to the reader with the right success message:
no message, the episode's own feedback, another goal's success message, "Thanks!"; and the policy (request only)
for scale. Also the chance that a sampled action differs (total variation). (docs/STATUS.md 2026-11-03)
  uv run python experiments/minicraft/check_reader_kl.py <learner .ckpt> [device]"""
import sys

import numpy as np
import torch
import torch.nn.functional as F_

from gft.minicraft.core import WorldConfig
from gft.minicraft.goals import GOAL_ID, MAX_GOALS, Requests
from gft.minicraft.learn import (CleanPolicy, obs_tensor, policy_actions, prompt_text, prompt_tokens, run_episodes,
                                 stack_steps)
from gft.minicraft.teacher import compose, episode_facts_all

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/pod/gft/single/gft_v3e.ckpt"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"
torch.set_num_threads(8)
cfg = WorldConfig(success_needs_done=True)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)["learner"]
clean = CleanPolicy(cfg.size, pixels=True, exact_text=True); clean.load_state_dict(sd["clean"]); clean.to(DEV).eval()
aux = CleanPolicy(cfg.size, pixels=True, exact_text=True); aux.load_state_dict(sd["aux"]); aux.to(DEV).eval()
bank = sd["succ_parts"]
rng = np.random.default_rng(3)
text = lambda parts: " ".join(x[2] for x in parts) or "Okay."
NAMES = ["make planks", "make rope", "make bricks", "gather wood", "dig clay"]
n = 24
g = np.full((len(NAMES) * n, MAX_GOALS), -1); g[:, 0] = np.repeat([GOAL_ID[x] for x in NAMES], n)
req = Requests(g, rng.integers(0, 2**31, len(g)))
env, log, snaps, acts, valid, sat, _ = run_episodes(cfg, req, lambda e, t: policy_actions(clean, e, req, DEV, rng))
own = [text(compose(f, "realistic", rng)) for f in episode_facts_all(log, env, req.goals, sat)]
grid, pf, state, a, b, _ = stack_steps(snaps, acts, valid)
keep = rng.choice(len(b), size=min(4000, len(b)), replace=False)
grid, pf, state, b = grid[keep], pf[keep], state[keep], b[keep]
texts = [req.text(i) for i in range(len(req))]
pick = lambda r: text(bank[r][-8:][int(rng.integers(len(bank[r][-8:])))])
right = [pick(t) for t in texts]
other = [pick([o for o in NAMES if o != t][int(rng.integers(len(NAMES) - 1))]) for t in texts]
conds = {"right success msg": right, "no message": [None] * len(req), "own feedback": own,
         "other goal's success msg": other, "'Thanks!'": ["Thanks!"] * len(req)}


def logp(model, msgs):
    with torch.no_grad():
        p = model.encode_prompt(prompt_tokens([prompt_text(texts[i], msgs[i]) for i in range(len(req))], DEV))
        out = []
        for j in range(0, len(b), 512):
            x, iv = obs_tensor(grid[j:j + 512], pf[j:j + 512], state[j:j + 512], DEV)
            out.append(F_.log_softmax(model(x, iv, p[torch.as_tensor(b[j:j + 512], device=DEV)])[0], -1))
        return torch.cat(out)


ref = logp(aux, right)
pol = logp(clean, [None] * len(req))
print(f"{len(b)} states from {len(req)} episodes ({', '.join(NAMES)}); policy success {sat[:, 0].mean():.2f}")
print(f"reader entropy (right msg) {-(ref.exp() * ref).sum(-1).mean():.3f} nats")
for name, lp in [(k, logp(aux, v)) for k, v in conds.items() if k != "right success msg"] + [("POLICY, request only", pol)]:
    kl = (ref.exp() * (ref - lp)).sum(-1)
    tv = 0.5 * (ref.exp() - lp.exp()).abs().sum(-1)
    print(f"  vs {name:26s} KL mean {kl.mean():.4f}  p95 {kl.quantile(0.95):.4f}  | P(sampled action differs) {tv.mean():.3f}")
