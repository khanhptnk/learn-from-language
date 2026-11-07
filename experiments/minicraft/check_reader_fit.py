"""Can the reader fit (request, feedback, trajectory)? (docs/STATUS.md 2026-11-03)

Data: single-task episodes of the GFT policy (a learner checkpoint), each with three feedback texts -- the realistic
teacher's, "complete" (every item collected / crafted with counts) and "narrated" (every notable event in time order,
with places: what was picked up where, made where, put where to no effect). Readers start from the checkpoint's
reader and are trained (behaviour cloning of the episodes' actions) under one prompt each: request only, request +
realistic, request + complete, request + narrated. Held-out action NLL tells how much each feedback lets the reader
predict the actions beyond the request; the generating policy's own NLL on the held-out actions (its entropy) is the
floor without extra information. Then a memorisation test: a few episodes, many passes -- the feedback can tell
episodes apart, the request alone cannot.
  python experiments/minicraft/check_reader_fit.py <learner .ckpt> [device] [n_train] [iters]"""
import copy
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F_

from gft.minicraft.core import WorldConfig
from gft.minicraft.goals import GOAL_ID, MAX_GOALS, Requests
from gft.minicraft.learn import (CleanPolicy, obs_tensor, policy_actions, prompt_text, prompt_tokens, run_episodes,
                                 stack_steps)
from gft.minicraft.teacher import compose, episode_facts_all, render

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/pod/gft/single/gft_v3e.ckpt"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"
N_TRAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
N_TEST, BATCH, LR = min(1000, max(32, N_TRAIN // 6)), 1024, 3e-4
MEMO_ITERS = min(3000, 5 * ITERS)
NAMES = ["gather wood", "gather fiber", "dig clay", "make planks", "make rope", "make bricks"]
PLACE = {"collected": 2, "crafted": 2, "used_on": 0, "craft_held": 0}   # template variants that name the place
torch.manual_seed(0)
cfg = WorldConfig(success_needs_done=True)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)["learner"]
policy = CleanPolicy(cfg.size, pixels=True, exact_text=True)
policy.load_state_dict(sd["clean"])
policy.to(DEV).eval()
reader0 = CleanPolicy(cfg.size, pixels=True, exact_text=True)
reader0.load_state_dict(sd["aux"])
reader0.to(DEV)
rng = np.random.default_rng(0)
text = lambda parts: " ".join(x[2] for x in parts) or "Okay."


def collect(n):
    """n episodes in chunks: steps (grid, pf, state, action, episode) and per episode the request and three texts."""
    out = {k: [] for k in ("grid", "pf", "state", "a", "b")}
    meta = []
    for s0 in range(0, n, 512):
        m = min(512, n - s0)
        g = np.full((m, MAX_GOALS), -1)
        g[:, 0] = [GOAL_ID[NAMES[i]] for i in rng.integers(len(NAMES), size=m)]
        req = Requests(g, rng.integers(0, 2 ** 31, m))
        env, log, snaps, acts, valid, sat, _ = run_episodes(cfg, req, lambda e, t: policy_actions(policy, e, req, DEV,
                                                                                                    rng))
        facts = episode_facts_all(log, env, req.goals, sat)
        for i, f in enumerate(facts):
            narr = " ".join(render(c[0], c, PLACE[c[0]]) for c, _, _ in f.events) or "Nothing happened."
            meta.append(dict(req=req.text(i), ok=bool(sat[i, 0]), realistic=text(compose(f, "realistic", rng)),
                             complete=text(compose(f, "complete", rng)), narrated=narr))
        grid, pf, state, a, b, _ = stack_steps(snaps, acts, valid)
        for k, v in (("grid", grid), ("pf", pf), ("state", state), ("a", a), ("b", b + s0)):
            out[k].append(v)
        print(f"  collected {s0 + m}/{n}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}, meta


def tokens(meta, cond):
    return prompt_tokens([prompt_text(m["req"], None if cond == "request" else m[cond]) for m in meta], DEV)


def nll(model, data, tok, idx=None):
    """Mean action NLL over the steps idx (default all), no gradients."""
    idx = np.arange(len(data["a"])) if idx is None else idx
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for j in range(0, len(idx), 4096):
            k = idx[j:j + 4096]
            x, iv = obs_tensor(data["grid"][k], data["pf"][k], data["state"][k], DEV)
            b = torch.as_tensor(data["b"][k], device=DEV)
            eps, inv = torch.unique(b, return_inverse=True)
            lp = F_.log_softmax(model(x, iv, model.encode_prompt(tok[eps])[inv])[0], -1)
            tot += -lp.gather(1, torch.as_tensor(data["a"][k], device=DEV)[:, None]).sum().item()
    return tot / len(idx)


def train(data, tok, iters, test=None, test_tok=None, every=1000):
    model = copy.deepcopy(reader0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = []
    r = np.random.default_rng(1)
    for it in range(1, iters + 1):
        model.train()
        k = r.integers(len(data["a"]), size=BATCH)
        x, iv = obs_tensor(data["grid"][k], data["pf"][k], data["state"][k], DEV)
        b = torch.as_tensor(data["b"][k], device=DEV)
        eps, inv = torch.unique(b, return_inverse=True)
        loss = F_.cross_entropy(model(x, iv, model.encode_prompt(tok[eps])[inv])[0],
                                torch.as_tensor(data["a"][k], device=DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if test is not None and (it % every == 0 or it == iters):
            sub = r.choice(len(data["a"]), size=min(50_000, len(data["a"])), replace=False)
            hist.append((it, nll(model, data, tok, sub), nll(model, test, test_tok)))
    return model, hist


t0 = time.time()
print(f"collecting {N_TRAIN} + {N_TEST} episodes", flush=True)
train_d, train_m = collect(N_TRAIN)
test_d, test_m = collect(N_TEST)
print(f"{len(train_d['a'])} / {len(test_d['a'])} steps; success {np.mean([m['ok'] for m in train_m]):.2f} "
      f"({time.time() - t0:.0f}s)", flush=True)
for c in ("realistic", "complete", "narrated"):
    w = np.array([len(m[c].split()) for m in train_m])
    print(f"  {c}: {w.mean():.0f} words on average, {np.mean(w > 120):.0%} over 120 (prompts keep 128 tokens); "
          f"e.g. {test_m[0][c][:300]}")
print(f"\nfloor: the generating policy's NLL on held-out actions (its entropy, request only): "
      f"{nll(policy, test_d, tokens(test_m, 'request')):.4f}")
print(f"start: the checkpoint's reader, request only: {nll(reader0, test_d, tokens(test_m, 'request')):.4f}", flush=True)
for cond in ("request", "realistic", "complete", "narrated"):
    t1 = time.time()
    _, hist = train(train_d, tokens(train_m, cond), ITERS, test_d, tokens(test_m, cond))
    print(f"\n[{cond}] ({time.time() - t1:.0f}s) iter: train NLL / held-out NLL")
    print("  " + "  ".join(f"{it}: {tr:.4f}/{te:.4f}" for it, tr, te in hist), flush=True)

print(f"\nmemorisation: 48 episodes, {MEMO_ITERS} iterations, train NLL (lower = fits the episodes' own actions)")
keep = np.isin(train_d["b"], np.arange(48))
small = {k: v[keep] for k, v in train_d.items()}
for cond in ("request", "realistic", "complete", "narrated"):
    m, _ = train(small, tokens(train_m, cond), MEMO_ITERS)
    print(f"  {cond:10s} {nll(m, small, tokens(train_m, cond)):.4f}", flush=True)
