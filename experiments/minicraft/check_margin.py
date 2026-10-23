"""Why is pi^fb's Stage II margin ~0? Offline check on a GFT checkpoint: can pi^fb learn to condition on feedback
given enough of pi's own episodes, and how does that compare with filtered BC on the same episodes?

  .venv/bin/python experiments/minicraft/check_margin.py --ckpt outputs/minicraft/pixels/gft_realistic_s0.ckpt \
      --obs pixels --out outputs/minicraft/margin/gft_pixels.json

1. Rebuild the learner (pi, pi^fb, curriculum) from the checkpoint.
2. Roll out pi on N tasks drawn from the curriculum (as Stage I does); teacher feedback for each episode.
3. For each data size n: (a) continue fitting the run's pi^fb on the first n episodes exactly as Stage I does
   (prompt = request + feedback, feedback-word dropout); (b) filtered BC: fit a copy of pi on the successful episodes
   among them (prompt = request). Same number of passes for both.
4. Evaluate on fresh curriculum tasks (Stage II distribution) with identical sampling noise: pi (request), pi^fb
   with ideal feedback G* and with the request alone, filtered BC (request). Overall and per-goal success.
If filtered BC beats pi but pi^fb|G* does not, pi^fb is failing to learn the success conditioning (learning problem);
if neither beats pi, the data has too few successes to learn from.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from train import base_policy, has_heldout  # noqa: E402

import gft.minicraft.learn as learn_mod  # noqa: E402
from gft.minicraft.core import WorldConfig  # noqa: E402
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import (GFT, CurriculumConfig, GFTConfig, cached_pool, obs_tensor, on_device,  # noqa: E402
                                 policy_actions, prompt_text, prompt_tokens, request_prompts, rng_state, run_episodes,
                                 set_rng_state, stack_steps)
from gft.minicraft.teacher import compose, episode_facts  # noqa: E402


def load_learner(ckpt, obs, profile, device, seed=0):
    """Rebuild a GFT learner (pi, pi^fb, curriculum, pool) from a train.py checkpoint of a graph-curriculum run."""
    cfg = WorldConfig()
    if obs == "pixels":
        learn_mod.FWD_CHUNK = 1024
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    base = base_policy(cfg, device, seed=0, kind="random", pixels=obs == "pixels")
    learner = GFT(cfg, pool, copy.deepcopy(base), device, GFTConfig(profile=profile), seed=seed,
                  curriculum=CurriculumConfig(enabled=True, mode="graph", mastery=0.7))
    learner.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["learner"])
    return cfg, pool, learner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs", choices=["symbolic", "pixels"], default="pixels")
    ap.add_argument("--profile", default="realistic")
    ap.add_argument("--sizes", default="256,2048,8192", help="numbers of pi's episodes to fit on")
    ap.add_argument("--epochs", type=int, default=4, help="passes over the data (the runs use --epochs-aux 4)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--word-dropout", type=float, default=0.1)
    ap.add_argument("--n-eval", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=512, help="episodes per rollout batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = a.device
    cfg, pool, learner = load_learner(a.ckpt, a.obs, a.profile, dev, a.seed)
    pi, aux0, cur = learner.clean, learner.aux, learner.cur
    rng = np.random.default_rng(a.seed)
    record = dict(args=vars(a), feedbacks=learner.teacher_calls, results={})
    print(f"checkpoint at {learner.teacher_calls} feedbacks", flush=True)

    # 2. pi's episodes on curriculum tasks, with teacher feedback.
    t0 = time.time()
    n_max = max(map(int, a.sizes.split(",")))
    data = dict(grid=[], pf=[], inv=[], a=[], b=[], tok=[], tok_req=[], n_req=[], ok=[])
    n_done = 0
    while n_done < n_max:
        req = pool.requests(cur.sample(rng, min(a.chunk, n_max - n_done)))
        p = request_prompts(pi, req, dev)
        env, log, snaps, acts, valid, sat, mask = run_episodes(
            cfg, req, lambda e, t: policy_actions(pi, e, req, dev, rng, prompt=p))
        facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
        texts = [" ".join(x[2] for x in compose(f, a.profile, rng)) or "Okay." for f in facts]
        grid, pf, inv, act, b, _ = stack_steps(snaps, acts, valid)
        data["grid"].append(grid), data["pf"].append(pf), data["inv"].append(inv), data["a"].append(act)
        data["b"].append(b + n_done)
        data["tok"] += [prompt_text(req.text(i), texts[i]) for i in range(len(req))]
        data["tok_req"] += [prompt_text(req.text(i)) for i in range(len(req))]
        data["ok"].append((sat | ~mask).all(1))
        n_done += len(req)
    tok = prompt_tokens(data["tok"], dev)
    tok_req = prompt_tokens(data["tok_req"], dev)
    n_req = (prompt_tokens(data["tok_req"], dev) > 0).sum(1)
    fb_pos = torch.arange(tok.shape[1], device=dev)[None, :] >= n_req[:, None]
    ok = np.concatenate(data["ok"])
    grid, pf, inv, act, b = (np.concatenate(data[k]) for k in ("grid", "pf", "inv", "a", "b"))
    grid, pf, inv, act, b = on_device(dev, grid, pf, inv, act, b)
    record["data"] = dict(episodes=n_max, steps=int(len(b)), success=float(ok.mean()))
    print(f"rollout {time.time() - t0:.0f}s: {n_max} episodes, {len(b)} steps, pi success {ok.mean():.3f}", flush=True)

    # 4. Evaluation tasks (Stage II distribution) and their ideal feedback.
    idx_e = cur.sample(np.random.default_rng(a.seed + 1), a.n_eval)
    req_e = pool.requests(idx_e)
    star = [" ".join(x[2] for x in compose(pool.ideal_facts(i, rng), a.profile, rng)) or "Okay." for i in idx_e]

    @torch.no_grad()
    def evaluate(model, feedback=None):
        model.eval()
        pr = request_prompts(model, req_e, dev, feedback=feedback)
        r = np.random.default_rng(7)
        saved = rng_state()
        torch.manual_seed(7)                    # same action-sampling noise for every policy compared
        _, _, _, _, _, s, m = run_episodes(cfg, req_e, lambda e, t: policy_actions(model, e, req_e, dev, r, prompt=pr),
                                           record=False)
        set_rng_state(saved)
        model.train()
        per_goal = {GOALS[k].phrase: float(s[req_e.goals == k].mean()) for k in np.unique(req_e.goals) if k >= 0}
        return dict(success=float((s | ~m).all(1).mean()), per_goal=per_goal)

    def show(name, r):
        print(f"  {name:28s} success {r['success']:.3f} | "
              + " ".join(f"{k.split()[-1]} {v:.2f}" for k, v in r["per_goal"].items()), flush=True)
        return r

    def fit(model, steps_sel, tokens, dropout):
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        n_steps = max(1, a.epochs * len(steps_sel) // a.minibatch)
        for _ in range(n_steps):
            mb = steps_sel[torch.randint(len(steps_sel), (a.minibatch,), device=dev)]
            eps, inverse = torch.unique(b[mb], return_inverse=True)
            tk = tokens[eps]
            if dropout > 0:
                drop = fb_pos[eps] & (torch.rand(tk.shape, device=dev) < dropout)
                tk = torch.where(drop, torch.zeros_like(tk), tk)
            loss = F_.cross_entropy(model(*obs_tensor(grid[mb], pf[mb], inv[mb], dev), model.encode_prompt(tk)[inverse])[0],
                                    act[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return n_steps

    res = record["results"]
    res["pi"] = show("pi (request)", evaluate(pi))
    res["pi_fb_run_gstar"] = show("run's pi^fb | G*", evaluate(aux0, star))
    res["pi_fb_run_request"] = show("run's pi^fb | request", evaluate(aux0))
    ok_t = torch.as_tensor(ok, device=dev)
    for n in map(int, a.sizes.split(",")):
        t0 = time.time()
        sel = torch.nonzero(b < n)[:, 0]
        aux = copy.deepcopy(aux0)
        k1 = fit(aux, sel, tok, a.word_dropout)
        sel_ok = torch.nonzero((b < n) & ok_t[b])[:, 0]
        bc = copy.deepcopy(pi)
        k2 = fit(bc, sel_ok, tok_req, 0.0) if len(sel_ok) else 0
        print(f"n={n} episodes ({int(ok[:n].sum())} successful); {k1} pi^fb / {k2} BC updates "
              f"({time.time() - t0:.0f}s)", flush=True)
        res[f"n{n}"] = dict(successes=int(ok[:n].sum()), pi_fb_updates=k1, bc_updates=k2,
                            pi_fb_gstar=show("pi^fb fit | G*", evaluate(aux, star)),
                            pi_fb_request=show("pi^fb fit | request", evaluate(aux)),
                            filtered_bc=show("filtered BC (request)", evaluate(bc)))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
