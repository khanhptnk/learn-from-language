"""Diagnostics for GFT's feedback-conditioned policy pi^fb: a copy of the base policy prompted with the request plus
feedback. Does Stage I teach it to use the feedback, or to memorize episodes? Used to choose pi^fb's training settings
without looking at final GFT results.

  .venv/bin/python experiments/minicraft/check_reader.py --lrs 3e-4,1e-3 --out outputs/minicraft/reader/prompt.json

1. Roll out the base policy (--base, see train.py BASES) on training tasks; teacher feedback for every episode.
2. For each learning rate, fit a fresh copy of the base as pi^fb on that fixed batch exactly as GFT's Stage I does
   (random step minibatches, feedback-word dropout). Report action log-loss on training vs held-out episodes, overall
   and on key steps (steps where something was collected, crafted or bridged), and on held-out episodes with the
   feedback removed from the prompt -- the gap between "held" and "held_no_fb" is how much pi^fb uses the feedback.
3. Stage II check at every evaluation point: roll out pi^fb prompted with ideal feedback on training tasks; compare
   request success and fraction of requirements met with the base policy's.
4. Steering probe after training: prompt pi^fb with the request plus ONE sentence ("You ended on the far side of the
   river.", ...) and measure how often that outcome happens, vs. the base and vs. a content-free control sentence.
   If feedback means anything to pi^fb, the described outcome should become more frequent.
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

from gft.minicraft.core import (EV_BRIDGE, EV_COLLECT, EV_CRAFT, GEM_ITEM, IRON_PICKAXE, ITEM_NAMES,  # noqa: E402
                                WorldConfig)
from gft.minicraft.learn import (Curriculum, CurriculumConfig, cached_pool, obs_tensor, on_device,  # noqa: E402
                                 policy_actions, prompt_text, prompt_tokens, request_prompts, rng_state, run_episodes,
                                 set_rng_state, stack_steps)
from gft.minicraft.teacher import compose, episode_facts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="random")
    ap.add_argument("--curriculum", choices=["on", "off"], default="off")
    ap.add_argument("--unlocked", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--profile", default="mixed")
    ap.add_argument("--lrs", default="3e-4,1e-3")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--evals", type=int, default=4, help="evaluation points during training")
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--word-dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-stage2", type=int, default=2048)
    ap.add_argument("--n-probe", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    cfg, dev = WorldConfig(), a.device
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    base = base_policy(cfg, dev, seed=0, kind=a.base)
    base.eval()
    rng = np.random.default_rng(a.seed)
    cur = Curriculum(pool, CurriculumConfig(enabled=a.curriculum == "on", start=a.unlocked))
    record = dict(args=vars(a), runs={})

    # 1. Fixed batch of base-policy episodes with teacher feedback.
    t0 = time.time()
    req = pool.requests(cur.sample(rng, a.episodes))
    prompt = request_prompts(base, req, dev)
    env, log, snaps, acts, valid, sat, mask = run_episodes(
        cfg, req, lambda e, t: policy_actions(base, e, req, dev, rng, prompt=prompt))
    facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
    collected = {ITEM_NAMES[i]: float(np.mean([f.collected[i] > 0 for f in facts])) for i in range(len(ITEM_NAMES))
                 if any(f.collected[i] > 0 for f in facts)}
    record["base"] = dict(success=float((sat | ~mask).all(1).mean()), collected=collected)
    print(f"rollout {time.time() - t0:.0f}s; base success {record['base']['success']:.3f}; episodes collecting: "
          + ", ".join(f"{k} {v:.3f}" for k, v in collected.items()), flush=True)
    texts = [" ".join(s for _, _, s in compose(f, a.profile, rng)) or "Okay." for f in facts]
    tok = prompt_tokens([prompt_text(req.text(b), texts[b]) for b in range(len(req))], dev)
    tok_req = prompt_tokens([prompt_text(req.text(b)) for b in range(len(req))], dev)
    n_req = (tok_req > 0).sum(1)
    fb_pos = torch.arange(tok.shape[1], device=dev)[None, :] >= n_req[:, None]      # feedback-word positions
    grid, pf, inv, act, b, t = stack_steps(snaps, acts, valid)
    key = np.isin(np.array([log[tt]["type"][bb] for bb, tt in zip(b, t)]), [EV_COLLECT, EV_CRAFT, EV_BRIDGE])
    grid, pf, inv = on_device(dev, grid, pf, inv)
    act, b_t = torch.as_tensor(act, device=dev), torch.as_tensor(b, device=dev)
    held_ep = torch.as_tensor(rng.random(len(req)) < 0.2, device=dev)
    held, key = held_ep[b_t], torch.as_tensor(key, device=dev)
    sets = dict(train=torch.nonzero(~held)[:, 0], held=torch.nonzero(held)[:, 0],
                key_train=torch.nonzero(~held & key)[:, 0], key_held=torch.nonzero(held & key)[:, 0])
    record["sizes"] = {k: int(len(v)) for k, v in sets.items()}
    print(f"{len(b)} steps; sizes {record['sizes']}", flush=True)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    eval_sets = {k: v[torch.randperm(len(v), generator=gen, device=dev)[:8192]] for k, v in sets.items()}
    eval_sets["train_no_fb"] = eval_sets["train"]            # fitting check: does feedback help on seen episodes?
    eval_sets["held_no_fb"] = eval_sets["held"]
    eval_sets["key_held_no_fb"] = eval_sets["key_held"]

    @torch.no_grad()
    def losses(model):
        out = {}
        for name, sel in eval_sets.items():
            tk = tok_req if name.endswith("no_fb") else tok
            tot = 0.0
            for j in range(0, len(sel), 4096):
                s = sel[j:j + 4096]
                eps, inverse = torch.unique(b_t[s], return_inverse=True)
                lg, _ = model(*obs_tensor(grid[s], pf[s], inv[s], dev), model.encode_prompt(tk[eps])[inverse])
                tot += F_.cross_entropy(lg, act[s], reduction="sum").item()
            out[name] = tot / max(len(sel), 1)
        return out

    # 3. Stage II tasks with ideal feedback from the success bank.
    idx2 = (rng.choice(np.nonzero(cur.deepest == a.unlocked - 1)[0], size=a.n_stage2) if a.curriculum == "on"
            else rng.choice(len(pool.seeds), size=a.n_stage2))
    req2 = pool.requests(idx2)
    star = [" ".join(s for _, _, s in compose(pool.ideal_facts(i, rng), a.profile, rng)) or "Okay." for i in idx2]

    @torch.no_grad()
    def stage2(model, with_feedback=True):
        p2 = request_prompts(model, req2, dev, feedback=star if with_feedback else None)
        r2 = np.random.default_rng(1)
        saved = rng_state()
        torch.manual_seed(1)                     # same action-sampling noise for every policy compared
        _, _, _, _, _, s2, m2 = run_episodes(cfg, req2, lambda e, t: policy_actions(
            model, e, req2, dev, r2, prompt=p2), record=False)
        set_rng_state(saved)
        return dict(success=float((s2 | ~m2).all(1).mean()), frac=float((s2 & m2).sum() / m2.sum()))

    # 4. Steering probe: one-sentence feedback -> does the described outcome become more frequent?
    probes = {"control": ("Thanks!", None),
              "crossed": ("You ended on the far side of the river.", lambda f: f.crossed),
              "bridged": ("You built a bridge over the river.", lambda f: f.bridged),
              "iron pickaxe": ("You made 1 iron pickaxe.", lambda f: f.crafted[IRON_PICKAXE] > 0),
              "gem": ("You collected 1 gem.", lambda f: f.collected[GEM_ITEM] > 0)}
    req3 = pool.requests(rng.choice(len(pool.seeds), size=max(a.n_probe, 1)))
    outcomes = {k: fn for k, (_, fn) in probes.items() if fn is not None}

    @torch.no_grad()
    def probe(model, sentence):
        p3 = request_prompts(model, req3, dev, feedback=None if sentence is None else [sentence] * len(req3))
        r3 = np.random.default_rng(2)
        saved = rng_state()
        torch.manual_seed(2)
        e3, l3, _, _, _, s3, _ = run_episodes(cfg, req3, lambda e, t: policy_actions(model, e, req3, dev, r3, prompt=p3),
                                             record=False)
        set_rng_state(saved)
        facts3 = [episode_facts(l3, i, e3, req3.goals[i], s3[i]) for i in range(len(req3))]
        return {k: float(np.mean([fn(f) for f in facts3])) for k, fn in outcomes.items()}

    def steering(model, label):
        rows = {"request only": probe(model, None)} | {name: probe(model, sent) for name, (sent, _) in probes.items()}
        print(f"  steering probe ({label}): rate of each outcome (columns) under each prompt (rows)", flush=True)
        print("    " + " " * 14 + "".join(f"{k:>14s}" for k in outcomes), flush=True)
        for name, r in rows.items():
            print(f"    {name:14s}" + "".join(f"{r[k]:14.3f}" for k in outcomes), flush=True)
        return rows

    record["base_probe"] = steering(base, "base") if a.n_probe else None
    record["base_eval"] = dict(losses=losses(base), stage2_request_only=stage2(base, False), stage2=stage2(base))
    print(f"base: {fmt(record['base_eval']['losses'])} | Stage II request only "
          f"{fmt(record['base_eval']['stage2_request_only'])} | with ideal feedback {fmt(record['base_eval']['stage2'])}",
          flush=True)

    # 2. Fit pi^fb (a copy of the base) as GFT's Stage I does.
    for lr in map(float, a.lrs.split(",")):
        torch.manual_seed(a.seed)
        aux = copy.deepcopy(base)
        opt = torch.optim.AdamW(aux.parameters(), lr=lr, weight_decay=a.weight_decay)
        rec = dict(curve=[])
        t0 = time.time()
        print(f"lr {lr:g}:", flush=True)
        tr = sets["train"]
        every = max(1, a.steps // a.evals)
        for step in range(a.steps + 1):
            if step % every == 0 or step == a.steps:
                ls = losses(aux)
                ls.update({f"stage2_{k}": v for k, v in stage2(aux).items()})
                rec["curve"].append(dict(step=step, seconds=time.time() - t0, **ls))
                print(f"  step {step:5d}  {fmt(ls)}  ({time.time() - t0:.0f}s)", flush=True)
            if step == a.steps:
                break
            mb = tr[torch.randint(len(tr), (a.minibatch,), device=dev, generator=gen)]
            eps, inverse = torch.unique(b_t[mb], return_inverse=True)
            tk = tok[eps]
            if a.word_dropout > 0:
                drop = fb_pos[eps] & (torch.rand(tk.shape, device=dev, generator=gen) < a.word_dropout)
                tk = torch.where(drop, torch.zeros_like(tk), tk)
            lg, _ = aux(*obs_tensor(grid[mb], pf[mb], inv[mb], dev), aux.encode_prompt(tk)[inverse])
            loss = F_.cross_entropy(lg, act[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
        rec["probe"] = steering(aux, f"pi^fb, lr {lr:g}, after {a.steps} steps") if a.n_probe else None
        record["runs"][f"{lr:g}"] = rec
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(record, indent=1))


def fmt(d):
    return "  ".join(f"{k} {v:.3f}" for k, v in d.items())


if __name__ == "__main__":
    main()
