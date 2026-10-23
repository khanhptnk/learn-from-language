"""Does pi^fb respond to feedback at all, and in which direction? Probes on a GFT checkpoint.

  .venv/bin/python experiments/minicraft/check_steering.py --ckpt outputs/minicraft/pixels/gft_realistic_s0.ckpt \
      --out outputs/minicraft/margin/steering_pixels.json

A. Sensitivity on pi's own states (curriculum tasks): KL(pi^fb | prompt || pi^fb | request) and how often the most
   likely action changes, for the teacher's real feedback (failed / successful episodes), another episode's feedback,
   the ideal feedback G* (tasks whose G* has a claim pi never earned vs ones it has), and KL(pi^fb | request || pi).
   ~0 everywhere = pi^fb ignores its feedback input.
B. Counterfactual steering on the frontier goal: roll out pi^fb on single-goal tasks under different one-message
   prompts (failure messages that name different missing items, G*, a content-free control) and count how far along
   the chain each rollout gets. A reader that uses feedback should behave differently under each; ideally "you still
   needed iron" (= it had the ore) should not look like "you still needed ore".
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from check_margin import load_learner  # noqa: E402

from gft.minicraft.core import IRON, ORE_ITEM, STONE, STONE_PICKAXE, WOOD_PICKAXE  # noqa: E402
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import (claim_type, obs_tensor, on_device, policy_actions, prompt_text,  # noqa: E402
                                 prompt_tokens, request_prompts, rng_state, run_episodes, set_rng_state, stack_steps)
from gft.minicraft.teacher import compose, episode_facts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--obs", choices=["symbolic", "pixels"], default="pixels")
    ap.add_argument("--profile", default="realistic")
    ap.add_argument("--n", type=int, default=512, help="pi episodes for part A")
    ap.add_argument("--goal", type=int, default=6, help="frontier goal for part B (6 = get some metal)")
    ap.add_argument("--n-probe", type=int, default=512, help="rollouts per prompt in part B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = a.device
    cfg, pool, learner = load_learner(a.ckpt, a.obs, a.profile, dev, a.seed)
    pi, aux, cur = learner.clean, learner.aux, learner.cur
    pi.eval(), aux.eval()
    rng = np.random.default_rng(a.seed)
    record = dict(args=vars(a), feedbacks=learner.teacher_calls)
    print(f"checkpoint at {learner.teacher_calls} feedbacks", flush=True)

    # A. Sensitivity on pi's states.
    idx = cur.sample(rng, a.n)
    req = pool.requests(idx)
    p = request_prompts(pi, req, dev)
    env, log, snaps, acts, valid, sat, mask = run_episodes(cfg, req, lambda e, t: policy_actions(pi, e, req, dev, rng,
                                                                                                 prompt=p))
    facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
    parts = [compose(f, a.profile, rng) for f in facts]
    real = [" ".join(x[2] for x in pt) or "Okay." for pt in parts]
    seen = set(sum(learner.seen_claims, Counter()))       # claim types in the run's recent Stage I feedback
    star_parts = [compose(pool.ideal_facts(i, rng), a.profile, rng) for i in idx]
    star = [" ".join(x[2] for x in pt) or "Okay." for pt in star_parts]
    star_unseen = np.array([any(claim_type(x[1]) not in seen for x in pt) for pt in star_parts])
    other = [real[i] for i in rng.permutation(len(real))]
    ok = (sat | ~mask).all(1)
    grid, pf, inv, _, b, _ = stack_steps(snaps, acts, valid)
    grid, pf, inv = on_device(dev, grid, pf, inv)
    b_t = torch.as_tensor(b, device=dev)

    @torch.no_grad()
    def logp(model, texts):
        pr = model.encode_prompt(prompt_tokens([prompt_text(req.text(i), None if texts is None else texts[i])
                                                for i in range(len(req))], dev))
        out = []
        for j in range(0, len(b), 2048):
            s = slice(j, j + 2048)
            out.append(F_.log_softmax(model(*obs_tensor(grid[s], pf[s], inv[s], dev), pr[b_t[s]])[0].float(), -1))
        return torch.cat(out)

    l_req, l_pi = logp(aux, None), logp(pi, None)

    def compare(lp, lq, ep_sel):
        sel = torch.as_tensor(ep_sel[b], device=dev)
        if not bool(sel.any()):
            return dict(kl=float("nan"), flip=float("nan"), steps=0)
        kl = (lp.exp() * (lp - lq)).sum(-1)[sel]
        flip = (lp.argmax(-1) != lq.argmax(-1))[sel].float()
        return dict(kl=float(kl.mean()), kl_p90=float(kl.quantile(0.9)), flip=float(flip.mean()), steps=int(sel.sum()))

    every = np.ones(len(req), bool)
    rows = {"real feedback, failed episodes": compare(logp(aux, real), l_req, ~ok),
            "real feedback, successful episodes": compare(logp(aux, real), l_req, ok),
            "another episode's feedback": compare(logp(aux, other), l_req, every),
            "G*, has claim never earned": compare(logp(aux, star), l_req, star_unseen),
            "G*, all claims earned before": compare(logp(aux, star), l_req, ~star_unseen),
            "pi^fb|request vs pi": compare(l_req, l_pi, every)}
    record["sensitivity"] = rows
    print(f"A. pi^fb sensitivity on {len(b)} states of {len(req)} pi episodes (success {ok.mean():.3f}); "
          f"KL(pi^fb|prompt || pi^fb|request) per step:", flush=True)
    for k, v in rows.items():
        print(f"  {k:38s} KL {v['kl']:.4f} (p90 {v.get('kl_p90', float('nan')):.4f})  argmax changes "
              f"{v['flip']:.3f}  [{v['steps']} steps]", flush=True)

    # B. Counterfactual prompts on single-goal frontier tasks.
    cand = np.nonzero((pool.goals[:, 0] == a.goal) & (pool.goals[:, 1] < 0))[0]
    idx3 = rng.choice(cand, size=a.n_probe)
    req3 = pool.requests(idx3)
    phrase = GOALS[a.goal].phrase
    star3 = [" ".join(x[2] for x in compose(pool.ideal_facts(i, rng), a.profile, rng)) or "Okay." for i in idx3]
    prompts = {"request only": None, "G* (ideal)": star3,
               f"you managed to {phrase}": [f"You managed to {phrase}."] * len(req3),
               "failed; still needed ore": [f"You did not {phrase}. You still needed ore."] * len(req3),
               "failed; still needed iron": [f"You did not {phrase}. You still needed iron."] * len(req3),
               "failed; needed stone pickaxe": [f"You did not {phrase}. You still needed a stone pickaxe."] * len(req3),
               "control: Thanks!": ["Thanks!"] * len(req3)}
    stages = {"wood pickaxe": lambda f: f.crafted[WOOD_PICKAXE] > 0, "stone": lambda f: f.collected[STONE] > 0,
              "stone pickaxe": lambda f: f.crafted[STONE_PICKAXE] > 0, "ore": lambda f: f.collected[ORE_ITEM] > 0,
              "iron": lambda f: f.crafted[IRON] > 0}

    @torch.no_grad()
    def probe(model, texts):
        pr = request_prompts(model, req3, dev, feedback=texts)
        r = np.random.default_rng(3)
        saved = rng_state()
        torch.manual_seed(3)                          # same sampling noise under every prompt
        e3, l3, _, _, _, s3, _ = run_episodes(cfg, req3, lambda e, t: policy_actions(model, e, req3, dev, r, prompt=pr))
        set_rng_state(saved)
        f3 = [episode_facts(l3, i, e3, req3.goals[i], s3[i]) for i in range(len(req3))]
        return {k: float(np.mean([fn(f) for f in f3])) for k, fn in stages.items()} | {"steps": float(np.mean(
            [f.steps for f in f3]))}

    res = {"pi (request)": probe(pi, None)} | {f"pi^fb | {k}": probe(aux, v) for k, v in prompts.items()}
    record["steering"] = res
    print(f"B. '{phrase}' tasks, {len(req3)} rollouts per prompt: share reaching each step of the chain", flush=True)
    print("  " + " " * 40 + "".join(f"{k:>14s}" for k in list(stages) + ["steps"]), flush=True)
    for k, v in res.items():
        print(f"  {k:40s}" + "".join(f"{v[s]:14.3f}" for s in list(stages)) + f"{v['steps']:14.1f}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
