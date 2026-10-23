"""Benchmark for GFT's feedback reader pi^fb on its actual task: predict pi's trajectory from (request, feedback).

  # 1. pi's episodes on curriculum tasks (from a GFT checkpoint), with teacher feedback; tasks split train / held-out
  .venv/bin/python experiments/minicraft/reader_bench.py collect --ckpt outputs/minicraft/pixels/gft_realistic_s0.ckpt \
      --episodes 16384 --data outputs/minicraft/reader_bench/pixels.pt
  # 2. fit a reader on the training episodes and measure, on held-out tasks, how much it uses the feedback
  .venv/bin/python experiments/minicraft/reader_bench.py train --data outputs/minicraft/reader_bench/pixels.pt \
      --out outputs/minicraft/reader_bench/base.json

Held-out metrics (episodes of tasks -- world + request -- never trained on), all in nats per EPISODE:
  gain_own    log p(trajectory | request, its feedback) - log p(trajectory | request): information the reader extracts
  gain_other  same with the feedback of another episode of the same request (should be < 0 if feedback is read)
  specific    gain_own - gain_other: what the episode's OWN feedback adds over any feedback (the prompt format alone
              can shift the loss, since the reader never trains on feedback-free prompts)
  retrieval   among K episodes of the same request with different feedback, how often the true feedback gives the
              trajectory the highest likelihood (chance = 1/K) -- "does it predict the corresponding trajectory?"
Per-step losses (train / held-out, with / without feedback) are reported too.
"""

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from check_margin import load_learner  # noqa: E402
from train import base_policy, parse_arch  # noqa: E402

from gft.minicraft.learn import (obs_tensor, on_device, policy_actions, prompt_text, prompt_tokens,  # noqa: E402
                                 request_prompts, run_episodes, stack_steps)
from gft.minicraft.teacher import compose, episode_facts  # noqa: E402


def collect(a):
    dev = a.device
    cfg, pool, learner = load_learner(a.ckpt, a.obs, a.profile, dev, a.seed)
    pi, cur = learner.clean, learner.cur
    pi.eval()
    rng = np.random.default_rng(a.seed)
    out = defaultdict(list)
    t0, n = time.time(), 0
    while n < a.episodes:
        idx = cur.sample(rng, min(a.chunk, a.episodes - n))
        req = pool.requests(idx)
        p = request_prompts(pi, req, dev)
        env, log, snaps, acts, valid, sat, mask = run_episodes(
            cfg, req, lambda e, t: policy_actions(pi, e, req, dev, rng, prompt=p))
        facts = [episode_facts(log, b, env, req.goals[b], sat[b]) for b in range(len(req))]
        out["feedback"] += [" ".join(x[2] for x in compose(f, a.profile, rng)) or "Okay." for f in facts]
        out["request"] += [req.text(b) for b in range(len(req))]
        grid, pf, inv, act, b, _ = stack_steps(snaps, acts, valid)
        for k, v in (("grid", grid), ("pf", pf), ("inv", inv), ("act", act), ("b", b + n)):
            out[k].append(torch.as_tensor(v))
        out["task"].append(torch.as_tensor(idx))
        out["success"].append(torch.as_tensor((sat | ~mask).all(1)))
        n += len(req)
    data = {k: torch.cat(v) if isinstance(v[0], torch.Tensor) else v for k, v in out.items()}
    data["grid"] = data["grid"].to(torch.uint8)
    data["meta"] = dict(ckpt=a.ckpt, obs=a.obs, profile=a.profile, feedbacks=learner.teacher_calls)
    Path(a.data).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, a.data)
    print(f"{n} episodes, {len(data['b'])} steps, success {data['success'].float().mean():.3f} "
          f"({time.time() - t0:.0f}s) -> {a.data}", flush=True)


def train(a):
    dev = a.device
    torch.manual_seed(a.seed)
    D = torch.load(a.data, weights_only=False)
    meta = D["meta"]
    if a.init == "base":                              # architecture comparison: each starts from its own text-free base
        from gft.minicraft.core import WorldConfig
        model = base_policy(WorldConfig(), dev, seed=0, kind="random", pixels=meta["obs"] == "pixels",
                            arch=parse_arch(a.arch))
    else:
        _, _, learner = load_learner(meta["ckpt"], meta["obs"], meta["profile"], dev, a.seed)
        model = copy.deepcopy(learner.aux if a.init == "aux" else learner.clean)
    rng = np.random.default_rng(a.seed)
    tasks = D["task"].numpy()
    held_task = rng.random(int(tasks.max()) + 1) < a.heldout
    held_ep = held_task[tasks]
    n_ep = len(tasks)
    req, fb = D["request"], D["feedback"]
    tok = prompt_tokens([prompt_text(req[i], fb[i]) for i in range(n_ep)], dev)
    tok_req = prompt_tokens([prompt_text(req[i]) for i in range(n_ep)], dev)
    n_req = (tok_req > 0).sum(1)
    fb_pos = torch.arange(tok.shape[1], device=dev)[None, :] >= n_req[:, None]
    grid, pf, inv, act, b = on_device(dev, D["grid"], D["pf"], D["inv"], D["act"], D["b"])
    held_step = torch.as_tensor(held_ep, device=dev)[b]
    tr = torch.nonzero(~held_step)[:, 0]
    if a.train_episodes:                              # data-size control: first N training episodes
        keep = np.nonzero(~held_ep)[0][:a.train_episodes]
        tr = tr[torch.isin(b[tr], torch.as_tensor(keep, device=dev))]
    starts = torch.searchsorted(b, torch.arange(n_ep + 1, device=dev))   # steps of episode e: starts[e]:starts[e+1]
    print(f"{n_ep} episodes ({int(held_ep.sum())} held-out, tasks never trained on); {len(tr)} training steps",
          flush=True)

    # held-out evaluation sets
    held = np.nonzero(held_ep)[0]
    gen = np.random.default_rng(a.seed + 1)
    ev = gen.choice(held, size=min(a.n_eval, len(held)), replace=False)
    by_req = defaultdict(list)
    for e in held:
        by_req[req[e]].append(e)
    other = {}                                          # another held-out episode of the same request, other feedback
    cands = {}                                          # retrieval candidates: true episode first
    for e in ev:
        pool_e = [c for c in by_req[req[e]] if fb[c] != fb[e]]
        if pool_e:
            other[e] = pool_e[gen.integers(len(pool_e))]
            texts, picks = {fb[e]}, []
            for c in gen.permutation(pool_e):
                if fb[c] not in texts and len(picks) < a.k - 1:
                    texts.add(fb[c])
                    picks.append(c)
            if len(picks) == a.k - 1:
                cands[e] = [e] + picks
    tr_ev = gen.choice(np.nonzero(~held_ep)[0], size=min(a.n_eval, int((~held_ep).sum())), replace=False)

    @torch.no_grad()
    def episode_logp(pairs):
        """Summed action log-prob of episode e's trajectory under prompt text t, for each (e, t)."""
        model.eval()
        out = torch.zeros(len(pairs), device=dev)
        P = model.encode_prompt(prompt_tokens([t for _, t in pairs], dev))
        steps = [torch.arange(starts[e], starts[e + 1], device=dev) for e, _ in pairs]
        pid = torch.cat([torch.full((len(s),), i, device=dev) for i, s in enumerate(steps)])
        steps = torch.cat(steps)
        for j in range(0, len(steps), a.fwd_chunk):
            s, p = steps[j:j + a.fwd_chunk], pid[j:j + a.fwd_chunk]
            lg = model(*obs_tensor(grid[s], pf[s], inv[s], dev), P[p])[0].float()
            out.index_add_(0, p, -F_.cross_entropy(lg, act[s], reduction="none"))
        model.train()
        return out

    def evaluate():
        r = {}
        for name, eps in (("train", tr_ev), ("held", ev)):
            own = episode_logp([(e, prompt_text(req[e], fb[e])) for e in eps])
            none = episode_logp([(e, prompt_text(req[e])) for e in eps])
            n_steps = float(sum(int(starts[e + 1] - starts[e]) for e in eps))
            r[f"{name}_loss_step"] = float(-own.sum()) / n_steps
            r[f"{name}_loss_step_nofb"] = float(-none.sum()) / n_steps
            r[f"{name}_gain_own"] = float((own - none).mean())
        eo = list(other)
        if eo:
            oth = episode_logp([(e, prompt_text(req[e], fb[other[e]])) for e in eo])
            own = episode_logp([(e, prompt_text(req[e], fb[e])) for e in eo])
            none = episode_logp([(e, prompt_text(req[e])) for e in eo])
            r["held_gain_other"] = float((oth - none).mean())
            r["held_specific"] = float((own - oth).mean())   # own vs another episode's feedback: the usable signal
        ec = list(cands)
        if ec:
            sc = episode_logp([(e, prompt_text(req[e], fb[c])) for e in ec for c in cands[e]]).view(len(ec), a.k)
            r["held_retrieval"] = float((sc.argmax(1) == 0).float().mean())
            r["retrieval_n"] = len(ec)
        return r

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_updates = max(1, int(a.epochs * len(tr) // a.minibatch))
    every = max(1, n_updates // a.evals)
    record = dict(args=vars(a), meta=meta, curve=[], chance=1 / a.k)
    t0 = time.time()
    for step in range(n_updates + 1):
        if step % every == 0 or step == n_updates:
            r = evaluate()
            record["curve"].append(dict(update=step, epoch=step * a.minibatch / len(tr), seconds=time.time() - t0, **r))
            print(f"update {step:6d} (epoch {step * a.minibatch / len(tr):5.2f}, {time.time() - t0:5.0f}s) "
                  + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(record, indent=1))
        if step == n_updates:
            break
        mb = tr[torch.randint(len(tr), (a.minibatch,), device=dev)]
        eps, inverse = torch.unique(b[mb], return_inverse=True)
        tk = tok[eps]
        if a.word_dropout > 0:
            drop = fb_pos[eps] & (torch.rand(tk.shape, device=dev) < a.word_dropout)
            tk = torch.where(drop, torch.zeros_like(tk), tk)
        loss = F_.cross_entropy(model(*obs_tensor(grid[mb], pf[mb], inv[mb], dev), model.encode_prompt(tk)[inverse])[0],
                                act[mb])
        opt.zero_grad()
        loss.backward()
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--obs", choices=["symbolic", "pixels"], default="pixels")
    c.add_argument("--profile", default="realistic")
    c.add_argument("--episodes", type=int, default=16384)
    c.add_argument("--chunk", type=int, default=512)
    c.add_argument("--data", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--init", choices=["pi", "aux", "base"], default="pi",
                   help="start from the run's pi or its pi^fb, or from the text-free base of --arch")
    t.add_argument("--arch", default="", help="with --init base: architecture, e.g. kind=gvin")
    t.add_argument("--heldout", type=float, default=0.15, help="share of tasks (world + request) held out")
    t.add_argument("--train-episodes", type=int, default=0, help="use only the first N training episodes (0 = all)")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--word-dropout", type=float, default=0.1)
    t.add_argument("--epochs", type=float, default=8)
    t.add_argument("--minibatch", type=int, default=1024)
    t.add_argument("--evals", type=int, default=8)
    t.add_argument("--n-eval", type=int, default=512, help="held-out (and training) episodes evaluated")
    t.add_argument("--k", type=int, default=4, help="retrieval candidates per episode")
    t.add_argument("--fwd-chunk", type=int, default=1024)
    t.add_argument("--out", default="")
    for p in (c, t):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else train(a)


if __name__ == "__main__":
    main()
