"""How strong is the language-pretrained base's grounding, relative to its own request channel?

  told vs asked   for goals G: seen requests R without G (new worlds). asked = request mode on R + G; told = message
                  mode on R with R's own failure message plus "You managed to G."; base = message mode on the own
                  message. Measures how often G is achieved under each (same worlds and noise). told/asked ~ 1 means
                  the sentence is grounded as well as the request phrase.
  reader bound    fresh corpus episodes (new worlds): per-step CE of the message mode under the episode's own message,
                  under another episode's message (same request), and of the request mode under the episode's true
                  behaviour goals (hindsight label, available where the demonstrator finished). The gap
                  swapped - own is the information the reader extracts; own - hindsight is what it misses.

  .venv/bin/python experiments/minicraft/check_lang_calibration.py --heldout-goals 8,9 --lang-steps 20000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).parent))
from check_lang_pretrain import message, roll  # noqa: E402
from train import base_policy, heldout_splits, parse_goals  # noqa: E402

from gft.minicraft.core import WorldConfig  # noqa: E402
from gft.minicraft.goals import GOALS, MAX_GOALS, Requests  # noqa: E402
from gft.minicraft.learn import FWD_CHUNK, lang_corpus, obs_tensor, prompt_text, prompt_tokens  # noqa: E402
from gft.minicraft.teacher import achieved  # noqa: E402


def with_goal(req, g):
    goals = np.full((len(req), MAX_GOALS), -1)
    for b in range(len(req)):
        row = sorted({int(x) for x in req.goals[b] if x >= 0} | {g}, reverse=True)
        goals[b, :len(row)] = row
    return Requests(goals, req.seeds)


@torch.no_grad()
def ce_under(model, corpus, texts, idx, dev):
    tok = prompt_tokens(texts, dev)
    b = torch.as_tensor(corpus["b"][idx], device=dev)
    tot = 0.0
    for j in range(0, len(idx), FWD_CHUNK):
        i = idx[j:j + FWD_CHUNK]
        x, iv = obs_tensor(corpus["grid"][i], corpus["pf"][i], corpus["inv"][i], dev)
        eps, inverse = torch.unique(b[j:j + FWD_CHUNK], return_inverse=True)
        logits, _ = model(x, iv, model.encode_prompt(tok[eps])[inverse])
        tot += float(F_.cross_entropy(logits, torch.as_tensor(corpus["a"][i], device=dev), reduction="sum"))
    return tot / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-goals", default="build a well,make a sling")
    ap.add_argument("--lang-episodes", type=int, default=0)
    ap.add_argument("--lang-steps", type=int, default=0)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--goals", default="make rope,make bricks,make planks,make a pickaxe,build a raft,build an oven,quarry stone,build a wall,build a fence")
    ap.add_argument("--corpus", type=int, default=2048, help="fresh episodes for the reader bound")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg, dev = WorldConfig(), a.device
    held = parse_goals(a.heldout_goals)
    model = base_policy(cfg, dev, seed=0, kind="lang", pixels=True, heldout_goals=held,
                        lang_episodes=a.lang_episodes, lang_steps=a.lang_steps).eval()
    rng = np.random.default_rng(0)
    seen = heldout_splits(a.n, tune=True, heldout_goals=held)["val_seen"]
    res = {}
    print(f"told vs asked ({a.n} seen requests, new worlds; achieved = G's item obtained during the episode)")
    print(f"  {'goal G':26s}{'n':>5s}{'base':>7s}{'told':>7s}{'asked':>7s}{'told/asked':>11s}")
    for g in sorted(parse_goals(a.goals)):
        sel = np.array([g not in seen.goals[b] and (seen.goals[b] >= 0).sum() < MAX_GOALS for b in range(len(seen))])
        sub = Requests(seen.goals[sel], seen.seeds[sel])
        f_own, _, _ = roll(model, cfg, sub, [prompt_text(sub.text(b)) for b in range(len(sub))], dev, 10 + g)
        own = [message(f, rng) for f in f_own]
        told_msgs = [m + f" You managed to {GOALS[g].phrase}." for m in own]
        plus = with_goal(sub, g)
        rows = {}
        for name, req, texts in (("base", sub, [prompt_text(sub.text(b), m) for b, m in enumerate(own)]),
                                 ("told", sub, [prompt_text(sub.text(b), m) for b, m in enumerate(told_msgs)]),
                                 ("asked", plus, [prompt_text(plus.text(b)) for b in range(len(plus))])):
            fs, _, _ = roll(model, cfg, req, texts, dev, 100 + g)
            rows[name] = float(np.mean([achieved(f, g) for f in fs]))
        rows["ratio"] = (rows["told"] - rows["base"]) / max(rows["asked"] - rows["base"], 1e-9)
        res[GOALS[g].phrase] = dict(n=int(sel.sum()), **rows)
        print(f"  {GOALS[g].phrase:26s}{sel.sum():5d}{rows['base']:7.3f}{rows['told']:7.3f}{rows['asked']:7.3f}"
              f"{rows['ratio']:11.2f}")
    print("  told/asked = (told - base) / (asked - base)")

    corpus, counts = lang_corpus(cfg, a.corpus, held, seed=12345)
    b_all = corpus["b"]
    has_h = np.array([h is not None for h in corpus["hindsight"]])
    idx_all = np.arange(len(b_all))[::max(1, len(b_all) // 100_000)]
    idx_h = idx_all[has_h[b_all[idx_all]]]
    swap = np.arange(len(corpus["request"]))
    by_req = {}
    for e, r in enumerate(corpus["request"]):
        by_req.setdefault(r, []).append(e)
    for e, r in enumerate(corpus["request"]):
        same = [o for o in by_req[r] if o != e]
        swap[e] = rng.choice(same) if same else e
    own_t = [prompt_text(r, m) for r, m in zip(corpus["request"], corpus["message"])]
    swp_t = [prompt_text(r, corpus["message"][swap[e]]) for e, r in enumerate(corpus["request"])]
    hind_t = [prompt_text(h or "") for h in corpus["hindsight"]]
    bound = dict(episodes=int(counts["episodes"]), finished=int(counts["hindsight"]),
                 all_own=ce_under(model, corpus, own_t, idx_all, dev), all_swapped=ce_under(model, corpus, swp_t, idx_all, dev),
                 fin_own=ce_under(model, corpus, own_t, idx_h, dev), fin_swapped=ce_under(model, corpus, swp_t, idx_h, dev),
                 fin_hindsight=ce_under(model, corpus, hind_t, idx_h, dev))
    res["reader_bound"] = bound
    print(f"\nreader bound ({bound['episodes']} fresh episodes, {bound['finished']} with a hindsight label), CE nats/step")
    print(f"  all episodes:      own message {bound['all_own']:.3f}   swapped message {bound['all_swapped']:.3f}   "
          f"gain {bound['all_swapped'] - bound['all_own']:+.3f}")
    print(f"  finished episodes: own message {bound['fin_own']:.3f}   swapped message {bound['fin_swapped']:.3f}   "
          f"true goals as request {bound['fin_hindsight']:.3f}")
    print(f"    the reader extracts {bound['fin_swapped'] - bound['fin_own']:.3f} of the "
          f"{bound['fin_swapped'] - bound['fin_hindsight']:.3f} nats/step that knowing the true goals gives "
          f"({(bound['fin_swapped'] - bound['fin_own']) / max(bound['fin_swapped'] - bound['fin_hindsight'], 1e-9):.0%})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
