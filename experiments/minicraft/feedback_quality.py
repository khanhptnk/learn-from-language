"""Measure the quality of MiniCraft teacher feedback, per profile.

  uv run python experiments/minicraft/feedback_quality.py      # ~1 min on CPU; writes results/minicraft/feedback_quality.md

Episodes come from a mix of policies on random 1-3-goal requests: the expert, noisy experts (expert action with
prob 0.7 / 0.4, else random), and random play. Measures:
  truthfulness       numbers and claims parsed back from the text vs. the episode's facts
  informativeness    AUROC of a bag-of-words logistic classifier predicting success / each requirement (held-out)
  separation         mean classifier score on successes minus failures; share of failures scored like successes
  G* calibration     AUROC of a classifier telling feedback on the ideal-feedback source's successes from feedback on
                     successes of a held-out policy (noisy0.4): expert-only source vs. success bank (expert + noisy0.7)
  request overlap    share of feedback word tokens that occur in request phrases
  diversity          words per message, distinct messages
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from gft.minicraft.core import ITEM_NAMES, N_WORLD_ACTIONS, WorldConfig
from gft.minicraft.expert import expert_action
from gft.minicraft.goals import GOALS, sample_requests
from gft.minicraft.learn import run_episodes
from gft.minicraft.teacher import PROFILES, episode_facts, feedback

OUT = Path("results/minicraft/feedback_quality.md")
WORD = re.compile(r"[a-z]+|\d+")


def episodes(cfg, n_per, rng):
    """Facts for episodes from four policies; returns list of (policy_name, Facts)."""
    out = []
    for name, p_expert in (("expert", 1.0), ("noisy0.7", 0.7), ("noisy0.4", 0.4), ("random", 0.0)):
        req = sample_requests(rng, n_per, rng.integers(0, 2 ** 31, size=n_per))

        def act(e, t):
            return np.array([expert_action(e, b, req.goals[b]) if (not e.done[b] and rng.random() < p_expert)
                             else int(rng.integers(N_WORLD_ACTIONS)) for b in range(len(req))])
        env, log, _, _, _, sat, _ = run_episodes(cfg, req, act, record=False)
        out += [(name, episode_facts(log, b, env, req.goals[b], sat[b])) for b in range(len(req))]
    return out


def truthful(msg, f):
    """Check every count claim of the form 'collected/picked up/made/crafted N item' against the facts."""
    errors = 0
    for verb, n, noun in re.findall(r"(collected|picked up|made|crafted) (\d+) ([a-z ]+?)(?:s)?\.", msg):
        item = next((i for i, nm in enumerate(ITEM_NAMES) if noun.rstrip("s") == nm or noun == nm), None)
        if item is None:
            continue
        src = f.collected if verb in ("collected", "picked up") else f.crafted
        errors += int(src[item] != int(n))
    for g, ok in zip(f.goals, f.satisfied):
        ph = GOALS[g].phrase
        errors += int((f"did not {ph}" in msg or f"Not done: {ph}" in msg) and ok)
        errors += int((f"You did {ph}." in msg or f"Done: {ph}." in msg) and not ok)
    return errors


def bow(msgs, vocab):
    X = np.zeros((len(msgs), len(vocab)), dtype=np.float32)
    for i, m in enumerate(msgs):
        for w in WORD.findall(m.lower()):
            j = vocab.get(w)
            if j is not None:
                X[i, j] += 1
    return np.log1p(X)


def auroc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    if pos.sum() in (0, len(y)):
        return float("nan")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def logistic_auroc(msgs, y, rng, steps=300):
    """Held-out AUROC (and held-out scores) of an L2-regularized bag-of-words logistic regression."""
    vocab = {w: i for i, w in enumerate(sorted({w for m in msgs for w in WORD.findall(m.lower())}))}
    X = torch.tensor(bow(msgs, vocab))
    yt = torch.tensor(y, dtype=torch.float32)
    idx = rng.permutation(len(y))
    tr, te = idx[: len(y) // 2], idx[len(y) // 2:]
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X[tr] @ w + b, yt[tr]) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        s = torch.sigmoid(X[te] @ w + b).numpy()
    return auroc(s, y[te]), s, te


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    cfg = WorldConfig()
    eps = episodes(cfg, 250, rng)
    request_words = {w for g in GOALS for w in WORD.findall(g.phrase)}
    success = np.array([all(f.satisfied) for _, f in eps])
    lines = ["# MiniCraft teacher feedback quality\n",
             f"{len(eps)} episodes (250 each from expert, noisy experts at 0.7 / 0.4, random); "
             f"success rate {success.mean():.2f}. Held-out halves for all classifier numbers.\n",
             "| Profile | truthfulness errors | AUROC success | AUROC per requirement | success − failure score "
             "| failures scored ≥ median success | G* calib. AUROC (expert-only) | G* calib. AUROC (bank) "
             "| request-word share | words/msg | distinct |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for prof in PROFILES:
        msgs = [feedback(f, prof, rng) for _, f in eps]
        errs = sum(truthful(m, f) for m, (_, f) in zip(msgs, eps))
        a_succ, s, te = logistic_auroc(msgs, success.astype(float), rng)
        gap = s[success[te]].mean() - s[~success[te]].mean()
        alias = (s[~success[te]] >= np.median(s[success[te]])).mean()
        # per-requirement: one message per (episode, goal), label = that goal satisfied, plus the goal phrase id
        rm, ry = [], []
        for m, (_, f) in zip(msgs, eps):
            for g, ok in zip(f.goals, f.satisfied):
                rm.append(f"goal{g} " + m)
                ry.append(float(ok))
        a_req, _, _ = logistic_auroc(rm, np.array(ry), rng)
        # G* calibration: can a classifier tell the ideal-feedback source's successes from a held-out policy's?
        def calib(source):
            sm = [(m, name in source) for m, (name, f) in zip(msgs, eps)
                  if all(f.satisfied) and (name in source or name == "noisy0.4")]
            return logistic_auroc([m for m, _ in sm], np.array([float(e) for _, e in sm]), rng)[0]
        a_cal, a_bank = calib({"expert"}), calib({"expert", "noisy0.7"})
        words = [w for m in msgs for w in WORD.findall(m.lower())]
        overlap = np.mean([w in request_words for w in words])
        lines.append(f"| {prof} | {errs} | {a_succ:.3f} | {a_req:.3f} | {gap:+.3f} | {alias:.2f} | {a_cal:.3f} | {a_bank:.3f} | "
                     f"{overlap:.2f} | {len(words) / len(msgs):.1f} | {len(set(msgs)) / len(msgs):.2f} |")
    lines += ["", "Examples (noisy0.4 episodes):", ""]
    ex = [f for name, f in eps if name == "noisy0.4"][:2]
    for f in ex:
        lines.append(f"- request goals {[GOALS[g].phrase for g in f.goals]}, satisfied {f.satisfied}")
        for prof in PROFILES:
            lines.append(f"  - *{prof}*: {feedback(f, prof, rng)}")
    lines.append(f"\n({time.time() - t0:.0f} s)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
