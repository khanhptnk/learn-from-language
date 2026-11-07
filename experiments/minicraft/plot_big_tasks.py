"""Per-task success over training, larger policy (2.9M) vs the default (1.05M), GFT (R3) and GRPO, seed 0
(2026-11-06): one panel per task, success on the evaluation's train split (temperature 1) at each evaluation; a
vertical tick marks when the run mastered the task (0.7 on its own episodes). Single seeds.
  .venv/bin/python experiments/minicraft/plot_big_tasks.py results/minicraft/big_tasks.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gft.minicraft.goals import GOAL_ID

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/minicraft/big_tasks.png"
BUDGET = 100352
RUNS = [("GFT (R3), 1.05M", "outputs/pod/gft/r3_live/r3_100k.json", "#2a78d6", ":"),
        ("GFT (R3), 2.9M", "outputs/pod/gft/big/gft_big.json", "#2a78d6", "-"),
        ("GRPO, 1.05M", "outputs/pod/gft/single/ppo_grpo.json", "#eda100", ":"),
        ("GRPO, 2.9M", "outputs/pod/gft/big/grpo_big.json", "#eda100", "-")]
TASKS = ["gather wood", "gather fiber", "dig clay", "make rope", "make bricks", "make planks", "make a pickaxe",
         "build a raft", "build an oven"]


def main():
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    for (label, path, col, ls) in RUNS:
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        ev = [e for e in d["evals"] if e["spent"] <= BUDGET + 500]
        done = {g: s for s, g in d["curriculum"]}
        for ax, t in zip(axes.flat, TASKS):
            ys = [e["train"]["per_goal"].get(t, [None])[0] for e in ev]
            xs = [e["spent"] for e, y in zip(ev, ys) if y is not None]
            ax.plot(xs, [y for y in ys if y is not None], ls, color=col, lw=1.6, label=label)
            if GOAL_ID[t] in done:
                ax.axvline(done[GOAL_ID[t]], color=col, ls=ls, lw=0.9, alpha=0.6)
    for ax, t in zip(axes.flat, TASKS):
        ax.set_title(t, fontsize=10)
        ax.axhline(0.7, color="#999", lw=0.7, ls=":")
        ax.set(ylim=(-0.03, 1.0), xlim=(0, BUDGET + 1000))
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("feedbacks")
    for ax in axes[:, 0]:
        ax.set_ylabel("success (train split)")
    axes.flat[0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Per-task success; vertical lines: task mastered (0.7 on own episodes). Single seeds.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    main()
