"""Can PPO with group advantage (GRPO) learn planks on single tasks? GRPO seeds 0-2 (100k feedbacks) with GFT v3e
(seed 0, 100k) for reference. Left: goals mastered (labels carry the last validation success, new worlds); right:
planks success on the evaluation's train split. Open marker = still running.
  .venv/bin/python experiments/minicraft/plot_grpo_seeds.py <dir with ppo_grpo_s1/s2.json> results/minicraft/grpo_seeds.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "outputs/pod/gft/single"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/minicraft/grpo_seeds.png"
RUNS = [("GRPO, seed 0", "outputs/pod/gft/single/ppo_grpo.json", "#eda100", "-"),
        ("GRPO, seed 1", f"{D}/ppo_grpo_s1.json", "#eda100", "--"),
        ("GRPO, seed 2", f"{D}/ppo_grpo_s2.json", "#eda100", ":"),
        ("GFT v3e, seed 0 (reference)", "outputs/pod/gft/single/gft_v3e.json", "#2a78d6", ":"),
        ("GFT R3 (default since 2026-11-04), seed 0", "outputs/pod/gft/r3_live/r3_100k.json", "#2a78d6", "-")]
BUDGET = 100352
sc = lambda x: x["success"] if isinstance(x, dict) else x


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, (label, path, col, ls) in enumerate(RUNS):
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        ev, end = d["evals"], d["evals"][-1]["spent"]
        xs, ys = [0], [0]
        for spent, _ in d["curriculum"]:
            xs += [spent, spent]
            ys += [ys[-1], ys[-1] + 1]
        xs.append(end)
        ys.append(ys[-1])
        off = 0.06 * (k - 2)
        a1.plot(xs, [y + off for y in ys], ls, color=col, lw=1.8, label=f"{label}: val {sc(ev[-1]['val_worlds']):.2f}")
        a1.plot(end, ys[-1] + off, "o", color=col, ms=5, mfc="white" if end < BUDGET - 500 else col)
        a2.plot([e["spent"] for e in ev], [e["train"]["per_goal"]["make planks"][0] for e in ev], ls, color=col,
                lw=1.6, marker="o", ms=2.5, label=label)
    a1.set(xlabel="feedbacks (teacher messages / PPO rewards)", ylabel="goals mastered",
           title="Goals mastered (0.7 on own episodes)", ylim=(0, 7.5), xlim=(0, BUDGET + 1000))
    a1.legend(fontsize=7, loc="upper left", frameon=False)
    a2.axhline(0.7, color="#999", lw=0.8, ls=":")
    a2.set(xlabel="feedbacks", ylabel="planks success (train split)", title="make planks", ylim=(0, 1.0),
           xlim=(0, BUDGET + 1000))
    a2.legend(fontsize=7, loc="upper left", frameon=False)
    for a in (a1, a2):
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(OUT)


if __name__ == "__main__":
    main()
