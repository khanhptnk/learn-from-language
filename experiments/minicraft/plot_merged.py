"""Merged GFT (R3 + symmetric scorer pairs, 2026-11-06), GFT with the built success message (2026-11-07), R3 and GRPO at 100k feedbacks, seed 0. Left: goals
mastered; right: validation success. Single seeds.
  .venv/bin/python experiments/minicraft/plot_merged.py results/minicraft/merged.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/minicraft/merged.png"
BUDGET = 100352
RUNS = [("GFT (R3)", "outputs/pod/gft/r3_live/r3_100k.json", "#2a78d6", ":"),
        ("GFT merged (R3 + symmetric scorer pairs)", "outputs/pod/gft/main/gft_merged.json", "#2a78d6", "-"),
        ("GFT built ideal + 1/2 improved, 1/2 ideal target", "outputs/pod/gft/main/gft_built.json", "#1baf7a", "-"),
        ("GFT built, verdict-only teacher (language ablation)", "outputs/pod/gft/main/gft_verdict.json", "#1baf7a", ":"),
        ("GFT built + reader writes messages", "outputs/pod/gft/main/gft_writer.json", "#6250d6", "-"),
        ("GRPO", "outputs/pod/gft/single/ppo_grpo.json", "#eda100", "-")]
sc = lambda x: x["success"] if isinstance(x, dict) else x


def main():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.4))
    for k, (label, path, col, ls) in enumerate(RUNS):
        if not Path(path).exists():
            continue
        d = json.load(open(path))
        ev = [e for e in d["evals"] if e["spent"] <= BUDGET + 500]
        end = min(BUDGET, ev[-1]["spent"])
        xs, ys = [0], [0]
        for spent, _ in d["curriculum"]:
            if spent > BUDGET:
                break
            xs += [spent, spent]
            ys += [ys[-1], ys[-1] + 1]
        xs.append(end)
        ys.append(ys[-1])
        off = 0.06 * (k - 1.5)
        a1.plot(xs, [y + off for y in ys], ls, color=col, lw=1.8, label=f"{label}: val {sc(ev[-1]['val_worlds']):.2f}")
        a1.plot(end, ys[-1] + off, "o", color=col, ms=5, mfc="white" if end < BUDGET - 500 else col)
        a2.plot([e["spent"] for e in ev], [sc(e["val_worlds"]) for e in ev], ls, color=col, lw=1.6, marker="o", ms=2.5,
                label=label)
        R = d.get("rounds", [])
        if R and "stage1_success" in R[0]:          # GFT: Stage I = one teacher message per episode
            x = [(i + 1) * 256 for i in range(len(R))]
            y = [r["stage1_success"] for r in R]
        elif R and "train_success" in R[0]:          # PPO / GRPO: one reward per episode
            x, y = [r["episodes"] for r in R], [r["train_success"] for r in R]
        else:
            x, y = [], []
        if y:
            w = max(1, round(5000 / (x[1] - x[0] if len(x) > 1 else 256)))   # rolling mean over ~5k feedbacks
            ys = [sum(y[max(0, i - w + 1):i + 1]) / len(y[max(0, i - w + 1):i + 1]) for i in range(len(y))]
            keep = [i for i in range(len(x)) if x[i] <= BUDGET + 500]
            a3.plot([x[i] for i in keep], [ys[i] for i in keep], ls, color=col, lw=1.6, label=label)
    a1.set(xlabel="feedbacks (teacher messages)", ylabel="goals mastered", title="Goals mastered (0.7 on own episodes)",
           ylim=(0, 7.5), xlim=(0, BUDGET + 500))
    a1.legend(fontsize=7, loc="upper left", frameon=False)
    a2.set(xlabel="feedbacks", ylabel="validation success (new worlds)", title="Validation success", ylim=(0, 0.6),
           xlim=(0, BUDGET + 500))
    a2.legend(fontsize=7, loc="upper left", frameon=False)
    a3.set(xlabel="feedbacks", ylabel="success on the curriculum's current tasks",
           title="Training success (own episodes, current tasks; ~5k rolling mean)", ylim=(0, 1.0),
           xlim=(0, BUDGET + 500))
    a3.legend(fontsize=7, loc="lower left", frameon=False)
    for a in (a1, a2, a3):
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(OUT)


if __name__ == "__main__":
    main()
