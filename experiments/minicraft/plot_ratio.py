"""Pass-ratio runs (pod/jobs_ratio.sh) against the base (batch5/alltasks): left, goals mastered vs feedbacks (labels
carry the last validation success, new worlds); right, validation success vs feedbacks. Open marker = still running.
  .venv/bin/python experiments/minicraft/plot_ratio.py <dir with R1-R4.json> results/minicraft/ratio.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "outputs/pod/gft/ratio"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/minicraft/ratio.png"
RUNS = [("base (4,1,2,3)", "outputs/pod/gft/batch5_live/alltasks.json", "#555555"),
        ("R1 (4,1,0,3) no reader self-imitation", f"{D}/R1.json", "#2a78d6"),
        ("R2 (4,1,4,3) more self-imitation", f"{D}/R2.json", "#1baf7a"),
        ("R3 (4,1,2,1) lighter scorer", f"{D}/R3.json", "#eda100"),
        ("R4 (4,4,4,4) one number", f"{D}/R4.json", "#e34948"),
        ("R5 (2,1,2,1) slower pi", f"{D}/R5.json", "#6250d6"),
        ("R6 (4,2,2,1) more reader passes", f"{D}/R6.json", "#008300"),
        ("S2 (8,2,4,2) = R3 x2", f"{D}/S2.json", "#000000")]
BUDGET = 40192
sc = lambda x: x["success"] if isinstance(x, dict) else x


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, (label, path, col) in enumerate(RUNS):
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
        a1.plot(xs, [y + off for y in ys], color=col, lw=1.8, label=f"{label}: val {sc(ev[-1]['val_worlds']):.2f}")
        a1.plot(end, ys[-1] + off, "o", color=col, ms=5, mfc="white" if end < BUDGET - 200 else col)
        a2.plot([e["spent"] for e in ev], [sc(e["val_worlds"]) for e in ev], color=col, lw=1.6, marker="o", ms=2.5)
    a1.set(xlabel="feedbacks", ylabel="goals mastered", title="Goals mastered (0.7 on own episodes)", ylim=(0, 6.5),
           xlim=(0, BUDGET + 500))
    a1.legend(fontsize=7, loc="upper left", frameon=False)
    a2.set(xlabel="feedbacks", ylabel="validation success (new worlds)", title="Validation success", ylim=(0, 0.5),
           xlim=(0, BUDGET + 500))
    for a in (a1, a2):
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(OUT)


if __name__ == "__main__":
    main()
