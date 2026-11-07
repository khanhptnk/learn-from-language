"""Is the reader only a second, lagging sampler? Three arms x seeds, single tasks, v3e settings, 40k feedbacks:
full GFT (reader samples Stage II), h2 (a copy of pi refreshed every 4 rounds samples it), policy (pi samples its own
retries). Left: goals mastered vs feedbacks (labels carry the last validation success, new worlds); right: planks
success on the evaluation's train split. Seed 0 comes from the cause runs (policy: single/gft_lite_policy, cut at
40k). Open marker = still running.
  .venv/bin/python experiments/minicraft/plot_seeds.py <seeds dir> results/minicraft/seeds_reader.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/pod/gft/seeds"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/minicraft/seeds_reader.png"
ARMS = [("full GFT (reader)", "#2a78d6", ["outputs/pod/gft/cause/control.json"] +
         [f"{SEEDS_DIR}/control_s{s}.json" for s in (1, 2)]),
        ("lagged copy of pi (h2)", "#1baf7a", ["outputs/pod/gft/cause/h2_lag.json"] +
         [f"{SEEDS_DIR}/h2_lag_s{s}.json" for s in (1, 2)]),
        ("pi samples own retries", "#e34948", ["outputs/pod/gft/single/gft_lite_policy.json"] +
         [f"{SEEDS_DIR}/policy_s{s}.json" for s in (1, 2)]),
        ("Stage II from all tasks", "#6250d6", ["outputs/pod/gft/batch5_live/alltasks.json"])]
LS = ["-", "--", ":"]
BUDGET = 40192
sc = lambda x: x["success"] if isinstance(x, dict) else x


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, (arm, col, paths) in enumerate(ARMS):
        for s, path in enumerate(paths):
            if not Path(path).exists():
                continue
            d = json.load(open(path))
            ev = [e for e in d["evals"] if e["spent"] <= BUDGET]
            end, last = ev[-1]["spent"], ev[-1]
            xs, ys = [0], [0]
            for spent, _ in d["curriculum"]:
                if spent > BUDGET:
                    break
                xs += [spent, spent]
                ys += [ys[-1], ys[-1] + 1]
            xs.append(end)
            ys.append(ys[-1])
            off = 0.05 * (3 * k + s - 4)                       # small offsets: the lines overlap
            a1.plot(xs, [y + off for y in ys], LS[s], color=col, lw=1.8,
                    label=f"{arm}, seed {s}: val {sc(last['val_worlds']):.2f}")
            a1.plot(end, ys[-1] + off, "o", color=col, ms=5, mfc="white" if end < BUDGET - 200 else col)
            a2.plot([e["spent"] for e in ev], [e["train"]["per_goal"]["make planks"][0] for e in ev], LS[s],
                    color=col, lw=1.6, marker="o", ms=2.5)
    a1.set(xlabel="feedbacks", ylabel="goals mastered", title="Goals mastered (0.7 on own episodes)", ylim=(0, 6.5),
           xlim=(0, BUDGET + 500))
    a1.legend(fontsize=6.5, loc="upper left", frameon=False)
    a2.axhline(0.7, color="#999", lw=0.8, ls=":")
    a2.set(xlabel="feedbacks", ylabel="planks success (train split)", title="make planks (line style = seed)",
           ylim=(0, 1.0), xlim=(0, BUDGET + 500))
    for a in (a1, a2):
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(OUT)


if __name__ == "__main__":
    main()
