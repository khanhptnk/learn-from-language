"""v3 runs (explore base, well/sling excluded, 1 seed): left, goals mastered vs feedbacks (step lines, labels carry
final validation success); right, the first crafting tier (planks, rope, bricks) on the evaluation's train split
(~40 episodes per goal per point), GFT v3e (solid) vs the tuned PPO, P4 group advantage (dashed).
  uv run python experiments/minicraft/plot_v3.py results/minicraft/v3_frontier.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("GFT v3e (keep top 50%, 4 passes)", "outputs/pod/gft/v3e/gft_explore_s0.json", "#2a78d6", "-"),
        ("GFT v3b", "outputs/pod/gft/v3b/gft_explore_s0.json", "#2a78d6", ":"),
        ("PPO P4 (group advantage)", "outputs/pod/gft/ppo_sweep/p4_group.json", "#eda100", "-"),
        ("PPO v3 (GAE baseline)", "outputs/pod/gft/v3/ppo_frac_explore_s0.json", "#eda100", ":")]
VS = [("GFT v3e", RUNS[0][1], "-"), ("PPO P4", RUNS[2][1], "--")]
FRONTIER = [("make planks", "#e34948"), ("make rope", "#1baf7a"), ("make bricks", "#6250d6")]
sc = lambda x: x["success"] if isinstance(x, dict) else x


def main(out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, (label, path, col, ls) in enumerate(RUNS):
        d = json.load(open(path))
        end, e = d["evals"][-1]["spent"], d["evals"][-1]
        xs, ys = [0], [0]
        for s, _ in d["curriculum"]:
            xs += [s, s]
            ys += [ys[-1], ys[-1] + 1]
        xs.append(end)
        ys.append(ys[-1])
        a1.plot(xs, [y + 0.06 * (k - 1.5) for y in ys], ls, color=col, lw=2,   # small offsets: the lines overlap
                label=f"{label}: val worlds {sc(e['val_worlds']):.2f}, combos {sc(e['val_combos']):.2f}, "
                      f"unseen {sc(e['val_unseen']):.2f}")
    for name, path, ls in VS:
        d = json.load(open(path))
        xs = [e["spent"] for e in d["evals"]]
        for goal, gc in FRONTIER:
            a2.plot(xs, [e["train"]["per_goal"][goal][0] for e in d["evals"]], ls, color=gc, lw=1.8,
                    marker="o" if ls == "-" else None, ms=3, label=f"{goal} ({name})")
    a1.set(xlabel="feedbacks (teacher messages / PPO rewards)", ylabel="goals mastered (of 12)",
           title="Goals mastered (0.7 on own episodes)", ylim=(0, 6))
    a1.legend(fontsize=7, loc="upper left", frameon=False)
    a2.axhline(0.7, color="#999", lw=0.8, ls=":")
    a2.text(100000, 0.71, "mastery 0.7", fontsize=7, color="#777", ha="right")
    a2.set(xlabel="feedbacks", ylabel="evaluation success (train split)",
           title="First crafting tier: GFT v3e vs tuned PPO", ylim=(0, 1.0))
    a2.legend(fontsize=7, loc="upper left", frameon=False, ncol=2)
    for a in (a1, a2):
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/minicraft/v3_frontier.png")
