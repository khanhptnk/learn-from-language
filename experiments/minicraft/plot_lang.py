"""Post-training from the language-pretrained base on the held-out tasks: success vs teacher calls, per arm.
Left: success on held-out requests (new worlds) and on seen requests (regression check). Right: success of the
the two held-out goals themselves.

  .venv/bin/python experiments/minicraft/plot_lang.py outputs/pod/lang results/minicraft/lang_posttrain.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("GFT (inferred target)", "gft_inferred_s0.json", "#6250d6", "o"), ("PPO (frac reward)", "ppo_s0.json", "#eda100", "s")]
SUBGOALS = [("build a well", "-"), ("make a sling", "--")]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
for label, f, color, mk in ARMS:
    r = json.load(open(src / f))
    ev = r["evals"]
    x = [e["teacher_calls"] if "gft" in f else e["episodes"] for e in ev]
    ax1.plot(x, [e["val_worlds"]["success"] for e in ev], "-", color=color, lw=2, marker=mk, ms=6, mec="#fcfcfb",
             label=f"{label}: held-out requests (well / sling)")
    ax1.plot(x, [e["val_seen"]["success"] for e in ev], ":", color=color, lw=2, marker=mk, ms=5, mec="#fcfcfb",
             label=f"{label}: seen requests")
    for g, ls in SUBGOALS:
        ax2.plot(x, [e["val_worlds"]["per_goal"][g][0] for e in ev], ls, color=color, lw=2, marker=mk, ms=5,
                 mec="#fcfcfb", label=f"{label}: {g}")
for ax, title in ((ax1, "Success on new worlds (both arms start from the same pretrained base)"),
                  (ax2, "The held-out goals themselves (success per goal)")):
    ax.set_facecolor("#fcfcfb")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("teacher calls (GFT) / episodes (PPO)", color="#52514e")
    ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
    ax.grid(True, axis="y", color="#e6e5e1", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.06))
ax1.set_ylabel("success", color="#52514e")
fig.text(0.01, 0.005, "New game, held-out tasks 'build a well' and 'make a sling' (requested in pretraining, never "
         "demonstrated). Both arms start from the same pretrained base; 50k teacher calls / episodes, seed 0.",
         fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
