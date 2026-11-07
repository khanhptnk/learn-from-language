"""Episodes to master each curriculum goal per arm: step lines (log x) with a marker at each mastery point, lines
running to each run's last episode (open marker = still running, tick = finished). Two panels: the main arms, and the
ablations of GFT+PPO (filter) with the full method and PPO as references. Legend labels carry validation success.

  .venv/bin/python experiments/minicraft/plot_mastery.py <dir with run jsons> results/minicraft/mastery_arms.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gft.minicraft.goals import GOALS as _GOALS  # noqa: E402
GOALS = [g.phrase for g in _GOALS]
# (label, file, episodes per curriculum-log unit, palette slot) -- categorical hues in fixed order per panel
MAIN = [("GFT+PPO, discriminator filter", "filter_gftppo_filter_s0.json", 1, "#2a78d6"),
        ("GFT+PPO, filter (dual head, rerun)", "filter_gftppo_filter_dual_s0.json", 1, "#eb6834"),
        ("GFT+PPO, text reader", "pixels_gftppo_realistic_s0.json", 2, "#1baf7a"),
        ("PPO, tuned (150k)", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100"),
        ("disc_rl: PPO on language reward only", "filter_discrl_s0.json", 1, "#008300"),
        ("GFT alone, filter (dual head)", "filter_gft_filter_dual_s0.json", 1, "#6250d6"),
        ("GFT alone, filter (cross-attn head)", "filter_gft_filter_xattn_s0.json", 1, "#e34948")]
ABL = [("GFT+PPO, filter (dual head) — reference", "filter_gftppo_filter_dual_s0.json", 1, "#eb6834"),
       ("PPO, tuned — reference", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100"),
       ("uniform weights (PPO + self-imitation)", "filter_abl_uniform_s0.json", 1, "#2a78d6"),
       ("oracle weights (perfect filter)", "filter_abl_oracle_s0.json", 1, "#1baf7a"),
       ("no ideal-message positives", "filter_abl_noideal_s0.json", 1, "#6250d6"),
       ("no language (all messages blank)", "filter_abl_nolang_s0.json", 1, "#e34948")]
# no critic: sequence-level credit only (GRPO with the true reward, GFT / RFT without PPO), against PPO (critic)
CRITIC = [("PPO, tuned (critic, per-step credit)", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100"),
          ("GRPO: PPO with group advantage, no critic (true reward)", "grpo_grpo_s0.json", 1, "#e34948"),
          ("GFT alone, raw weights (language)", "filter_gft_filter_dual_s0.json", 1, "#6250d6"),
          ("GFT alone, z-scored weights (language)", "zs_gft_zs_s0.json", 1, "#2a78d6"),
          ("GFT alone, big step (language)", "bigstep_gft_big_s0.json", 1, "#008300"),
          ("RFT alone: perfect weights, no PPO (true reward)", "rft_rft_s0.json", 1, "#1baf7a")]
# GFT alone with different feedback readers (reader runs: 2 environment episodes per teacher call)
READER = [("sentence reader, pretrained bge-small text", "sreaderbge_gft_sreader_bge_s0.json", 2, "#eb6834"),
          ("sentence reader, text from scratch", "sreader_gft_sreader_s0.json", 2, "#2a78d6"),
          ("old reader (copy of pi prompted with the feedback)", "pixels_gft_realistic_s0.json", 2, "#6250d6"),
          ("discriminator filter (no reader)", "filter_gft_filter_dual_s0.json", 1, "#1baf7a"),
          ("PPO, tuned — reference", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100")]
# discriminator without PPO: episode weights vs per-step prefix credit vs discriminator reward + RL
PREFIX = [("GFT alone, discriminator, per-step prefix credit (+/- 25%)", "prefix_gft_prefix_s0.json", 1, "#eb6834"),
          ("GFT alone, discriminator, episode softmax weights", "filter_gft_filter_dual_s0.json", 1, "#1baf7a"),
          ("RL (PPO) on the discriminator reward", "filter_discrl_s0.json", 1, "#6250d6"),
          ("RFT alone: perfect episode weights, no PPO", "rft_rft_s0.json", 1, "#2a78d6"),
          ("PPO on the true reward — reference", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100")]
# ADEL as written (one model pi(a | s, request [SEP] description), hindsight MLE only, random init, annealed explorer)
ADEL = [("ADEL, descriptive teacher with quantities", "adel_qty_adel_qty_s0.json", 1, "#eb6834"),
        ("ADEL, descriptive teacher (count-free)", "adel_desc_adel_desc_s0.json", 1, "#2a78d6"),
        ("ADEL-style, realistic teacher (reader + policy)", "adel_nodamp_ideal_s0.json", 1, "#6250d6"),
        ("PPO, tuned — reference", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100")]
# ADEL with the realistic teacher: c* = the teacher's success message vs that message plus inferred progress
ADEL_REAL = [("ADEL, realistic teacher, inferred ideal, pretrained text (bge)", "abl2_bge_explorer_s0.json", 1, "#1baf7a"),
             ("ADEL, realistic teacher, inferred ideal (+ progress)", "adel_real_inferred_s0.json", 1, "#eb6834"),
             ("ADEL, realistic teacher, ideal = success message", "adel_real_teacher_s0.json", 1, "#2a78d6"),
             ("ADEL, descriptive teacher (count-free) — reference", "adel_desc_adel_desc_s0.json", 1, "#6250d6"),
             ("PPO, tuned — reference", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100")]
# GFT alone with the discriminator (episode weights) against PPO on the true reward, all PPO seeds (one hue)
GFT_VS_PPO = [("GFT alone, discriminator, episode softmax weights (seed 0)", "filter_gft_filter_dual_s0.json", 1,
               "#6250d6"),
              ("PPO on the true reward, seed 0", "filter_ppo_t0.8_150k_s0.json", 1, "#eda100"),
              ("PPO on the true reward, seed 1", "seeds_ppo_s1.json", 1, "#eda100"),
              ("PPO on the true reward, seed 2", "seeds_ppo_s2.json", 1, "#eda100")]
STOPPED = {"filter_gft_filter_xattn_s0.json", "adel_qty_adel_qty_s0.json", "adel_real_inferred_s0.json",
           "adel_real_teacher_s0.json", "bigstep_gft_big_s0.json",
           "sreader_gft_sreader_s0.json"}                 # killed by hand before their budget
# seed study: (arm label, [files of seeds 0, 1, 2], colour); seed-0 files are the earlier runs of the same setting
SEEDS = [("PPO", ["filter_ppo_t0.8_150k_s0.json", "seeds_ppo_s1.json", "seeds_ppo_s2.json"], "#eda100"),
         ("PPO + reward-weighted self-imitation", ["filter_abl_oracle_s0.json", "seeds_rewardsi_s1.json",
                                                   "seeds_rewardsi_s2.json"], "#1baf7a"),
         ("GFT+PPO, language filter", ["filter_gftppo_filter_dual_s0.json", "seeds_gftppo_s1.json",
                                       "seeds_gftppo_s2.json"], "#eb6834"),
         ("PPO + uniform self-imitation", ["filter_abl_uniform_s0.json", "seeds_uniform_s1.json",
                                           "seeds_uniform_s2.json"], "#2a78d6")]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
only = sys.argv[3] if len(sys.argv) > 3 else "both"          # "main" | "ablation" | "both" | "seeds" | ...
panels = {"main": [(MAIN, "Episodes to master each curriculum goal (pixels, realistic teacher, 1 seed)")],
          "ablation": [(ABL, "Ablations of GFT+PPO (filter, dual head), 100k-episode budget")],
          "seeds": [(SEEDS, "Seed study: 3 seeds per arm (thin lines = seeds), 100k-episode budget")],
          "critic": [(CRITIC, "Without a critic: GRPO vs GFT alone vs RFT alone, against PPO (seed 0)")],
          "reader": [(READER, "GFT alone: feedback readers vs the discriminator (seed 0)")],
          "adel": [(ADEL, "ADEL: hindsight MLE on teacher descriptions, by teacher (seed 0)")],
          "adel_real": [(ADEL_REAL, "ADEL with the realistic teacher: inferred ideal vs success message (seed 0)")],
          "gft_vs_ppo": [(GFT_VS_PPO, "GFT alone (discriminator) vs PPO, 150k / 100k teacher calls")],
          "prefix": [(PREFIX,"Discriminator without PPO: per-step prefix credit vs episode weights (seed 0)")]}
panels["both"] = panels["main"] + panels["ablation"]
chosen = panels[only]
fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 5.6 * len(chosen)), dpi=150, sharex=True, squeeze=False)
axes = axes[:, 0]
fig.patch.set_facecolor("#fcfcfb")


def style(ax, title):
    ax.set_xscale("log")
    ax.set_xticks([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 6e5])
    ax.set_xticklabels(["1k", "3k", "10k", "30k", "100k", "300k", "600k"])
    ax.set_xlim(700, 7.5e5)
    ax.set_yticks(range(1, len(GOALS) + 1))
    ax.set_yticklabels(GOALS)
    ax.set_ylim(0.5, 8.6)
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
    ax.grid(True, axis="x", color="#e6e5e1", lw=0.8, zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, title="validation success = new worlds",
              title_fontsize=7.5)


def draw_seeds(ax, groups, title):
    """One colour per arm, one thin step line per seed; legend lists each seed's validation at 100k."""
    ax.set_facecolor("#fcfcfb")
    for label, files, color in groups:
        vals, lines = [], []
        for f in files:
            p = src / f
            if not p.exists():
                vals.append("-")
                continue
            r = json.load(open(p))
            pts = sorted(r.get("curriculum", []))
            ev = r["evals"]
            last = min(ev[-1]["episodes"], 100_000)
            v = {e["episodes"] // 1000: e["val_worlds"]["success"] for e in ev}
            vals.append(f"{v[100]:.2f}" if 100 in v else (f"{v[last // 1000]:.2f}@{last // 1000}k" if v else "-"))
            pts = [(e, g) for e, g in pts if e <= 100_000]
            xs = [max(e, 300) for e, _ in pts] + [last]
            ys = [g + 1 for _, g in pts] + [pts[-1][1] + 1 if pts else 0]
            lines.append(ax.step(xs, ys, where="post", color=color, lw=1.5, alpha=0.85, zorder=3)[0])
            ax.plot(xs[:-1], ys[:-1], "o", color=color, ms=5, mec="#fcfcfb", mew=1, alpha=0.9, zorder=4)
            if last < 100_000:                                   # still running: open marker at the line end
                ax.plot([xs[-1]], [ys[-1]], "o", color=color, ms=7, mfc="#fcfcfb", mew=1.6, zorder=5)
        if lines:
            lines[0].set_label(f"{label} — val@100k seeds 0/1/2: {' / '.join(vals)}")
    style(ax, title)


def draw(ax, runs, title):
    ax.set_facecolor("#fcfcfb")
    for label, f, per, color in runs:
        p = src / f
        if not p.exists():
            continue
        r = json.load(open(p))
        pts = sorted((e * per, g) for e, g in r.get("curriculum", []))
        done = "seconds" in r or f in STOPPED
        ev = r["evals"]
        last = ev[-1]["episodes"] if ev else 0
        xs = [max(e, 300) for e, _ in pts] + [last]
        ys = [g + 1 for _, g in pts] + [pts[-1][1] + 1 if pts else 0]
        key = "val_worlds" if ev and "val_worlds" in ev[-1] else "new_worlds"
        val = {e["episodes"] // 1000: e[key]["success"] for e in ev}
        status = " (stopped)" if f in STOPPED else ("" if "seconds" in r else " (running)")
        tag = f"val {val[last // 1000]:.2f} at {last // 1000}k (best {max(val.values()):.2f})" if val else ""
        ax.step(xs, ys, where="post", color=color, lw=2, zorder=3, label=f"{label}{status} — {tag}")
        ax.plot(xs[:-1], ys[:-1], "o", color=color, ms=7, mec="#fcfcfb", mew=1.5, zorder=4)
        if done:
            ax.plot([last], [ys[-1]], "|", color=color, ms=12, mew=2, zorder=4)
        else:
            ax.plot([last], [ys[-1]], "o", mfc="#fcfcfb", mec=color, mew=2, ms=8, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 6e5])
    ax.set_xticklabels(["1k", "3k", "10k", "30k", "100k", "300k", "600k"])
    ax.set_xlim(700, 7.5e5)
    ax.set_yticks(range(1, len(GOALS) + 1))
    ax.set_yticklabels(GOALS)
    ax.set_ylim(0.5, 8.6)
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=11)
    ax.grid(True, axis="x", color="#e6e5e1", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, title="validation success = new worlds",
              title_fontsize=7.5)


for ax, (runs, title) in zip(axes, chosen):
    (draw_seeds if runs is SEEDS else draw)(ax, runs, title)
axes[-1].set_xlabel("environment episodes (one teacher or verifier call per episode)", color="#52514e")
fig.text(0.01, 0.004, "Lines run to each run's last episode: open marker = still going, tick = finished or stopped. "
         "Text-reader run: 2 episodes per feedback.", fontsize=7.5, color="#52514e")
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
