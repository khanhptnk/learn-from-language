# Results

Summaries of finished runs. Raw per-run JSON lives in the gitignored `outputs/`.

| File | Experiment | Reproduce |
|---|---|---|
| `tier0/summary.json` | Exact-population GLFT: Section 7 example, aliasing, bound tightness, kappa x m, comparators | `uv run python experiments/tier0/run_tier0.py` |
| `l1/tuned.json`, `l1/tune.log` | Per-arm hyperparameters for the L1 regime map | `uv run python experiments/l1/sweep_l1.py tune` |
| `l1/regime.log` | L1 regime map: N x m x q x kappa x feedback type, AUC | `uv run python experiments/l1/sweep_l1.py regime` |
| `l1/horizon.log` | L1-H depth sweep: fixed N = 64 with k = 1..6, and b = 4 with N = 4^k | `uv run python experiments/l1/horizon_l1.py run` |
| `l1/intents.log` | L1-I teacher-intent profiles x depth x q | `uv run python experiments/l1/intents_l1.py run` |
| `l1/intents_rerun_scorer_vague.log` | GLFT re-tuned for scorer/vague after the EVAL = m fix (supersedes those rows above) | `uv run python experiments/l1/intents_l1.py rerun` |

All L1 runs use tabular, count-based learners (`glft/l1_tabular.py`, `glft/l1_horizon.py`, `glft/l1_intents.py`).
