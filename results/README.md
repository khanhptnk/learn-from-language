# Results

Summaries of finished runs. Raw per-run JSON lives in the gitignored `outputs/`.

| File | Experiment | Reproduce |
|---|---|---|
| `tier0/summary.json` | Exact-population GFT: Section 7 example, aliasing, bound tightness, kappa x m, comparators | `uv run python experiments/tier0/run_tier0.py` |
| `l1/tuned.json`, `l1/tune.log` | Per-arm hyperparameters for the L1 regime map | `uv run python experiments/l1/sweep_l1.py tune` |
| `l1/regime.log` | L1 regime map: N x m x q x kappa x feedback type, AUC | `uv run python experiments/l1/sweep_l1.py regime` |
| `l1/horizon.log` | L1-H depth sweep: fixed N = 64 with k = 1..6, and b = 4 with N = 4^k | `uv run python experiments/l1/horizon_l1.py run` |
| `l1/intents.log` | L1-I teacher-intent profiles x depth x q | `uv run python experiments/l1/intents_l1.py run` |
| `l1/intents_rerun_scorer_vague.log` | GFT re-tuned for scorer/vague after the EVAL = m fix (supersedes those rows above) | `uv run python experiments/l1/intents_l1.py rerun` |
| `l1/intents_extend.log` | Batch GFT with distribution distillation (`gft-dist`, proposed) and RL with the optimistic parser | `uv run python experiments/l1/intents_l1.py extend` |
| `l1/verify_intents_exact.log` | Exact-population convergence for every intent profile, chatter-only, liars, style preferences | `uv run python experiments/l1/verify_intents_exact.py` |
| `l1/sgd_regime.txt`, `l1/sgd_horizon.txt`, `l1/sgd_intents.txt`, `l1/sgd_intents_compact.txt` | **Incremental GFT (`gft-sgd`: Batch GFT with a few SGD steps per fit, Algorithm 2 of the revised paper; batch size 1 gives Online GFT, Algorithm 3)** | `uv run python experiments/l1/online_l1.py {regime,horizon,intents}` |
| `theory/gft_verification.md` | **Numerical verification of every result in `GFT_revised_2026-10-20.pdf`** (exact population, FCP/SDPO, gradient identities, finite-sample theorems) | `uv run python experiments/theory/verify_gft.py` |

All L1 runs are tabular. The method under test is incremental GFT (`gft-sgd`, `gft/l1_online.py`).
Earlier L1 GFT results (`gft`, `gft-dist` in `gft/l1_tabular.py`, `gft/l1_horizon.py`, `gft/l1_intents.py`)
use count-based full refits with replay (Algorithm 1 style) and are kept as references.

The method was renamed from GLFT to GFT (General Feedback Training) with the revised paper. Result logs written before
the rename (`tier0/`, `l1/`) use the old names (`glft`, `glft-sgd`, ...) for the same arms.
