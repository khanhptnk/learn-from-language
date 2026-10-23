"""Expert episode lengths (steps until DONE) per goal and for the phase-1 request mix, vs the step limit.

  uv run python experiments/minicraft/expert_lengths.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import has_heldout  # noqa: E402

from gft.minicraft.core import WorldConfig  # noqa: E402
from gft.minicraft.expert import ExpertPool  # noqa: E402
from gft.minicraft.goals import GOALS, Requests, sample_requests  # noqa: E402
from gft.minicraft.learn import run_episodes  # noqa: E402


def lengths(cfg, req, expert):
    big = WorldConfig(**{**cfg.__dict__, "max_steps": 1000})            # no cut-off, to see true lengths
    _, _, _, _, valid, sat, mask = run_episodes(big, req, lambda e, t: expert(e, req.goals, np.nonzero(~e.done)[0]),
                                                record=False)
    return valid.sum(1), (sat | ~mask).all(1)


if __name__ == "__main__":
    cfg, rng = WorldConfig(), np.random.default_rng(0)
    expert = ExpertPool()
    print(f"step limit {cfg.max_steps}; expert steps incl. DONE (median / 90th pct / max), 300 worlds each")
    for g, goal in enumerate(GOALS):
        goals = np.full((300, 3), -1)
        goals[:, 0] = g
        n, ok = lengths(cfg, Requests(goals, rng.integers(0, 2 ** 31, size=300)), expert)
        print(f"  {goal.phrase:24s} {int(np.median(n)):4d} {int(np.percentile(n, 90)):4d} {n.max():4d}  "
              f"over limit {np.mean(n > cfg.max_steps):.2f}  solved {ok.mean():.2f}")
    req = sample_requests(rng, 2000, rng.integers(0, 2 ** 31, size=2000), combo_filter=lambda gs: not has_heldout(gs))
    n, ok = lengths(cfg, req, expert)
    k = (req.goals >= 0).sum(1)
    print(f"phase-1 request mix (2000): median {int(np.median(n))}, 90th pct {int(np.percentile(n, 90))}, max {n.max()}, "
          f"over limit {np.mean(n > cfg.max_steps):.3f}")
    for g, goal in enumerate(GOALS):
        m = (req.goals == g).any(1)
        print(f"  requests containing '{goal.phrase}': median {int(np.median(n[m]))}, over limit {np.mean(n[m] > cfg.max_steps):.2f}")
    expert.close()
