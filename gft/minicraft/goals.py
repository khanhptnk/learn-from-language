"""High-level goals (what the agent is told) and their hidden requirements (what is checked)."""

from dataclasses import dataclass, field

import numpy as np

from gft.minicraft.core import (GEM_ITEM, IRON, IRON_PICKAXE, PLANK, STICK, STONE, STONE_PICKAXE, WOOD,
                                WOOD_PICKAXE)

CROSS = -1  # pseudo-item: "agent ends east of the river"


@dataclass(frozen=True)
class Goal:
    phrase: str      # shown to the agent; never names the recipe chain
    item: int        # inventory item checked, or CROSS
    count: int = 1
    depth: int = 1   # rough number of distinct sub-steps (for curricula and difficulty tiers)


GOALS = (
    Goal("gather firewood", WOOD, 2, 1),
    Goal("make some planks", PLANK, 2, 2),
    Goal("make a handle", STICK, 1, 3),
    Goal("get ready to mine", WOOD_PICKAXE, 1, 5),
    Goal("collect building stone", STONE, 2, 6),
    Goal("upgrade your tools", STONE_PICKAXE, 1, 7),
    Goal("get some metal", IRON, 1, 9),
    Goal("make an iron tool", IRON_PICKAXE, 1, 11),
    Goal("get across the river", CROSS, 1, 13),
    Goal("fetch the gem", GEM_ITEM, 1, 15),
)
N_GOALS = len(GOALS)
MAX_GOALS = 3


@dataclass(frozen=True)
class Alias:
    phrase: str      # a new high-level request
    goals: tuple     # the existing goals it requires


# Phase-2 requests: new phrases for goal combinations never requested together in phase 1 (the held-out pairs).
# The phrases avoid the feedback vocabulary, so their meaning must be learned; the events they need are familiar.
ALIASES = (
    Alias("go treasure hunting", (9, 0)),       # fetch the gem + gather firewood
    Alias("outfit the workshop", (7, 1)),       # make an iron tool + make some planks
    Alias("prepare the dig", (6, 3)),           # get some metal + get ready to mine
    Alias("build an outpost", (8, 4)),          # get across the river + collect building stone
)


@dataclass
class Requests:
    """Batch of requests: goal ids (B, MAX_GOALS) padded with -1, the world seed of each, and optionally an alias id
    per request (-1 = none) whose phrase replaces the goal phrases."""
    goals: np.ndarray
    seeds: np.ndarray
    alias: np.ndarray = field(default=None)

    def __len__(self):
        return len(self.seeds)

    def text(self, b):
        if self.alias is not None and self.alias[b] >= 0:
            return ALIASES[self.alias[b]].phrase
        return " and ".join(GOALS[g].phrase for g in self.goals[b] if g >= 0)


def alias_requests(k, seeds):
    """Requests for alias k on the given world seeds."""
    goals = np.full((len(seeds), MAX_GOALS), -1, dtype=np.int64)
    goals[:, :len(ALIASES[k].goals)] = ALIASES[k].goals
    return Requests(goals, np.asarray(seeds), np.full(len(seeds), k))


def sample_requests(rng, n, seeds, goal_pool=None, max_goals=MAX_GOALS, combo_filter=None):
    """n requests of 1..max_goals distinct goals from `goal_pool` (default all), on the given world seeds.
    `combo_filter(frozenset) -> bool` rejects combinations (used for held-out compositional splits)."""
    pool = np.arange(N_GOALS) if goal_pool is None else np.asarray(goal_pool)
    goals = np.full((n, MAX_GOALS), -1, dtype=np.int64)
    for b in range(n):
        while True:
            k = int(rng.integers(1, max_goals + 1))
            gs = rng.choice(pool, size=min(k, len(pool)), replace=False)
            if combo_filter is None or combo_filter(frozenset(gs.tolist())):
                break
        goals[b, :len(gs)] = np.sort(gs)[::-1]                  # deepest first (the order the expert pursues)
    return Requests(goals, np.asarray(seeds))


GOAL_ITEM = np.array([g.item for g in GOALS])
GOAL_COUNT = np.array([g.count for g in GOALS])


def requirement_status(env, req: Requests):
    """(B, MAX_GOALS) bool satisfied, and (B, MAX_GOALS) bool mask of real goals."""
    mask = req.goals >= 0
    g = np.where(mask, req.goals, 0)
    item, count = GOAL_ITEM[g], GOAL_COUNT[g]
    have = np.take_along_axis(env.inv, np.where(item >= 0, item, 0), axis=1) >= count
    crossed = (env.pos[:, 1] > env.cfg.river_col)[:, None]
    return np.where(item == CROSS, crossed, have) & mask, mask


def requirement_status_reference(env, req: Requests):
    """Unvectorized reference implementation (kept for the equivalence test)."""
    mask = req.goals >= 0
    sat = np.zeros(req.goals.shape, dtype=bool)
    for j in range(MAX_GOALS):
        g = req.goals[:, j]
        for gid in np.unique(g[g >= 0]):
            goal = GOALS[gid]
            rows = g == gid
            if goal.item == CROSS:
                sat[rows, j] = env.pos[rows, 1] > env.cfg.river_col
            else:
                sat[rows, j] = env.inv[rows, goal.item] >= goal.count
    return sat, mask
