"""High-level goals (what the agent is told) and their hidden requirements (what is checked)."""

from dataclasses import dataclass, field

import numpy as np

from gft.minicraft.core import (BRICK, CLAY, FENCE, FIBER, OVEN, PICKAXE, PLANK, RAFT, ROPE, SLING, STONE, WALL_ITEM,
                                WELL, WOOD)


@dataclass(frozen=True)
class Goal:
    phrase: str      # shown to the agent; names the thing, never the recipe
    item: int        # inventory item checked
    count: int = 1
    depth: int = 1   # number of expert ops from nothing (for curricula and difficulty tiers)


GOALS = (                                     # ids in depth order (curricula and plots rely on it)
    Goal("gather wood", WOOD, 2, 2),
    Goal("gather fiber", FIBER, 2, 2),
    Goal("dig clay", CLAY, 2, 2),
    Goal("make rope", ROPE, 1, 2),
    Goal("make bricks", BRICK, 1, 2),
    Goal("make planks", PLANK, 2, 4),
    Goal("make a pickaxe", PICKAXE, 1, 5),
    Goal("build a raft", RAFT, 1, 5),
    Goal("build an oven", OVEN, 1, 5),
    Goal("build a well", WELL, 1, 5),
    Goal("quarry stone", STONE, 1, 6),
    Goal("make a sling", SLING, 1, 9),
    Goal("build a wall", WALL_ITEM, 1, 9),
    Goal("build a fence", FENCE, 1, 9),
)
N_GOALS = len(GOALS)
MAX_GOALS = 3
GOAL_ID = {g.phrase: i for i, g in enumerate(GOALS)}


@dataclass(frozen=True)
class Alias:
    phrase: str      # a new high-level request
    goals: tuple     # the existing goals it requires


# Phase-2 requests: new phrases for goal combinations never requested together in phase 1 (the held-out pairs).
# The phrases avoid the feedback vocabulary, so their meaning must be learned; the events they need are familiar.
ALIASES = (
    Alias("set up camp", (GOAL_ID["build a wall"], GOAL_ID["gather wood"])),
    Alias("prepare the harbour", (GOAL_ID["build a raft"], GOAL_ID["dig clay"])),
    Alias("start the garden", (GOAL_ID["build a fence"], GOAL_ID["make rope"])),
    Alias("open the kitchen", (GOAL_ID["build an oven"], GOAL_ID["quarry stone"])),
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


def compatible(gs):
    return True


def sample_requests(rng, n, seeds, goal_pool=None, max_goals=MAX_GOALS, combo_filter=None):
    """n requests of 1..max_goals distinct goals from `goal_pool` (default all), on the given world seeds.
    `combo_filter(frozenset) -> bool` rejects combinations (used for held-out compositional splits)."""
    pool = np.arange(N_GOALS) if goal_pool is None else np.asarray(goal_pool)
    goals = np.full((n, MAX_GOALS), -1, dtype=np.int64)
    for b in range(n):
        while True:
            k = int(rng.integers(1, max_goals + 1))
            gs = rng.choice(pool, size=min(k, len(pool)), replace=False)
            fs = frozenset(gs.tolist())
            if compatible(fs) and (combo_filter is None or combo_filter(fs)):
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
    have = np.take_along_axis(env.inv, item, axis=1) >= count
    return have & mask, mask


def requirement_status_reference(env, req: Requests):
    """Unvectorized reference implementation (kept for the equivalence test)."""
    mask = req.goals >= 0
    sat = np.zeros(req.goals.shape, dtype=bool)
    for j in range(MAX_GOALS):
        g = req.goals[:, j]
        for gid in np.unique(g[g >= 0]):
            goal = GOALS[gid]
            rows = g == gid
            sat[rows, j] = env.inv[rows, goal.item] >= goal.count
    return sat, mask
