"""MiniCraft: a small deterministic crafting world (see docs/minicraft_design.md).

Batched numpy environment: B independent worlds step together. Every step records one event per world
(what the action did, and why it failed if it did), which the teacher later turns into feedback.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- cells
EMPTY, WALL, TREE, ROCK, ORE, WATER, BRIDGE, WORKBENCH, FURNACE, GEM = range(10)
CELL_NAMES = ["empty", "wall", "tree", "rock", "ore", "water", "bridge", "workbench", "furnace", "gem"]
N_CELLS = len(CELL_NAMES)
PASSABLE = np.zeros(N_CELLS, dtype=bool)
PASSABLE[[EMPTY, BRIDGE]] = True

# ---------------------------------------------------------------- items
(WOOD, PLANK, STICK, WOOD_PICKAXE, STONE, STONE_PICKAXE, ORE_ITEM, IRON, IRON_PICKAXE, BRIDGE_ITEM,
 GEM_ITEM) = range(11)
ITEM_NAMES = ["wood", "plank", "stick", "wood pickaxe", "stone", "stone pickaxe", "ore", "iron", "iron pickaxe",
              "bridge", "gem"]
N_ITEMS = len(ITEM_NAMES)
TOOLS = (WOOD_PICKAXE, STONE_PICKAXE, IRON_PICKAXE)

# Using a resource cell: cell -> (item gained, tool required or -1)
COLLECT = {TREE: (WOOD, -1), ROCK: (STONE, WOOD_PICKAXE), ORE: (ORE_ITEM, STONE_PICKAXE), GEM: (GEM_ITEM, IRON_PICKAXE)}


@dataclass(frozen=True)
class Recipe:
    output: int
    inputs: tuple            # ((item, count), ...) in the order the expert gathers them
    station: int = -1        # cell type that must be adjacent, or -1


RECIPES = (
    Recipe(PLANK, ((WOOD, 1),), WORKBENCH),
    Recipe(STICK, ((PLANK, 1),)),
    Recipe(WOOD_PICKAXE, ((STICK, 1), (PLANK, 1)), WORKBENCH),
    Recipe(STONE_PICKAXE, ((STICK, 1), (STONE, 1)), WORKBENCH),
    Recipe(IRON, ((ORE_ITEM, 1), (WOOD, 1)), FURNACE),
    Recipe(IRON_PICKAXE, ((STICK, 1), (IRON, 1)), WORKBENCH),
    Recipe(BRIDGE_ITEM, ((IRON, 1), (PLANK, 2)), WORKBENCH),
)
RECIPE_FOR = {r.output: i for i, r in enumerate(RECIPES)}

# ---------------------------------------------------------------- actions
UP, RIGHT, DOWN, LEFT, USE = range(5)
CRAFT0 = 5
N_WORLD_ACTIONS = CRAFT0 + len(RECIPES)                     # actions that act on the world (random noise draws these)
DONE = N_WORLD_ACTIONS                                       # the agent declares it is finished: the episode ends
N_ACTIONS = N_WORLD_ACTIONS + 1
ACTION_NAMES = (["up", "right", "down", "left", "use"] + [f"craft {ITEM_NAMES[r.output]}" for r in RECIPES]
                + ["done"])
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])          # (row, col) for up, right, down, left

# ---------------------------------------------------------------- events
EV_MOVE, EV_BLOCKED, EV_COLLECT, EV_USE_FAIL, EV_USE_NOTHING, EV_CRAFT, EV_CRAFT_FAIL, EV_BRIDGE, EV_DONE = range(1, 10)
EVENT_NAMES = ["none", "move", "blocked", "collect", "use_fail", "use_nothing", "craft", "craft_fail", "bridge", "done"]


@dataclass(frozen=True)
class WorldConfig:
    size: int = 13
    n_trees: int = 12
    n_rocks: int = 4
    n_ore: int = 3
    max_steps: int = 200
    explicit_done: bool = True   # episodes end when the agent takes DONE (or at max_steps); success is checked then.
                                 # False: the episode ends as soon as all requirements hold (counts incidental success)

    @property
    def river_col(self):
        return self.size - 4


def generate_world(cfg: WorldConfig, rng: np.random.Generator):
    """One world: (grid, agent position, facing). West of the river: stations, resources, agent; east: the gem.
    Resampled until every object has a reachable neighbour cell and the river bank is reachable."""
    S, c = cfg.size, cfg.river_col
    while True:
        grid = np.zeros((S, S), dtype=np.int8)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = WALL
        grid[1:-1, c] = WATER
        east = [(r, x) for r in range(1, S - 1) for x in range(c + 2, S - 1)]   # not on the bank: bridges stay usable
        gr, gx = east[rng.integers(len(east))]
        grid[gr, gx] = GEM
        west = [(r, x) for r in range(1, S - 1) for x in range(1, c)]
        kinds = [WORKBENCH, FURNACE] + [TREE] * cfg.n_trees + [ROCK] * cfg.n_rocks + [ORE] * cfg.n_ore
        cells = rng.choice(len(west), size=len(kinds) + 1, replace=False)
        for k, i in zip(kinds, cells[:-1]):
            grid[west[i]] = k
        pos = np.array(west[cells[-1]])
        reach = reachable(grid, pos)
        objects = np.argwhere((grid != EMPTY) & (grid != WALL) & (grid != WATER) & (grid != GEM))
        ok = all(any(reach[tuple(o + d)] for d in DIRS) for o in objects)
        ok &= reach[1:-1, c - 1].any()
        if ok:
            return grid, pos, int(rng.integers(4))


def reachable(grid, pos):
    """Boolean map of passable cells reachable from pos (pos itself included)."""
    seen = np.zeros(grid.shape, dtype=bool)
    seen[tuple(pos)] = True
    stack = [tuple(pos)]
    while stack:
        r, x = stack.pop()
        for dr, dx in DIRS:
            n = (r + dr, x + dx)
            if not seen[n] and PASSABLE[grid[n]]:
                seen[n] = True
                stack.append(n)
    return seen


class MiniCraft:
    """B worlds stepping in lockstep. State: grid (B,S,S), pos (B,2), face (B,), inv (B,N_ITEMS), t (B,), done (B,).

    `step(actions)` returns a dict of per-world event fields:
      type      event code (EV_*)
      item      collected/crafted item, or the cell type acted on / blocked by
      tool      for failed uses: the tool (or bridge) that was needed, else -1
      station   for failed crafts: the missing station cell type, else -1
      missing   for failed crafts: the first missing input item, else -1
    """

    def __init__(self, cfg: WorldConfig, seeds):
        self.cfg = cfg
        self.seeds = np.asarray(seeds)
        B, S = len(self.seeds), cfg.size
        self.grid = np.zeros((B, S, S), dtype=np.int8)
        self.pos = np.zeros((B, 2), dtype=np.int64)
        self.face = np.zeros(B, dtype=np.int64)
        for b, s in enumerate(self.seeds):
            self.grid[b], self.pos[b], self.face[b] = generate_world(cfg, np.random.default_rng(int(s)))
        self.inv = np.zeros((B, N_ITEMS), dtype=np.int64)
        self.t = np.zeros(B, dtype=np.int64)
        self.done = np.zeros(B, dtype=bool)

    @property
    def B(self):
        return len(self.seeds)

    def _cell(self, rc):
        return self.grid[np.arange(self.B), rc[:, 0], rc[:, 1]]

    def station_adjacent(self, station):
        """(B,) bool: a cell of type `station` is one of the 4 neighbours of the agent."""
        out = np.zeros(self.B, dtype=bool)
        for d in DIRS:
            out |= self._cell(self.pos + d) == station
        return out

    def step(self, actions):
        actions = np.asarray(actions)
        B = self.B
        ev = {k: np.full(B, -1, dtype=np.int64) for k in ("item", "tool", "station", "missing")}
        ev["type"] = np.zeros(B, dtype=np.int64)
        live = ~self.done
        rows = np.arange(B)
        # Moves (also set facing; blocked moves only turn).
        mv = live & (actions < USE)
        if mv.any():
            self.face[mv] = actions[mv]
            target = self.pos + DIRS[self.face]
            cell = self._cell(target)
            ok = mv & PASSABLE[cell]
            self.pos[ok] = target[ok]
            ev["type"][ok] = EV_MOVE
            blk = mv & ~PASSABLE[cell]
            ev["type"][blk] = EV_BLOCKED
            ev["item"][blk] = cell[blk]
        # Use the faced cell.
        us = live & (actions == USE)
        if us.any():
            target = self.pos + DIRS[self.face]
            cell = self._cell(target)
            ev["item"][us] = cell[us]
            ev["type"][us] = EV_USE_NOTHING
            for kind, (item, tool) in COLLECT.items():
                m = us & (cell == kind)
                if not m.any():
                    continue
                have = np.ones(B, dtype=bool) if tool < 0 else self.inv[:, tool] > 0
                got = m & have
                self.inv[got, item] += 1
                self.grid[rows[got], target[got, 0], target[got, 1]] = EMPTY
                ev["type"][got] = EV_COLLECT
                ev["item"][got] = item
                fail = m & ~have
                ev["type"][fail] = EV_USE_FAIL
                ev["tool"][fail] = tool
            w = us & (cell == WATER)
            put = w & (self.inv[:, BRIDGE_ITEM] > 0)
            self.inv[put, BRIDGE_ITEM] -= 1
            self.grid[rows[put], target[put, 0], target[put, 1]] = BRIDGE
            ev["type"][put] = EV_BRIDGE
            nob = w & ~put
            ev["type"][nob] = EV_USE_FAIL
            ev["tool"][nob] = BRIDGE_ITEM
        # Crafts.
        for k, rec in enumerate(RECIPES):
            m = live & (actions == CRAFT0 + k)
            if not m.any():
                continue
            ev["item"][m] = rec.output
            has_station = np.ones(B, dtype=bool) if rec.station < 0 else self.station_adjacent(rec.station)
            missing = np.full(B, -1, dtype=np.int64)
            for item, n in reversed(rec.inputs):                # report the first missing input in recipe order
                missing[self.inv[:, item] < n] = item
            ok = m & has_station & (missing < 0)
            for item, n in rec.inputs:
                self.inv[ok, item] -= n
            self.inv[ok, rec.output] += 1
            ev["type"][ok] = EV_CRAFT
            fail = m & ~ok
            ev["type"][fail] = EV_CRAFT_FAIL
            ev["station"][fail & ~has_station] = rec.station
            ev["missing"][fail] = missing[fail]
        # Declare done: the episode ends after this step.
        dn = live & (actions == DONE)
        ev["type"][dn] = EV_DONE
        self.done |= dn
        self.t[live] += 1
        return ev
