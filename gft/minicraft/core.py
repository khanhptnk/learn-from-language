"""MiniCraft: a small deterministic crafting world (see docs/minicraft_design.md).

Four resource branches (wood, fiber, clay, stone) that combine into composite things at four stations. Crafting makes
exactly what was put in: each station has an input slot; generic actions select an inventory kind (a visible cursor),
put one of it into the adjacent station's slot, take the slot back, and craft, which makes the recipe whose inputs equal
the slot's contents at that station (else nothing). No task needs an action of its own, so a held-out task is new only
in which known skills it combines.

Batched numpy environment: B independent worlds step together. Every step records one event per world
(what the action did, and why it failed if it did), which the teacher later turns into feedback.
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- cells
EMPTY, WALL, TREE, REEDS, CLAYPIT, ROCK, WORKBENCH, LOOM, KILN, SITE = range(10)
CELL_NAMES = ["empty", "wall", "tree", "reeds", "clay pit", "rock", "workbench", "loom", "kiln", "site"]
N_CELLS = len(CELL_NAMES)
PASSABLE = np.zeros(N_CELLS, dtype=bool)
PASSABLE[EMPTY] = True
STATIONS = (WORKBENCH, LOOM, KILN, SITE)

# ---------------------------------------------------------------- items
(WOOD, FIBER, CLAY, STONE, PLANK, ROPE, BRICK, PICKAXE, RAFT, SLING, OVEN, WELL, WALL_ITEM, FENCE) = range(14)
ITEM_NAMES = ["wood", "fiber", "clay", "stone", "plank", "rope", "brick", "pickaxe", "raft", "sling", "oven", "well",
              "wall", "fence"]
N_ITEMS = len(ITEM_NAMES)
TOOLS = (PICKAXE,)

# Using a resource cell: cell -> (item gained, tool required or -1)
COLLECT = {TREE: (WOOD, -1), REEDS: (FIBER, -1), CLAYPIT: (CLAY, -1), ROCK: (STONE, PICKAXE)}


@dataclass(frozen=True)
class Recipe:
    output: int
    inputs: tuple            # ((item, count), ...)
    station: int             # cell type that must be adjacent


# Intermediates first, then the composites (two branches each). Craft makes the recipe of the adjacent station whose
# inputs equal the station slot's contents exactly; no two recipes of a station share inputs.
RECIPES = (
    Recipe(PLANK, ((WOOD, 1),), WORKBENCH),
    Recipe(ROPE, ((FIBER, 1),), LOOM),
    Recipe(BRICK, ((CLAY, 1),), KILN),
    Recipe(PICKAXE, ((PLANK, 2),), WORKBENCH),
    Recipe(RAFT, ((PLANK, 1), (ROPE, 1)), WORKBENCH),
    Recipe(SLING, ((ROPE, 1), (STONE, 1)), WORKBENCH),
    Recipe(OVEN, ((PLANK, 1), (BRICK, 1)), KILN),
    Recipe(WELL, ((ROPE, 1), (BRICK, 1)), SITE),
    Recipe(WALL_ITEM, ((BRICK, 1), (STONE, 1)), SITE),
    Recipe(FENCE, ((PLANK, 1), (STONE, 1)), SITE),
)
RECIPE_FOR = {r.output: i for i, r in enumerate(RECIPES)}
COMPOSITES = (RAFT, SLING, OVEN, WELL, WALL_ITEM, FENCE)
RECIPE_IN = np.zeros((len(RECIPES), N_ITEMS), dtype=np.int64)       # input counts per recipe
for _k, _r in enumerate(RECIPES):
    for _j, _c in _r.inputs:
        RECIPE_IN[_k, _j] = _c
assert len({(r.station, tuple(RECIPE_IN[k])) for k, r in enumerate(RECIPES)}) == len(RECIPES), \
    "two recipes of one station share inputs: craft would be ambiguous"
STATION_INDEX = {s: i for i, s in enumerate(STATIONS)}

# ---------------------------------------------------------------- actions
UP, RIGHT, DOWN, LEFT, USE, SELECT, PUT, TAKE, CRAFT = range(9)
N_WORLD_ACTIONS = CRAFT + 1                                  # actions that act on the world (random noise draws these)
DONE = N_WORLD_ACTIONS                                       # the agent declares it is finished: the episode ends
N_ACTIONS = N_WORLD_ACTIONS + 1
ACTION_NAMES = ["up", "right", "down", "left", "use", "select", "put", "take", "craft", "done"]
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])          # (row, col) for up, right, down, left

# ---------------------------------------------------------------- events
EV_MOVE, EV_BLOCKED, EV_COLLECT, EV_USE_FAIL, EV_USE_NOTHING, EV_CRAFT, EV_CRAFT_FAIL, EV_DONE = range(1, 9)
EV_SELECT, EV_PUT, EV_PUT_FAIL, EV_TAKE, EV_TAKE_FAIL = range(9, 14)
EVENT_NAMES = ["none", "move", "blocked", "collect", "use_fail", "use_nothing", "craft", "craft_fail", "done",
               "select", "put", "put_fail", "take", "take_fail"]
AGENT_FEATS = 3 * N_ITEMS        # agent_state(): inventory, cursor one-hot, adjacent station slot


@dataclass(frozen=True)
class WorldConfig:
    size: int = 13
    n_trees: int = 8
    n_reeds: int = 6
    n_clay: int = 5
    n_rocks: int = 4
    max_steps: int = 200
    explicit_done: bool = True   # episodes end when the agent takes DONE (or at max_steps); success is checked then.
                                 # False: the episode ends as soon as all requirements hold (counts incidental success)
    success_needs_done: bool = False  # a requirement counts as met only if it holds at the end AND the agent took
                                      # DONE (running out of steps with everything in hand is not a success)


def generate_world(cfg: WorldConfig, rng: np.random.Generator):
    """One world: (grid, agent position, facing). Resampled until every object has a reachable neighbour cell."""
    S = cfg.size
    while True:
        grid = np.zeros((S, S), dtype=np.int8)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = WALL
        inside = [(r, x) for r in range(1, S - 1) for x in range(1, S - 1)]
        kinds = (list(STATIONS) + [TREE] * cfg.n_trees + [REEDS] * cfg.n_reeds + [CLAYPIT] * cfg.n_clay
                 + [ROCK] * cfg.n_rocks)
        cells = rng.choice(len(inside), size=len(kinds) + 1, replace=False)
        for k, i in zip(kinds, cells[:-1]):
            grid[inside[i]] = k
        pos = np.array(inside[cells[-1]])
        reach = reachable(grid, pos)
        objects = np.argwhere((grid != EMPTY) & (grid != WALL))
        if not all(any(reach[tuple(o + d)] for d in DIRS) for o in objects):
            continue
        # no cell touches two stations: the craft action must be unambiguous about where the agent is crafting
        nbrs = [{tuple(o + d) for d in DIRS} for o in np.argwhere(np.isin(grid, STATIONS))]
        if all(not (a & b) for i, a in enumerate(nbrs) for b in nbrs[i + 1:]):
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


_WORLD_CACHE = {}


def craftable(slot, station):
    """Index of the recipe craft makes at `station` from slot contents `slot` (1-D counts): the recipe of that station
    whose inputs equal the contents exactly, or -1."""
    hit = np.nonzero([r.station == station for r in RECIPES] & (RECIPE_IN == np.asarray(slot)).all(1))[0]
    return int(hit[0]) if len(hit) else -1


class MiniCraft:
    """B worlds stepping in lockstep. State: grid (B,S,S), pos (B,2), face (B,), inv (B,N_ITEMS), cursor (B,) (the
    selected item kind, -1 none), slot (B,len(STATIONS),N_ITEMS) (each station's input slot), t (B,), done (B,).

    `step(actions)` returns a dict of per-world event fields:
      type      event code (EV_*)
      item      collected / crafted / selected / put item, or the cell type acted on / blocked by; -1 if none
      tool      for failed uses: the tool that was needed, else -1
      station   the adjacent station for put / take / craft (failed or not), or -1 (no station)
      count     items taken back by take, else -1
    """

    def __init__(self, cfg: WorldConfig, seeds):
        self.cfg = cfg
        self.seeds = np.asarray(seeds)
        B, S = len(self.seeds), cfg.size
        self.grid = np.zeros((B, S, S), dtype=np.int8)
        self.pos = np.zeros((B, 2), dtype=np.int64)
        self.face = np.zeros(B, dtype=np.int64)
        key = repr(cfg)                               # worlds are a deterministic function of (config, seed): cache them
        for b, s in enumerate(self.seeds):
            w = _WORLD_CACHE.get((key, int(s)))
            if w is None:
                w = _WORLD_CACHE[(key, int(s))] = generate_world(cfg, np.random.default_rng(int(s)))
            self.grid[b], self.pos[b], self.face[b] = w[0].copy(), w[1], w[2]
        self.inv = np.zeros((B, N_ITEMS), dtype=np.int64)
        self.cursor = np.full(B, -1, dtype=np.int64)
        self.slot = np.zeros((B, len(STATIONS), N_ITEMS), dtype=np.int64)
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

    def adjacent_stations(self):
        """(B, len(STATIONS)) bool: which stations neighbour the agent."""
        return np.stack([self.station_adjacent(s) for s in STATIONS], 1)

    def adjacent_station(self):
        """(B,) index into STATIONS of the station next to the agent, or -1 (no cell touches two stations)."""
        adj = self.adjacent_stations()
        return np.where(adj.any(1), adj.argmax(1), -1)

    def agent_state(self):
        """(B, AGENT_FEATS) what the agent sees besides the map: inventory counts, the cursor (one-hot), and the
        contents of the adjacent station's slot (zeros away from a station)."""
        B = self.B
        cur = np.zeros((B, N_ITEMS), dtype=np.int64)
        sel = self.cursor >= 0
        cur[np.nonzero(sel)[0], self.cursor[sel]] = 1
        si = self.adjacent_station()
        near = np.zeros((B, N_ITEMS), dtype=np.int64)
        at = si >= 0
        near[at] = self.slot[np.nonzero(at)[0], si[at]]
        return np.concatenate([self.inv, cur, near], 1)
    def step(self, actions):
        actions = np.asarray(actions)
        assert ((actions >= 0) & (actions < N_ACTIONS)).all(), "action out of range"
        B = self.B
        ev = {k: np.full(B, -1, dtype=np.int64) for k in ("item", "tool", "station", "count")}
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
        si = self.adjacent_station()
        # Select: the cursor moves to the next item kind held (in item order, wrapping), or -1 if nothing is held.
        se = live & (actions == SELECT)
        for b in np.nonzero(se)[0]:
            held = np.nonzero(self.inv[b] > 0)[0]
            nxt = held[held > self.cursor[b]]
            self.cursor[b] = nxt[0] if len(nxt) else (held[0] if len(held) else -1)
            ev["type"][b], ev["item"][b] = EV_SELECT, self.cursor[b]
        # Put: one of the selected item into the adjacent station's slot.
        pu = live & (actions == PUT)
        for b in np.nonzero(pu)[0]:
            c = self.cursor[b]
            ev["station"][b] = STATIONS[si[b]] if si[b] >= 0 else -1
            ev["item"][b] = c
            if si[b] >= 0 and c >= 0 and self.inv[b, c] > 0:
                self.inv[b, c] -= 1
                self.slot[b, si[b], c] += 1
                ev["type"][b] = EV_PUT
            else:
                ev["type"][b] = EV_PUT_FAIL
        # Take: the adjacent station's slot back into the inventory.
        tk = live & (actions == TAKE)
        for b in np.nonzero(tk)[0]:
            ev["station"][b] = STATIONS[si[b]] if si[b] >= 0 else -1
            n = int(self.slot[b, si[b]].sum()) if si[b] >= 0 else 0
            if n:
                self.inv[b] += self.slot[b, si[b]]
                self.slot[b, si[b]] = 0
                ev["type"][b], ev["count"][b] = EV_TAKE, n
            else:
                ev["type"][b] = EV_TAKE_FAIL
        # Craft: the recipe of the adjacent station whose inputs equal its slot's contents.
        cr = live & (actions == CRAFT)
        for b in np.nonzero(cr)[0]:
            ev["type"][b] = EV_CRAFT_FAIL
            if si[b] < 0:
                continue                                     # no station around (station stays -1)
            ev["station"][b] = STATIONS[si[b]]
            k = craftable(self.slot[b, si[b]], STATIONS[si[b]])
            if k >= 0:
                self.slot[b, si[b]] = 0
                self.inv[b, RECIPES[k].output] += 1
                ev["type"][b], ev["item"][b] = EV_CRAFT, RECIPES[k].output
        # Declare done: the episode ends after this step.
        dn = live & (actions == DONE)
        ev["type"][dn] = EV_DONE
        self.done |= dn
        self.t[live] += 1
        return ev
