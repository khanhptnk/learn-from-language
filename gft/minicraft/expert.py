"""Expert planner for MiniCraft: greedy recipe expansion + BFS navigation, replanned every step. Plans count the items
in station slots as held (they can be taken back); at a station the expert takes back anything that does not belong,
selects and puts the recipe's missing inputs one at a time, and crafts once the slot holds exactly the inputs. Before
any step that is not a craft (collecting, or declaring done) it takes back whatever is left in station slots, so a
goal's items and the tools are in hand."""

import multiprocessing as mp
import os
from collections import deque
from functools import lru_cache

import numpy as np

from gft.minicraft.core import (COLLECT, CRAFT, DIRS, DONE, N_ITEMS, N_WORLD_ACTIONS, PASSABLE, PUT, RECIPE_FOR,
                                RECIPE_IN, RECIPES, SELECT, STATION_INDEX, STATIONS, TAKE, USE)
from gft.minicraft.goals import GOALS

RAW = {item: (cell, tool) for cell, (item, tool) in COLLECT.items()}


def _ops_from_nothing(item):
    return len(plan(np.zeros(N_ITEMS, np.int64), [], items=((item, 1),)))


def plan(inv, goals, items=None):
    """Ordered list of sub-tasks that turns `inv` into one satisfying `goals` (deepest goal first), or the (item,
    count) pairs `items`. Ops: ("collect", cell), ("craft", recipe_index).

    A recipe's inputs are gathered deepest branch first, and each branch is finished before the next starts."""
    inv = inv.copy()
    ops = []

    def need(item, n, depth=0):
        if depth > 40:
            raise RecursionError("recipe expansion did not terminate")
        while inv[item] < n:
            if item in RAW:
                cell, tool = RAW[item]
                if tool >= 0 and inv[tool] < 1:
                    need(tool, 1, depth + 1)
                ops.append(("collect", cell))
            else:
                k = RECIPE_FOR[item]
                rec = RECIPES[k]
                order = sorted(rec.inputs, key=lambda jc: -ITEM_DEPTH.get(jc[0], 0))
                for j, c in order:
                    need(j, c, depth + 1)
                for j, c in rec.inputs:
                    inv[j] -= c
                ops.append(("craft", k))
            inv[item] += 1

    # deepest goal first (ties: later id first)
    targets = items if items is not None else [(GOALS[g].item, GOALS[g].count) for g in
                                               sorted((g for g in goals if g >= 0), key=lambda g: (-GOALS[g].depth, -g))]
    for item, count in targets:
        need(item, count)
    return ops


ITEM_DEPTH = {}
for _i in range(N_ITEMS):                   # ops from an empty inventory, in dependency order (inputs before outputs)
    ITEM_DEPTH[_i] = _ops_from_nothing(_i)


_PASS = PASSABLE.tolist()


def _adjacent_mask(grid, kind):
    """(S, S) bool: cell has a 4-neighbour of type `kind`. The border is wall, so wrap-around never matters for
    cells the agent can stand on."""
    m = grid == kind
    adj = np.zeros_like(m)
    adj[1:] |= m[:-1]          # neighbour above
    adj[:-1] |= m[1:]          # below
    adj[:, 1:] |= m[:, :-1]    # left
    adj[:, :-1] |= m[:, 1:]    # right
    return adj


def _bfs_first_step(grid, start, goal):
    """First move direction (0-3) on a shortest passable path from start to a cell where `goal` (S, S bool) holds,
    or None. Returns -1 if start itself is a goal. BFS over flat lists (the border is wall, so flat neighbours of
    passable cells never wrap)."""
    S = grid.shape[0]
    goal = goal.ravel().tolist()
    s = start[0] * S + start[1]
    if goal[s]:
        return -1
    passable = [_PASS[x] for x in grid.ravel().tolist()]
    offsets = [int(dr) * S + int(dc) for dr, dc in DIRS]
    first = [-1] * (S * S)
    seen = [False] * (S * S)
    seen[s] = True
    q = deque()
    for d, off in enumerate(offsets):
        n = s + off
        if passable[n] and not seen[n]:
            seen[n] = True
            first[n] = d
            q.append(n)
    while q:
        i = q.popleft()
        if goal[i]:
            return first[i]
        for off in offsets:
            n = i + off
            if not seen[n] and passable[n]:
                seen[n] = True
                first[n] = first[i]
                q.append(n)
    return None


def _adjacent_dir(grid, r, c, kind):
    """Direction (0-3) from (r, c) to an adjacent cell of type `kind`, or -1."""
    for d, (dr, dc) in enumerate(DIRS):
        if grid[r + dr, c + dc] == kind:
            return d
    return -1


@lru_cache(maxsize=1 << 16)
def _first_op(inv, goals):
    """First op of plan() (cached: the same inventories and requests recur constantly)."""
    ops = plan(np.array(inv), goals)
    return ops[0] if ops else None


def _craft_target(grid, pos, inv, cursor, slot, k):
    """Craft op k: (action, None) if the agent can act where it stands, else (None, station cell type to walk to).
    At the recipe's station: take back a slot that holds anything extra, put the first missing input (selecting it
    first), craft when the slot holds exactly the inputs. A missing input that sits in another station's slot is
    fetched from there (take)."""
    st = RECIPES[k].station
    si, need = STATION_INDEX[st], RECIPE_IN[k]
    short = [j for j in np.nonzero(need)[0] if inv[j] + slot[si, j] < need[j]]     # inputs left in other slots
    if short:
        j = short[0]
        t = next(STATIONS[u] for u in range(len(STATIONS)) if u != si and slot[u, j] > 0)
        return (TAKE, None) if _adjacent_dir(grid, *pos, t) >= 0 else (None, t)
    if _adjacent_dir(grid, *pos, st) < 0:
        return None, st
    c = slot[si]
    if (c == need).all():
        return CRAFT, None
    if (c > need).any():
        return TAKE, None
    j = int(np.nonzero(c < need)[0][0])
    return (PUT if cursor == j else SELECT), None


def _stranded(slot):
    """Station cell types whose slot holds something."""
    return [STATIONS[u] for u in range(len(STATIONS)) if slot[u].any()]


def _union_adjacent(grid, kinds):
    m = np.zeros(grid.shape, bool)
    for k in kinds:
        m |= _adjacent_mask(grid, k)
    return m


def expert_action(env, b, goals, rng=None):
    """Next action of the expert in world b of `env` for the given goal ids."""
    return expert_action_state(env.grid[b], env.pos[b], int(env.face[b]), env.inv[b], int(env.cursor[b]),
                               env.slot[b], goals, rng)


def expert_action_state(grid, pos, face, inv, cursor, slot, goals, rng=None):
    """expert_action from the raw state of one world (picklable inputs, for parallel labelling)."""
    pos = (int(pos[0]), int(pos[1]))
    op = _first_op(tuple((inv + slot.sum(0)).tolist()), tuple(int(g) for g in goals))
    left = _stranded(slot) if op is None or op[0] != "craft" else []
    if left:                                              # goal items / tools left in a slot: take them back first
        if any(_adjacent_dir(grid, *pos, k) >= 0 for k in left):
            return TAKE
        step = _bfs_first_step(grid, pos, _union_adjacent(grid, left))
    elif op is None:                                      # plan complete: declare done
        return DONE
    elif op[0] == "craft":
        a, walk_to = _craft_target(grid, pos, inv, cursor, slot, op[1])
        if a is not None:
            return a
        step = _bfs_first_step(grid, pos, _adjacent_mask(grid, walk_to))
    else:
        kind = op[1]
        d = _adjacent_dir(grid, *pos, kind)
        if d >= 0:
            return USE if face == d else d                     # moving into a non-passable cell only turns
        step = _bfs_first_step(grid, pos, _adjacent_mask(grid, kind))
    if step is None or step < 0:                          # nothing reachable (resource exhausted): give up
        return int(rng.integers(N_WORLD_ACTIONS)) if rng is not None else DONE
    return step


def _label_chunk(args):
    grid, pos, face, inv, cursor, slot, goals = args
    return [expert_action_state(grid[i], pos[i], int(face[i]), inv[i], int(cursor[i]), slot[i], goals[i])
            for i in range(len(grid))]


def available_cpus():
    """CPUs this process may actually use: the container's cgroup CPU quota if set (cloud pods report the host's
    CPUs in os.cpu_count() and sched_getaffinity), else the affinity mask."""
    quota = None
    try:
        q, period = open("/sys/fs/cgroup/cpu.max").read().split()           # cgroup v2: "<quota|max> <period>"
        if q != "max":
            quota = int(q) / int(period)
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())       # cgroup v1; -1 = unlimited
            if q > 0:
                quota = q / int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        except (OSError, ValueError):
            pass
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, min(n, int(quota) if quota else n))


class ExpertPool:
    """Expert actions for many worlds at once, split across worker processes; identical to calling expert_action on
    each world. Workers are started with forkserver, so they never inherit a CUDA context."""

    def __init__(self, workers=None):
        self.workers = workers or available_cpus()
        self.pool = mp.get_context("forkserver").Pool(self.workers) if self.workers > 1 else None

    def __call__(self, env, goals, idx=None):
        """Actions (B,) for the worlds in `idx` (default: all); other entries are 0."""
        idx = np.arange(len(goals)) if idx is None else np.asarray(idx)
        out = np.zeros(len(goals), dtype=np.int64)
        if len(idx) == 0:
            return out
        chunks = [c for c in np.array_split(idx, min(len(idx), 4 * self.workers)) if len(c)]
        args = [(env.grid[c], env.pos[c], env.face[c], env.inv[c], env.cursor[c], env.slot[c], goals[c]) for c in chunks]
        res = self.pool.map(_label_chunk, args) if self.pool else [_label_chunk(x) for x in args]
        out[np.concatenate(chunks)] = np.concatenate([np.asarray(r, dtype=np.int64) for r in res])
        return out

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


# ---------------------------------------------------------------- reference implementation (equivalence test)

def _bfs_first_step_reference(grid, start, is_goal):
    """First move direction (0-3) on a shortest passable path from start to a cell with is_goal(r, c), or None.
    Returns -1 if start itself is a goal."""
    if is_goal(*start):
        return -1
    S = grid.shape[0]
    first = np.full((S, S), -1, dtype=np.int64)
    seen = np.zeros((S, S), dtype=bool)
    seen[start] = True
    q = deque()
    for d, (dr, dc) in enumerate(DIRS):
        n = (start[0] + dr, start[1] + dc)
        if PASSABLE[grid[n]] and not seen[n]:
            seen[n] = True
            first[n] = d
            q.append(n)
    while q:
        r, c = q.popleft()
        if is_goal(r, c):
            return int(first[r, c])
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if not seen[n] and PASSABLE[grid[n]]:
                seen[n] = True
                first[n] = first[r, c]
                q.append(n)
    return None


def expert_action_reference(env, b, goals, rng=None):
    """Per-cell implementation of expert_action, kept to test the fast one against."""
    grid, pos, face = env.grid[b], tuple(env.pos[b]), int(env.face[b])
    ops = plan(env.inv[b] + env.slot[b].sum(0), goals)
    left = _stranded(env.slot[b]) if not ops or ops[0][0] != "craft" else []
    if left:                                              # goal items / tools left in a slot: take them back first
        if any(_adjacent_dir(grid, *pos, k) >= 0 for k in left):
            return TAKE
        step = _bfs_first_step_reference(grid, pos, lambda r, c: any(_adjacent_dir(grid, r, c, k) >= 0 for k in left))
        if step is None or step < 0:
            return int(rng.integers(N_WORLD_ACTIONS)) if rng is not None else DONE
        return step
    if not ops:                                           # plan complete: declare done
        return DONE
    op = ops[0]
    if op[0] == "craft":
        a, kind = _craft_target(grid, pos, env.inv[b], int(env.cursor[b]), env.slot[b], op[1])
        if a is not None:
            return a
    else:
        kind = op[1]
        d = _adjacent_dir(grid, *pos, kind)
        if d >= 0:
            return USE if face == d else d
    step = _bfs_first_step_reference(grid, pos, lambda r, c: _adjacent_dir(grid, r, c, kind) >= 0)
    if step is None or step < 0:
        return int(rng.integers(N_WORLD_ACTIONS)) if rng is not None else DONE
    return step
