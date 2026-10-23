"""Expert planner for MiniCraft: greedy recipe expansion + BFS navigation, replanned every step."""

import multiprocessing as mp
import os
from collections import deque
from functools import lru_cache

import numpy as np

from gft.minicraft.core import (BRIDGE, BRIDGE_ITEM, COLLECT, CRAFT0, DIRS, DONE, GEM, GEM_ITEM, IRON_PICKAXE, N_WORLD_ACTIONS,
                                PASSABLE, RECIPE_FOR, RECIPES, USE, WATER)
from gft.minicraft.goals import CROSS, GOALS

RAW = {item: (cell, tool) for cell, (item, tool) in COLLECT.items()}


def plan(inv, goals, crossed, bridged):
    """Ordered list of sub-tasks that turns `inv` into one satisfying `goals` (deepest goal first).

    Ops: ("collect", cell), ("craft", recipe_index), ("bridge",), ("cross",).
    """
    inv = inv.copy()
    ops = []
    state = {"crossed": crossed, "bridged": bridged}

    def need(item, n, depth=0):
        if depth > 40:
            raise RecursionError("recipe expansion did not terminate")
        while inv[item] < n:
            if item == GEM_ITEM:
                need(IRON_PICKAXE, 1, depth + 1)
                ensure_crossed(depth + 1)
                ops.append(("collect", GEM))
            elif item in RAW:
                cell, tool = RAW[item]
                if tool >= 0 and inv[tool] < 1:
                    need(tool, 1, depth + 1)
                ops.append(("collect", cell))
            else:
                k = RECIPE_FOR[item]
                rec = RECIPES[k]
                while any(inv[j] < c for j, c in rec.inputs):
                    for j, c in rec.inputs:
                        need(j, c, depth + 1)
                for j, c in rec.inputs:
                    inv[j] -= c
                ops.append(("craft", k))
            inv[item] += 1

    def ensure_crossed(depth):
        if state["crossed"]:
            return
        if not state["bridged"]:
            need(BRIDGE_ITEM, 1, depth + 1)
            inv[BRIDGE_ITEM] -= 1
            ops.append(("bridge",))
            state["bridged"] = True
        ops.append(("cross",))
        state["crossed"] = True

    # "Be east of the river" is about the final position, so pursue it last.
    for g in sorted((g for g in goals if g >= 0), key=lambda g: GOALS[g].item == CROSS):
        goal = GOALS[g]
        if goal.item == CROSS:
            ensure_crossed(0)
        else:
            need(goal.item, goal.count)
    return ops


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
def _first_op(inv, goals, crossed, bridged):
    """First op of plan() (cached: the same inventories and requests recur constantly)."""
    ops = plan(np.array(inv), goals, crossed, bridged)
    return ops[0] if ops else None


def expert_action(env, b, goals, rng=None):
    """Next action of the expert in world b of `env` for the given goal ids."""
    return expert_action_state(env.grid[b], env.pos[b], int(env.face[b]), env.inv[b], env.cfg.river_col, goals, rng)


def expert_action_state(grid, pos, face, inv, river, goals, rng=None):
    """expert_action from the raw state of one world (picklable inputs, for parallel labelling)."""
    pos = (int(pos[0]), int(pos[1]))
    crossed = pos[1] > river
    bridged = bool((grid[:, river] == BRIDGE).any())
    op = _first_op(tuple(inv.tolist()), tuple(int(g) for g in goals), crossed, bridged)
    if op is None:                                        # plan complete: declare done
        return DONE
    if op[0] == "craft":
        rec = RECIPES[op[1]]
        if rec.station < 0 or _adjacent_dir(grid, *pos, rec.station) >= 0:
            return CRAFT0 + op[1]
        step = _bfs_first_step(grid, pos, _adjacent_mask(grid, rec.station))
    elif op[0] in ("collect", "bridge"):
        kind = WATER if op[0] == "bridge" else op[1]
        d = _adjacent_dir(grid, *pos, kind)
        if d >= 0:
            return USE if face == d else d                     # moving into a non-passable cell only turns
        step = _bfs_first_step(grid, pos, _adjacent_mask(grid, kind))
    else:  # cross
        step = _bfs_first_step(grid, pos, np.broadcast_to(np.arange(grid.shape[1]) > river, grid.shape))
    if step is None or step < 0:
        return int(rng.integers(N_WORLD_ACTIONS)) if rng is not None else 0
    return step


def _label_chunk(args):
    grid, pos, face, inv, river, goals = args
    return [expert_action_state(grid[i], pos[i], int(face[i]), inv[i], river, goals[i]) for i in range(len(grid))]


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
        args = [(env.grid[c], env.pos[c], env.face[c], env.inv[c], env.cfg.river_col, goals[c]) for c in chunks]
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


def _adjacent_target_reference(grid, r, c, kind):
    """Direction (0-3) from (r, c) to an adjacent cell of type `kind`, or -1."""
    for d, (dr, dc) in enumerate(DIRS):
        if grid[r + dr, c + dc] == kind:
            return d
    return -1


def expert_action_reference(env, b, goals, rng=None):
    """Original per-cell implementation of expert_action, kept to test the fast one against."""
    grid, pos, face = env.grid[b], tuple(env.pos[b]), int(env.face[b])
    river = env.cfg.river_col
    crossed = pos[1] > river
    bridged = bool((grid[:, river] == BRIDGE).any())
    ops = plan(env.inv[b], goals, crossed, bridged)
    if not ops:                                           # plan complete: declare done
        return DONE
    op = ops[0]
    if op[0] == "craft":
        rec = RECIPES[op[1]]
        if rec.station < 0 or _adjacent_target_reference(grid, *pos, rec.station) >= 0:
            return CRAFT0 + op[1]
        step = _bfs_first_step_reference(grid, pos, lambda r, c: _adjacent_target_reference(grid, r, c, rec.station) >= 0)
    elif op[0] in ("collect", "bridge"):
        kind = WATER if op[0] == "bridge" else op[1]
        d = _adjacent_target_reference(grid, *pos, kind)
        if d >= 0:
            return USE if face == d else d                     # moving into a non-passable cell only turns
        step = _bfs_first_step_reference(grid, pos, lambda r, c: _adjacent_target_reference(grid, r, c, kind) >= 0)
    else:  # cross
        step = _bfs_first_step_reference(grid, pos, lambda r, c: c > river)
    if step is None or step < 0:
        return int(rng.integers(N_WORLD_ACTIONS)) if rng is not None else 0
    return step
