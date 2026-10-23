"""Check that feedback is accurate: every sentence must parse into a claim that is true of the trajectory.

Ground truth comes from *replaying* the episode (world seed + actions) and reading state changes -- inventory deltas,
grid changes, whether the agent moved -- independently of the event log and of the teacher's fact extraction.
"Still needed" claims are checked against a plan computed from the replayed end state (the plan defines what
remains to be produced for an unmet goal).
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gft.minicraft.core import (BRIDGE, BRIDGE_ITEM, CELL_NAMES, COLLECT, CRAFT0, DIRS, DONE, GEM_ITEM, ITEM_NAMES, RECIPES,
                                USE, WATER, MiniCraft)
from gft.minicraft.expert import plan
from gft.minicraft.goals import CROSS, GOALS
from gft.minicraft.teacher import TEMPLATES, render

NAME_TO_ITEM = {n: i for i, n in enumerate(ITEM_NAMES)}
NAME_TO_CELL = {n: i for i, n in enumerate(CELL_NAMES)}
PHRASE_TO_GOAL = {g.phrase: i for i, g in enumerate(GOALS)}
TARGET_TO_TOOL = {"a rock": "wood pickaxe", "the ore": "stone pickaxe", "the gem": "iron pickaxe", "the water": "bridge"}


@dataclass
class Truth:
    goals: list
    satisfied: list
    collected: Counter = field(default_factory=Counter)
    crafted: Counter = field(default_factory=Counter)
    use_fail_tools: set = field(default_factory=set)
    station_fail: Counter = field(default_factory=Counter)
    missing_fail: set = field(default_factory=set)            # (output, any input missing at a failed attempt)
    bridged: bool = False
    crossed: bool = False
    steps: int = 0
    blocked: int = 0
    wasted: int = 0
    final_inv: np.ndarray = None
    still_needed: set = field(default_factory=set)            # items a plan from the replayed end state must produce


def replay(cfg, seed, goals, actions):
    """Re-run the episode and derive the truth from state changes."""
    env = MiniCraft(cfg, [seed])
    tr = Truth(goals=[int(g) for g in goals if g >= 0], satisfied=[])
    for a in actions:
        inv0, grid0, pos0, face0 = env.inv[0].copy(), env.grid[0].copy(), env.pos[0].copy(), int(env.face[0])
        env.step([int(a)])
        inv1, grid1, pos1 = env.inv[0], env.grid[0], env.pos[0]
        tr.steps += 1
        if a < USE:
            tr.blocked += int((pos1 == pos0).all())
        elif a == USE:
            tgt = tuple(pos0 + DIRS[face0])
            cell = int(grid0[tgt])
            gained = np.nonzero(inv1 > inv0)[0]
            if len(gained):
                tr.collected[int(gained[0])] += 1
            elif cell == WATER and grid1[tgt] == BRIDGE:
                tr.bridged = True
            elif cell in COLLECT:
                tr.use_fail_tools.add(COLLECT[cell][1])
            elif cell == WATER:
                tr.use_fail_tools.add(BRIDGE_ITEM)
            else:
                tr.wasted += 1
        elif a == DONE:                                     # declares the episode finished; changes nothing
            pass
        else:
            rec = RECIPES[a - CRAFT0]
            if inv1[rec.output] > inv0[rec.output]:
                tr.crafted[rec.output] += 1
            else:
                near = [int(grid0[tuple(pos0 + d)]) for d in DIRS]
                if rec.station >= 0 and rec.station not in near:
                    tr.station_fail[rec.station] += 1
                else:
                    tr.missing_fail.update((rec.output, j) for j, c in rec.inputs if inv0[j] < c)
    tr.crossed = bool(env.pos[0, 1] > cfg.river_col)
    tr.final_inv = env.inv[0].copy()
    for g in tr.goals:
        goal = GOALS[g]
        tr.satisfied.append(tr.crossed if goal.item == CROSS else bool(env.inv[0, goal.item] >= goal.count))
    bridged = bool((env.grid[0][:, cfg.river_col] == BRIDGE).any())
    for g, ok in zip(tr.goals, tr.satisfied):
        if not ok:
            for op in plan(tr.final_inv, [g], tr.crossed, bridged):
                if op[0] == "craft":
                    tr.still_needed.add(RECIPES[op[1]].output)
                elif op[0] == "collect":
                    tr.still_needed.add(next(it for c, (it, _) in COLLECT.items() if c == op[1]))
                elif op[0] == "bridge":
                    tr.still_needed.add(BRIDGE_ITEM)
    return tr


# ---------------------------------------------------------------- parsing sentences back into claims

def _template_regex(t):
    parts = re.split(r"(\{\w+\})", t)
    rx = "".join(f"(?P<{p[1:-1]}>.+?)" if p.startswith("{") else re.escape(p) for p in parts)
    return re.compile("^" + rx + "$")


PARSERS = [(kind, v, _template_regex(t)) for kind, ts in TEMPLATES.items() for v, t in enumerate(ts)]


def _item_np(s):
    s = re.sub(r"^(a|an) ", "", s.lower())
    return NAME_TO_ITEM[s]


def _n_item(s):
    n, name = s.split(" ", 1)
    name = name if name in NAME_TO_ITEM else name[:-1]
    return NAME_TO_ITEM[name], int(n)


def _times(s):
    return 1 if s == "once" else 2 if s == "twice" else int(s.split()[0])


def parse(sentence):
    """(kind, claim) for a sentence produced by the templates, or None."""
    for kind, _, rx in PARSERS:
        m = rx.match(sentence)
        if not m:
            continue
        g = m.groupdict()
        try:
            if kind in ("collected", "crafted"):
                return kind, (kind, *_n_item(g["n_item"]))
            if kind in ("goal_done", "goal_not_done", "subgoal"):
                return kind, (kind, PHRASE_TO_GOAL[g["phrase"]])
            if kind == "use_fail":
                tool = g.get("tool") or g["tool_np"]
                return kind, (kind, _item_np(tool))
            if kind == "station_fail":
                if "times" in g:
                    return kind, (kind, NAME_TO_CELL[g["station"]], _times(g["times"]))
                return kind, (kind, NAME_TO_CELL[re.sub(r"^(a|an) ", "", g["station_np"])], None)
            if kind == "missing_fail":
                return kind, (kind, _item_np(g["item_np"]), _item_np(g["missing_np"]))
            if kind == "steps":
                return kind, (kind, int(g["n"]))
            if kind in ("blocked", "wasted"):
                return kind, (kind, _times(g["times"]))
            if kind == "needed":
                return kind, (kind, _item_np(g["item_np"]))
            if kind == "used_up":
                name = g["item"]
                return kind, (kind, NAME_TO_ITEM[name if name in NAME_TO_ITEM else name[:-1]])
            if kind == "hint_recipe":
                return kind, (kind, _item_np(g["Item_np"]))
            if kind == "hint_raw":
                return kind, (kind, NAME_TO_ITEM[g["item"]])
            if kind == "eval_count":
                return kind, (kind, int(g["k"]), int(g["m"]))
            return kind, (kind,)
        except (KeyError, ValueError):
            continue
    return None


def check(sentence, tr: Truth):
    """(ok, reason) for one sentence against the replayed truth."""
    p = parse(sentence)
    if p is None:
        return False, "unparsed"
    kind, c = p
    if kind == "collected":
        return tr.collected[c[1]] == c[2], f"collected {tr.collected[c[1]]}"
    if kind == "crafted":
        return tr.crafted[c[1]] == c[2], f"crafted {tr.crafted[c[1]]}"
    if kind == "bridged":
        return tr.bridged, ""
    if kind == "crossed":
        return tr.crossed, ""
    if kind == "nothing":
        return not tr.collected and not tr.crafted, ""
    if kind in ("goal_done", "goal_not_done"):
        if c[1] not in tr.goals:
            return False, "goal not in request"
        ok = tr.satisfied[tr.goals.index(c[1])]
        return ok == (kind == "goal_done"), f"satisfied={ok}"
    if kind == "subgoal":                                  # reached during the episode (item obtained often enough)
        goal = GOALS[c[1]]
        if goal.item == CROSS:
            return tr.crossed, ""
        got = tr.collected[goal.item] + tr.crafted[goal.item]
        return got >= goal.count, f"obtained {got}"
    if kind == "use_fail":
        return c[1] in tr.use_fail_tools, f"{tr.use_fail_tools}"
    if kind == "station_fail":
        n = tr.station_fail[c[1]]
        return (n == c[2]) if c[2] is not None else n > 0, f"station failures {n}"
    if kind == "missing_fail":
        return (c[1], c[2]) in tr.missing_fail, ""
    if kind == "steps":
        return tr.steps == c[1], f"steps {tr.steps}"
    if kind == "blocked":
        return tr.blocked == c[1], f"blocked {tr.blocked}"
    if kind == "wasted":
        return tr.wasted == c[1], f"wasted {tr.wasted}"
    if kind == "used_up":                                  # obtained enough of a requested item, not kept
        need = max((GOALS[g].count for g in tr.goals if GOALS[g].item == c[1]), default=None)
        got = tr.collected[c[1]] + tr.crafted[c[1]]
        return need is not None and got >= need and tr.final_inv[c[1]] < need, f"got {got}"
    if kind == "needed":
        return c[1] in tr.still_needed, f"still needed {sorted(ITEM_NAMES[i] for i in tr.still_needed)}"
    if kind in ("hint_recipe", "hint_raw"):
        return sentence == render(kind, c, 0), "hint does not match the recipe book"
    if kind == "eval_count":
        return (c[1], c[2]) == (sum(tr.satisfied), len(tr.satisfied)), ""
    if kind == "eval_all":
        return all(tr.satisfied), ""
    if kind == "eval_not_all":
        return not all(tr.satisfied), ""
    if kind == "vague_tools":
        return bool(tr.use_fail_tools), ""
    if kind == "vague_crafting":
        return bool(tr.station_fail or tr.missing_fail), ""
    return True, ""                                         # chatter


def split_sentences(msg):
    return [s for s in re.split(r"(?<=[.!])\s+", msg.strip()) if s]


def verify_message(msg, tr: Truth):
    """List of (sentence, ok, reason); a message is accurate iff every sentence is ok."""
    return [(s, *check(s, tr)) for s in split_sentences(msg)]
