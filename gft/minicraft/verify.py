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

from gft.minicraft.core import (CELL_NAMES, COLLECT, CRAFT, DIRS, DONE, ITEM_NAMES, RECIPE_FOR, RECIPES, STATIONS, USE,
                                MiniCraft)
from gft.minicraft.expert import plan
from gft.minicraft.goals import GOALS
from gft.minicraft.teacher import SEQUENCE, TEMPLATES, render

NAME_TO_ITEM = {n: i for i, n in enumerate(ITEM_NAMES)}
NAME_TO_CELL = {n: i for i, n in enumerate(CELL_NAMES)}
PHRASE_TO_GOAL = {g.phrase: i for i, g in enumerate(GOALS)}


@dataclass
class Truth:
    goals: list
    satisfied: list
    collected: Counter = field(default_factory=Counter)
    crafted: Counter = field(default_factory=Counter)
    use_fail_tools: set = field(default_factory=set)
    station_fail: int = 0                                      # crafts away from any station
    nothing_fail: Counter = field(default_factory=Counter)     # station -> crafts there that made nothing
    craft_held: set = field(default_factory=set)               # (station, items held) of crafts that made nothing
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
        slot0 = env.slot[0].copy()
        env.step([int(a)])
        inv1, pos1 = env.inv[0], env.pos[0]
        tr.steps += 1
        if a < USE:
            tr.blocked += int((pos1 == pos0).all())
        elif a == USE:
            cell = int(grid0[tuple(pos0 + DIRS[face0])])
            gained = np.nonzero(inv1 > inv0)[0]
            if len(gained):
                tr.collected[int(gained[0])] += 1
            elif cell in COLLECT:
                tr.use_fail_tools.add(COLLECT[cell][1])
            else:
                tr.wasted += 1
        elif a == CRAFT:
            gained = np.nonzero(inv1 > inv0)[0]
            if len(gained):
                tr.crafted[int(gained[0])] += 1
            else:
                near = [int(grid0[tuple(pos0 + d)]) for d in DIRS]
                stations = [s for s in STATIONS if s in near]
                if stations:
                    tr.nothing_fail[stations[0]] += 1
                    held = slot0[STATIONS.index(stations[0])]
                    tr.craft_held.add((stations[0], tuple((int(i), int(held[i])) for i in np.nonzero(held > 0)[0])))
                else:
                    tr.station_fail += 1
        elif a == DONE:                                     # declares the episode finished; changes nothing
            pass
    tr.final_inv = env.inv[0].copy()
    pressed_done = len(actions) > 0 and int(actions[-1]) == DONE
    for g in tr.goals:
        goal = GOALS[g]
        tr.satisfied.append(bool(env.inv[0, goal.item] >= goal.count) and (pressed_done or not cfg.success_needs_done))
    for g, ok in zip(tr.goals, tr.satisfied):
        if not ok:
            for op in plan(tr.final_inv, [g]):
                if op[0] == "craft":
                    tr.still_needed.add(RECIPES[op[1]].output)
                else:
                    tr.still_needed.add(next(it for c, (it, _) in COLLECT.items() if c == op[1]))
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
    """'3 planks' -> (plank, 3); 'a plank' / 'wood' -> (item, 1)."""
    if s in NAME_TO_ITEM or re.match(r"^(a|an) ", s):
        return _item_np(s), 1
    n, name = s.split(" ", 1)
    name = name if name in NAME_TO_ITEM else name[:-1]
    return NAME_TO_ITEM[name], int(n)


def _times(s):
    return 1 if s == "once" else 2 if s == "twice" else int(s.split()[0])


_PREFIX = re.compile(r"^(?:" + "|".join(re.escape(p) for p in SEQUENCE) + r")(.)(.*)$")


def strip_prefix(sentence):
    """'Then you made a plank.' -> 'You made a plank.' (teacher.sequenced undone)."""
    m = _PREFIX.match(sentence)
    return m[1].upper() + m[2] if m else sentence


def _kinds(s):
    """'wood, planks and a pickaxe' -> item ids (item_kind_np undone)."""
    out = []
    for part in re.split(r", | and ", s):
        name = re.sub(r"^(a|an) ", "", part)
        out.append(NAME_TO_ITEM[name if name in NAME_TO_ITEM else name[:-1]])
    return tuple(out)


def parse(sentence):
    """(kind, claim) for a sentence produced by the templates, or None."""
    sentence = strip_prefix(sentence)
    for kind, _, rx in PARSERS:
        m = rx.match(sentence)
        if not m:
            continue
        g = m.groupdict()
        try:
            if kind in ("collected", "crafted"):
                return kind, (kind, *_n_item(g["n_item"]))
            if kind == "used_on":
                return kind, (kind, _kinds(g["in_kind"])[0], _item_np(g["out_np"]))
            if kind == "craft_held":
                return kind, (kind, NAME_TO_CELL[re.sub(r"^(a|an|the) ", "", g["station_np"])],
                              tuple(sorted(_n_item(x) for x in re.split(r", | and ", g["held"]))))
            if kind in ("goal_done", "goal_not_done", "subgoal"):
                return kind, (kind, PHRASE_TO_GOAL[g["phrase"]])
            if kind == "use_fail":
                tool = g.get("tool") or g["tool_np"]
                return kind, (kind, _item_np(tool))
            if kind == "station_fail":
                return kind, (kind, _times(g["times"]))
            if kind == "nothing_fail":
                return kind, (kind, NAME_TO_CELL[re.sub(r"^(a|an|the) ", "", g["station_np"])], _times(g["times"]))
            if kind == "steps":
                return kind, (kind, int(g["n"]))
            if kind in ("blocked", "wasted"):
                return kind, (kind, _times(g["times"]))
            if kind == "needed":
                return kind, (kind, _item_np(g.get("item_np") or g["Item_np"]))
            if kind == "used_up":
                name = re.sub(r"^(a|an) ", "", g.get("item") or g["some"])
                return kind, (kind, NAME_TO_ITEM[name if name in NAME_TO_ITEM else name[:-1]])
            if kind == "short":
                return kind, (kind, PHRASE_TO_GOAL[g["phrase"]], int(g["have"]))
            if kind == "no_done":
                return kind, (kind, PHRASE_TO_GOAL[g["phrase"]])
            if kind == "hint_recipe":
                return kind, (kind, _item_np(g.get("Item_np") or g["item_np"]))
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
    sentence = strip_prefix(sentence)
    if kind in ("collected", "crafted", "used_on", "craft_held"):   # the place / inputs must match the recipe book
        if not any(sentence == render(kind, c, v) for v in range(len(TEMPLATES[kind]))):
            return False, "place or inputs do not match the recipe book"
    if kind == "collected":
        return tr.collected[c[1]] == c[2], f"collected {tr.collected[c[1]]}"
    if kind == "crafted":
        return tr.crafted[c[1]] == c[2], f"crafted {tr.crafted[c[1]]}"
    if kind == "used_on":
        rec = RECIPES[RECIPE_FOR[c[2]]]
        ok = tr.crafted[c[2]] >= 1 and c[1] in {j for j, _ in rec.inputs} and tr.collected[c[1]] + tr.crafted[c[1]] >= 1
        return ok, f"crafted {tr.crafted[c[2]]}, got {tr.collected[c[1]] + tr.crafted[c[1]]}"
    if kind == "craft_held":
        return (c[1], c[2]) in tr.craft_held, f"failed crafts holding {sorted(tr.craft_held)}"
    if kind == "nothing":
        return not tr.collected and not tr.crafted, ""
    if kind in ("goal_done", "goal_not_done"):
        if c[1] not in tr.goals:
            return False, "goal not in request"
        ok = tr.satisfied[tr.goals.index(c[1])]
        return ok == (kind == "goal_done"), f"satisfied={ok}"
    if kind == "subgoal":                                  # reached during the episode (item obtained often enough)
        goal = GOALS[c[1]]
        got = tr.collected[goal.item] + tr.crafted[goal.item]
        return got >= goal.count, f"obtained {got}"
    if kind == "use_fail":
        return c[1] in tr.use_fail_tools, f"{tr.use_fail_tools}"
    if kind == "station_fail":
        return tr.station_fail == c[1], f"crafts away from a station {tr.station_fail}"
    if kind == "nothing_fail":
        return tr.nothing_fail[c[1]] == c[2], f"empty crafts at {CELL_NAMES[c[1]]}: {tr.nothing_fail[c[1]]}"
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
    if kind in ("short", "no_done"):                       # an unmet goal: too few of its item held, or all of it
        if c[1] not in tr.goals or tr.satisfied[tr.goals.index(c[1])]:
            return False, "goal not in request or met"
        goal = GOALS[c[1]]
        held = int(tr.final_inv[goal.item])
        if kind == "no_done":
            return held >= goal.count, f"held {held}"
        ok = held == c[2] < goal.count and any(sentence == render(kind, c, v) for v in range(len(TEMPLATES[kind])))
        return ok, f"held {held} of {goal.count}"
    if kind == "needed":
        return c[1] in tr.still_needed, f"still needed {sorted(ITEM_NAMES[i] for i in tr.still_needed)}"
    if kind in ("hint_recipe", "hint_raw"):
        return any(sentence == render(kind, c, v) for v in range(len(TEMPLATES[kind]))), \
            "hint does not match the recipe book"
    if kind == "eval_count":
        return (c[1], c[2]) == (sum(tr.satisfied), len(tr.satisfied)), ""
    if kind == "eval_all":
        return all(tr.satisfied), ""
    if kind == "eval_not_all":
        return not all(tr.satisfied), ""
    if kind == "vague_tools":
        return bool(tr.use_fail_tools), ""
    if kind == "vague_crafting":
        return bool(tr.station_fail or tr.nothing_fail), ""
    return True, ""                                         # chatter


def split_sentences(msg):
    return [s for s in re.split(r"(?<=[.!])\s+", msg.strip()) if s]


def reported_fraction(msg, goals):
    """Share of the requested goals (ids, -1 padding ignored) that the message does not report as not done: the
    scalar a reader of the realistic teacher's message gets, which names every unmet goal ("Not done: G.")."""
    goals = [int(g) for g in goals if g >= 0]
    unmet = {q[1][1] for s in split_sentences(msg) if (q := parse(s)) and q[0] == "goal_not_done"}
    return 1.0 - len(unmet & set(goals)) / len(goals)


def verify_message(msg, tr: Truth):
    """List of (sentence, ok, reason); a message is accurate iff every sentence is ok."""
    return [(s, *check(s, tr)) for s in split_sentences(msg)]
