"""Template teacher for MiniCraft: episode event log -> facts -> claims -> mixed-intent feedback text.

Feedback talks about low-level events and mechanics (wood, planks, workbench, pickaxes, failed crafts), mostly not in
the request's vocabulary. Each sentence realizes one structured *claim* through a template; `gft.minicraft.verify`
parses sentences back into claims and checks them against a replay of the trajectory, so accuracy is testable.
A profile sets how likely each intent is to be voiced.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gft.minicraft.core import (BRIDGE, BRIDGE_ITEM, CELL_NAMES, EV_BLOCKED, EV_BRIDGE, EV_COLLECT, EV_CRAFT,
                                EV_CRAFT_FAIL, EV_USE_FAIL, EV_USE_NOTHING, ITEM_NAMES, RECIPE_FOR, RECIPES)
from gft.minicraft.expert import RAW, plan
from gft.minicraft.goals import CROSS, GOALS

INTENTS = ("descriptive", "confirm", "diagnostic", "process", "corrective", "hint", "evaluative", "vague", "chatter")
MASS = {"wood", "stone", "ore", "iron"}
TOOL_TARGET = {"wood pickaxe": "a rock", "stone pickaxe": "the ore", "iron pickaxe": "the gem", "bridge": "the water"}
BLOCKED_MIN, WASTED_MIN, MAX_MISSING_CLAIMS = 10, 5, 3


@dataclass
class Facts:
    goals: list                      # goal ids of the request
    satisfied: list                  # bool per goal
    collected: Counter = field(default_factory=Counter)
    crafted: Counter = field(default_factory=Counter)
    use_fail_tool: Counter = field(default_factory=Counter)       # tool (or bridge item) that was missing
    craft_fail_station: Counter = field(default_factory=Counter)  # station that was not adjacent
    craft_fail_missing: Counter = field(default_factory=Counter)  # (output item, first missing input)
    bridged: bool = False
    crossed: bool = False
    steps: int = 0
    blocked: int = 0
    wasted_uses: int = 0
    next_needed: dict = field(default_factory=dict)               # unmet goal -> first item still needed
    final_inv: tuple = ()                                         # inventory at the end of the episode


def episode_facts(log, b, env, goals, satisfied):
    """Facts of episode b from a log of per-step event dicts (as returned by MiniCraft.step)."""
    f = Facts(goals=[int(g) for g in goals if g >= 0], satisfied=[bool(s) for s, g in zip(satisfied, goals) if g >= 0])
    for ev in log:
        t = ev["type"][b]
        if t == EV_COLLECT:
            f.collected[int(ev["item"][b])] += 1
        elif t == EV_CRAFT:
            f.crafted[int(ev["item"][b])] += 1
        elif t == EV_USE_FAIL:
            f.use_fail_tool[int(ev["tool"][b])] += 1
        elif t == EV_CRAFT_FAIL:
            if ev["station"][b] >= 0:
                f.craft_fail_station[int(ev["station"][b])] += 1
            else:
                f.craft_fail_missing[(int(ev["item"][b]), int(ev["missing"][b]))] += 1
        elif t == EV_BRIDGE:
            f.bridged = True
        elif t == EV_BLOCKED:
            f.blocked += 1
        elif t == EV_USE_NOTHING:
            f.wasted_uses += 1
    f.steps = int(env.t[b])
    f.final_inv = tuple(int(x) for x in env.inv[b])
    river = env.cfg.river_col
    f.crossed = bool(env.pos[b, 1] > river)
    bridged_now = bool((env.grid[b][:, river] == BRIDGE).any())
    for g, ok in zip(f.goals, f.satisfied):
        if not ok:
            item = _first_missing_item(plan(env.inv[b], [g], f.crossed, bridged_now))
            if item is not None:
                f.next_needed[g] = item
    return f


def _first_missing_item(ops):
    """Item produced by the first op of a plan (what the agent lacked first), or None for moves/crossing."""
    if not ops:
        return None
    op = ops[0]
    if op[0] == "craft":
        return RECIPES[op[1]].output
    if op[0] == "collect":
        return next(item for item, (cell, _) in RAW.items() if cell == op[1])
    if op[0] == "bridge":
        return BRIDGE_ITEM
    return None


# ---------------------------------------------------------------- noun phrases

def np_item(item):
    """'a plank', 'an iron pickaxe', 'wood' (mass nouns take no article)."""
    name = ITEM_NAMES[item]
    return name if name in MASS else ("an " if name[0] in "aeiou" else "a ") + name


def n_item(count, item):
    """'1 plank', '3 planks', '5 wood'."""
    name = ITEM_NAMES[item]
    return f"{count} {name}" if count == 1 or name in MASS else f"{count} {name}s"


def times(c):
    return "once" if c == 1 else "twice" if c == 2 else f"{c} times"


def a_cell(cell):
    name = CELL_NAMES[cell]
    return ("an " if name[0] in "aeiou" else "a ") + name


# ---------------------------------------------------------------- templates: claim kind -> sentence variants
# Placeholders are filled by `render`; `gft.minicraft.verify` compiles the same templates into parsers.

TEMPLATES = {
    "collected": ["You collected {n_item}.", "You picked up {n_item}."],
    "crafted": ["You made {n_item}.", "You crafted {n_item}."],
    "bridged": ["You built a bridge over the river.", "You bridged the river."],
    "crossed": ["You ended on the far side of the river.", "You finished across the river."],
    "nothing": ["You did not collect or make anything.", "Nothing was collected or made."],
    "goal_done": ["You did {phrase}.", "Done: {phrase}."],
    "subgoal": ["Along the way you managed to {phrase}.", "You managed to {phrase}."],
    "used_up": ["You used up the {item} you needed.", "You had {item} at some point but did not keep {it}."],
    "goal_not_done": ["You did not {phrase}.", "Not done: {phrase}."],
    "use_fail": ["You tried to work {target} without {tool_np}.", "{Target} would not give way: you had no {tool}."],
    "station_fail": ["Crafting failed {times} because no {station} was next to you.",
                     "You were not standing by {station_np} when crafting."],
    "missing_fail": ["You tried to make {item_np} without {missing_np}."],
    "steps": ["You took {n} steps.", "That run lasted {n} steps."],
    "blocked": ["You bumped into things {times}."],
    "wasted": ["You used empty ground {times}."],
    "needed": ["You still needed {item_np}.", "You should have gotten {item_np} first."],
    "hint_recipe": ["{Item_np} is made from {inputs}{where}."],
    "hint_raw": ["You get {item} from {cell_np}{with_tool}."],
    "eval_count": ["{k} of {m} done."],
    "eval_all": ["Great job!"],
    "eval_not_all": ["Not quite there."],
    "vague_tools": ["Something went wrong with your tools."],
    "vague_crafting": ["Some of your crafting was off."],
    "chatter": ["Thanks!", "Thanks for trying.", "Okay."],
}


def render(kind, claim, variant):
    """Sentence for a claim (kind, *args) using template `variant`."""
    args = claim[1:]
    v = {}
    if kind in ("collected", "crafted"):
        v["n_item"] = n_item(args[1], args[0])
    elif kind in ("goal_done", "goal_not_done", "subgoal"):
        v["phrase"] = GOALS[args[0]].phrase
    elif kind == "use_fail":
        tool = ITEM_NAMES[args[0]]
        v.update(target=TOOL_TARGET[tool], Target=TOOL_TARGET[tool].capitalize(), tool=tool, tool_np=np_item(args[0]))
    elif kind == "station_fail":
        v.update(times=times(args[1]), station=CELL_NAMES[args[0]], station_np=a_cell(args[0]))
    elif kind == "missing_fail":
        v.update(item_np=np_item(args[0]), missing_np=np_item(args[1]))
    elif kind == "steps":
        v["n"] = args[0]
    elif kind in ("blocked", "wasted"):
        v["times"] = times(args[0])
    elif kind == "needed":
        v["item_np"] = np_item(args[0])
    elif kind == "used_up":
        mass = ITEM_NAMES[args[0]] in MASS
        v.update(item=ITEM_NAMES[args[0]] if mass else ITEM_NAMES[args[0]] + "s", it="it" if mass else "them")
    elif kind == "hint_recipe":
        rec = RECIPES[RECIPE_FOR[args[0]]]
        v.update(Item_np=np_item(args[0])[0].upper() + np_item(args[0])[1:],
                 inputs=" and ".join(np_item(j) for j, _ in rec.inputs),
                 where=f" at {a_cell(rec.station)}" if rec.station >= 0 else "")
    elif kind == "hint_raw":
        cell, tool = RAW[args[0]]
        v.update(item=ITEM_NAMES[args[0]], cell_np=a_cell(cell), with_tool=f" with {np_item(tool)}" if tool >= 0 else "")
    elif kind == "eval_count":
        v.update(k=args[0], m=args[1])
    return TEMPLATES[kind][variant].format(**v)


def _part_items(g):
    """Items the expert produces on the way to goal g from nothing, plus g's own item: what a summary of g covers."""
    items = set() if GOALS[g].item == CROSS else {GOALS[g].item}
    for op in plan(np.zeros(len(ITEM_NAMES), np.int64), [g], False, False):
        if op[0] == "craft":
            items.add(RECIPES[op[1]].output)
        elif op[0] == "collect":
            items.add(next(i for i, (c, _) in RAW.items() if c == op[1]))
        elif op[0] == "bridge":
            items.add(BRIDGE_ITEM)
    return items


PART_ITEMS = {g: _part_items(g) for g in range(len(GOALS))}
SUMMARY_P = 0.5           # chance that an achieved, unrequested sub-goal is described by its summary, not its parts


def achieved(f: Facts, g):
    """Sub-goal g was reached during the episode: its item obtained `count` times in total, or the river crossed."""
    goal = GOALS[g]
    return f.crossed if goal.item == CROSS else f.collected[goal.item] + f.crafted[goal.item] >= goal.count


def claims(f: Facts, rng, summary_p=SUMMARY_P):
    """All true claims about the episode, grouped by intent: {intent: [claim tuple, ...]}.

    Mixed granularity: each achieved sub-goal that was not requested is, with probability summary_p, described by its
    summary ("You managed to get some metal.") instead of its parts -- the event claims about the items it covers are
    dropped. Deeper sub-goals are considered first; a sub-goal already covered by a summary is not summarized again."""
    out = {k: [] for k in INTENTS}
    covered = set()
    for g in sorted(range(len(GOALS)), key=lambda g: -GOALS[g].depth):
        if g in f.goals or not achieved(f, g):
            continue
        own = PART_ITEMS[g] if GOALS[g].item == CROSS else {GOALS[g].item}
        if own <= covered or rng.random() >= summary_p:
            continue
        out["descriptive"].append(("subgoal", g))
        covered |= PART_ITEMS[g]
    for item, c in sorted(f.collected.items()):
        if item not in covered:
            out["descriptive"].append(("collected", item, c))
    for item, c in sorted(f.crafted.items()):
        if item not in covered:
            out["descriptive"].append(("crafted", item, c))
    if f.bridged:
        out["descriptive"].append(("bridged",))
    if f.crossed:
        out["descriptive"].append(("crossed",))
    if not f.collected and not f.crafted:
        out["descriptive"].append(("nothing",))
    for g, ok in zip(f.goals, f.satisfied):
        out["confirm"].append(("goal_done" if ok else "goal_not_done", g))
    for tool in sorted(f.use_fail_tool):
        out["diagnostic"].append(("use_fail", tool))
    for station, c in sorted(f.craft_fail_station.items()):
        out["diagnostic"].append(("station_fail", station, c))
    missing = sorted(f.craft_fail_missing.items(), key=lambda kv: -kv[1])[:MAX_MISSING_CLAIMS]  # summarize
    for (item, miss), _ in missing:
        out["diagnostic"].append(("missing_fail", item, miss))
    out["process"].append(("steps", f.steps))
    if f.blocked > BLOCKED_MIN:
        out["process"].append(("blocked", f.blocked))
    if f.wasted_uses > WASTED_MIN:
        out["process"].append(("wasted", f.wasted_uses))
    for g, item in f.next_needed.items():
        out["corrective"].append(("needed", item))
        out["hint"].append(("hint_recipe", item) if item in RECIPE_FOR else ("hint_raw", item))
    k, m = sum(f.satisfied), len(f.satisfied)
    out["evaluative"].append(("eval_count", k, m))
    out["evaluative"].append(("eval_all",) if k == m else ("eval_not_all",))
    if f.use_fail_tool:
        out["vague"].append(("vague_tools",))
    if f.craft_fail_station or f.craft_fail_missing:
        out["vague"].append(("vague_crafting",))
    out["chatter"].append(("chatter",))
    return out


def mentions(claim, g):
    """The claim speaks about goal g's outcome: its item collected/crafted, g summarized or confirmed done, or (for the
    river) bridging/crossing."""
    item = GOALS[g].item
    kind = claim[0]
    if kind in ("collected", "crafted"):
        return claim[1] == item
    if kind in ("subgoal", "goal_done"):
        return claim[1] == g
    if kind in ("bridged", "crossed"):
        return item == CROSS
    return False


SIGNIFICANT = (EV_COLLECT, EV_CRAFT, EV_BRIDGE, EV_CRAFT_FAIL, EV_USE_FAIL)


def event_arrays(log):
    """Per-step event fields of a rollout as (T, B) arrays (built once per rollout for claim_steps)."""
    return {k: np.stack([ev[k] for ev in log]) for k in ("type", "item", "tool", "station", "missing")}


def claim_steps(log, b, claim):
    """Time steps of episode b that a claim is about, or None for claims about the whole episode. For an event
    claim: each matching event step plus the steps leading to it since the previous significant event (the approach,
    e.g. walking to the tree before chopping it). `log` is a rollout's event log or its event_arrays."""
    ev = event_arrays(log) if isinstance(log, list) else log
    if len(ev["type"]) == 0:
        return None
    typ = ev["type"][:, b]
    T = len(typ)
    get = lambda k: ev[k][:, b]
    kind = claim[0]
    if kind == "collected":
        hit = (typ == EV_COLLECT) & (get("item") == claim[1])
    elif kind == "crafted":
        hit = (typ == EV_CRAFT) & (get("item") == claim[1])
    elif kind == "bridged":
        hit = typ == EV_BRIDGE
    elif kind == "use_fail":
        hit = (typ == EV_USE_FAIL) & (get("tool") == claim[1])
    elif kind == "station_fail":
        hit = (typ == EV_CRAFT_FAIL) & (get("station") == claim[1])
    elif kind == "missing_fail":
        hit = (typ == EV_CRAFT_FAIL) & (get("item") == claim[1]) & (get("missing") == claim[2])
    elif kind == "subgoal":
        hit = (np.isin(typ, (EV_COLLECT, EV_CRAFT)) & np.isin(get("item"), list(PART_ITEMS[claim[1]]))) | (
            (typ == EV_BRIDGE) & (GOALS[claim[1]].item == CROSS))
    elif kind == "crossed":
        br = np.nonzero(typ == EV_BRIDGE)[0]
        return np.arange(br[-1] + 1 if len(br) else 0, T)
    else:
        return None
    events = np.nonzero(hit)[0]
    if len(events) == 0:
        return None
    sig = np.nonzero(np.isin(typ, SIGNIFICANT))[0]
    starts = np.searchsorted(sig, events)                   # previous significant event = sig[starts - 1]
    lo = np.where(starts > 0, sig[np.maximum(starts - 1, 0)] + 1, 0)
    return np.unique(np.concatenate([np.arange(l, e + 1) for l, e in zip(lo, events)]))


PROFILES = {
    # intent -> probability that each candidate claim of that intent is voiced
    "reporter": dict(descriptive=0.8, confirm=0.3, process=0.2),
    "diagnoser": dict(descriptive=0.4, diagnostic=0.9, confirm=0.3),
    "coach": dict(corrective=0.8, hint=0.6, confirm=0.4, evaluative=0.3),
    "grader": dict(confirm=0.9, evaluative=0.5),
    "describe": dict(descriptive=1.0),          # every outcome claim, nothing else (sanity checks)
    "realistic": dict(),                         # structured like human feedback: see compose_realistic
    "mixed": dict(descriptive=0.5, confirm=0.4, diagnostic=0.5, process=0.3, corrective=0.4, hint=0.3,
                  evaluative=0.3, vague=0.3, chatter=0.3),
}


def _say(intent, claim, rng):
    return intent, claim, render(claim[0], claim, int(rng.integers(len(TEMPLATES[claim[0]]))))


def compose_realistic(f: Facts, rng):
    """A message structured the way people give feedback: the outcome first, then on failure the one or two things
    that matter (what was used up or still missing, how far the agent got, its most frequent mistake), on success
    brief praise and sometimes a summary; chatter and step counts only occasionally. Never empty."""
    out = []
    done = [g for g, ok in zip(f.goals, f.satisfied) if ok]
    unmet = [g for g, ok in zip(f.goals, f.satisfied) if not ok]
    # outcome
    if not unmet:
        out.append(_say("evaluative", ("eval_all",), rng))
        if len(f.goals) > 1 and rng.random() < 0.5:
            out.append(_say("evaluative", ("eval_count", len(done), len(f.goals)), rng))
        if rng.random() < 0.5:                                   # brief summary of what was done
            g = max(f.goals, key=lambda g: GOALS[g].depth)
            out.append(_say("confirm", ("goal_done", g), rng))
    else:
        for g in unmet:
            out.append(_say("confirm", ("goal_not_done", g), rng))
        for g in done:
            if rng.random() < 0.5:
                out.append(_say("confirm", ("goal_done", g), rng))
        # what went wrong, for up to two unmet goals
        for g in unmet[:2]:
            item = GOALS[g].item
            if item != CROSS and achieved(f, g) and f.final_inv[item] < GOALS[g].count:
                out.append(_say("corrective", ("used_up", item), rng))
            elif g in f.next_needed:
                need = f.next_needed[g]
                hint = ("hint_recipe", need) if need in RECIPE_FOR else ("hint_raw", need)
                out.append(_say("hint", hint, rng) if rng.random() < 0.3 else _say("corrective", ("needed", need), rng))
        # how far the agent got: the deepest sub-goal reached (requested ones are already covered above)
        reached = [g for g in range(len(GOALS)) if achieved(f, g) and g not in f.goals]
        if reached and rng.random() < 0.6:
            out.append(_say("descriptive", ("subgoal", max(reached, key=lambda g: GOALS[g].depth)), rng))
        # its most frequent mistake
        fails = [(c, ("station_fail", st, c)) for st, c in f.craft_fail_station.items()]
        fails += [(c, ("missing_fail", it, ms)) for (it, ms), c in f.craft_fail_missing.items()]
        fails += [(c, ("use_fail", tool)) for tool, c in f.use_fail_tool.items()]
        if fails and rng.random() < 0.4:
            out.append(_say("diagnostic", max(fails)[1], rng))
    if rng.random() < 0.1:
        out.append(_say("process", ("steps", f.steps), rng))
    if rng.random() < 0.2:
        out.append(_say("chatter", ("chatter",), rng))
    return out


def compose(f: Facts, profile, rng):
    """Voiced claims of one message as (intent, claim, sentence), in random order (structured for 'realistic')."""
    if profile == "realistic":
        return compose_realistic(f, rng)
    probs = PROFILES[profile]
    picked = []
    for intent, cs in claims(f, rng).items():
        p = probs.get(intent, 0.0)
        for c in dict.fromkeys(cs):                           # deduplicate identical claims
            if rng.random() < p:
                picked.append((intent, c, render(c[0], c, int(rng.integers(len(TEMPLATES[c[0]]))))))
    rng.shuffle(picked)
    return picked


def feedback(f: Facts, profile, rng):
    """One feedback message; 'Okay.' (a chatter claim) if nothing else is voiced."""
    parts = compose(f, profile, rng)
    return " ".join(s for _, _, s in parts) if parts else "Okay."
