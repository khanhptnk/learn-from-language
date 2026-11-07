"""Template teacher for MiniCraft: episode event log -> facts -> claims -> mixed-intent feedback text.

Feedback talks about low-level events and mechanics (wood, planks, workbench, pickaxes, failed crafts), mostly not in
the request's vocabulary. Each sentence realizes one structured *claim* through a template; `gft.minicraft.verify`
parses sentences back into claims and checks them against a replay of the trajectory, so accuracy is testable.
A profile sets how likely each intent is to be voiced.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gft.minicraft.core import (CELL_NAMES, EV_BLOCKED, EV_COLLECT, EV_CRAFT, EV_CRAFT_FAIL, EV_PUT, EV_TAKE,
                                EV_USE_FAIL, EV_USE_NOTHING, ITEM_NAMES, N_ITEMS, RECIPE_FOR, RECIPES)
from gft.minicraft.expert import ITEM_DEPTH, RAW, plan
from gft.minicraft.goals import GOALS

INTENTS = ("descriptive", "confirm", "diagnostic", "process", "corrective", "hint", "evaluative", "vague", "chatter")
MASS = {"wood", "fiber", "clay", "stone", "rope"}
TOOL_TARGET = {"pickaxe": "a rock"}
BLOCKED_MIN, WASTED_MIN = 10, 5


@dataclass
class Facts:
    goals: list                      # goal ids of the request
    satisfied: list                  # bool per goal
    collected: Counter = field(default_factory=Counter)
    crafted: Counter = field(default_factory=Counter)
    use_fail_tool: Counter = field(default_factory=Counter)       # tool (or bridge item) that was missing
    craft_fail_station: Counter = field(default_factory=Counter)  # station stood by with nothing to make there
    craft_fail_nostation: int = 0                                 # crafts attempted away from any station
    steps: int = 0
    blocked: int = 0
    wasted_uses: int = 0
    next_needed: dict = field(default_factory=dict)               # unmet goal -> first item still needed
    missing: dict = field(default_factory=dict)                   # unmet goal -> all items its plan still produces
    final_inv: tuple = ()                                         # inventory at the end of the episode
    events: list = field(default_factory=list)                    # narration candidates (claim, t0, t1), in time order


def episode_facts(log, b, env, goals, satisfied):
    """Facts of episode b from a log of per-step event dicts (as returned by MiniCraft.step)."""
    ev = {k: np.stack([e[k][b] for e in log])[:, None] for k in EVENT_FIELDS} if log else None
    return _facts(ev, 0, goals, satisfied, int(env.t[b]), env.inv[b])


def episode_facts_all(log, env, goals, satisfied):
    """Facts of every episode of a rollout from the event arrays (one pass over numpy columns per episode instead of
    a Python loop over every step of every episode)."""
    ev = event_arrays(log) if log else None
    return [_facts(ev, b, goals[b], satisfied[b], int(env.t[b]), env.inv[b]) for b in range(env.B)]


def slot_steps(typ, item, station):
    """(T, N_ITEMS): the contents of the slot of the station each step acts on, before the step (zeros for steps at no
    station), replayed from one episode's events: put adds its item, take and a successful craft empty the slot."""
    slots, out = {}, np.zeros((len(typ), N_ITEMS), np.int64)
    for t in np.nonzero(station >= 0)[0]:
        s = slots.setdefault(int(station[t]), np.zeros(N_ITEMS, np.int64))
        out[t] = s
        if typ[t] == EV_PUT:
            s[item[t]] += 1
        elif typ[t] in (EV_TAKE, EV_CRAFT):
            s[:] = 0
    return out


def _facts(ev, b, goals, satisfied, steps, inv):
    f = Facts(goals=[int(g) for g in goals if g >= 0], satisfied=[bool(s) for s, g in zip(satisfied, goals) if g >= 0])
    if ev is not None:
        typ = ev["type"][:, b]
        item = ev["item"][:, b]
        for it, c in zip(*np.unique(item[typ == EV_COLLECT], return_counts=True)):
            f.collected[int(it)] = int(c)
        for it, c in zip(*np.unique(item[typ == EV_CRAFT], return_counts=True)):
            f.crafted[int(it)] = int(c)
        for tl, c in zip(*np.unique(ev["tool"][:, b][typ == EV_USE_FAIL], return_counts=True)):
            f.use_fail_tool[int(tl)] = int(c)
        cf = typ == EV_CRAFT_FAIL
        st = ev["station"][:, b][cf]
        for s_, c in zip(*np.unique(st[st >= 0], return_counts=True)):
            f.craft_fail_station[int(s_)] = int(c)
        f.craft_fail_nostation = int((st < 0).sum())
        f.blocked = int((typ == EV_BLOCKED).sum())
        f.wasted_uses = int((typ == EV_USE_NOTHING).sum())
        _fill_events(f, ev, b)
    f.steps = steps
    f.final_inv = tuple(int(x) for x in inv)
    _fill_missing(f, inv)
    return f


def _fill_events(f: Facts, ev, b):
    """Narration candidates of episode b with the steps they are about: every item collected and crafted (with its
    count), each input that went into something made ("used_on"), and each failed craft at a station with what its
    slot held then, with counts ("craft_held": 2 planks make a pickaxe, 1 or 3 do not). Spans come from claim_steps;
    sorted by their last step."""
    typ, item, station = ev["type"][:, b], ev["item"][:, b], ev["station"][:, b]
    cands = [("collected", it, c) for it, c in sorted(f.collected.items())]
    cands += [("crafted", it, c) for it, c in sorted(f.crafted.items())]
    for out_item in sorted(f.crafted):
        for j, _ in RECIPES[RECIPE_FOR[out_item]].inputs:
            cands.append(("used_on", j, out_item))
    slot = slot_steps(typ, item, station)          # failed crafts with what the slot held then
    seen = set()
    for t in np.nonzero((typ == EV_CRAFT_FAIL) & (station >= 0))[0]:
        held = tuple((int(i), int(slot[t, i])) for i in np.nonzero(slot[t] > 0)[0])
        if held and (int(station[t]), held) not in seen:
            seen.add((int(station[t]), held))
            cands.append(("craft_held", int(station[t]), held))
    events = []
    for c in cands:
        span = claim_steps(ev, b, c)
        if span is not None and len(span):
            events.append((c, int(span[0]), int(span[-1])))
    f.events = sorted(events, key=lambda x: (x[2], x[1]))


def _fill_missing(f: Facts, inv):
    """next_needed / missing of every unmet goal from the expert's plan at the end state."""
    for g, ok in zip(f.goals, f.satisfied):
        if ok:
            continue
        ops = plan(inv, [g])
        item = _first_missing_item(ops)
        if item is not None:
            f.next_needed[g] = item
        items = []
        for op in ops:
            x = _first_missing_item([op])
            if x is not None and x not in items:
                items.append(x)
        if items:
            f.missing[g] = items


def _first_missing_item(ops):
    """Item produced by the first op of a plan (what the agent lacked first), or None for an empty plan."""
    if not ops:
        return None
    op = ops[0]
    if op[0] == "craft":
        return RECIPES[op[1]].output
    return next(item for item, (cell, _) in RAW.items() if cell == op[1])


# ---------------------------------------------------------------- noun phrases

def np_item(item):
    """'a plank', 'an oven', 'wood' (mass nouns take no article)."""
    name = ITEM_NAMES[item]
    return name if name in MASS else ("an " if name[0] in "aeiou" else "a ") + name


def n_item(count, item):
    """'a plank', '3 planks', 'wood', '5 wood'."""
    name = ITEM_NAMES[item]
    if count == 1:
        return np_item(item)
    return f"{count} {name}" if name in MASS else f"{count} {name}s"


def times(c):
    return "once" if c == 1 else "twice" if c == 2 else f"{c} times"


def a_cell(cell):
    name = CELL_NAMES[cell]
    if name == "reeds":                              # plural: "the reeds"
        return "the " + name
    return ("an " if name[0] in "aeiou" else "a ") + name


def recipe_inputs(rec):
    """'a plank and rope', '2 planks': one unit's inputs."""
    return " and ".join(np_item(j) if c == 1 else n_item(c, j) for j, c in rec.inputs)


SEQUENCE = ("First, ", "Then ", "After that, ", "Later ", "Next, ")   # narration prefixes (verify.strip_prefix undoes them)


def sequenced(sentence, k, rng):
    """Sentence k of a narration with a sequence word in front ("Then you made ...")."""
    pre = SEQUENCE[0] if k == 0 else SEQUENCE[1 + int(rng.integers(len(SEQUENCE) - 1))]
    return pre + sentence[0].lower() + sentence[1:]


# ---------------------------------------------------------------- templates: claim kind -> sentence variants
# Placeholders are filled by `render`; `gft.minicraft.verify` compiles the same templates into parsers.

TEMPLATES = {
    "collected": ["You collected {n_item}.", "You picked up {n_item}.", "You picked up {n_item} at {cell_np}.",
                  "You got {n_item} from {cell_np}.", "At {cell_np} you collected {n_item}.",
                  "You took {n_item} from {cell_np}."],
    "crafted": ["You made {n_item}.", "You crafted {n_item}.", "You made {n_item} at {station_np}.",
                "You made {n_item} at {station_np} from {inputs}.", "At {station_np} you turned {inputs} into {n_item}.",
                "You crafted {n_item} at {station_np}, using {inputs}.", "You used {station_np} to make {n_item} from {inputs}."],
    "used_on": ["You had {in_kind} but used {it} on {out_np}.", "You got {in_kind}, then turned {it} into {out_np}.",
                "Your {in_kind} went into {out_np}.", "The {in_kind} you got became {out_np}."],
    "craft_held": ["You pressed craft at {station_np} with only {held} in it, so nothing happened.",
                   "At {station_np} nothing could be made from {held}.",
                   "You put {held} in {station_np}, which makes nothing."],
    "nothing": ["You did not collect or make anything.", "Nothing was collected or made."],
    "goal_done": ["You did {phrase}.", "Done: {phrase}."],
    "subgoal": ["Along the way you managed to {phrase}.", "You managed to {phrase}."],
    "used_up": ["You used up the {item} you needed.", "You had {some} at some point but did not keep {it}.",
                "You got {some} but used {it} up."],
    "goal_not_done": ["You did not {phrase}.", "Not done: {phrase}.", "You never managed to {phrase}.",
                      "Still to do: {phrase}."],
    "use_fail": ["You tried to work {target} without {tool_np}.", "{Target} would not give way: you had no {tool}."],
    "station_fail": ["Crafting failed {times} because no station was next to you.",
                     "You pressed craft {times} away from any station."],
    "nothing_fail": ["Crafting at {station_np} failed {times}: what you put in made nothing.",
                     "You pressed craft {times} at {station_np} with nothing to make."],
    "steps": ["You took {n} steps.", "That run lasted {n} steps."],
    "blocked": ["You bumped into things {times}."],
    "wasted": ["You used empty ground {times}."],
    "needed": ["You still needed {item_np}.", "You should have gotten {item_np} first.", "{Item_np} was still missing.",
               "Without {item_np} this was not going to work.", "Next time get {item_np} before anything else."],
    # an unmet goal whose item the agent holds, but too few of it (at the end), or all of it -- the episode just did
    # not end by DONE (WorldConfig.success_needs_done); both name the goal, so a reader can tell what finishes it
    "short": ["You had {have} of the {count} {items} needed to {phrase}.",
              "You needed {count} {items} to {phrase} but had only {have}."],
    "no_done": ["You had what you needed to {phrase} but never said you were done.",
                "You never said you were done, so {phrase} did not count."],
    "hint_recipe": ["{Item_np} is made from {inputs}{where}.", "To make {item_np} you need {inputs}{where}."],
    "hint_raw": ["You get {item} from {cell_np}{with_tool}.", "You can get {item} from {cell_np}{with_tool}."],
    "eval_count": ["{k} of {m} done."],
    "eval_all": ["Great job!", "Nice work!", "That worked."],       # not "Well done!": "well" is a held-out item's name
    "eval_not_all": ["Not quite there."],
    "vague_tools": ["Something went wrong with your tools."],
    "vague_crafting": ["Some of your crafting was off."],
    "chatter": ["Thanks!", "Thanks for trying.", "Okay."],
}


def render(kind, claim, variant):
    """Sentence for a claim (kind, *args) using template `variant`."""
    args = claim[1:]
    v = {}
    if kind == "collected":
        v.update(n_item=n_item(args[1], args[0]), cell_np=a_cell(RAW[args[0]][0]))
    elif kind == "crafted":
        rec = RECIPES[RECIPE_FOR[args[0]]]
        v.update(n_item=n_item(args[1], args[0]), station_np=a_cell(rec.station), inputs=recipe_inputs(rec))
    elif kind == "used_on":                       # (input item, item it went into)
        name = ITEM_NAMES[args[0]]
        v.update(in_kind=item_kind_np(args[0]), it="it" if name in MASS else "them", out_np=np_item(args[1]))
    elif kind == "craft_held":                    # (station, ((item, count), ...) in its slot)
        v.update(station_np=a_cell(args[0]), held=and_list([n_item(c, i) for i, c in args[1]]))
    elif kind in ("goal_done", "goal_not_done", "subgoal"):
        v["phrase"] = GOALS[args[0]].phrase
    elif kind == "use_fail":
        tool = ITEM_NAMES[args[0]]
        v.update(target=TOOL_TARGET[tool], Target=TOOL_TARGET[tool].capitalize(), tool=tool, tool_np=np_item(args[0]))
    elif kind == "station_fail":
        v["times"] = times(args[0])
    elif kind == "nothing_fail":
        v.update(times=times(args[1]), station_np=a_cell(args[0]))
    elif kind == "steps":
        v["n"] = args[0]
    elif kind in ("blocked", "wasted"):
        v["times"] = times(args[0])
    elif kind == "needed":
        v.update(item_np=np_item(args[0]), Item_np=np_item(args[0])[0].upper() + np_item(args[0])[1:])
    elif kind == "used_up":                       # singular when the goal takes one ("the stick", "a stick")
        name = ITEM_NAMES[args[0]]
        one = max((g.count for g in GOALS if g.item == args[0]), default=2) == 1
        v.update(item=name if name in MASS or one else name + "s",
                 some=name if name in MASS else np_item(args[0]) if one else name + "s",
                 it="it" if name in MASS or one else "them")
    elif kind == "short":                         # (goal, how many of its item were held at the end)
        goal = GOALS[args[0]]
        name = ITEM_NAMES[goal.item]
        v.update(have=args[1], count=goal.count, items=name if name in MASS else name + "s", phrase=goal.phrase)
    elif kind == "no_done":
        v["phrase"] = GOALS[args[0]].phrase
    elif kind == "hint_recipe":
        rec = RECIPES[RECIPE_FOR[args[0]]]
        v.update(Item_np=np_item(args[0])[0].upper() + np_item(args[0])[1:], item_np=np_item(args[0]),
                 inputs=recipe_inputs(rec), where=f" at {a_cell(rec.station)}" if rec.station >= 0 else "")
    elif kind == "hint_raw":
        cell, tool = RAW[args[0]]
        v.update(item=ITEM_NAMES[args[0]], cell_np=a_cell(cell), with_tool=f" with {np_item(tool)}" if tool >= 0 else "")
    elif kind == "eval_count":
        v.update(k=args[0], m=args[1])
    return TEMPLATES[kind][variant].format(**v)


def _part_items(g):
    """Items the expert produces on the way to goal g from nothing, plus g's own item: what a summary of g covers."""
    items = {GOALS[g].item}
    for op in plan(np.zeros(len(ITEM_NAMES), np.int64), [g]):
        items.add(RECIPES[op[1]].output if op[0] == "craft" else next(i for i, (c, _) in RAW.items() if c == op[1]))
    return items


PART_ITEMS = {g: _part_items(g) for g in range(len(GOALS))}
SUMMARY_P = 0.5           # chance that an achieved, unrequested sub-goal is described by its summary, not its parts


def held(f: Facts, item):
    """How many of `item` the agent held at the end (0 for facts without a final inventory)."""
    return f.final_inv[item] if f.final_inv else 0


def achieved(f: Facts, g):
    """Sub-goal g was reached during the episode: its item obtained `count` times in total."""
    goal = GOALS[g]
    return f.collected[goal.item] + f.crafted[goal.item] >= goal.count


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
        own = {GOALS[g].item}
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
    if not f.collected and not f.crafted:
        out["descriptive"].append(("nothing",))
    for g, ok in zip(f.goals, f.satisfied):
        out["confirm"].append(("goal_done" if ok else "goal_not_done", g))
    for tool in sorted(f.use_fail_tool):
        out["diagnostic"].append(("use_fail", tool))
    if f.craft_fail_nostation:
        out["diagnostic"].append(("station_fail", f.craft_fail_nostation))
    for station, c in sorted(f.craft_fail_station.items()):
        out["diagnostic"].append(("nothing_fail", station, c))
    out["process"].append(("steps", f.steps))
    if f.blocked > BLOCKED_MIN:
        out["process"].append(("blocked", f.blocked))
    if f.wasted_uses > WASTED_MIN:
        out["process"].append(("wasted", f.wasted_uses))
    for g, ok in zip(f.goals, f.satisfied):
        h, goal = held(f, GOALS[g].item), GOALS[g]
        if not ok and h >= goal.count:
            out["corrective"].append(("no_done", g))
        elif not ok and 0 < h and not achieved(f, g):
            out["corrective"].append(("short", g, h))
    said = set()
    for g, items in f.missing.items():
        for item in items:
            if item not in said:
                said.add(item)
                out["corrective"].append(("needed", item))
                out["hint"].append(("hint_recipe", item) if item in RECIPE_FOR else ("hint_raw", item))
    k, m = sum(f.satisfied), len(f.satisfied)
    out["evaluative"].append(("eval_count", k, m))
    out["evaluative"].append(("eval_all",) if k == m else ("eval_not_all",))
    if f.use_fail_tool:
        out["vague"].append(("vague_tools",))
    if f.craft_fail_station or f.craft_fail_nostation:
        out["vague"].append(("vague_crafting",))
    out["chatter"].append(("chatter",))
    return out


def mentions(claim, g):
    """The claim speaks about goal g's outcome: its item collected/crafted, g summarized or confirmed done."""
    item = GOALS[g].item
    kind = claim[0]
    if kind in ("collected", "crafted"):
        return claim[1] == item
    if kind in ("subgoal", "goal_done"):
        return claim[1] == g
    return False


SIGNIFICANT = (EV_COLLECT, EV_CRAFT, EV_CRAFT_FAIL, EV_USE_FAIL)


EVENT_FIELDS = ("type", "item", "tool", "station")


def event_arrays(log):
    """Per-step event fields of a rollout as (T, B) arrays (built once per rollout for claim_steps)."""
    return {k: np.stack([ev[k] for ev in log]) for k in EVENT_FIELDS}


def claim_steps(log, b, claim):
    """Time steps of episode b that a claim is about, or None for claims about the whole episode. For an event
    claim: each matching event step plus the steps leading to it since the previous significant event (the approach,
    e.g. walking to the tree before chopping it). `log` is a rollout's event log or its event_arrays. A craft_held
    claim picks only the failures whose slot held exactly its items."""
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
    elif kind == "use_fail":
        hit = (typ == EV_USE_FAIL) & (get("tool") == claim[1])
    elif kind == "station_fail":
        hit = (typ == EV_CRAFT_FAIL) & (get("station") < 0)
    elif kind == "nothing_fail":
        hit = (typ == EV_CRAFT_FAIL) & (get("station") == claim[1])
    elif kind == "subgoal":
        hit = np.isin(typ, (EV_COLLECT, EV_CRAFT)) & np.isin(get("item"), list(PART_ITEMS[claim[1]]))
    elif kind == "used_on":                                 # from getting the input to the craft that consumed it
        got = np.nonzero(np.isin(typ, (EV_COLLECT, EV_CRAFT)) & (get("item") == claim[1]))[0]
        made = np.nonzero((typ == EV_CRAFT) & (get("item") == claim[2]))[0]
        if len(got) == 0 or len(made) == 0 or made[-1] < got[0]:
            return None
        hit = np.zeros(T, bool)
        hit[got[0]] = hit[made[made >= got[0]][0]] = True
        events = np.nonzero(hit)[0]
        return np.arange(_approach_start(typ, events[0]), events[-1] + 1)
    elif kind == "craft_held":
        hit = (typ == EV_CRAFT_FAIL) & (get("station") == claim[1])
        held = np.zeros(N_ITEMS, np.int64)
        for i, c in claim[2]:
            held[i] = c
        hit &= (slot_steps(typ, get("item"), get("station")) == held).all(1)
    else:
        return None
    events = np.nonzero(hit)[0]
    if len(events) == 0:
        return None
    sig = np.nonzero(np.isin(typ, SIGNIFICANT))[0]
    starts = np.searchsorted(sig, events)                   # previous significant event = sig[starts - 1]
    lo = np.where(starts > 0, sig[np.maximum(starts - 1, 0)] + 1, 0)
    return np.unique(np.concatenate([np.arange(l, e + 1) for l, e in zip(lo, events)]))


def _approach_start(typ, event):
    """First step of the approach to `event`: the step after the previous significant event."""
    prev = np.nonzero(np.isin(typ[:event], SIGNIFICANT))[0]
    return int(prev[-1]) + 1 if len(prev) else 0


PROFILES = {
    # intent -> probability that each candidate claim of that intent is voiced
    "reporter": dict(descriptive=0.8, confirm=0.3, process=0.2),
    "diagnoser": dict(descriptive=0.4, diagnostic=0.9, confirm=0.3),
    "coach": dict(corrective=0.8, hint=0.6, confirm=0.4, evaluative=0.3),
    "grader": dict(confirm=0.9, evaluative=0.5),
    "describe": dict(descriptive=1.0),          # every outcome claim, nothing else (sanity checks)
    "complete": dict(),                          # every item collected / crafted with counts, no summaries, fixed
                                                 # wording and order: feedback that pins down what happened
    "realistic": dict(),                         # structured like human feedback: see compose_realistic
    "verdict": dict(),                           # the realistic teacher's praise and verdicts only (no causes,
                                                 # hints or narration): the language ablation
    "mixed": dict(descriptive=0.5, confirm=0.4, diagnostic=0.5, process=0.3, corrective=0.4, hint=0.3,
                  evaluative=0.3, vague=0.3, chatter=0.3),
}


def _say(intent, claim, rng):
    return intent, claim, render(claim[0], claim, int(rng.integers(len(TEMPLATES[claim[0]]))))


def compose_realistic(f: Facts, rng):
    """A message structured the way people give feedback: the outcome first, then on failure the one or two things
    that matter (what was used up or still missing, how far the agent got, its most frequent mistake), on success
    praise and what was done (always the deepest goal, the others and the sub-goals reached on the way sometimes);
    chatter and step counts only occasionally. A goal's own item is never "still needed" (that restates the goal):
    how it is made or found is said instead. Narration dwells on the request's items. Never empty."""
    out = []
    done = [g for g, ok in zip(f.goals, f.satisfied) if ok]
    unmet = [g for g, ok in zip(f.goals, f.satisfied) if not ok]
    # outcome
    if not unmet:
        out.append(_say("evaluative", ("eval_all",), rng))
        if len(f.goals) > 1 and rng.random() < 0.5:
            out.append(_say("evaluative", ("eval_count", len(done), len(f.goals)), rng))
        for g in sorted(f.goals, key=lambda g: -GOALS[g].depth):  # what was done: the deepest goal always, others sometimes
            deepest = g == max(f.goals, key=lambda g: GOALS[g].depth)
            if deepest or rng.random() < 0.3:
                out.append(_say("confirm", ("goal_done", g), rng))
        reached = sorted([g for g in range(len(GOALS)) if achieved(f, g) and g not in f.goals],
                         key=lambda g: -GOALS[g].depth)
        for g in reached[:2]:                                    # how it got there: up to two sub-goals on the way
            if rng.random() < 0.5:
                out.append(_say("descriptive", ("subgoal", g), rng))
    else:
        for g in unmet:
            # "You did not make some planks." is false of an agent that made them and used them up, or holds some or
            # all of them: say "Not done" (the remark below explains it)
            kept_short = achieved(f, g) or held(f, GOALS[g].item) > 0
            out.append(("confirm", ("goal_not_done", g), render("goal_not_done", ("goal_not_done", g), 1))
                       if kept_short else _say("confirm", ("goal_not_done", g), rng))
        for g in done:
            if rng.random() < 0.5:
                out.append(_say("confirm", ("goal_done", g), rng))
        # what went wrong, for up to two unmet goals (a remark two goals share is said once): that it was all there
        # but the run did not end by DONE, what was used up, how many of too few were held, or one of the things
        # still missing -- usually the next one, sometimes a later one
        said = set()
        for g in unmet[:2]:
            item = GOALS[g].item
            if held(f, item) >= GOALS[g].count:
                part = _say("corrective", ("no_done", g), rng)
            elif achieved(f, g) and held(f, item) < GOALS[g].count:
                part = _say("corrective", ("used_up", item), rng)
            elif held(f, item) > 0:
                part = _say("corrective", ("short", g, held(f, item)), rng)
            elif g in f.missing:
                items = f.missing[g]
                later = [x for x in items[1:] if f.final_inv[x] == 0]   # a later step, of something not in hand
                need = items[0] if not later or rng.random() < 0.5 else later[int(rng.integers(len(later)))]
                hint = ("hint_recipe", need) if need in RECIPE_FOR else ("hint_raw", need)
                if need == item or rng.random() < 0.3:          # the goal's own item: how it is made, not "still
                    part = _say("hint", hint, rng)               # needed" (which restates the goal)
                elif f.collected[need] + f.crafted[need] > 0:   # it got some: not "you should have gotten X first"
                    v = (0, 2, 3)[int(rng.integers(3))]
                    part = ("corrective", ("needed", need), render("needed", ("needed", need), v))
                else:
                    part = _say("corrective", ("needed", need), rng)
            else:
                continue
            if part[1] not in said:
                said.add(part[1])
                out.append(part)
        # how far the agent got: the deepest sub-goal reached (requested ones are already covered above)
        reached = [g for g in range(len(GOALS)) if achieved(f, g) and g not in f.goals]
        if reached and rng.random() < 0.6:
            out.append(_say("descriptive", ("subgoal", max(reached, key=lambda g: GOALS[g].depth)), rng))
        # its most frequent mistake
        fails = [(c, ("nothing_fail", st, c)) for st, c in f.craft_fail_station.items()]
        fails += [(f.craft_fail_nostation, ("station_fail", f.craft_fail_nostation))] if f.craft_fail_nostation else []
        fails += [(c, ("use_fail", tool)) for tool, c in f.use_fail_tool.items()]
        if fails and rng.random() < 0.4:
            out.append(_say("diagnostic", max(fails)[1], rng))
    out += narrate(f, rng, (1, 3) if not unmet else (1, 2), focus=set().union(*(PART_ITEMS[g] for g in f.goals)))
    if rng.random() < 0.1:
        out.append(_say("process", ("steps", f.steps), rng))
    if rng.random() < 0.2:
        out.append(_say("chatter", ("chatter",), rng))
    return out


NARRATE_WEIGHT = {"collected": 1.0, "used_on": 1.5, "craft_held": 1.0}   # crafted: 1 + item depth
OFF_FOCUS = 0.2           # weight factor of an event about none of the request's items


def _event_items(c):
    """Items a narration candidate is about."""
    if c[0] == "craft_held":
        return {i for i, _ in c[2]}
    return {c[1], c[2]} if c[0] == "used_on" else {c[1]}


def narrate(f: Facts, rng, k_range=(1, 3), p_sequence=0.5, focus=None):
    """What happened, in the words a watcher uses: between k_range[0] and k_range[1] of the episode's notable events
    (f.events: things picked up and where, things made and where from what, inputs that went into something, failed
    crafts with what was held), weighted toward deep crafts, told in time order, sometimes with sequence words
    ("First, ... Then ..."). Events about none of the `focus` items (the request's, when given) weigh OFF_FOCUS times
    as much. Parts are (intent, claim, sentence); an empty list when nothing happened."""
    cands = f.events
    if not cands:
        return []
    w = np.array([1.0 + ITEM_DEPTH[c[1]] if c[0] == "crafted" else NARRATE_WEIGHT[c[0]] for c, _, _ in cands])
    if focus is not None:
        w *= [1.0 if _event_items(c) & focus else OFF_FOCUS for c, _, _ in cands]
    k = min(len(cands), int(rng.integers(k_range[0], k_range[1] + 1)))
    pick = sorted(rng.choice(len(cands), size=k, replace=False, p=w / w.sum()).tolist())   # time order
    seq = rng.random() < p_sequence and k > 1
    out = []
    for j, i in enumerate(pick):
        c = cands[i][0]
        if c[0] == "crafted":                  # "from <inputs>" names one unit's inputs: only for a single craft
            lo, hi = 2, len(TEMPLATES["crafted"]) if c[2] == 1 else 3
        elif c[0] == "collected":
            lo, hi = 2, len(TEMPLATES["collected"])
        else:
            lo, hi = 0, len(TEMPLATES[c[0]])
        s = render(c[0], c, int(rng.integers(lo, hi)))
        out.append(("descriptive", c, sequenced(s, j, rng) if seq else s))
    return out


def flip_toward_success(parts, rng):
    """Hard negative for a feedback message: the same (intent, claim, sentence) parts with ONE failure claim about a
    goal turned into its success counterpart -- "Not done: get some metal." -> "Done: get some metal." -- and the
    corrective / hint claims that explain that failure ("You still needed ore.") dropped. This is what the teacher
    would have said about a trajectory one fact better than the actual one. Returns None if nothing can be flipped."""
    unmet = [i for i, (_, c, _) in enumerate(parts) if c[0] == "goal_not_done"]
    if not unmet:
        return None
    i = unmet[int(rng.integers(len(unmet)))]
    g = parts[i][1][1]
    out = []
    for j, (intent, c, sent) in enumerate(parts):
        if j == i:
            out.append(_say("confirm", ("goal_done", g), rng))
        elif c[0] in ("needed", "used_up", "hint_recipe", "hint_raw", "nothing_fail", "use_fail", "station_fail"):
            continue                                   # explanations of failures: no longer warranted
        elif c[0] == "eval_not_all" and len(unmet) == 1:
            out.append(_say("evaluative", ("eval_all",), rng))
        else:
            out.append((intent, c, sent))
    return out


def improve_feedback(f: Facts, parts, prereqs, rng):
    """The feedback the teacher would give a trajectory ONE step better than the actual one (a local Stage II target,
    in place of the ideal message): for an unmet requested goal, the next sub-goal on its prerequisite chain that the
    episode did not reach becomes reached. If that next step is the goal itself, its "Not done" turns into "Done"
    (flip_toward_success); otherwise "You managed to <next sub-goal>." replaces the progress sentence. Explanations of
    the old failure (still needed, used up, hints, mistakes) are dropped -- they describe the worse trajectory.
    Successful episodes keep their feedback. prereqs: {goal: set of goals it requires} (learn.goal_prerequisites).
    Uses the teacher's message structure; with free-form feedback this edit would be an LLM rewrite."""
    unmet = [g for g, ok in zip(f.goals, f.satisfied) if not ok]
    if not unmet:
        return list(parts)
    g = unmet[int(rng.integers(len(unmet)))]
    todo = sorted((h for h in prereqs[g] if not achieved(f, h)), key=lambda h: GOALS[h].depth)
    if not todo:                                       # every prerequisite reached: the next step is the goal
        out = []
        for intent, c, sent in parts:
            if c == ("goal_not_done", g):
                out.append(_say("confirm", ("goal_done", g), rng))
            elif c[0] in ("needed", "used_up", "hint_recipe", "hint_raw", "nothing_fail", "use_fail", "station_fail"):
                continue
            elif c[0] == "eval_not_all" and len(unmet) == 1:
                out.append(_say("evaluative", ("eval_all",), rng))
            else:
                out.append((intent, c, sent))
        if len(unmet) == 1 and not any(c[0] == "eval_all" for _, c, _ in out):
            out.insert(0, _say("evaluative", ("eval_all",), rng))   # all requested goals now done
            out = [x for x in out if x[1][0] != "eval_not_all"]
        return out
    nxt = todo[0]
    out = [(intent, c, sent) for intent, c, sent in parts
           if c[0] not in ("needed", "used_up", "hint_recipe", "hint_raw", "nothing_fail", "use_fail", "station_fail",
                           "subgoal")]
    out.append(_say("descriptive", ("subgoal", nxt), rng))
    return out


def true_claims(f: Facts):
    """Every claim the teacher could truthfully make about the episode (all intents, parts and summaries), as a
    claim set -- the reference for "is this message true of this trajectory?"."""
    out = set()
    for summary_p in (0.0, 1.0):                   # with and without sub-goal summaries: both are true
        for cs in claims(f, np.random.default_rng(0), summary_p=summary_p).values():
            out |= claim_set([(None, c, None) for c in cs])
    out |= claim_set([(None, c, None) for c, _, _ in f.events])     # narration: what went into what, failed crafts
    return frozenset(out)


def claim_set(parts):
    """The claims of a message without wording or counts: what it asserts (for false-negative checks)."""
    out = set()
    for _, c, _ in parts:
        kind = c[0]
        if kind in ("collected", "crafted", "station_fail", "use_fail", "goal_done", "goal_not_done", "subgoal",
                    "needed", "hint_recipe", "hint_raw", "used_up"):
            out.add(c[:2])
        elif kind in ("nothing_fail", "used_on", "craft_held"):
            out.add(c[:3])
        elif kind not in ("chatter", "steps", "eval_count"):
            out.add((kind,))
    return frozenset(out)


TOOLS = {ITEM_NAMES[i] for i in range(N_ITEMS) if ITEM_NAMES[i] not in MASS and i >= 7}   # countable one-offs: "a pickaxe", "a raft"


def item_kind_np(item):
    """Count-free noun phrase: mass nouns bare ('wood'), tools with an article ('a wood pickaxe'), other countable
    materials in the plural ('planks', 'sticks')."""
    name = ITEM_NAMES[item]
    return name if name in MASS else np_item(item) if name in TOOLS else name + "s"


def and_list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def compose_descriptive(f: Facts, rng):
    """A message that always describes what was achieved, in the same words whether or not the request was met: the
    verdict ("Great job!" / "Not done: X" for each unmet goal), then which items were collected and which were made
    (without counts, in a fixed order, with a random paraphrase of the verb), what it ended with of each requested
    item ("You ended with enough stone." / "You ended with only 1 stone."), a bridge or crossing, then, on failure,
    what was still needed. Every success on a request gets about the same description and partial attempts
    get a subset of it, so the ideal feedback can be composed from sentences that partial attempts also receive."""
    unmet = [g for g, ok in zip(f.goals, f.satisfied) if not ok]
    out = [_say("evaluative", ("eval_all",), rng)] if not unmet else \
        [_say("confirm", ("goal_not_done", g), rng) for g in unmet]
    collected = [i for i in range(N_ITEMS) if f.collected[i] > 0]
    made = [i for i in range(N_ITEMS) if f.crafted[i] > 0]
    if collected:
        verb = ("collected", "picked up")[int(rng.integers(2))]
        out.append(("descriptive", ("collected_kinds", tuple(collected)),
                    f"You {verb} {and_list([item_kind_np(i) for i in collected])}."))
    if made:
        verb = ("made", "crafted")[int(rng.integers(2))]
        out.append(("descriptive", ("crafted_kinds", tuple(made)),
                    f"You {verb} {and_list([item_kind_np(i) for i in made])}."))
    # what it ended with of each requested item, relative to what the request needs ("enough stone" / "only 1 stone"):
    # the kinds above cannot tell one stone from enough, or sticks that were used up from sticks that were kept
    kept, short = [], []
    for g, ok in zip(f.goals, f.satisfied):
        goal = GOALS[g]
        name = ITEM_NAMES[goal.item]
        plural = name if name in MASS else name + "s"
        if ok:
            kept.append(f"enough {plural}" if goal.count > 1 else np_item(goal.item))
        else:
            have = f.final_inv[goal.item] if f.final_inv else 0
            short.append(f"only {n_item(have, goal.item)}" if have else f"no {plural}")
    for part in (kept, short):
        if part:
            out.append(("descriptive", ("ended_with", tuple(part)), f"You ended with {and_list(part)}."))
    for g in unmet[:2]:
        if g in f.next_needed:
            out.append(_say("corrective", ("needed", f.next_needed[g]), rng))
    return out


VERDICT_KINDS = ("eval_all", "eval_not_all", "eval_count", "goal_done", "goal_not_done")


def compose(f: Facts, profile, rng):
    """Voiced claims of one message as (intent, claim, sentence), in random order (structured for 'realistic' and
    'descriptive')."""
    if profile == "realistic":
        return compose_realistic(f, rng)
    if profile == "verdict":                                 # the language ablation: the realistic teacher's outcome
        return [x for x in compose_realistic(f, rng) if x[1][0] in VERDICT_KINDS]   # sentences only
    if profile == "descriptive":
        return compose_descriptive(f, rng)
    if profile == "complete":
        return [("descriptive", c, render(c[0], c, 0)) for c in claims(f, rng, summary_p=0.0)["descriptive"]]
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
