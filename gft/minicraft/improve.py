"""The improved message: what the teacher would say about an attempt one step better than the one it described,
inferred from the message text alone (no world, no facts, no task graph). Ported from the old curriculum's
improve.py (run 17g's recipe) to this game's teacher.

The teacher states the cause of a failure in item words ("You still needed a plank.", "To make a raft you need a
plank and rope at a workbench.", "You get stone from a rock with a pickaxe.", "You used up the planks you needed.")
and also narrates, in the same words, what a run got ("You made a plank.", "You collected 2 stone."). One step better
means the stated cause is fixed, so the rewrite turns the first cause sentence into its narration counterpart:

    still needed / recipe hint for / source hint for X          -> You made X. (a crafted item) / You collected X.
    used up X / did not keep X (one goal left)                  -> Done: G.
    had k of the n Xs needed to G                               -> Done: G. You made n Xs. (the narration if received)
    never said you were done, so G did not count                -> Done: G. (every such G: one DONE finishes them all)
    (no cause stated) tried to work C without the tool T        -> You made T.

"Not done" verdicts stay, in the neutral wording ("Not done: G." / "Still to do: G."; the better attempt may hold
some of G's item), and flip to done only for the goal whose own item was used up ("You used up the planks you
needed." -- the goal's last problem). With several goals left, the goal is the one received feedback tied to the
item: messages with exactly one goal not done that said "used up X" (the teacher says it only of an unmet goal's own
item); without such evidence there is no improved message.

Nothing else is dropped to hint at a change: every other sentence stays, and the improved message differs from the
original only by the cause and its counterpart. When every goal is done, sentences false of a success go ("Not quite
there.", "Thanks for trying.") and the message takes the teacher's success form (praise first). The other remarks are
judged by the feedback received so far (`Received`): they are added in order while some received message has said
all of them with the new sentence. Verdicts always stay (dropping one would hint that its goal was done). If a "not
done" verdict was never received together with the new sentence, the new sentence may have finished that goal: if
received feedback tied the fixed item to that goal ("used up X" said with it the one goal left: X is its item), the
goal flips to done (the finishing step); otherwise the text cannot tell and there is no improved message. (The old
curriculum flipped on co-occurrence -- the new sentence received with the goal achieved, never with it unmet -- which
here marked e.g. "make a pickaxe" done for "You picked up stone." in 18% of improved messages.)
The new sentence goes where the teacher narrates a gain: last, before a closing chatter sentence.

Every sentence the rewrite writes is a wording received before; if one was never received, there is no improved
message. A success has no improved message (this teacher names no process slip a better success would avoid).
"""

from collections import Counter

from gft.minicraft.core import RECIPE_FOR
from gft.minicraft.goals import GOALS
from gft.minicraft.teacher import TEMPLATES, render
from gft.minicraft.verify import parse, split_sentences

NEUTRAL = (1, 3)                                    # "Not done: G." / "Still to do: G."
CAUSES = ("no_done", "short", "used_up", "needed", "hint_recipe", "hint_raw")
FINISH = ("no_done", "short")                      # causes that name their goal: fixing one finishes it
FALLBACK = ("use_fail",)
VERDICTS = ("goal_not_done", "goal_done")
FAILED_ONLY = ("eval_not_all", "chatter")   # false of a success (dropped when one is made)


def key(c):
    """A claim without wording, counts or steps: what received sentences are matched by."""
    return c[:2]


class Received:
    """The feedback received so far, as the claim sets of its messages, indexed by claim."""

    def __init__(self):
        self.ids, self.postings, self.sentences = {}, {}, Counter()
        self.item_goal = {}                         # item -> Counter of goals: "used up X" said with exactly one goal
                                                    # not done, so X is that goal's item

    def observe(self, text):
        sents = split_sentences(text)
        self.sentences.update(sents)
        ps = [p for s in sents if (p := parse(s)) is not None]
        self._add(frozenset(key(p[1]) for p in ps))
        unmet = [c[1] for k, c in ps if k == "goal_not_done"]
        if len(unmet) == 1:
            for k, c in ps:
                if k == "used_up":
                    self.item_goal.setdefault(c[1], Counter())[unmet[0]] += 1

    def goal_of(self, item):
        """The goal whose item this is, as received feedback showed it (None without evidence or if it is mixed)."""
        c = self.item_goal.get(item)
        return next(iter(c)) if c and len(c) == 1 else None

    def _add(self, ks):
        if ks not in self.ids:
            i = self.ids[ks] = len(self.ids)
            for k in ks:
                self.postings.setdefault(k, set()).add(i)

    def inside(self, claims):
        """Some received message said all of these claims (and maybe more)."""
        sets = sorted((self.postings.get(key(c), set()) for c in claims), key=len)
        return bool(sets) and bool(set.intersection(*sets))

    def state_dict(self):
        return dict(shapes=list(self.ids), sentences=dict(self.sentences),
                    item_goal={k: dict(v) for k, v in self.item_goal.items()})

    def load_state_dict(self, sd):
        self.__init__()
        for ks in sd["shapes"]:
            self._add(ks)
        self.sentences.update(sd["sentences"])
        self.item_goal = {k: Counter(v) for k, v in sd.get("item_goal", {}).items()}


def _got(item):
    """The narration claim that the run got one `item`: made, for a crafted item, else collected."""
    return ("crafted" if item in RECIPE_FOR else "collected", item, 1)


def _fix(c):
    """The narration claim saying the cause c was fixed, or None when a verdict flips instead (used_up: the one unmet
    goal)."""
    if c[0] in ("needed", "hint_recipe", "hint_raw", "use_fail"):
        return _got(c[1])
    return None


def _say_like(claim, rng, received):
    """A verdict in a wording received for another goal ("Done: G." before G was ever done), None if none was."""
    seen = [v for v in range(len(TEMPLATES[claim[0]]))
            if any(received.sentences[render(claim[0], (claim[0], g), v)] for g in range(len(GOALS)))]
    return ("confirm", claim, render(claim[0], claim, seen[int(rng.integers(len(seen)))])) if seen else None


def _say_received(intent, claim, rng, received, variants=None):
    """The claim in a wording received before (among its templates, or `variants` of them), None if none was;
    without received feedback, as the teacher words it."""
    vs = range(len(TEMPLATES[claim[0]])) if variants is None else variants
    if received is None:
        v = list(vs)[int(rng.integers(len(vs)))]
        return intent, claim, render(claim[0], claim, v)
    seen = [t for t in (render(claim[0], claim, v) for v in vs) if received.sentences[t]]
    return (intent, claim, seen[int(rng.integers(len(seen)))]) if seen else None


def improve_feedback(parts, rng, received=None):
    """The improved message for a message given as (intent, claim, sentence) parts; only the sentences are read.
    received: a Received of the feedback so far (None: keep every kind of sentence, the teacher's wordings). Returns
    parts (claims parsed back from text), or None if the message states no cause to fix, describes a success, or a needed
    wording was never received."""
    sents = [(s, parse(s)) for _, _, s in parts]
    sents = [(s, p[0], p[1]) for s, p in sents if p is not None]
    unmet = [c[1] for _, k, c in sents if k == "goal_not_done"]
    if not unmet:
        return None
    cause = next((j for j, (_, k, _) in enumerate(sents) if k in CAUSES), None)
    if cause is None:
        cause = next((j for j, (_, k, _) in enumerate(sents) if k in FALLBACK), None)
    if cause is None:
        return None
    c = sents[cause][2]
    if c[0] in FINISH:
        return _finish(sents, cause, rng, received)
    fix = _fix(c)
    if c[0] == "used_up":                           # the goal's own item: that goal is done
        g_x = received.goal_of(c[1]) if received is not None else None
        flips = [unmet[0]] if len(unmet) == 1 else [g_x] if g_x in unmet else None
    else:
        flips = []
    if flips is None:
        return None
    still = [g for g in unmet if g not in flips]
    if fix:
        new = _say_received("descriptive", fix, rng, received)
        dones = []
    else:
        dones = [_say_received("confirm", ("goal_done", g), rng, received) for g in flips]
        new = dones[0] if dones and all(dones) else None
    if new is None:
        return None
    out = []
    for j, (s, k, cl) in enumerate(sents):         # everything stays but the fixed cause (and the flipped verdict)
        if j == cause or (k == "goal_not_done" and cl[1] not in still):
            continue
        if not still and k in FAILED_ONLY:          # a success now: these would be false of it
            continue
        if k == "goal_not_done":                    # neutral wording: the better attempt may hold some of G's item
            out.append(_neutral(s, k, cl, rng, received))
        else:
            out.append((None, cl, s))
    if not fix:                                     # done verdicts come after the "not done" ones
        at = sum(x[1][0] == "goal_not_done" for x in out)
        out[at:at] = dones
    if received is not None and fix:                # "Still to do: G." never said with the new sentence: it finishes G
        ends = [x[1][1] for x in out if x[1][0] == "goal_not_done" and not received.inside([x[1], new[1]])]
        if any(received.goal_of(fix[1]) != g for g in ends):   # ... only if received feedback tied the fixed item to
            return None                             # G ("used up X" with G the one goal left); else the text cannot tell
        if ends:                                    # G's own item made: making it finishes G, G flips
            still = [g for g in still if g not in ends]
            out = [x for x in out if not (x[1][0] == "goal_not_done" and x[1][1] in ends)
                   and (still or x[1][0] not in FAILED_ONLY)]
            dones = [_say_received("confirm", ("goal_done", g), rng, received) or _say_like(("goal_done", g), rng,
                                                                                            received) for g in ends]
            if not all(dones):
                return None
            at = sum(x[1][0] == "goal_not_done" for x in out)
            out[at:at] = dones
    elif received is not None and any(x[1][0] == "goal_not_done" and not received.inside([x[1], new[1]]) for x in out):
        return None                                 # "Still to do: G." never said with the new sentence (here a done
                                                    # verdict): it may have finished G, which the text cannot tell
    if received is not None:                        # remarks: added in order while some received message has said
        kept = [new]                                # them all with the new sentence; verdicts always stay
        for x in out:
            if x is new or x[1][0] in VERDICTS or any(x is d for d in dones):
                continue
            if received.inside([y[1] for y in kept] + [x[1]]):
                kept.append(x)
        out = [x for x in out if x[1][0] in VERDICTS or any(x is y for y in kept)]
    if not still:                                   # every goal done: the teacher's success form
        praise = _say_received("evaluative", ("eval_all",), rng, received)
        return [praise] + out + ([new] if fix else []) if praise else None
    if not fix:
        return out
    at = len(out) - (len(out) > 0 and out[-1][1][0] == "chatter")   # where the teacher narrates: before a closing
    return out[:at] + [new] + out[at:]                              # chatter sentence


def _neutral(s, k, cl, rng, received):
    """A "not done" verdict part in a neutral wording (the better attempt may hold some of the goal's item)."""
    v = next(v for v in range(len(TEMPLATES[k])) if render(k, cl, v) == s)
    nv = None if v in NEUTRAL else _say_received("confirm", cl, rng, received, NEUTRAL)
    return nv if nv else (None, cl, s)


def _finish(sents, cause, rng, received):
    """The improved message for a cause that names its goal G ("You had 1 of the 2 planks needed to make planks.",
    "You never said you were done, so make planks did not count."): G flips to done, the cause goes, and a short count
    gets its narration counterpart ("You made 2 planks.") when that wording was received. A missing DONE flips every
    goal said to be ready (pressing done once finishes them all). The rest follows improve_feedback: other verdicts
    stay (neutral wording), remarks stay while some received message said them with the new verdict, a success takes
    the teacher's success form. None if a needed wording was never received."""
    c = sents[cause][2]
    unmet = [cl[1] for _, k, cl in sents if k == "goal_not_done"]
    flips = [c[1]] if c[0] == "short" else [cl[1] for _, k, cl in sents if k == "no_done"]
    flips = [g for g in dict.fromkeys(flips) if g in unmet]
    if c[1] not in flips:
        return None
    dones = []
    for g in flips:
        d = _say_received("confirm", ("goal_done", g), rng, received)
        d = d if d is not None else _say_like(("goal_done", g), rng, received)
        if d is None:
            return None
        dones.append(d)
    new = None
    if c[0] == "short":
        item, n = GOALS[c[1]].item, GOALS[c[1]].count
        new = _say_received("descriptive", ("crafted" if item in RECIPE_FOR else "collected", item, n), rng, received)
    still = [g for g in unmet if g not in flips]
    gone = lambda k, cl: (k in FINISH and cl[1] in flips) or (k == "goal_not_done" and cl[1] in flips) \
        or (not still and k in FAILED_ONLY)
    out = [_neutral(s, k, cl, rng, received) if k == "goal_not_done" else (None, cl, s)
           for s, k, cl in sents if not gone(k, cl)]
    at = sum(x[1][0] == "goal_not_done" for x in out)
    out[at:at] = dones
    if received is not None:                        # remarks: kept while received with the new verdict (and narration)
        kept = [dones[0]] + ([new] if new else [])
        for x in out:
            if x[1][0] in VERDICTS or any(x is d for d in dones):
                continue
            if received.inside([y[1] for y in kept] + [x[1]]):
                kept.append(x)
        out = [x for x in out if x[1][0] in VERDICTS or any(x is y for y in kept)]
    if not still:
        praise = _say_received("evaluative", ("eval_all",), rng, received)
        if praise is None:
            return None
        out = [praise] + out
    if new:
        at = len(out) - (len(out) > 0 and out[-1][1][0] == "chatter")
        out = out[:at] + [new] + out[at:]
    return out


def improve_text(text, rng, received=None):
    """improve_feedback on a message string; None without a cause to fix."""
    out = improve_feedback([(None, None, s) for s in split_sentences(text)], rng, received)
    return None if out is None else " ".join(x[2] for x in out)
