"""Exact word vocabulary of the MiniCraft prompts (requests, aliases, teacher feedback of every profile).

Word ids: 1..len(WORDS) for these words in this order, then numbers 0..255, then hashed buckets for anything else.
Append new words at the END only: ids of existing words must not change (trained text embeddings depend on them).
tests/test_minicraft_learn.py checks that every word the teacher and the requests produce is listed.
"""

WORDS = (
    "a", "across", "all", "along", "an", "and", "anything", "at", "because", "blocked", "bridge", "bridged",
    "bridges", "build", "building", "built", "bumped", "but", "by", "cell", "collect", "collected", "count", "craft",
    "crafted", "crafting", "crossed", "did", "dig", "done", "empty", "ended", "failed", "far", "feedback", "fetch",
    "finished", "firewood", "first", "for", "from", "furnace", "gather", "gem", "gems", "get", "give", "go", "goal",
    "gotten", "great", "ground", "had", "handle", "handles", "have", "hunting", "inputs", "into", "iron", "is", "it",
    "item", "job", "keep", "lasted", "made", "make", "managed", "metal", "mine", "missing", "needed", "next", "no",
    "not", "nothing", "of", "off", "okay", "on", "once", "or", "ore", "outfit", "outpost", "over", "pickaxe",
    "pickaxes", "picked", "plank", "planks", "point", "prepare", "quite", "raw", "ready", "recipe", "request",
    "river", "rock", "run", "should", "side", "some", "something", "standing", "station", "step", "steps", "stick",
    "sticks", "still", "stone", "subgoal", "target", "thanks", "that", "the", "them", "there", "things", "times",
    "to", "took", "tool", "tools", "treasure", "tree", "trees", "tried", "trying", "twice", "up", "upgrade", "use",
    "used", "wall", "was", "wasted", "water", "way", "went", "were", "when", "where", "with", "without", "wood",
    "work", "workbench", "workshop", "would", "wrong", "you", "your",
)
NUMBERS = 256        # "0" .. "255" get their own ids after WORDS
UNK_BUCKETS = 64     # other words: hashed into this many shared ids
