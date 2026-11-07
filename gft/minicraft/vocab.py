"""Exact word vocabulary of the MiniCraft prompts (requests, aliases, teacher feedback of every profile).

Word ids: 1..len(WORDS) for these words in this order, then numbers 0..255, then hashed buckets for anything else.
Append new words at the END only: ids of existing words must not change (trained text embeddings depend on them).
tests/test_minicraft_learn.py checks that every word the teacher and the requests produce is listed.
"""

WORDS = (
    'a', 'along', 'an', 'any', 'anything', 'at', 'away', 'be', 'because', 'before',
    'brick', 'bricks', 'build', 'bumped', 'but', 'camp', 'can', 'carried', 'clay', 'collect',
    'collected', 'could', 'craft', 'crafted', 'crafting', 'did', 'dig', 'do', 'done', 'else',
    'empty', 'ended', 'enough', 'failed', 'feedback', 'fence', 'fences', 'fiber', 'first', 'for',
    'from', 'garden', 'gather', 'get', 'give', 'going', 'got', 'gotten', 'great', 'ground',
    'had', 'harbour', 'have', 'into', 'is', 'job', 'keep', 'kiln', 'kitchen', 'lasted',
    'loom', 'made', 'make', 'managed', 'missing', 'need', 'needed', 'never', 'next', 'no',
    'not', 'nothing', 'of', 'off', 'okay', 'once', 'only', 'open', 'or', 'oven',
    'ovens', 'pickaxe', 'pickaxes', 'picked', 'pit', 'plank', 'planks', 'point', 'prepare', 'pressed',
    'quarry', 'quite', 'raft', 'rafts', 'reeds', 'request', 'rock', 'rope', 'run', 'set',
    'should', 'site', 'sling', 'slings', 'some', 'something', 'start', 'station', 'steps', 'still',
    'stone', 'thanks', 'that', 'the', 'there', 'things', 'this', 'time', 'times', 'to',
    'took', 'tools', 'tree', 'tried', 'trying', 'twice', 'up', 'used', 'wall', 'walls',
    'was', 'way', 'well', 'wells', 'went', 'with', 'without', 'wood', 'work', 'workbench',
    'worked', 'would', 'wrong', 'you', 'your',
    # narration (2026-10-28): places, inputs, sequence words
    'after', 'and', 'became', 'happened', 'holding', 'it', 'later', 'nice', 'on', 'so',
    'them', 'then', 'turned', 'using',
    # counts and DONE (2026-11-02): "You had 1 of the 2 planks needed ...", "You never said you were done, ..."
    'what', 'said', 'were', 'count',
    # pace (2026-11-04): "You did make rope quicker than usual." / "..., but it took longer than usual."
    'than', 'usual', 'quicker', 'faster', 'longer', 'slower',
)
NUMBERS = 256        # "0" .. "255" get their own ids after WORDS
UNK_BUCKETS = 64     # other words: hashed into this many shared ids
