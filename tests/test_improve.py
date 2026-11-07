"""The improved message (improve.py): one step better than the teacher's message, from its text alone."""
import numpy as np
import pytest

from gft.minicraft.core import N_WORLD_ACTIONS, WorldConfig
from gft.minicraft.expert import expert_action
from gft.minicraft.goals import sample_requests
from gft.minicraft.improve import Received, improve_text
from gft.minicraft.learn import run_episodes
from gft.minicraft.teacher import compose, episode_facts_all
from gft.minicraft.verify import parse, split_sentences


def imp(text, seed=0, received=None):
    return improve_text(text, np.random.default_rng(seed), received)


MADE_PLANK = {"You made a plank.", "You crafted a plank.", "You made a plank at a workbench.",
              "You made a plank at a workbench from wood.", "At a workbench you turned wood into a plank.",
              "You crafted a plank at a workbench, using wood.", "You used a workbench to make a plank from wood."}
GOT_STONE = {"You collected stone.", "You picked up stone.", "You picked up stone at a rock.",
             "You got stone from a rock.", "At a rock you collected stone.", "You took stone from a rock."}
MADE_PICKAXE = {"You made a pickaxe.", "You crafted a pickaxe.", "You made a pickaxe at a workbench.",
                "You made a pickaxe at a workbench from 2 planks.", "At a workbench you turned 2 planks into a pickaxe.",
                "You crafted a pickaxe at a workbench, using 2 planks.",
                "You used a workbench to make a pickaxe from 2 planks."}


@pytest.mark.parametrize("cause, fix", [
    ("You still needed a plank.", MADE_PLANK),
    ("Without a plank this was not going to work.", MADE_PLANK),
    ("To make a plank you need wood at a workbench.", MADE_PLANK),
    ("You can get stone from a rock with a pickaxe.", GOT_STONE),
    ("You tried to work a rock without a pickaxe.", MADE_PICKAXE),        # no cause stated: the tool it lacked
])
def test_cause_becomes_its_narration_counterpart(cause, fix):
    out = split_sentences(imp(f"You did not build a fence. Done: make rope. {cause} You collected 2 wood. "
                              "Thanks for trying."))
    assert out[0] in ("Not done: build a fence.", "Still to do: build a fence.")   # neutral wording
    assert out[1:3] == ["Done: make rope.", "You collected 2 wood."]               # nothing else dropped
    assert out[3] in fix and out[4] == "Thanks for trying." and len(out) == 5      # the fix, before the closing


def test_only_the_cause_changes():
    text = ("Still to do: build a raft. Not done: make planks. Done: gather wood. You still needed rope. "
            "You pressed craft twice away from any station. You collected 3 wood.")
    out = split_sentences(imp(text))
    src = split_sentences(text)
    assert [s for s in src if s != "You still needed rope."] == out[:-1]           # every other sentence stays
    assert parse(out[-1])[1] == ("collected", 1, 1) or parse(out[-1])[1][:2] == ("crafted", 5)


def test_used_up_item_finishes_the_one_goal_left():
    out = split_sentences(imp("You did not make planks. You used up the planks you needed. You collected 2 wood. "
                              "Thanks for trying."))
    assert out[0] in ("Great job!", "Nice work!")                                  # the success form
    assert out[1] in ("You did make planks.", "Done: make planks.")
    assert "Thanks for trying." not in out and "You collected 2 wood." in out
    # with two goals left, the goal is the one received feedback tied to the item; without evidence, no message
    assert imp("Not done: make planks. Not done: build a raft. You used up the planks you needed.") is None


def test_no_improved_message_without_a_cause_for_a_success_or_without_a_received_wording():
    assert imp("Not done: build a raft. You collected 2 wood.") is None             # no cause stated
    assert imp("Great job! Done: build a raft.") is None                             # a success
    rec = Received()
    rec.observe("Not done: build a raft. You collected 2 wood.")
    assert imp("Not done: build a raft. You still needed rope.", received=rec) is None   # "made rope" never received


def test_a_fix_finishes_a_goal_only_if_received_feedback_tied_its_item_to_it():
    rec = Received()
    for msg in ("Great job! Done: make rope. You made rope.", "Not done: make rope. You still needed rope.",
                "Still to do: make rope. You collected 2 fiber."):
        rec.observe(msg)
    assert imp("Still to do: make rope. You still needed rope.", received=rec) is None   # no tie: the text cannot tell
    rec.observe("Not done: make rope. You used up the rope you needed.")                # rope is make rope's item
    out = split_sentences(imp("Still to do: make rope. You still needed rope.", received=rec))
    assert out[0] == "Great job!" and "Done: make rope." in out and "You made rope." in out
    rec.observe("Not done: make rope. You made rope.")       # now also said with the goal unmet: no flip, the verdict
    assert imp("Still to do: make rope. You still needed rope.", received=rec) == "Still to do: make rope. You made rope."


def test_a_fix_never_finishes_a_goal_by_co_occurrence_alone():
    """Stone always came with a pickaxe made, but stone is not the pickaxe goal's item: no false "Done"."""
    rec = Received()
    for msg in ("Great job! Done: make a pickaxe. Done: quarry stone. You collected stone.",
                "Not done: make a pickaxe. You can get stone from a rock with a pickaxe."):
        rec.observe(msg)
    assert imp("Not done: make a pickaxe. You can get stone from a rock with a pickaxe.", received=rec) is None


def test_improved_messages_are_in_the_teachers_language():
    """On real rollouts (a noisy expert), every improved message parses, every sentence of it was received before,
    and only failures get one."""
    cfg = WorldConfig()
    rng = np.random.default_rng(0)
    req = sample_requests(rng, 96, np.arange(96))
    def act(e, t):
        return np.array([expert_action(e, b, req.goals[b]) if rng.random() > 0.4 else int(rng.integers(N_WORLD_ACTIONS))
                         for b in range(len(req))])
    env, log, _, _, _, sat, mask = run_episodes(cfg, req, act)
    facts = episode_facts_all(log, env, req.goals, sat)
    texts = [" ".join(x[2] for x in compose(f, "realistic", rng)) for f in facts]
    rec = Received()
    for t in texts:
        rec.observe(t)
    made = 0
    for t, ok in zip(texts, (sat | ~mask).all(1)):
        out = imp(t, received=rec)
        if out is None:
            continue
        made += 1
        assert not ok
        for s in split_sentences(out):
            assert parse(s) is not None and rec.sentences[s] > 0, s
    assert made >= 2


def test_a_short_count_finishes_its_goal_with_the_full_count():
    rec = Received()
    for msg in ("Great job! Done: make planks. You made 2 planks.", "Not done: make planks. You collected 3 wood."):
        rec.observe(msg)
    out = split_sentences(imp("Not done: make planks. You had 1 of the 2 planks needed to make planks. "
                              "You collected 3 wood.", received=rec))
    assert out[0] == "Great job!" and "Done: make planks." in out and out[-1] == "You made 2 planks."
    assert not any("needed to make planks" in s for s in out)
    # the narration only if that wording was received; the done verdict in a wording received for another goal
    rec2 = Received()
    rec2.observe("Great job! Done: gather wood. You collected 2 wood.")
    rec2.observe("Not done: make planks. You collected 3 wood.")
    out = split_sentences(imp("Not done: make planks. You had 1 of the 2 planks needed to make planks.", received=rec2))
    assert out == ["Great job!", "Done: make planks."]


def test_a_missing_done_flips_every_ready_goal_and_keeps_the_rest():
    text = ("Not done: gather wood. Not done: dig clay. Still to do: make rope. You never said you were done, so "
            "gather wood did not count. You had what you needed to dig clay but never said you were done. "
            "You still needed rope. You collected 2 wood.")
    out = split_sentences(imp(text))
    assert out[0] in ("Not done: make rope.", "Still to do: make rope.")
    assert {"Done: gather wood.", "You did gather wood."} & set(out) and {"Done: dig clay.", "You did dig clay."} & set(out)
    assert not any("said you were done" in s for s in out) and "You still needed rope." in out
