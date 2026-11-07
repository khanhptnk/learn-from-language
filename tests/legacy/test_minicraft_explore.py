"""Pretraining v3 data (gft/minicraft/explore.py): random courses of primitive skills, no recipes, no text."""
import ast
from pathlib import Path

import numpy as np

from gft.minicraft.core import AGENT_FEATS, N_ACTIONS, PLANK, SLING, WELL, WorldConfig
from gft.minicraft.explore import explore_corpus

CFG = WorldConfig()


def test_explorer_policy_never_reads_the_recipe_book_or_requests():
    """The behaviour (PrimitiveExplorer) uses no recipe, goal or plan; RECIPES only feeds the coverage report."""
    src = Path("gft/minicraft/explore.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "PrimitiveExplorer")
    names = {n.id for n in ast.walk(cls) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(cls)
                                                                         if isinstance(n, ast.Attribute)}
    assert not names & {"RECIPES", "RECIPE_FOR", "RECIPE_IN", "craftable", "plan", "GOALS", "goals", "expert_action"}


def test_corpus_is_deterministic_drops_heldout_items_and_covers_primitives():
    steps, cov = explore_corpus(CFG, 96, [PLANK], seed=3, chunk=48)          # planks held out: they are common
    steps2, cov2 = explore_corpus(CFG, 96, [PLANK], seed=3, chunk=48)
    assert all(np.array_equal(steps[k], steps2[k]) for k in steps) and cov == cov2
    assert cov["drop_rate"] > 0 and cov["crafted_per_episode"]["plank"] == 0     # dropped, not hidden
    assert cov["episodes"] == 96 - round(cov["drop_rate"] * 96) and len(np.unique(steps["b"])) == cov["episodes"]
    assert steps["state"].shape == (cov["steps"], AGENT_FEATS) and steps["a"].max() < N_ACTIONS
    assert set(cov["options"]) == {"gather", "craft", "wander", "done"}
    acts = np.bincount(steps["a"], minlength=N_ACTIONS)
    assert (acts > 0).all()                                    # every primitive action occurs, done included
    assert cov["distinct_craft_attempts"] > 20 and 0 < cov["craft_success"] < 0.5


def test_seen_crafts_happen_by_chance_and_heldout_ones_never_stay():
    steps, cov = explore_corpus(CFG, 128, [WELL, SLING], seed=0, chunk=64)
    made = cov["crafted_per_episode"]
    assert made["well"] == 0 and made["sling"] == 0
    assert min(made["plank"], made["rope"], made["brick"]) > 0.02   # first-tier crafts happen by chance


def test_pretrain_explorer_fits_the_corpus_without_a_prompt():
    import torch
    from gft.minicraft.explore import pretrain_explorer
    torch.manual_seed(0)
    steps, _ = explore_corpus(CFG, 16, [WELL, SLING], seed=0, chunk=16)
    model, losses = pretrain_explorer(steps, CFG, "cpu", n_steps=101, batch=64, arch=dict(d=32))
    assert losses[-1] < losses[0]
    assert all(p.grad is None for p in model.prompt.parameters())       # the text pathway was never used
