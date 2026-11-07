"""experiments/minicraft/train.py options: the default run setting, --arm grpo and --exclude-goals."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/minicraft"))
import train as T  # noqa: E402
from gft.minicraft.goals import GOAL_ID  # noqa: E402


class _Stop(Exception):
    pass


def _parsed(argv, monkeypatch):
    got, real = [], argparse.ArgumentParser.parse_args
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self, args=None, ns=None: got.append(real(self, args, ns)) or got[-1])
    monkeypatch.setattr(T, "WorldConfig", lambda **kw: (_ for _ in ()).throw(_Stop()))
    monkeypatch.setattr(sys, "argv", ["train.py"] + argv)
    with pytest.raises(_Stop):
        T.main()
    return vars(got[-1])


def test_defaults_are_the_run_setting_and_explicit_flags_win(monkeypatch):
    a = _parsed("--arm gft --out x.json".split(), monkeypatch)
    assert a["curriculum"] == "graph" and a["mastery"] == 0.7 and a["cur_confirm"] and a["cur_review"] == 0.5
    assert a["success_needs_done"] and a["tune"] and a["budget_unit"] == "feedbacks" and a["budget"] == 100_000
    assert a["base"] == "explore" and a["obs"] == "pixels" and a["exclude_goals"] == "build a well,make a sling"
    b = _parsed("--arm gft --mastery 0.8 --no-cur-confirm --no-tune --out x.json".split(), monkeypatch)
    assert b["mastery"] == 0.8 and not b["cur_confirm"] and not b["tune"]


def test_grpo_is_ppo_with_the_group_advantage(monkeypatch):
    got = {}
    monkeypatch.setattr(T, "PPOConfig", lambda **kw: got.update(kw) or (_ for _ in ()).throw(_Stop()))
    monkeypatch.setattr(T, "base_policy", lambda *a, **kw: T.CleanPolicy(13, pixels=True))
    monkeypatch.setattr(T, "cached_pool", lambda *a, **kw: _FakePool())
    monkeypatch.setattr(T, "heldout_splits", lambda *a, **kw: {})
    monkeypatch.setattr(sys, "argv", "train.py --arm grpo --device cpu --out x.json".split())
    with pytest.raises(_Stop):
        T.main()
    assert got["advantage"] == "group" and got["gamma"] == 1.0 and got["entropy_target"] == 0.8
    assert got["reward"] == "frac"


class _FakePool:
    seeds = [0]

    def requests(self, idx):
        return None


def test_excluded_goals_are_only_in_their_own_split():
    ex = frozenset({GOAL_ID["build a well"], GOAL_ID["make a sling"]})
    for tune in (True, False):
        sp = T.heldout_splits(48, tune, exclude_goals=ex)
        unseen = "val_unseen" if tune else "new_unseen"
        for name, r in sp.items():
            has = [bool({int(g) for g in r.goals[b] if g >= 0} & ex) for b in range(len(r.seeds))]
            assert all(has) if name == unseen else not any(has), name
            assert (np.asarray(r.seeds) >= 2 ** 31).all()                       # never a training world


def test_requests_are_single_tasks_by_default(monkeypatch):
    """The agent is always prompted with one task: --max-goals defaults to 1, every split holds single-goal requests
    (the unseen-goal split too) and there is no held-out-combination split."""
    a = _parsed("--arm gft --out x.json".split(), monkeypatch)
    assert a["max_goals"] == 1
    ex = frozenset({GOAL_ID["build a well"], GOAL_ID["make a sling"]})
    sp = T.heldout_splits(48, True, max_goals=1, exclude_goals=ex)
    assert set(sp) == {"val_worlds", "val_unseen"}
    for r in sp.values():
        assert ((r.goals >= 0).sum(1) == 1).all()
