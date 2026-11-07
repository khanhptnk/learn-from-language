"""Gates for the grounded base (--base grounded, gft/minicraft/pretrain.py) before any post-training run.

  G1 data        the corpus audit and coverage stats saved next to the base (drop rate, modes, sentence kinds)
  G2 BC          request-mode success on seen requests (unseen worlds), per goal, vs a p = 0.9 noisy expert
  G3 grounding   from the pretraining log: masked-word accuracy with vs without the frames per word class (the gap),
                 message information gain
  G4 composition the steering probes: for each probe composite P, episodes on the same worlds under
                   request R alone                     (R = "gather wood": P is not asked for)
                   request R + "Great job! You made P at <station> from <inputs>."   (a description of P)
                   request P alone                     (the phrase itself; for a proxy held-out P it is a new word)
                   request P + the description
                 share of episodes that make P, and expert ops left to P at the end (paired, with SE). With
                 --heldout-goals including a seen composite (e.g. "build a fence"), that probe is the real test:
                 a sentence made of grounded words about a thing never made or asked for.

  .venv/bin/python experiments/minicraft/check_grounded.py --heldout-goals "build a well,make a sling" \\
      [--probes "build a raft,build a fence"] [--lang-episodes N --lang-steps T --arch ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import base_path, base_policy, heldout_splits, parse_arch, parse_goals  # noqa: E402

from gft.minicraft.core import N_WORLD_ACTIONS, RECIPE_FOR, RECIPES, WorldConfig  # noqa: E402
from gft.minicraft.expert import ExpertPool, plan  # noqa: E402
from gft.minicraft.goals import GOALS, GOAL_ID, Requests  # noqa: E402
from gft.minicraft.learn import (policy_actions, prompt_text, prompt_tokens, rng_state, run_episodes,  # noqa: E402
                                 set_rng_state)
from gft.minicraft.teacher import a_cell, np_item, recipe_inputs  # noqa: E402


def roll(model, cfg, req, texts, dev, seed):
    with torch.no_grad():
        p = model.encode_prompt(prompt_tokens(texts, dev))
    rng = np.random.default_rng(seed)
    saved = rng_state()
    torch.manual_seed(seed)
    env, _, _, _, _, sat, mask = run_episodes(cfg, req, lambda e, t: policy_actions(model, e, req, dev, rng, prompt=p),
                                              record=False)
    set_rng_state(saved)
    return env, sat, mask


def expert_runs(cfg, req, p_expert, seed):
    rng = np.random.default_rng(seed)
    pool = ExpertPool()

    def act(e, t):
        ex = pool(e, req.goals, np.nonzero(~e.done)[0])
        return np.where(rng.random(len(req)) < p_expert, ex, rng.integers(N_WORLD_ACTIONS, size=len(req)))
    env, _, _, _, _, sat, mask = run_episodes(cfg, req, act, record=False)
    pool.close()
    return env, sat, mask


def description(goal):
    """The teacher's narration of making goal's item: 'Great job! You made a fence at a site from a plank and stone.'"""
    assert goal.item in RECIPE_FOR, f"{goal.phrase}: probes must be crafted items"
    rec = RECIPES[RECIPE_FOR[goal.item]]
    return f"Great job! You made {np_item(goal.item)} at {a_cell(rec.station)} from {recipe_inputs(rec)}."


def ops_to(env, item):
    """Expert ops left to one `item` from each end inventory."""
    return np.array([len(plan(inv, [], items=((item, 1),))) for inv in env.inv])


def paired(x, y):
    d = np.asarray(y, float) - np.asarray(x, float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-goals", default="build a well,make a sling")
    ap.add_argument("--probes", default="build a raft,build an oven,build a wall,build a fence")
    ap.add_argument("--anchor", default="gather wood", help="the request under which a probe's description is appended")
    ap.add_argument("--lang-episodes", type=int, default=0)
    ap.add_argument("--lang-steps", type=int, default=0)
    ap.add_argument("--arch", default="")
    ap.add_argument("--pretrain-opts", default="")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.base, a.obs = "grounded", "pixels"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WorldConfig()
    held = parse_goals(a.heldout_goals)
    model = base_policy(cfg, dev, 0, kind="grounded", pixels=True, arch=parse_arch(a.arch), heldout_goals=held,
                        lang_episodes=a.lang_episodes, lang_steps=a.lang_steps,
                        pretrain_opts=parse_arch(a.pretrain_opts)).eval()
    path = base_path(cfg, 0, "grounded", "outputs/minicraft/base", True, parse_arch(a.arch), held, a.lang_episodes,
                     a.lang_steps, parse_arch(a.pretrain_opts))[0]
    info = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    out = dict(heldout=sorted(held), base=str(path))
    print("== G1 data ==")
    print(json.dumps(info.get("counts", {})))
    print(json.dumps(info.get("stats", {}), indent=None)[:2000])
    out["data"] = dict(counts=info.get("counts"), stats=info.get("stats"))
    print("== G3 grounding (pretraining log, last row) ==")
    last = (info.get("loss") or [{}])[-1]
    keys = [k for k in last if k.startswith(("gap_", "with_frames", "without_frames", "info_gain", "heldout"))]
    print({k: round(last[k], 3) for k in keys})
    out["grounding"] = {k: last[k] for k in keys}
    # G2: seen requests on unseen worlds
    print("== G2 request-mode BC on seen requests (unseen worlds) ==")
    sp = heldout_splits(a.n, True, heldout_goals=held)
    req = sp["val_seen"]
    env, sat, mask = roll(model, cfg, req, [prompt_text(req.text(b)) for b in range(len(req))], dev, 1)
    env_x, sat_x, mask_x = expert_runs(cfg, req, 0.9, 1)
    succ = float((sat | ~mask).all(1).mean())
    succ_x = float((sat_x | ~mask_x).all(1).mean())
    per_goal = {}
    for gid, g in enumerate(GOALS):
        m = req.goals == gid
        if m.any():
            per_goal[g.phrase] = (round(float(sat[m].mean()), 3), round(float(sat_x[m].mean()), 3), int(m.sum()))
    print(f"success {succ:.3f} (p=0.9 expert {succ_x:.3f}); per goal (policy, expert, n):")
    for k, v in per_goal.items():
        print(f"  {k:16s} {v[0]:.3f}  {v[1]:.3f}  n={v[2]}")
    out["bc"] = dict(success=succ, expert=succ_x, per_goal=per_goal)
    # G4: composition probes
    print("== G4 composition probes ==")
    rng = np.random.default_rng(3)
    seeds = rng.integers(2 ** 31, 2 ** 32, size=a.n)
    anchor = GOAL_ID[a.anchor]
    out["probes"] = {}
    probes = [p.strip() for p in a.probes.split(",") if p.strip()]
    for k, phrase in enumerate(probes):
        g = GOAL_ID[phrase]
        goal = GOALS[g]
        desc = description(goal)
        # control: the description of a *different* composite (the next probe), so that "steers" means steering
        # toward P specifically and not just "any made-something sentence makes the policy craft more"
        other = description(GOALS[GOAL_ID[probes[(k + 1) % len(probes)]]]) if len(probes) > 1 else None
        conds = {"R": (anchor, None), "R+desc": (anchor, desc), "P": (g, None), "P+desc": (g, desc)}
        if other:
            conds["R+other"] = (anchor, other)
        res = {}
        for name, (gid, msg) in conds.items():
            rq = Requests(np.array([[gid, -1, -1]] * a.n), seeds)
            env, sat, mask = roll(model, cfg, rq, [prompt_text(rq.text(b), msg) for b in range(a.n)], dev, 5)
            res[name] = dict(made=float((env.inv[:, goal.item] >= 1).mean()), ops=ops_to(env, goal.item),
                             steps=float(env.t.mean()))
        d_r, se_r = paired(res["R"]["ops"], res["R+desc"]["ops"])
        d_p, se_p = paired(res["P"]["ops"], res["P+desc"]["ops"])
        d_c, se_c = paired(res["R+other"]["ops"], res["R+desc"]["ops"]) if other else (float("nan"), float("nan"))
        tag = " (held out)" if g in held else ""
        print(f"{phrase}{tag}: \"{desc}\"")
        print(f"  made:  R {res['R']['made']:.3f}  R+desc {res['R+desc']['made']:.3f}"
              + (f"  R+other {res['R+other']['made']:.3f}" if other else "")
              + f"  |  P {res['P']['made']:.3f}  P+desc {res['P+desc']['made']:.3f}")
        print(f"  ops left to it:  R {res['R']['ops'].mean():.2f} -> R+desc {res['R+desc']['ops'].mean():.2f} "
              f"({d_r:+.2f} ± {se_r:.2f})"
              + (f"; vs R+other {res['R+other']['ops'].mean():.2f} ({d_c:+.2f} ± {se_c:.2f})" if other else "")
              + f"  |  P {res['P']['ops'].mean():.2f} -> P+desc {res['P+desc']['ops'].mean():.2f} ({d_p:+.2f} ± {se_p:.2f})")
        print("  steps:  " + "  ".join(f"{k} {v['steps']:.0f}" for k, v in res.items()))
        out["probes"][phrase] = dict(heldout=g in held, description=desc, other=other,
                                     **{k: dict(made=v["made"], ops=float(v["ops"].mean()), steps=v["steps"])
                                        for k, v in res.items()},
                                     delta_R=(d_r, se_r), delta_P=(d_p, se_p), delta_control=(d_c, se_c))
    verdict = []
    for phrase, r in out["probes"].items():
        steer = r["delta_R"][0] < 0 and abs(r["delta_R"][0]) > 2 * r["delta_R"][1]
        specific = r["delta_control"][0] < 0 and abs(r["delta_control"][0]) > 2 * r["delta_control"][1]
        verdict.append(f"{phrase}: {'steers' if steer else 'no steering'} from the description alone; "
                       f"{'specific' if specific else 'NOT specific'} (vs another composite's description)"
                       + (" [held out: the composition test]" if r["heldout"] else ""))
    print("== verdict ==")
    print("\n".join(verdict))
    out["verdict"] = verdict
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
        print("saved", a.out)


if __name__ == "__main__":
    main()
