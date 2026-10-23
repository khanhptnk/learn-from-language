"""Train one learner on MiniCraft and log everything needed for analysis.

  uv run python experiments/minicraft/train.py --arm gft --profile mixed --seed 0 --budget 50000 --out outputs/mc/gft_s0.json

Splits (sample-based success = all requirements met, and fraction of requirements met):
  train      tasks from the training pool (seen worlds and requests)
  new_worlds unseen world seeds, same request distribution (held-out goal pairs excluded)
  new_combos unseen world seeds, requests containing a goal pair held out of training
With --tune, val_worlds / val_combos (same construction, disjoint seeds) replace the two test splits, so
hyperparameters are chosen without looking at test data.
Curriculum (on by default, identical for every arm): requests start with the four easy goals; the next goal by depth
unlocks once the learner's own recent success on the newest goal reaches --cur-threshold.
The base policy (BC on easy goals) and the training pool are cached on disk, so every arm starts identically;
`--prepare-only` builds them once (the launcher does this before starting parallel jobs).

Output JSON (rewritten atomically after every evaluation, so partial runs are usable):
  meta     git commit, torch/numpy versions, device name, start time, full arguments
  evals    per evaluation: episodes, teacher/G*/verifier calls, wall time, and per split: success, fraction met,
           per-episode success (for bootstrap CIs), per-goal success, mean episode length
  rounds   per training round: learner diagnostics (GFT: Stage I/II success, losses, entropies, coverage of ideal
           feedback by recent actual feedback, reader loss on seen vs held-out episodes; PPO: losses, entropy, KL,
           clip fraction), unlocked curriculum goals, timings; example (request, feedback) pairs at evaluation rounds
  curriculum  unlock log [(episodes, goal id)]
Checkpoint: <out>.pt (latest clean policy). Resume: <out>.ckpt holds the full training state (networks, optimizers,
counters, numpy/torch/CUDA RNG states, the record so far, next evaluation point), written at every evaluation;
rerunning the same command continues from it, so an interrupted run ends exactly where an uninterrupted one would.
"""

import argparse
import contextlib
import copy
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from gft.minicraft.core import WorldConfig
from gft.minicraft.goals import ALIASES, GOALS, alias_requests, sample_requests
from gft.minicraft.learn import (GFT, PPO, AliasSchedule, CleanPolicy, CurriculumConfig, GFTConfig, PPOConfig,
                                 cached_alias_pool, cached_pool, concat_pools, evaluate, pretrain_base, rng_state,
                                 set_rng_state)

BASE_VERSION = "v7"   # bump when CleanPolicy's architecture or pretraining changes (invalidates cached base policies)

PHRASE = {g.phrase: i for i, g in enumerate(GOALS)}
HELDOUT_PAIRS = [frozenset({PHRASE["fetch the gem"], PHRASE["gather firewood"]}),
                 frozenset({PHRASE["make an iron tool"], PHRASE["make some planks"]}),
                 frozenset({PHRASE["get some metal"], PHRASE["get ready to mine"]}),
                 frozenset({PHRASE["get across the river"], PHRASE["collect building stone"]})]


def has_heldout(gs):
    return any(p <= gs for p in HELDOUT_PAIRS)


# Base policies (shared starting point of every learner):
#   easy      BC on expert runs of the four easy goals only; never sees deep goals (low-level skill missing)
#   random    BC on expert runs of random requests over all goals, independent of the actual request -- does random
#             meaningful things (gather, craft, mine, cross) and ignores what is asked; an initialization with all the
#             low-level skills but no idea which subgoals a request needs. Pretrained without text (prompt vector 0;
#             the text pathway starts untrained but trainable), so no text carries meaning yet: requests must be
#             learned by every method, feedback by GFT. Demos are clean expert runs (no random actions) ending with DONE
#   mixture   as random, but follows the actual request with probability 0.3
#   permuted  BC on expert runs that execute a fixed scrambled meaning of each request -- competent and coherent,
#             but systematically misreads requests
BASES = {
    "easy": dict(),
    "random": dict(goal_pool=None, behaviour="mixture", q=0.0, n_episodes=10_000, steps=10_000, noise=0.0),
    "mixture": dict(goal_pool=None, behaviour="mixture", q=0.3, n_episodes=10_000, steps=10_000),
    "permuted": dict(goal_pool=None, behaviour="permuted", n_episodes=10_000, steps=10_000),
}


def parse_arch(spec):
    """'kind=gvin,rec=8' -> CleanPolicy kwargs (ints, floats, true/false, strings)."""
    def val(v):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return v
    return {k: val(v) for k, v in (kv.split("=") for kv in spec.split(",") if kv)}


def base_policy(cfg, device, seed, kind="easy", cache_dir="outputs/minicraft/base", pixels=False, arch=None):
    """Text-free pretrained base of the given architecture (arch: CleanPolicy kwargs; default = POLICY_ARCH)."""
    arch = dict(arch or {})
    tag = ("" if kind == "easy" else f"_{kind}") + ("_pixels" if pixels else "") + "".join(
        f"_{k}{v}" for k, v in sorted(arch.items()))
    path = Path(cache_dir) / f"base_{BASE_VERSION}{tag}_s{seed}_size{cfg.size}.pt"
    model = CleanPolicy(cfg.size, pixels=pixels, **arch).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    model = pretrain_base(cfg, device, seed=seed, combo_filter=lambda gs: not has_heldout(gs),
                          arch=dict(pixels=pixels, **arch), **BASES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["gft", "ppo", "gftppo"], required=True,
                    help="gftppo: GFT whose policy also gets a PPO update on each round's Stage I episodes")
    ap.add_argument("--profile", default="realistic", help="teacher: realistic (default) | mixed | describe | ...")
    ap.add_argument("--reward", default="frac", help="PPO reward: bin | frac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=50_000, help="environment episodes")
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--episodes-per-round", type=int, default=0, help="GFT Stage I / PPO batch (0 = learner default)")
    ap.add_argument("--stage2-per-round", type=int, default=0, help="GFT Stage II batch (0 = default)")
    ap.add_argument("--damp", type=float, default=0.3, help="GFT damped distillation weight on pi^fb (1 = undamped)")
    ap.add_argument("--lr-aux", type=float, default=0, help="GFT pi^fb learning rate (0 = --lr)")
    ap.add_argument("--lr-clean", type=float, default=0, help="GFT pi learning rate (0 = --lr)")
    ap.add_argument("--epochs-aux", type=int, default=1, help="GFT Stage I passes per round (pi^fb)")
    ap.add_argument("--epochs-clean", type=int, default=1, help="GFT Stage II passes per round (pi)")
    ap.add_argument("--replay-rounds", type=int, default=1, help="GFT: pi^fb trains on the last k rounds' Stage I data")
    ap.add_argument("--aux-entropy", type=float, default=0.0, help="GFT: entropy bonus on pi^fb's Stage I fit")
    ap.add_argument("--stage2-temperature", type=float, default=1.0, help="GFT: pi^fb sampling temperature in Stage II")
    ap.add_argument("--clean-entropy", type=float, default=0.0, help="GFT: entropy bonus on pi's distillation")
    ap.add_argument("--entropy-target", type=float, default=None,
                    help="nats per step: automatic entropy coefficient for pi (GFT distillation and PPO updates)")
    ap.add_argument("--arch", default="", help="policy architecture, e.g. kind=gvin (docs/architecture_design.md)")
    ap.add_argument("--no-feedback", action="store_true",
                    help="ablation: pi^fb is prompted with the request alone (no teacher text) in both GFT stages")
    ap.add_argument("--base-share", type=float, default=0.0,
                    help="diagnostic: share of GFT Stage I episodes acted by the frozen base policy")
    ap.add_argument("--masked-claims", type=int, default=0,
                    help="GFT Stage I: extra one-sentence examples per episode, loss only on the sentence's steps")
    ap.add_argument("--ppo-epochs", type=int, default=2, help="PPO passes per batch")
    ap.add_argument("--gamma", type=float, default=0.99, help="PPO discount")
    ap.add_argument("--base", default="easy", choices=list(BASES), help="shared base policy (see BASES)")
    ap.add_argument("--curriculum", choices=["on", "off", "blocks", "graph"], default="on",
                    help="on: success-gated unlock in depth order; blocks: goals introduced in blocks on a fixed feedback "
                         "budget; graph: a goal opens once all its prerequisite goals are mastered")
    ap.add_argument("--mastery", type=float, default=0.7, help="graph curriculum: success that counts as mastered")
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--block-budget", type=int, default=10_000, help="feedbacks per block (teacher messages / rewards)")
    ap.add_argument("--obs", choices=["symbolic", "pixels"], default="pixels",
                    help="pixels: the policy sees an RGB rendering of the map (render.py) instead of symbolic cells")
    ap.add_argument("--goal-pool", default="", help="restrict tasks to these goal ids, e.g. 0,1,2,3 (sanity checks)")
    ap.add_argument("--max-goals", type=int, default=3, help="goals per request (1-3)")
    ap.add_argument("--phase2", choices=["none", "aliases"], default="none",
                    help="aliases: train on the base tasks until converged, then introduce new requests (ALIASES) "
                         "one at a time; --budget then caps phase 1 only")
    ap.add_argument("--alias-budget", type=int, default=5_000, help="feedbacks per new alias request in phase 2")
    ap.add_argument("--alias-pool", type=int, default=200, help="training tasks per alias")
    ap.add_argument("--converge-evals", type=int, default=4,
                    help="phase 1 has converged when training-task success has not improved by more than "
                         "--converge-tol over the last this-many evaluations")
    ap.add_argument("--converge-tol", type=float, default=0.01)
    ap.add_argument("--eval-every2", type=int, default=0, help="evaluation interval in phase 2 (episodes; 0 = same)")
    ap.add_argument("--cur-threshold", type=float, default=0.5, help="success on the newest goal needed to unlock the next")
    ap.add_argument("--tune", action="store_true", help="evaluate on validation splits instead of the test splits")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (validate against fp32 before relying on it)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy networks")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise-reproducible runs (deterministic torch/cuDNN algorithms; slower). Without it, runs and "
                         "resumes agree up to floating-point reduction order (~1e-7 in weights)")
    ap.add_argument("--prepare-only", action="store_true", help="build the cached pool and base policy, then exit")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(a.seed)
    cfg = WorldConfig()
    dev = a.device
    t0 = time.time()
    goal_pool = [int(x) for x in a.goal_pool.split(",")] if a.goal_pool else None
    task_kw = dict(goal_pool=goal_pool, max_goals=a.max_goals) if goal_pool or a.max_goals != 3 else {}
    pool = cached_pool(cfg, a.pool_size, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs),
                       **task_kw)
    if a.obs == "pixels":                             # rendered images are ~20x larger than the symbolic grids
        import gft.minicraft.learn as learn_mod
        learn_mod.FWD_CHUNK = 1024
    base = base_policy(cfg, dev, seed=0, kind=a.base, pixels=a.obs == "pixels", arch=parse_arch(a.arch))
    if a.phase2 == "aliases":
        alias_pool = cached_alias_pool(cfg, a.alias_pool, seed=7)
    if a.prepare_only:
        print(f"prepared: pool {len(pool.seeds)} tasks, base policy cached ({time.time() - t0:.0f}s)")
        return
    splits = dict(train=pool.requests(np.arange(min(a.n_eval, len(pool.seeds)))),
                  **heldout_splits(a.n_eval, a.tune, **task_kw))
    if a.phase2 == "aliases":
        pool = concat_pools(pool, alias_pool)
        splits.update(alias_splits(a.n_eval, a.tune))
    model = copy.deepcopy(base)
    cur = CurriculumConfig(enabled=a.curriculum != "off", threshold=a.cur_threshold,
                           mode={"blocks": "blocks", "graph": "graph"}.get(a.curriculum, "gate"),
                           block_size=a.block_size, block_budget=a.block_budget, mastery=a.mastery)
    if a.phase2 == "aliases":
        cur = AliasSchedule(pool, budget=a.alias_budget)
    if a.arm in ("gft", "gftppo"):
        gk = dict(profile=a.profile, lr_aux=a.lr_aux or a.lr, lr_clean=a.lr_clean or a.lr, damp=a.damp,
                  epochs_aux=a.epochs_aux, epochs_clean=a.epochs_clean, replay_rounds=a.replay_rounds,
                  masked_claims=a.masked_claims, base_share=a.base_share, aux_entropy=a.aux_entropy,
                  stage2_temperature=a.stage2_temperature, clean_entropy=a.clean_entropy,
                  entropy_target=a.entropy_target, no_feedback=a.no_feedback)
        gk.update({k: v for k, v in (("episodes", a.episodes_per_round), ("stage2", a.stage2_per_round)) if v})
        ppo = (PPOConfig(reward=a.reward, lr=a.lr, epochs=a.ppo_epochs, gamma=a.gamma, entropy_target=a.entropy_target)
               if a.arm == "gftppo" else None)
        learner = GFT(cfg, pool, model, dev, GFTConfig(**gk), seed=a.seed, curriculum=cur, ppo=ppo)
        clean = learner.clean
    else:
        pk = dict(reward=a.reward, lr=a.lr, epochs=a.ppo_epochs, gamma=a.gamma, entropy_target=a.entropy_target)
        if a.episodes_per_round:
            pk["episodes"] = a.episodes_per_round
        learner = PPO(cfg, pool, model, dev, PPOConfig(**pk), seed=a.seed, curriculum=cur)
        clean = learner.model
    if a.compile:                                   # the deployed policy, and GFT's pi^fb (same network)
        for m in [clean] + ([learner.aux] if a.arm != "ppo" else []):
            m.forward = torch.compile(m.forward)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (a.bf16 and dev.startswith("cuda")) else contextlib.nullcontext()
    out = Path(a.out)
    ckpt = out.with_suffix(".ckpt")
    record = dict(meta=meta(dev, a), evals=[], rounds=[])
    nxt = a.eval_every
    resumed = False
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        learner.load_state_dict(st["learner"])
        set_rng_state(st["torch_rng"])
        record, nxt, resumed = st["record"], st["nxt"], True
        if "seconds" in record:
            if a.phase2 == "aliases" or learner.episodes >= a.budget:
                print(f"{out} already finished; nothing to do", flush=True)
                return
            record["extended"] = record.get("extended", []) + [dict(from_episodes=learner.episodes, to=a.budget,
                                                                     previous_seconds=record.pop("seconds"))]
            print(f"extending finished run from {learner.episodes} to {a.budget} episodes", flush=True)
        record.setdefault("resumes", []).append(dict(at_episodes=learner.episodes,
                                                     time=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"resumed from {ckpt} at {learner.episodes} episodes", flush=True)

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        for path, write in ((out, lambda p: p.write_text(json.dumps(record))),
                            (out.with_suffix(".pt"), lambda p: torch.save(clean.state_dict(), p)),
                            (ckpt, lambda p: torch.save(dict(learner=learner.state_dict(), torch_rng=rng_state(),
                                                             record=record, nxt=nxt), p))):
            tmp = path.with_suffix(path.suffix + ".tmp")
            write(tmp)
            os.replace(tmp, path)

    def log_eval():
        row = dict(episodes=learner.episodes, wall=time.time() - t0,
                   teacher_calls=getattr(learner, "teacher_calls", 0), gstar=getattr(learner, "gstar", 0),
                   verifier_calls=getattr(learner, "verifier_calls", 0))
        with amp:
            for name, req in splits.items():
                row[name] = evaluate(clean, cfg, req, dev, seed=1, detail=True)
        record["evals"].append(row)
        record["curriculum"] = learner.cur.unlock_log
        print(json.dumps({k: (v["success"] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
        save()

    feedbacks = lambda: learner.teacher_calls if a.arm != "ppo" else learner.episodes

    def phase1_converged():
        """Training-task success has not improved by more than converge_tol over the last converge_evals evals."""
        h = [e["train"]["success"] for e in record["evals"]]
        k = a.converge_evals
        return len(h) > k and max(h[-k:]) <= max(h[:-k]) + a.converge_tol

    def running():
        if a.phase2 == "aliases":
            return not learner.cur.finished(feedbacks())
        return learner.episodes < a.budget

    if not resumed:
        log_eval()
    while running():
        with amp:
            info = learner.round()
        at_eval = learner.episodes >= nxt
        if not at_eval:
            info.pop("examples", None)
        record["rounds"].append(dict(episodes=learner.episodes, **info))
        if at_eval:
            in_phase2 = a.phase2 == "aliases" and learner.cur.phase2_at is not None
            nxt += (a.eval_every2 or a.eval_every) if in_phase2 else a.eval_every
            if a.phase2 == "aliases" and learner.cur.phase2_at is None:
                if phase1_converged() or learner.episodes >= a.budget:
                    learner.cur.start_phase2(feedbacks())
                    record["phase2_start"] = dict(episodes=learner.episodes, feedbacks=feedbacks(),
                                                  converged=phase1_converged())
                    nxt = learner.episodes + (a.eval_every2 or a.eval_every)
            log_eval()
    record["seconds"] = time.time() - t0
    save()


def alias_splits(n, tune):
    """Per alias request: n requests on unseen worlds (separate seed streams for test and validation)."""
    out = {}
    for k in range(len(ALIASES)):
        rng = np.random.default_rng((40_000 if tune else 30_000) + k)
        out[f"alias{k}"] = alias_requests(k, rng.integers(2 ** 31, 2 ** 32, size=n))
    return out


def heldout_splits(n, tune, goal_pool=None, max_goals=3):
    """Unseen-world splits. Test and validation seeds come from different streams in [2^31, 2^32) (the training pool
    uses [0, 2^31)) and are checked to be disjoint. With a restricted goal pool / single-goal requests (sanity
    checks) the held-out-combination split is dropped."""
    restricted = goal_pool is not None or max_goals != 3

    def make(stream):
        rng = np.random.default_rng(stream)
        worlds = sample_requests(rng, n, rng.integers(2 ** 31, 2 ** 32, size=n), goal_pool=goal_pool, max_goals=max_goals,
                                 combo_filter=lambda gs: not has_heldout(gs))
        combos = None if restricted else sample_requests(rng, n, rng.integers(2 ** 31, 2 ** 32, size=n),
                                                         combo_filter=has_heldout)
        return worlds, combos
    test, val = make(10_000), make(20_000)
    assert not set(test[0].seeds) & set(val[0].seeds)
    out = dict(val_worlds=val[0], val_combos=val[1]) if tune else dict(new_worlds=test[0], new_combos=test[1])
    return {k: v for k, v in out.items() if v is not None}


def meta(dev, a):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "", None
    return dict(args=vars(a), git_commit=commit, git_dirty=dirty, torch=torch.__version__, numpy=np.__version__,
                python=platform.python_version(), host=platform.node(), start=time.strftime("%Y-%m-%d %H:%M:%S"),
                device=torch.cuda.get_device_name(0) if dev.startswith("cuda") else platform.processor() or "cpu")


if __name__ == "__main__":
    main()
