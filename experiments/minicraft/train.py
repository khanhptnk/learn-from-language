"""Train one learner on MiniCraft and log everything needed for analysis.

  uv run python experiments/minicraft/train.py --arm gft --out outputs/minicraft/gft.json
  uv run python experiments/minicraft/train.py --arm grpo --out outputs/minicraft/grpo.json

Arms: gft (gft/minicraft/gft_message.py, the one GFT), grpo (PPO with group-standardized episode rewards, no critic,
gamma 1, entropy target 0.8) and ppo (PPO with a critic and GAE; its flags below). The defaults are the run setting of
every current experiment: realistic teacher, pixels, single-goal requests, the explore base with "build a well" and
"make a sling" never requested in training, the graph curriculum (mastery 0.7, two-window confirmation, review 0.5),
success needs DONE, the budget in feedbacks (teacher messages for GFT, rewards for PPO) with evaluation on that
budget, and validation splits (--no-tune: the test splits).

Splits (sample-based success = all requirements met, and fraction of requirements met):
  train       tasks from the training pool (seen worlds and requests)
  val_worlds  unseen world seeds, same request distribution
  val_unseen  unseen world seeds, requests containing an excluded goal (never requested in training)
With --no-tune, new_worlds / new_unseen (same construction, disjoint seeds) replace them.
The base policy and the training pool are cached on disk, so every arm starts identically; `--prepare-only` builds
them once (the launcher does this before starting parallel jobs).

Output JSON (rewritten atomically after every evaluation, so partial runs are usable):
  meta     git commit, torch/numpy versions, device name, start time, full arguments
  evals    per evaluation: episodes, teacher/G* calls, wall time, and per split: success, fraction met, per-episode
           success (for bootstrap CIs), per-goal success, mean episode length
  rounds   per training round: learner diagnostics (GFT: Stage I/II success, scorer and reader losses, kept attempts;
           PPO: losses, entropy, KL, clip fraction), unlocked curriculum goals, timings; example (request, feedback)
           pairs at evaluation rounds
  curriculum  unlock log [(time, goal id)]; time is teacher calls for GFT and rewards (= episodes) for PPO
Checkpoint: <out>.pt (latest clean policy). Resume: <out>.ckpt holds the full training state (networks, optimizers,
counters, numpy/torch/CUDA RNG states, the record so far, next evaluation point), written at every evaluation;
rerunning the same command continues from it, so an interrupted run ends exactly where an uninterrupted one would.
"""

import argparse
import contextlib
import copy
import hashlib
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
from gft.minicraft.pretrain import GROUNDED_ARCH, GroundedPolicy, audit, build_corpus, pretrain_grounded
from gft.minicraft.gft_message import GFT, GFTConfig
from gft.minicraft.learn import (PPO, AliasSchedule, CleanPolicy, CurriculumConfig, PPOConfig, cached_alias_pool,
                                 cached_pool, concat_pools, evaluate, lang_corpus, pretrain_base, pretrain_lang,
                                 rng_state, set_rng_state)

BASE_VERSION = "v10"   # bump when CleanPolicy's architecture, pretraining or the goal list changes (invalidates cached bases)

PHRASE = {g.phrase: i for i, g in enumerate(GOALS)}
HELDOUT_PAIRS = [frozenset({PHRASE["build a wall"], PHRASE["gather wood"]}),
                 frozenset({PHRASE["build a raft"], PHRASE["dig clay"]}),
                 frozenset({PHRASE["build a fence"], PHRASE["make rope"]}),
                 frozenset({PHRASE["build an oven"], PHRASE["quarry stone"]})]


def has_heldout(gs):
    return any(p <= gs for p in HELDOUT_PAIRS)


def parse_goals(spec):
    """'8,9' or 'get across the river,fetch the gem' -> frozenset of goal ids."""
    return frozenset(int(x) if x.strip().isdigit() else PHRASE[x.strip()] for x in spec.split(",") if x.strip())


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
#   lang      simulated language pretraining (learn.lang_corpus / pretrain_lang): noisy experts at mixed skill, one
#             network trained in a request mode (hindsight requests) and a message mode (request + the teacher's
#             message); needs --heldout-goals, which are requested in the corpus but never achieved
BASES = {
    "easy": dict(),
    "random": dict(goal_pool=None, behaviour="mixture", q=0.0, n_episodes=10_000, steps=10_000, noise=0.0),
    "mixture": dict(goal_pool=None, behaviour="mixture", q=0.3, n_episodes=10_000, steps=10_000),
    "permuted": dict(goal_pool=None, behaviour="permuted", n_episodes=10_000, steps=10_000),
    "lang": dict(n_episodes=30_000, steps=10_000),
    # grounded: pretraining v2 (gft/minicraft/pretrain.py): explorer-mixture corpus with narration, held-out goals
    # never requested; GroundedPolicy (token-level text fusion) trained with BC + masked words / actions given the
    # frames. Needs --heldout-goals and --obs pixels.
    "grounded": dict(n_episodes=20_000, steps=10_000),
    # explore: pretraining v3 (gft/minicraft/explore.py): random courses of primitive skills that know no recipe or
    # goal, no requests, no text; unconditional BC. Episodes that make an item of a --heldout-goals goal are dropped.
    "explore": dict(n_episodes=20_000, steps=10_000),
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


def base_path(cfg, seed, kind, cache_dir, pixels, arch, heldout_goals, lang_episodes, lang_steps, pretrain_opts=None):
    """Cache file of a base policy, and the lang pretraining sizes. pretrain_opts: pretrain_grounded kwargs
    (objective ablations), part of the tag."""
    if kind == "grounded":                     # spelling out a default must not fork the cache
        arch = {k: v for k, v in arch.items() if GROUNDED_ARCH.get(k) != v}
    tag = ("" if kind == "easy" else f"_{kind}") + ("_pixels" if pixels else "") + "".join(
        f"_{k}{v}" for k, v in sorted(arch.items()))
    if pretrain_opts:
        tag += "_opt" + "".join(f"_{k}{v}" for k, v in sorted(pretrain_opts.items()))
    size = None
    if kind in ("lang", "grounded", "explore"):
        assert heldout_goals or kind == "explore", f"--base {kind} needs --heldout-goals"
        size = dict(BASES[kind], **{k: v for k, v in (("n_episodes", lang_episodes), ("steps", lang_steps)) if v})
        tag += f"_h{'-'.join(map(str, sorted(heldout_goals or ())))}_n{size['n_episodes']}_t{size['steps']}"
    return Path(cache_dir) / f"base_{BASE_VERSION}{tag}_s{seed}_size{cfg.size}.pt", size


def base_policy(cfg, device, seed, kind="easy", cache_dir="outputs/minicraft/base", pixels=False, arch=None,
                heldout_goals=None, lang_episodes=0, lang_steps=0, pretrain_opts=None):
    """Pretrained base of the given architecture (arch: CleanPolicy kwargs; default = POLICY_ARCH): text-free, or for
    kind=lang / grounded language-pretrained with the held-out goals (lang_episodes / lang_steps override the BASES
    sizes; pretrain_opts: pretrain_grounded kwargs for objective ablations)."""
    arch = dict(arch or {})
    path, size = base_path(cfg, seed, kind, cache_dir, pixels, arch, heldout_goals, lang_episodes, lang_steps,
                           pretrain_opts)
    if kind == "grounded":
        assert pixels, "--base grounded is a pixel policy"
        full_arch = {**GROUNDED_ARCH, **arch}                          # one source of defaults (pretrain_grounded's)
        if path.exists():                                              # a cached base: build exactly what was trained
            saved = json.loads(path.with_suffix(".json").read_text()).get("full_arch") if path.with_suffix(".json").exists() else None
            if saved is None:                                          # older cache without the record: infer the reader
                keys = torch.load(path, map_location="cpu").keys()
                saved = {**full_arch, "reader_frames": "cheap" if "frame_fc.weight" in keys else "full"}
                if saved != full_arch:
                    print(f"note: cached base {path.name} predates the architecture record; using its reader_frames="
                          f"{saved['reader_frames']}", flush=True)
                    full_arch = saved
            assert saved == full_arch, f"cached base {path} was trained with {saved}, requested {full_arch}: " \
                                       f"pass the matching --arch or delete the cache"
        model = GroundedPolicy(cfg.size, **full_arch).to(device)
    else:
        model = CleanPolicy(cfg.size, pixels=pixels, **arch).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    if kind == "grounded":
        t0 = time.time()
        corpus, counts, stats = build_corpus(cfg, size["n_episodes"], heldout_goals, seed=seed)
        audit(corpus, counts, heldout_goals)
        print(f"grounded corpus: {counts} ({time.time() - t0:.0f}s)\n  {json.dumps(stats)}", flush=True)
        model, hist = pretrain_grounded(cfg, device, corpus, steps=size["steps"], seed=seed, arch=arch, log_every=500,
                                        **(pretrain_opts or {}))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(dict(counts={k: int(v) for k, v in counts.items()}, stats=stats,
                                                             loss=hist, seconds=time.time() - t0, arch=arch,
                                                             full_arch=full_arch, pretrain_opts=pretrain_opts or {})))
    elif kind == "lang":
        t0 = time.time()
        corpus, counts = lang_corpus(cfg, size["n_episodes"], heldout_goals, seed=seed)
        print(f"lang corpus: {counts} ({time.time() - t0:.0f}s)", flush=True)
        model, hist = pretrain_lang(cfg, device, corpus, steps=size["steps"], seed=seed,
                                    arch=dict(pixels=pixels, **arch), log_every=500)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(dict(counts={k: int(v) for k, v in counts.items()},
                                                             loss=hist, seconds=time.time() - t0)))
    elif kind == "explore":
        from gft.minicraft.explore import explore_corpus, pretrain_explorer
        t0 = time.time()
        held_items = [GOALS[g].item for g in heldout_goals or ()]
        corpus, cov = explore_corpus(cfg, size["n_episodes"], held_items, seed=seed)
        print(f"explore corpus ({time.time() - t0:.0f}s): {json.dumps(cov)}", flush=True)
        model, hist = pretrain_explorer(corpus, cfg, device, n_steps=size["steps"], seed=seed,
                                        arch=dict(pixels=pixels, **arch))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(dict(coverage=cov, loss=hist, seconds=time.time() - t0)))
    else:
        model = pretrain_base(cfg, device, seed=seed, combo_filter=lambda gs: not has_heldout(gs),
                              arch=dict(pixels=pixels, **arch), **BASES[kind])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


GRPO = dict(advantage="group", gamma=1.0, entropy_target=0.8, reward="frac")   # --arm grpo (2026-11-05 runs)
EXCLUDED = "build a well,make a sling"   # never requested in training: the unseen-goal split


def main():
    B = argparse.BooleanOptionalAction
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["gft", "grpo", "ppo"], required=True,
                    help="gft: gft_message.GFT; grpo: PPO, group advantage, gamma 1, entropy target 0.8; ppo: GAE")
    ap.add_argument("--profile", default="realistic", help="teacher: realistic (default) | verdict (its praise and verdicts only: the language ablation) | mixed | ...")
    ap.add_argument("--heldout-goals", default="",
                    help="post-train only on requests containing these goals (e.g. 8,9 or names), with a base that "
                         "never achieved them; needs --curriculum off and --exclude-goals ''")
    ap.add_argument("--exclude-goals", default=EXCLUDED,
                    help="goals never requested in training (base explore never makes them); evaluated in *_unseen")
    ap.add_argument("--lang-episodes", type=int, default=0, help="--base lang: corpus size (0 = BASES)")
    ap.add_argument("--lang-steps", type=int, default=0, help="--base lang: BC steps (0 = BASES)")
    ap.add_argument("--pretrain-opts", default="",
                    help="--base grounded: pretrain_grounded kwargs for objective ablations, e.g. align_frac=0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=100_000, help="see --budget-unit")
    ap.add_argument("--budget-unit", default="feedbacks", choices=["episodes", "feedbacks"],
                    help="feedbacks: teacher messages (GFT) / rewards (PPO); episodes: all environment episodes")
    ap.add_argument("--eval-every", type=int, default=5_000, help="in --budget-unit")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--episodes-per-round", type=int, default=0, help="GFT Stage I / PPO batch (0 = learner default)")
    ap.add_argument("--stage2-per-round", type=int, default=0, help="GFT Stage II attempts (0 = default)")
    ap.add_argument("--arch", default="", help="policy architecture, e.g. d=512,ch=64,trunk=4")
    # PPO only (grpo fixes advantage, gamma, entropy target and reward)
    ap.add_argument("--reward", default="frac", help="PPO: frac | bin | parse")
    ap.add_argument("--lr", type=float, default=3e-4, help="PPO")
    ap.add_argument("--ppo-epochs", type=int, default=2, help="PPO passes per batch")
    ap.add_argument("--gamma", type=float, default=0.99, help="PPO discount")
    ap.add_argument("--entropy-target", type=float, default=None, help="PPO: entropy controller target (nats/step)")
    ap.add_argument("--value-warmup", type=int, default=0, help="PPO: first rounds fit only the critic")
    # the run setting
    ap.add_argument("--base", default="explore", choices=list(BASES), help="shared base policy (see BASES)")
    ap.add_argument("--curriculum", choices=["on", "off", "blocks", "graph"], default="graph",
                    help="graph: goals open when their prerequisites are mastered; on: depth gate; blocks; off")
    ap.add_argument("--mastery", type=float, default=0.7, help="graph curriculum: success that counts as mastered")
    ap.add_argument("--cur-review", type=float, default=0.5, help="graph curriculum: share of tasks on mastered goals")
    ap.add_argument("--cur-confirm", action=B, default=True, help="graph curriculum: mastery holds for two windows")
    ap.add_argument("--block-size", type=int, default=2)
    ap.add_argument("--block-budget", type=int, default=10_000, help="feedbacks per block (teacher messages / rewards)")
    ap.add_argument("--cur-threshold", type=float, default=0.5, help="gate curriculum: success to unlock the next goal")
    ap.add_argument("--obs", choices=["symbolic", "pixels"], default="pixels")
    ap.add_argument("--goal-pool", default="", help="restrict tasks to these goal ids, e.g. 0,1,2,3 (sanity checks)")
    ap.add_argument("--max-goals", type=int, default=1, help="goals per request")
    ap.add_argument("--phase2", choices=["none", "aliases"], default="none",
                    help="aliases: after phase 1 converges, train on new alias requests")
    ap.add_argument("--alias-budget", type=int, default=5_000, help="feedbacks per new alias request in phase 2")
    ap.add_argument("--alias-pool", type=int, default=200, help="training tasks per alias")
    ap.add_argument("--converge-evals", type=int, default=4,
                    help="phase 1 has converged when train success has not improved over this many evaluations")
    ap.add_argument("--converge-tol", type=float, default=0.01)
    ap.add_argument("--eval-every2", type=int, default=0, help="evaluation interval in phase 2 (0 = same)")
    ap.add_argument("--success-needs-done", action=B, default=True, help="a task succeeds only when the agent says DONE")
    ap.add_argument("--tune", action=B, default=True, help="evaluate on validation splits (--no-tune: test splits)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (validate against fp32 before relying on it)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy networks")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic CUDA kernels (bit-identical resumes on GPU; slower)")
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
    cfg = WorldConfig(success_needs_done=a.success_needs_done)
    dev = a.device
    t0 = time.time()
    goal_pool = [int(x) for x in a.goal_pool.split(",")] if a.goal_pool else None
    task_kw = dict(goal_pool=goal_pool, max_goals=a.max_goals) if goal_pool or a.max_goals != 3 else {}
    held = parse_goals(a.heldout_goals) if a.heldout_goals else None
    excl = parse_goals(a.exclude_goals) if a.exclude_goals else frozenset()
    assert not (held and excl), "--heldout-goals and --exclude-goals are different designs: pass --exclude-goals ''"
    assert not held or a.curriculum == "off", "--heldout-goals: every task contains a held-out goal; use --curriculum off"
    if held:                                          # post-training on new tasks: requests containing a held-out goal
        pool = cached_pool(cfg, a.pool_size, seed=0, tag=f"heldout-goals-{'-'.join(map(str, sorted(held)))}-v2",
                           combo_filter=lambda gs: bool(gs & held), **task_kw)
    else:
        pool = cached_pool(cfg, a.pool_size, seed=0,
                           tag="no-heldout-pairs-v2" + (f"-excl{'-'.join(map(str, sorted(excl)))}" if excl else ""),
                           combo_filter=lambda gs: not has_heldout(gs) and not gs & excl, **task_kw)
    if a.obs == "pixels":                             # rendered images are ~20x larger than the symbolic grids
        import gft.minicraft.learn as learn_mod
        learn_mod.FWD_CHUNK = 1024
    assert not excl or a.base in ("explore", "easy"), \
        "--exclude-goals needs a base that never made them: explore (drops such episodes) or easy (four easy goals)"
    base = base_policy(cfg, dev, seed=0, kind=a.base, pixels=a.obs == "pixels", arch=parse_arch(a.arch),
                       heldout_goals=held or (excl if a.base == "explore" else None),
                       lang_episodes=a.lang_episodes, lang_steps=a.lang_steps,
                       pretrain_opts=parse_arch(a.pretrain_opts))
    if a.phase2 == "aliases":
        alias_pool = cached_alias_pool(cfg, a.alias_pool, seed=7)
    if a.prepare_only:
        print(f"prepared: pool {len(pool.seeds)} tasks, base policy cached ({time.time() - t0:.0f}s)")
        return
    splits = dict(train=pool.requests(np.arange(min(a.n_eval, len(pool.seeds)))),
                  **heldout_splits(a.n_eval, a.tune, heldout_goals=held, exclude_goals=excl, **task_kw))
    if a.base != "grounded":                          # same base, words by exact vocabulary id (its text pathway was
        exact = CleanPolicy(cfg.size, pixels=a.obs == "pixels", exact_text=True,   # never trained: no loss)
                            **parse_arch(a.arch)).to(dev)
        own = exact.state_dict()
        res = exact.load_state_dict({k: v for k, v in base.state_dict().items()
                                     if k in own and own[k].shape == v.shape}, strict=False)
        assert all(k.startswith("prompt.") for k in res.missing_keys), res.missing_keys
        base = exact
    if a.phase2 == "aliases":
        pool = concat_pools(pool, alias_pool)
        splits.update(alias_splits(a.n_eval, a.tune))
    model = copy.deepcopy(base)
    cur = CurriculumConfig(enabled=a.curriculum != "off", threshold=a.cur_threshold,
                           mode={"blocks": "blocks", "graph": "graph"}.get(a.curriculum, "gate"),
                           block_size=a.block_size, block_budget=a.block_budget, mastery=a.mastery,
                           review=a.cur_review, confirm=a.cur_confirm)
    if a.phase2 == "aliases":
        cur = AliasSchedule(pool, budget=a.alias_budget)
    if a.arm == "gft":
        gk = {k: v for k, v in (("episodes", a.episodes_per_round), ("stage2", a.stage2_per_round)) if v}
        learner = GFT(cfg, pool, model, dev, GFTConfig(profile=a.profile, **gk), seed=a.seed, curriculum=cur)
        clean = learner.clean
    else:
        pk = dict(reward=a.reward, lr=a.lr, epochs=a.ppo_epochs, gamma=a.gamma, entropy_target=a.entropy_target,
                  advantage="gae", profile=a.profile, value_warmup=a.value_warmup)
        if a.arm == "grpo":
            pk.update(GRPO)
        if a.episodes_per_round:
            pk["episodes"] = a.episodes_per_round
        learner = PPO(cfg, pool, model, dev, PPOConfig(**pk), seed=a.seed, curriculum=cur)
        clean = learner.model
    if a.compile:                                   # the deployed policy, and GFT's pi^fb (same network)
        for m in [clean] + ([learner.aux] if hasattr(learner, "aux") else []):
            m.forward = torch.compile(m.forward)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (a.bf16 and dev.startswith("cuda")) else contextlib.nullcontext()
    out = Path(a.out)
    ckpt = out.with_suffix(".ckpt")
    record = dict(meta=meta(dev, a), evals=[], rounds=[])
    nxt = a.eval_every
    resumed = False
    feedbacks = lambda: learner.teacher_calls if a.arm == "gft" else learner.episodes
    spent = lambda: feedbacks() if a.budget_unit == "feedbacks" else learner.episodes      # what --budget counts
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        learner.load_state_dict(st["learner"])
        set_rng_state(st["torch_rng"])
        record, nxt, resumed = st["record"], st["nxt"], True
        if "seconds" in record:
            if a.phase2 == "aliases" or spent() >= a.budget:
                print(f"{out} already finished; nothing to do", flush=True)
                return
            record["extended"] = record.get("extended", []) + [dict(from_episodes=learner.episodes, to=a.budget,
                                                                     unit=a.budget_unit,
                                                                     previous_seconds=record.pop("seconds"))]
            print(f"extending finished run from {spent()} to {a.budget} {a.budget_unit}", flush=True)
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
        row = dict(episodes=learner.episodes, spent=spent(), wall=time.time() - t0,
                   teacher_calls=getattr(learner, "teacher_calls", 0), gstar=getattr(learner, "gstar", 0),
                   verifier_calls=getattr(learner, "verifier_calls", 0))
        with amp:
            for name, req in splits.items():
                row[name] = evaluate(clean, cfg, req, dev, seed=1, detail=True)
        record["evals"].append(row)
        record["curriculum"] = learner.cur.unlock_log
        print(json.dumps({k: (v["success"] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
        save()

    def phase1_converged():
        """Training-task success has not improved by more than converge_tol over the last converge_evals evals."""
        h = [e["train"]["success"] for e in record["evals"]]
        k = a.converge_evals
        return len(h) > k and max(h[-k:]) <= max(h[:-k]) + a.converge_tol

    def running():
        if a.phase2 == "aliases":
            return not learner.cur.finished(feedbacks())
        return spent() < a.budget

    if not resumed:
        log_eval()
    while running():
        with amp:
            info = learner.round()
        at_eval = spent() >= nxt
        if not at_eval:
            info.pop("examples", None)
        record["rounds"].append(dict(episodes=learner.episodes, **info))
        done = not running()
        if at_eval:
            in_phase2 = a.phase2 == "aliases" and learner.cur.phase2_at is not None
            nxt += (a.eval_every2 or a.eval_every) if in_phase2 else a.eval_every
            if a.phase2 == "aliases" and learner.cur.phase2_at is None:
                if phase1_converged() or spent() >= a.budget:
                    learner.cur.start_phase2(feedbacks())
                    record["phase2_start"] = dict(episodes=learner.episodes, feedbacks=feedbacks(),
                                                  converged=phase1_converged())
                    nxt = learner.episodes + (a.eval_every2 or a.eval_every)
            log_eval()
        elif done:                                    # the budget ends between evaluations: evaluate where it ends
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


def heldout_splits(n, tune, goal_pool=None, max_goals=3, heldout_goals=None, exclude_goals=frozenset()):
    """Unseen-world splits. Test and validation seeds come from different streams in [2^31, 2^32) (the training pool
    uses [0, 2^31)) and are checked to be disjoint. With a restricted goal pool / single-goal requests (sanity
    checks) the held-out-combination split is dropped. With held-out goals: *_worlds = requests containing one (the
    new tasks), *_seen = requests without (regression check). With excluded goals: *_worlds and *_combos leave them
    out, and *_unseen = requests containing one (never requested in training)."""
    restricted = goal_pool is not None or max_goals != 3
    if heldout_goals:
        new, seen = (lambda gs: bool(gs & heldout_goals)), (lambda gs: not gs & heldout_goals)
    else:
        new, seen = (lambda gs: not has_heldout(gs) and not gs & exclude_goals), \
            (None if restricted else (lambda gs: has_heldout(gs) and not gs & exclude_goals))

    def make(stream):
        rng = np.random.default_rng(stream)
        worlds = sample_requests(rng, n, rng.integers(2 ** 31, 2 ** 32, size=n), goal_pool=goal_pool, max_goals=max_goals,
                                 combo_filter=new)
        other = None if seen is None else sample_requests(rng, n, rng.integers(2 ** 31, 2 ** 32, size=n),
                                                          combo_filter=seen)
        unseen = None if not exclude_goals else sample_requests(
            rng, n, rng.integers(2 ** 31, 2 ** 32, size=n), max_goals=max_goals,
            combo_filter=lambda gs: bool(gs & exclude_goals))
        return worlds, other, unseen
    test, val = make(10_000), make(20_000)
    assert not set(test[0].seeds) & set(val[0].seeds)
    name = "seen" if heldout_goals else "combos"
    out = dict(val_worlds=val[0], **{f"val_{name}": val[1]}, val_unseen=val[2]) if tune else \
        dict(new_worlds=test[0], **{f"new_{name}": test[1]}, new_unseen=test[2])
    return {k: v for k, v in out.items() if v is not None}


def code_hash():
    """sha1 of the package and experiment sources: identifies the code where there is no git checkout (pods)."""
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha1()
    for f in sorted([*root.glob("gft/**/*.py"), *root.glob("experiments/minicraft/*.py")]):
        h.update(str(f.relative_to(root)).encode() + f.read_bytes())
    return h.hexdigest()


def meta(dev, a):
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        commit = rev.stdout.strip() if rev.returncode == 0 else None
        dirty = (bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
                 if commit else None)
    except OSError:
        commit, dirty = None, None
    return dict(args=vars(a), git_commit=commit, git_dirty=dirty, code_sha1=code_hash(), torch=torch.__version__,
                numpy=np.__version__,
                python=platform.python_version(), host=platform.node(), start=time.strftime("%Y-%m-%d %H:%M:%S"),
                device=torch.cuda.get_device_name(0) if dev.startswith("cuda") else platform.processor() or "cpu")


if __name__ == "__main__":
    main()
