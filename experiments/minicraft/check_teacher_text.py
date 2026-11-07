"""Hunt for false or self-contradictory teacher messages (realistic profile).

Episodes of noisy experts at several skill levels (expert action w.p. p, else random). For every message:
  semantic   each sentence checked by the verifier against a replay of the episode (gft.minicraft.verify)
  literal    sentences or pairs whose plain-English reading is false or contradicts another sentence, although each
             is true under the game's definitions:
    did_not_but_had     "You did not <goal>." although the goal's item was obtained often enough (then used up)
    managed_but_needed  "You managed to <S>." and "You still needed <S's item>." (it had it, then used it up)
    needed_but_holds    "You still needed X." although the agent holds X at the end
    should_but_got      "You should have gotten X first." although the agent did get X during the episode
    plural_for_one      "You had sticks at some point ..." for a goal that takes one
    duplicate           the same sentence twice
    unassigned          an unmet goal among the first two without a corrective while a later one has one (the
                        learner, which assigns correctives to goals in order, would pair them wrongly)
Prints counts per check and examples.

  uv run python experiments/minicraft/check_teacher_text.py [episodes]
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import has_heldout  # noqa: E402

from gft.minicraft.core import ITEM_NAMES, WorldConfig  # noqa: E402
from gft.minicraft.expert import expert_action  # noqa: E402
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import N_WORLD_ACTIONS, cached_pool, run_episodes  # noqa: E402
from gft.minicraft.teacher import achieved, compose, episode_facts_all  # noqa: E402
from gft.minicraft.verify import parse, replay, split_sentences, verify_message  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.6, 0.9]
CORRECTIVE = ("needed", "hint_recipe", "hint_raw", "used_up", "short", "no_done")


def literal(f, text):
    sents = [(s, parse(s)) for s in split_sentences(text)]
    out = []
    got = lambda x: f.collected[x] + f.crafted[x]
    unmet = [g for g, ok in zip(f.goals, f.satisfied) if not ok]
    needed = {q[1][1] for _, q in sents if q and q[0] == "needed"}
    for s, q in sents:
        if not q:
            continue
        k, c = q
        if k == "goal_not_done" and s.startswith("You did not") and achieved(f, c[1]):
            out.append("did_not_but_had")
        if k == "subgoal" and GOALS[c[1]].item in needed:
            out.append("managed_but_needed")
        if k == "needed" and f.final_inv[c[1]] > 0:
            out.append("needed_but_holds")
        if k == "needed" and s.startswith("You should have gotten") and got(c[1]) > 0:
            out.append("should_but_got")
        if k == "used_up" and ITEM_NAMES[c[1]] + "s" in s and all(GOALS[g].count == 1 for g in f.goals
                                                                  if GOALS[g].item == c[1]):
            out.append("plural_for_one")
    if len({s for s, _ in sents}) < len(sents):
        out.append("duplicate")
    n_corr = sum(1 for _, q in sents if q and q[0] in CORRECTIVE)
    if len(unmet) >= 2 and n_corr >= 1:
        has = [g in f.next_needed or achieved(f, g) for g in unmet[:2]]
        if not has[0] and has[1]:
            out.append("unassigned")
    return out


def main():
    cfg = WorldConfig()
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    rng = np.random.default_rng(0)
    c, ex, kinds = Counter(), defaultdict(list), Counter()
    for s0 in range(0, N, 256):
        n = min(256, N - s0)
        idx = rng.integers(0, len(pool.seeds), n)
        req = pool.requests(idx)
        p = np.array(LEVELS)[rng.integers(len(LEVELS), size=n)]

        def act(e, t):
            return np.array([expert_action(e, b, req.goals[b]) if (not e.done[b] and rng.random() < p[b])
                             else int(rng.integers(N_WORLD_ACTIONS)) for b in range(n)])
        env, log, _, acts, valid, sat, _ = run_episodes(cfg, req, act, record=False)
        facts = episode_facts_all(log, env, req.goals, sat)
        for b, f in enumerate(facts):
            text = " ".join(x[2] for x in compose(f, "realistic", rng)) or "Okay."
            c["messages"] += 1
            c["failures"] += not all(f.satisfied)
            tr = replay(cfg, req.seeds[b], req.goals[b], acts[b][valid[b]])
            for s, ok, why in verify_message(text, tr):
                c["sentences"] += 1
                kinds[(parse(s) or ("unparsed",))[0]] += 1
                if not ok:
                    c["semantic_false"] += 1
                    if len(ex["semantic_false"]) < 5:
                        ex["semantic_false"].append(f"{req.text(b)} | {text} | false: {s} ({why})")
            for k in literal(f, text):
                c[k] += 1
                if len(ex[k]) < 3:
                    ex[k].append(f"{req.text(b)} | {text}")
        print(f"  {c['messages']}/{N}", flush=True)
    print(f"\n{c['messages']} messages ({c['failures']} failures), {c['sentences']} sentences")
    print(f"  semantic: false sentences {c['semantic_false']}")
    for k in ("did_not_but_had", "managed_but_needed", "needed_but_holds", "should_but_got", "plural_for_one",
              "duplicate", "unassigned"):
        print(f"  literal {k:20s} {c[k]:5d} messages ({c[k] / max(1, c['failures']):.3f} of failures)")
    print("  sentence kinds: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    for k, xs in ex.items():
        print(f"\n  [{k}]")
        for x in xs:
            print(f"    {x}")


if __name__ == "__main__":
    main()
