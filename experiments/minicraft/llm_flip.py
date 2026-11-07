"""Ask Claude to turn a feedback message into the description of a better attempt, and check what it writes.

Episodes come from noisy experts (expert action w.p. p, else random); the realistic teacher comments. Claude gets the
teacher's earlier messages as its history (no rules of the game), then one failure's request and feedback, and is
asked to "modify this feedback so that it becomes the description of a better one, make sure it describes a valid
trajectory, pick a negative comment and either drop or modify it".

Each rewrite is parsed back into claims (the teacher's templates) and checked against the real episode extended by
k = 0, 1, 2, ... steps of the expert's plan for its unmet goals (one item collected or crafted per step):
    valid, better   every claim is true at some k >= 1 (and not all at k = 0): a reachable, better attempt
    valid, same     every claim is true of the episode itself (k = 0), e.g. only a remark was dropped
    invalid         no such k
    unparsed        a sentence outside the teacher's language (not checked; shown)
Also: which kind of remark was changed (dropped / modified / added), and how far ahead (smallest k).

  uv run --with anthropic python experiments/minicraft/llm_flip.py --n 100 --out outputs/minicraft/llm_flip/run.json
"""

import argparse
import copy
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import has_heldout  # noqa: E402

from gft.minicraft.core import COLLECT, RECIPES, WorldConfig  # noqa: E402
from gft.minicraft.expert import expert_action, plan  # noqa: E402
from gft.minicraft.goals import GOALS  # noqa: E402
from gft.minicraft.learn import N_WORLD_ACTIONS, cached_pool, run_episodes  # noqa: E402
from gft.minicraft.teacher import (_first_missing_item, achieved, claim_set, compose, episode_facts_all,  # noqa: E402
                                   true_claims)
from gft.minicraft.verify import parse, split_sentences  # noqa: E402

MODEL = "claude-sonnet-5-5"
INSTRUCTION = ("Modify this feedback so that it becomes the description of a better one. Make sure it describes a "
               "valid trajectory. Pick a negative comment and either drop or modify it.")
CELL_ITEM = {cell: item for cell, (item, _) in COLLECT.items()}


def episodes(cfg, pool, n, levels, rng):
    idx = rng.integers(0, len(pool.seeds), n)
    req = pool.requests(idx)
    p = np.array(levels)[rng.integers(len(levels), size=n)]

    def act(e, t):
        return np.array([expert_action(e, b, req.goals[b]) if (not e.done[b] and rng.random() < p[b])
                         else int(rng.integers(N_WORLD_ACTIONS)) for b in range(n)])
    env, log, _, _, _, sat, _ = run_episodes(cfg, req, act, record=False)
    facts = episode_facts_all(log, env, req.goals, sat)
    return [(req.text(b), f, " ".join(x[2] for x in compose(f, "realistic", rng)) or "Okay.")
            for b, f in enumerate(facts)]


def extend(f):
    """The episode followed by the expert's plan for its unmet goals: Facts after 0, 1, 2, ... steps."""
    out = [f]
    unmet = [g for g, ok in zip(f.goals, f.satisfied) if not ok]
    for op in plan(np.array(f.final_inv), unmet):
        f = copy.deepcopy(f)
        inv = list(f.final_inv)
        if op[0] == "collect":
            it = CELL_ITEM[op[1]]
            inv[it] += 1
            f.collected[it] += 1
        else:
            r = RECIPES[op[1]]
            for j, c in r.inputs:
                inv[j] -= c
            inv[r.output] += 1
            f.crafted[r.output] += 1
        f.final_inv = tuple(inv)
        f.satisfied = [inv[GOALS[g].item] >= GOALS[g].count for g in f.goals]
        f.next_needed = {}
        for g, ok in zip(f.goals, f.satisfied):
            if not ok and (x := _first_missing_item(plan(np.array(inv), [g]))) is not None:
                f.next_needed[g] = x
        out.append(f)
    return out


def claims_of(text):
    """(claim set, count claims [(k, m)], unparsed sentences) of a message."""
    parts, counts, unparsed = [], [], []
    for s in split_sentences(text):
        q = parse(s)
        if q is None:
            unparsed.append(s)
        elif q[0] == "eval_count":
            counts.append(q[1][1:])
        else:
            parts.append((None, q[1], s))
    return claim_set(parts), counts, unparsed


def used_up_true(f, x):
    """The teacher's rule: a requested, unmet goal on item x whose x was obtained often enough but not kept."""
    return any(GOALS[g].item == x and not ok and achieved(f, g) and f.final_inv[x] < GOALS[g].count
               for g, ok in zip(f.goals, f.satisfied))


def holds(cs, counts, f):
    ok_counts = all((k, m) == (sum(f.satisfied), len(f.satisfied)) for k, m in counts)
    used = {c for c in cs if c[0] == "used_up"}
    return ok_counts and all(used_up_true(f, c[1]) for c in used) and cs - used <= true_claims(f)


def ask(client, system, request, feedback):
    r = client.beta.messages.create(
        model=MODEL, max_tokens=16000, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Request: {request}\nFeedback: {feedback}\n\n{INSTRUCTION}\n"
                                              "Reply with only the modified feedback inside <feedback></feedback>."}])
    if r.stop_reason == "refusal":
        return None, r.usage
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"<feedback>(.*?)</feedback>", text, re.S)
    return (m.group(1).strip() if m else text.strip()), r.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="failures to rewrite")
    ap.add_argument("--history", type=int, default=200, help="earlier teacher messages shown to Claude")
    ap.add_argument("--levels", default="0.1,0.2,0.3,0.6,0.9", help="expert-action probabilities of the episodes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/minicraft/llm_flip/run.json")
    ap.add_argument("--dry", action="store_true", help="no API calls: f-hat = f (checks the checker: all 'same')")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = WorldConfig()
    pool = cached_pool(cfg, 1000, seed=0, tag="no-heldout-pairs-v1", combo_filter=lambda gs: not has_heldout(gs))
    levels = [float(v) for v in a.levels.split(",")]
    hist = episodes(cfg, pool, a.history, levels, rng)
    system = ("You are an agent learning a crafting game from a teacher's feedback. These are messages the teacher "
              "gave on your earlier attempts:\n\n" + "\n".join(f"Request: {r}\nFeedback: {t}\n" for r, _, t in hist))
    fails = []
    while len(fails) < a.n:
        fails += [e for e in episodes(cfg, pool, 128, levels, rng) if not all(e[1].satisfied)]
    fails = fails[:a.n]
    if a.dry:
        answers = [(e[2], None) for e in fails]
    else:
        client = anthropic.Anthropic()
        with ThreadPoolExecutor(a.workers) as ex:
            answers = list(ex.map(lambda e: ask(client, system, e[0], e[2]), fails))

    rows, c = [], Counter()
    usage = Counter()
    for (r, f, text), (fhat, u) in zip(fails, answers):
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            usage[k] += getattr(u, k, 0) or 0
        if fhat is None:
            c["refused"] += 1
            continue
        old, new = split_sentences(text), split_sentences(fhat)
        dropped = [s for s in old if s not in new]
        added = [s for s in new if s not in old]
        kinds = lambda ss: [(parse(s) or ("unparsed",))[0] for s in ss]
        cs, counts, unparsed = claims_of(fhat)
        path = extend(f)
        ks = [k for k, fk in enumerate(path) if holds(cs, counts, fk)]
        verdict = ("unparsed" if unparsed else "invalid" if not ks else "same" if ks[0] == 0 else "better")
        c[verdict] += 1
        c["edit:" + ("drop" if dropped and not added else "modify" if dropped and added else
                     "add" if added else "none")] += 1
        for k in kinds(dropped):
            c["changed:" + k] += 1
        rows.append(dict(request=r, f=text, fhat=fhat, verdict=verdict, k=ks[0] if ks else None, path_len=len(path) - 1,
                         dropped=dropped, added=added, unparsed=unparsed))
    n = sum(v for k, v in c.items() if k in ("better", "same", "invalid", "unparsed"))
    print(f"{n} rewrites by {MODEL} ({c['refused']} refused), history of {a.history} teacher messages")
    for v in ("better", "same", "invalid", "unparsed"):
        print(f"  {v:9s} {c[v]:4d}  {c[v] / max(1, n):.2f}")
    ks = [r["k"] for r in rows if r["verdict"] == "better"]
    if ks:
        print(f"  better: smallest k (expert steps ahead) mean {np.mean(ks):.1f}, median {np.median(ks):.0f}; "
              f"k = 1 for {sum(k == 1 for k in ks)}")
    print("  edits: " + ", ".join(f"{k[5:]} {v}" for k, v in sorted(c.items()) if k.startswith("edit:")))
    print("  remark changed: " + ", ".join(f"{k[8:]} {v}" for k, v in c.most_common() if k.startswith("changed:")))
    print(f"  tokens: {dict(usage)}")
    for v in ("better", "same", "invalid", "unparsed"):
        for r in [r for r in rows if r["verdict"] == v][:3]:
            tag = v if r["k"] is None else f"{v}, k={r['k']}"
            print(f"\n  [{tag}] {r['request']}\n    f:    {r['f']}\n"
                  f"    fhat: {r['fhat']}" + (f"\n    unparsed: {r['unparsed']}" if r["unparsed"] else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(model=MODEL, instruction=INSTRUCTION, counts=dict(c), usage=dict(usage), rows=rows),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
