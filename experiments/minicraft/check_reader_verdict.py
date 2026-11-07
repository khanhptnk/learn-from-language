"""How does the reader help? (docs/STATUS.md 2026-11-03)
  python experiments/minicraft/check_reader_verdict.py <learner .ckpt> [device]
 Single-task GFT checkpoint (100k). Same tasks, same worlds, K attempts each under:
policy+request; reader+request; reader+request+right success message; reader+request+success message of another
goal; reader+request+the episode's own failure feedback; reader+request+"Thanks!".
Tier 2: the Stage I failures among planks / rope / bricks requests. Tier 3: pickaxe / raft / oven requests (the
right success message is the teacher's message about an expert success in that world: none was ever received)."""
import sys

import numpy as np
import torch
from gft.minicraft.core import WorldConfig
from gft.minicraft.expert import expert_action
from gft.minicraft.goals import GOAL_ID, MAX_GOALS, Requests
from gft.minicraft.learn import CleanPolicy, run_episodes, policy_actions, prompt_text, prompt_tokens
from gft.minicraft.teacher import compose, episode_facts_all
CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/pod/gft/single/gft_v3e.ckpt"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"
torch.set_num_threads(4)
cfg = WorldConfig(success_needs_done=True)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)["learner"]
clean = CleanPolicy(cfg.size, pixels=True, exact_text=True); clean.load_state_dict(sd["clean"]); clean.to(DEV).eval()
aux = CleanPolicy(cfg.size, pixels=True, exact_text=True); aux.load_state_dict(sd["aux"]); aux.to(DEV).eval()
bank = sd["succ_parts"]
rng = np.random.default_rng(11)
text = lambda parts: " ".join(x[2] for x in parts) or "Okay."
T2, T3 = ["make planks", "make rope", "make bricks"], ["make a pickaxe", "build a raft", "build an oven"]


def reqs(names, n):
    g = np.full((len(names) * n, MAX_GOALS), -1); g[:, 0] = np.repeat([GOAL_ID[x] for x in names], n)
    return Requests(g, rng.integers(0, 2**31, len(g)))


def expert_msgs(req):
    env, log, _, _, _, sat, _ = run_episodes(cfg, req, lambda e, t: np.array(
        [expert_action(e, b, req.goals[b]) for b in range(len(req))]))
    f = episode_facts_all(log, env, req.goals, sat)
    assert sat[:, 0].all()
    return [text(compose(x, "realistic", rng)) for x in f]


def bank_msg(r):
    bk = bank[r][-8:]
    return text(bk[int(rng.integers(len(bk)))])


def run(label, req, names, K):
    texts = [req.text(i) for i in range(len(req))]
    env, log, _, _, _, sat, _ = run_episodes(cfg, req, lambda e, t: policy_actions(clean, e, req, DEV, rng))
    fb = [text(compose(x, "realistic", rng)) for x in episode_facts_all(log, env, req.goals, sat)]
    keep = np.nonzero(~sat[:, 0])[0] if label == "tier 2" else np.arange(len(req))
    print(f"\n[{label}] policy success on all tasks:",
          {nm: round(float(sat[req.goals[:, 0] == GOAL_ID[nm], 0].mean()), 2) for nm in names}, "| tasks used", len(keep),
          flush=True)
    right = [bank_msg(texts[i]) for i in keep] if label == "tier 2" else expert_msgs(
        Requests(req.goals[keep], req.seeds[keep]))
    others = {nm: [o for o in T2 if o != nm] for nm in names}
    wrong = [bank_msg(others[texts[i]][int(rng.integers(len(others[texts[i]])))]) for i in keep]
    print("  e.g. right:", right[0], "| wrong:", wrong[0], "| own failure:", fb[keep[0]], flush=True)
    rr = np.repeat(keep, K)
    reqK = Requests(req.goals[rr], req.seeds[rr])
    rep = lambda xs: [x for x in xs for _ in range(K)]
    conds = [("policy, request", clean, None), ("reader, request", aux, [None] * len(rr)),
             ("reader + right success msg", aux, rep(right)), ("reader + other goal's success", aux, rep(wrong)),
             ("reader + own failure feedback", aux, rep([fb[i] for i in keep])), ("reader + 'Thanks!'", aux, ["Thanks!"] * len(rr))]
    for name, model, msgs in conds:
        r2 = np.random.default_rng(0)
        if msgs is None:
            act = lambda e, t: policy_actions(model, e, reqK, DEV, r2)
        else:
            with torch.no_grad():
                p = model.encode_prompt(prompt_tokens([prompt_text(reqK.text(i), msgs[i]) for i in range(len(rr))], DEV))
            act = lambda e, t, p=p: policy_actions(model, e, reqK, DEV, r2, prompt=p)
        _, _, _, _, valid, s2, _ = run_episodes(cfg, reqK, act)
        per = {nm.split()[-1]: round(float(s2[reqK.goals[:, 0] == GOAL_ID[nm], 0].mean()), 3) for nm in names}
        print(f"  {name:32s} {s2[:, 0].mean():.3f}  {per}  steps {valid.sum(1).mean():.0f}", flush=True)


run("tier 2", reqs(T2, 96), T2, K=8)
run("tier 3", reqs(T3, 48), T3, K=6)
