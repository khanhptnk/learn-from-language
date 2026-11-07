"""Can pi fit one round's kept set, and does fitting it reproduce the kept success? (docs/STATUS.md 2026-11-04)

Resumes a full GFT checkpoint, runs one round and stops it just before pi's distillation, keeping the snapshot
(saved to outputs/minicraft/kept_fit/snapshot.pt; later runs reuse it): pi_0, the Stage II retries, the kept ones and
their steps. Tasks are split 80/20 within each request kind (solvable = some retry of that request succeeded); a copy of
pi_0 is trained by plain BC (no damping, stay term or entropy bonus) on the kept steps of the 80% tasks, and after each
stage reports, for solvable requests:
  NLL / accuracy of the kept actions (train tasks, held-out tasks) and pi's entropy there,
  success on the train tasks' own worlds (memorising the kept episodes is allowed here), on the held-out tasks' worlds,
  and on fresh worlds with the same requests (16 samples per task, temperature 1), plus per-request success.
  .venv/bin/python experiments/minicraft/check_kept_fit.py <run .ckpt> [train.py args of that run]"""
import copy
import json
import runpy
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

import gft.minicraft.gft_message as gm
from gft.minicraft.goals import Requests
from gft.minicraft.learn import CleanPolicy, obs_tensor, policy_actions, run_episodes

torch.set_num_threads(8)
OUT = Path(__file__).resolve().parents[2] / "outputs/minicraft/kept_fit"
OUT.mkdir(parents=True, exist_ok=True)
SNAP_PATH = OUT / "snapshot.pt"


class Stop(Exception):
    pass


def take_snapshot(src_ckpt, run_args):
    shutil.copy(src_ckpt, OUT / "run.ckpt")       # train.py resumes from <out>.ckpt; never touch the original
    st = torch.load(OUT / "run.ckpt", map_location="cpu", weights_only=False)

    def pad(x, old, new):                         # a checkpoint from before the pace words: zero rows for them
        if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == old:
            return torch.cat([x, x.new_zeros(new - old, x.shape[1])])
        return {k: pad(v, old, new) for k, v in x.items()} if isinstance(x, dict) else x

    old = st["learner"]["clean"]["prompt.tok.weight"].shape[0]
    new = CleanPolicy(13, pixels=True, exact_text=True).state_dict()["prompt.tok.weight"].shape[0]
    if new > old:
        for k in ("clean", "aux", "opt_aux", "opt_clean", "scorer", "opt_scorer"):
            if k in st["learner"]:
                st["learner"][k] = pad(st["learner"][k], old, new)
        torch.save(st, OUT / "run.ckpt")
    del st
    snap = {}

    def grab(self, kept_steps, grid, pf, state, a_t, b_t, old_probs, tok_req, *rest):
        f = sys._getframe(1).f_locals             # _stage2's locals
        snap.update(pi0=copy.deepcopy(self.clean).cpu(), cfg=self.cfg, lr=self.g.lr_clean, episodes=self.episodes,
                    grid=grid.cpu(), pf=pf.cpu(), state=state.cpu(), a=a_t.cpu(), b=b_t.cpu(), rtok=tok_req.cpu(),
                    goals=f["reqc"].goals, seeds=f["reqc"].seeds, task_of=f["task_of"], kept=f["kept"],
                    ok=(f["sat2"] | ~f["mask2"]).all(1), s=f["s_att"])
        raise Stop

    gm.GFT._distil_kept = grab
    sys.argv = ["train.py", *run_args, "--device", "cpu", "--threads", "8", "--budget", "10000000",
                "--out", str(OUT / "run.json")]
    try:
        runpy.run_path(str(Path(__file__).parent / "train.py"), run_name="__main__")
    except Stop:
        pass
    torch.save(snap, SNAP_PATH)
    return snap


S = (torch.load(SNAP_PATH, weights_only=False) if SNAP_PATH.exists() else take_snapshot(Path(sys.argv[1]), sys.argv[2:]))
dev, cfg = "cpu", S["cfg"]
grid, pf, state, a, b, rtok = S["grid"], S["pf"], S["state"], S["a"], S["b"], S["rtok"]
task_of, kept, ok = S["task_of"], S["kept"], S["ok"]
reqc = Requests(S["goals"], S["seeds"])
fresh = Requests(S["goals"], np.random.default_rng(123).integers(2 ** 31, 2 ** 32 - 1, len(reqc)))
n_tasks, K = task_of.max() + 1, int((task_of == 0).sum())
req_of_task = [reqc.text(int(np.nonzero(task_of == k)[0][0])) for k in range(n_tasks)]
solv_req = {r for r in set(req_of_task) if ok[[req_of_task[t] == r for t in task_of]].any()}
solv_t = np.array([r in solv_req for r in req_of_task])
rng = np.random.default_rng(0)
test_t = np.zeros(n_tasks, bool)                  # 20% held out within solvable and within unsolvable tasks
for grp in (solv_t, ~solv_t):
    idx = rng.permutation(np.nonzero(grp)[0])
    test_t[idx[:max(1, round(0.2 * len(idx)))] if len(idx) > 1 else []] = True
b_np = b.numpy()
ep_tr, ep_te, ep_sv = ~test_t[task_of], test_t[task_of], solv_t[task_of]
kept_tr = np.nonzero(kept[b_np] & ep_tr[b_np])[0]                 # BC data: every kept step of the train tasks
kept_tr_s = np.nonzero(kept[b_np] & ep_tr[b_np] & ep_sv[b_np])[0]  # NLL reported on solvable requests' kept steps
kept_te_s = np.nonzero(kept[b_np] & ep_te[b_np] & ep_sv[b_np])[0]
print(f"\nsnapshot at {S['episodes']} episodes: {n_tasks} tasks x {K} retries; solvable requests {sorted(solv_req)}: "
      f"{solv_t.sum()} tasks ({(solv_t & ~test_t).sum()} train / {(solv_t & test_t).sum()} held out), retry success "
      f"{ok[ep_sv].mean():.2f}, kept {kept[ep_sv].sum()} (success {ok[kept & ep_sv].mean():.2f}); unsolvable: "
      f"{(~solv_t).sum()} tasks, kept {kept[~ep_sv].sum()}. BC on {len(kept_tr)} kept steps of all train tasks",
      flush=True)


@torch.no_grad()
def step_stats(pol, steps):
    if len(steps) == 0:
        return float("nan"), float("nan"), float("nan")
    nll, acc, ent = [], [], []
    for j in torch.as_tensor(steps).split(1024):
        lp = F_.log_softmax(pol(*obs_tensor(grid[j], pf[j], state[j], dev), pol.encode_prompt(rtok[b[j]]))[0], -1)
        nll.append(-lp.gather(1, a[j, None])[:, 0])
        acc.append((lp.argmax(-1) == a[j]).float())
        ent.append(-(lp.exp() * lp).sum(-1))
    m = lambda x: float(torch.cat(x).mean())
    return m(nll), m(acc), m(ent)


@torch.no_grad()
def success(pol, req):
    torch.manual_seed(0)
    r = np.random.default_rng(1)
    _, _, _, _, _, sat, mask = run_episodes(cfg, req, lambda e, t: policy_actions(pol, e, req, dev, r), record=False)
    return (sat | ~mask).all(1)


def per_req(s, sel):
    return {r.split()[-1]: round(float(s[sel & np.array([reqc.text(i) == r for i in range(len(reqc))])].mean()), 2)
            for r in sorted(solv_req)}


def report(tag, pol, ep):
    pol.eval()
    t0 = time.time()
    tr, te = step_stats(pol, kept_tr_s), step_stats(pol, kept_te_s)
    so, sf = success(pol, reqc), success(pol, fresh)
    row = dict(tag=tag, epochs=ep, nll_train=tr[0], acc_train=tr[1], ent_train=tr[2], nll_test=te[0], acc_test=te[1],
               ent_test=te[2], succ_train_worlds=float(so[ep_sv & ep_tr].mean()),
               succ_heldout_worlds=float(so[ep_sv & ep_te].mean()), succ_fresh=float(sf[ep_sv].mean()),
               per_request_train_worlds=per_req(so, ep_tr), per_request_fresh=per_req(sf, np.ones(len(sf), bool)),
               unsolvable_train_worlds=float(so[~ep_sv].mean()))
    print(f"{tag:>8s}  NLL {tr[0]:.3f}/{te[0]:.3f}  acc {tr[1]:.2f}/{te[1]:.2f}  H {tr[2]:.2f}/{te[2]:.2f}   success "
          f"train-worlds {row['succ_train_worlds']:.2f}  held-out-worlds {row['succ_heldout_worlds']:.2f}  fresh "
          f"{row['succ_fresh']:.2f}  (unsolvable {row['unsolvable_train_worlds']:.2f})   ({time.time() - t0:.0f}s)\n"
          f"{'':>10s}per request, train worlds {row['per_request_train_worlds']}  fresh {row['per_request_fresh']}",
          flush=True)
    pol.train()
    return row


print(f"\n{'':>8s}  solvable requests: NLL train/held-out tasks, accuracy, entropy on kept steps; success at T=1 "
      f"(retries: {ok[ep_sv & ep_tr].mean():.2f} / {ok[ep_sv & ep_te].mean():.2f}; kept: "
      f"{ok[kept & ep_sv & ep_tr].mean():.2f} / {ok[kept & ep_sv & ep_te].mean():.2f})\n{'':>10s}retries per request "
      f"{per_req(ok, np.ones(len(ok), bool))}", flush=True)
rows = [report("pi_0", S["pi0"], 0)]
pol = copy.deepcopy(S["pi0"])
opt = torch.optim.Adam(pol.parameters(), lr=S["lr"])
done = 0
for target in (1, 4, 16, 64):
    for _ in range(target - done):
        for j in torch.as_tensor(rng.permutation(kept_tr)).split(256):
            lp = F_.log_softmax(pol(*obs_tensor(grid[j], pf[j], state[j], dev), pol.encode_prompt(rtok[b[j]]))[0], -1)
            loss = -lp.gather(1, a[j, None])[:, 0].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
    done = target
    rows.append(report(f"BC x{target}", pol, target))
    (OUT / "kept_fit.json").write_text(json.dumps(rows, indent=1))
