"""Does the scorer's s rank Stage II retries by success or by length? (docs/STATUS.md 2026-11-05)

Resumes a GFT checkpoint for a few rounds on CPU (training as usual) and, in each round's Stage II, records every
retry's s = D(retry, m*) - D(source, m*), its success and its length (steps). Within a task (one world, one m*) the source
term is constant, so s orders the retries by D(retry, m*). Pairs are formed within tasks only and reported as
  AUC success:      P(s_success > s_failure)
  AUC short:        P(s_shorter > s_longer), over all pairs of different length, and within successes / failures
  quit beats succeed: P(s_failure > s_success) for pairs where the failure is the shorter episode
0.5 = no preference.
  .venv/bin/python experiments/minicraft/check_scorer_length.py <run .ckpt> <rounds> <tag> [train.py args of that run]"""
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import gft.minicraft.gft_message as gm
from gft.minicraft.core import DONE
from gft.minicraft.learn import CleanPolicy

torch.set_num_threads(int(os.environ.get("THREADS", "8")))
src_ckpt, R, tag, run_args = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4:]
OUT = Path(__file__).resolve().parents[2] / "outputs/minicraft/scorer_length" / tag
OUT.mkdir(parents=True, exist_ok=True)
FRESH = str(src_ckpt) == "none"                   # "none": train from the base model, recording every round
RESUME = not FRESH and src_ckpt.resolve() == (OUT / "run.ckpt").resolve()   # continue this tag's own run and records
if not FRESH and not RESUME:
    shutil.copy(src_ckpt, OUT / "run.ckpt")


def pad(x, old, new):                             # a checkpoint from before the pace words: zero rows for them
    if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == old:
        return torch.cat([x, x.new_zeros(new - old, x.shape[1])])
    return {k: pad(v, old, new) for k, v in x.items()} if isinstance(x, dict) else x


st = None if FRESH else torch.load(OUT / "run.ckpt", map_location="cpu", weights_only=False)
old = new = 0 if FRESH else st["learner"]["clean"]["prompt.tok.weight"].shape[0]
new = new if FRESH else CleanPolicy(13, pixels=True, exact_text=True).state_dict()["prompt.tok.weight"].shape[0]
if new > old:
    for k in ("clean", "aux", "opt_aux", "opt_clean", "scorer", "opt_scorer"):
        if k in st["learner"]:
            st["learner"][k] = pad(st["learner"][k], old, new)
    torch.save(st, OUT / "run.ckpt")
del st
if FRESH:
    (OUT / "run.ckpt").unlink(missing_ok=True)
    (OUT / "run.json").unlink(missing_ok=True)


class Stop(Exception):
    pass


rows, s1rows = [], []                             # one per Stage II retry / Stage I episode
if RESUME and (OUT / "rows.json").exists():
    rows, s1rows = json.loads((OUT / "rows.json").read_text()), json.loads((OUT / "stage1.json").read_text())
first = [True]
orig = gm.GFT._distil_kept


def grab(self, *args):
    f = sys._getframe(1).f_locals
    if first[0]:                                  # records past the checkpoint (a run stopped after it) are redone
        first[0] = False
        rows[:] = [r for r in rows if r["round"] < self.rounds]
        s1rows[:] = [r for r in s1rows if r["round"] < self.rounds]
    n_t = int(f["task_of"].max()) + 1
    base = (max(r["task"] for r in rows) + 1) if rows else 0
    ok = (f["sat2"] | ~f["mask2"]).all(1)
    acts2 = np.asarray(f["acts2"])
    s1, parts = f["s1"], f["parts"]
    L1 = np.bincount(np.asarray(s1["b"]), minlength=len(s1["idx"]))
    last1 = np.asarray(s1["a"])[np.cumsum(L1) - 1]
    for e in range(len(L1)):
        s1rows.append(dict(round=self.rounds, req=f["req"].text(e), ok=not self._failed(parts[e]), len=int(L1[e]),
                           done=bool(last1[e] == DONE)))
    L = np.asarray(f["valid2"]).sum(1) if np.asarray(f["valid2"]).ndim == 2 else None
    if L is None:                                 # steps per episode from the stacked steps
        L = np.bincount(f["b"], minlength=len(ok))
    for i in range(len(ok)):
        k = int(f["task_of"][i])
        star = f["star_txt"][k]
        rows.append(dict(task=base + k, req=f["reqc"].text(i), s=float(f["s_att"][i]), ok=bool(ok[i]), len=int(L[i]),
                         done=bool(acts2[i, L[i] - 1] == DONE) if L[i] > 0 else False,
                         kept=bool(f["kept"][i]), star=star,
                         round=self.rounds))
    print(f"round {self.rounds}: {n_t} tasks, {len(ok)} retries", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows))
    (OUT / "stage1.json").write_text(json.dumps(s1rows))
    if len({r["round"] for r in rows}) >= R:
        raise Stop
    return orig(self, *args)


gm.GFT._distil_kept = grab
sys.argv = ["train.py", *run_args, "--device", os.environ.get("DEV", "cpu"), "--threads", os.environ.get("THREADS", "8"), "--budget", "10000000",
            "--out", str(OUT / "run.json")]
try:
    runpy.run_path(str(Path(__file__).parent / "train.py"), run_name="__main__")
except Stop:
    pass


def pairs(sel_a, sel_b, rs):
    """P(s_a > s_b) (+0.5 ties) over pairs within a task, a from sel_a and b from sel_b (a != b)."""
    win = tot = 0.0
    for t in {r["task"] for r in rs}:
        T = [r for r in rs if r["task"] == t]
        for a in T:
            for b in T:
                if a is not b and sel_a(a, b) and sel_b(a, b):
                    win += (a["s"] > b["s"]) + 0.5 * (a["s"] == b["s"])
                    tot += 1
    return (round(win / tot, 3) if tot else None), int(tot)


def report(rs, name):
    rs = [r for r in rs if np.isfinite(r["s"])]
    out = dict(
        retries=len(rs), success=round(float(np.mean([r["ok"] for r in rs])), 3) if rs else None,
        auc_success=pairs(lambda a, b: a["ok"], lambda a, b: not b["ok"], rs),
        auc_short_all=pairs(lambda a, b: a["len"] < b["len"], lambda a, b: True, rs),
        auc_short_successes=pairs(lambda a, b: a["ok"] and a["len"] < b["len"], lambda a, b: b["ok"], rs),
        auc_short_failures=pairs(lambda a, b: not a["ok"] and a["len"] < b["len"], lambda a, b: not b["ok"], rs),
        quit_beats_succeed=pairs(lambda a, b: not a["ok"] and a["len"] < b["len"], lambda a, b: b["ok"], rs),
        kept=sum(r["kept"] for r in rs),
        kept_success=round(float(np.mean([r["ok"] for r in rs if r["kept"]])), 3) if any(r["kept"] for r in rs) else None,
        mean_len_all=round(float(np.mean([r["len"] for r in rs])), 1) if rs else None,
        mean_len_kept=round(float(np.mean([r["len"] for r in rs if r["kept"]])), 1) if any(r["kept"] for r in rs) else None)
    print(f"{name:>14s} " + "  ".join(f"{k} {v}" for k, v in out.items()), flush=True)
    return out


res = {"all": report(rows, "all")}
(OUT / "summary.json").write_text(json.dumps(res, indent=1))
