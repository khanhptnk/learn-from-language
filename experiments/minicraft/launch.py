"""Run many train.py jobs, N at a time on one machine (e.g. several seeds/arms sharing one GPU).

  uv run python experiments/minicraft/launch.py --jobs jobs.txt --parallel 4 --logdir outputs/mc/logs

`jobs.txt`: one train.py argument string per line (without the script name); finished jobs (output file exists) are
skipped (outputs are written incrementally; a job counts as finished once its output records the total time), so a
launch can be resumed. Prints a progress line whenever a job starts or ends.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def out_path(args):
    toks = shlex.split(args)
    return Path(toks[toks.index("--out") + 1])


def finished(args):
    """A job is finished when its output records the total run time (outputs are written incrementally) and it
    reached the requested --budget (a larger budget extends a finished run)."""
    p = out_path(args)
    try:
        r = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return False
    if "seconds" not in r:
        return False
    toks = shlex.split(args)
    if "--budget" in toks and "--phase2" not in toks and r.get("evals"):
        return r["evals"][-1]["episodes"] >= int(toks[toks.index("--budget") + 1])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--logdir", default="outputs/minicraft/logs")
    a = ap.parse_args()
    jobs = [l.strip() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    todo = [j for j in jobs if not finished(j)]
    Path(a.logdir).mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} with {a.parallel} in parallel",
          flush=True)
    running, done, failed, t0 = [], 0, 0, time.time()
    script = str(Path(__file__).with_name("train.py"))
    if todo:   # build the shared cached pool and base policy once, before jobs race to create them
        toks = shlex.split(todo[0])
        i = toks.index("--out")
        prep = toks[:i] + toks[i + 2:]
        print(f"[{time.time() - t0:7.0f}s] preparing shared pool and base policy", flush=True)
        subprocess.run([sys.executable, script, *prep, "--prepare-only"], check=True)
    while todo or running:
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            log = open(Path(a.logdir) / (out_path(j).stem + ".log"), "w")
            running.append((subprocess.Popen([sys.executable, script, *shlex.split(j)], stdout=log, stderr=log), j, log))
            print(f"[{time.time() - t0:7.0f}s] start {out_path(j).name}", flush=True)
        for p, j, log in list(running):
            if p.poll() is not None:
                running.remove((p, j, log))
                log.close()
                done += p.returncode == 0
                failed += p.returncode != 0
                print(f"[{time.time() - t0:7.0f}s] {'done' if p.returncode == 0 else 'FAILED'} {out_path(j).name} "
                      f"({done} done, {failed} failed, {len(todo) + len(running)} left)", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
