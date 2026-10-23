# Project status (as of 2026-10-21)

Goal: validate and develop GFT (General Feedback Training; previously GLFT) -- learning from language feedback of
diverse, mixed intents -- against RL (GRPO / RLOO / PPO), from cheap CPU settings up to LLM agents with human feedback.
Paper under test: `GFT_revised_2026-10-20.pdf` (34 pp). Earlier versions: `GLFT_end_to_end_audited.pdf`,
`GLFT-paper.pdf` (untracked).

Branch `l1-experiments`; latest commits: `fa1d6e6` (proof internals), `148c8f9`, `5af13fe` (verification + rename).
Nothing pushed. `outputs/` (raw JSON) is gitignored; summaries are in `results/` (index: `results/README.md`).

## Done

1. **Tier 0 (exact theory, CPU).** `gft/algos/gft_exact.py`, `experiments/tier0/`.
2. **Theory verification of the revised paper (CPU).** `gft/theory.py`, `tests/test_theory_gft.py`,
   `experiments/theory/verify_gft.py` -> `results/theory/gft_verification.md`. Everything passes. Exact-population
   results, FCP/SDPO, gradient identities, Eq. 89, rejection sampling checked as stated; Thm 8.1 proof steps and
   Thm 8.2 / D.1 hypotheses checked on examples; finite-sample conclusions only qualitatively (practical schedules;
   Eq. 70 vacuous at those sizes).
3. **L1 tabular studies (CPU).** One-shot opaque-mapping bandit (`gft/l1_tabular.py`), depth/horizon (`l1_horizon.py`),
   teacher-intent profiles (`l1_intents.py`), incremental GFT (`l1_online.py`, arm `gft-sgd`).

## Key findings so far

- GFT beats terminal-reward RL when base competence is low and feedback describes what actually happened (achieved
  outcomes); advantage grows with outcome-space size, number of requirements, horizon. Mechanism: hindsight
  grounding pooled across requests (needs a compositional model; unfactored control fails).
- RL wins when feedback is effectively a scalar (counts) or global complaints; pass/fail-only GFT generally loses.
- Prescriptive intents (corrections, hints) give GFT only "this failed" -- content unused (paper agrees, Sec. 12).
- Count-based GFT was depth-invariant; incremental GFT is not, but still beats RL at low competence.
- Finite-sample failure mode: support collapse from noisy fits. Fixes in incremental GFT: auxiliary model as
  clean-policy logits + feedback terms (prior x likelihood), expected-gradient (logit) distillation, tuned step for
  the "not mentioned" term (`unm_lr_frac`). These moved probes to 0.99–1.00.
- **The committed `results/l1/sgd_*` sweeps predate those fixes** (understate current `gft-sgd`); the rerun was stopped.

## Decisions

- Theory/convergence verification on CPU first (done); serious experiments on RunPod GPU.
- **No LLM agents** (2026-10-21): policies are small neural nets; an LLM may be the teacher (feedback writer).
- **GPU-stage environment: MiniCraft** (`gft/minicraft/`, design `docs/minicraft_design.md`), replacing the
  BabyAI-Req / Crafter-Goals plan: high-level requests ("fetch the gem") whose hidden requirements need long recipe
  chains; feedback talks about low-level events, not the request's words. Template teacher (all intents) is verified
  truthful by replay (0 errors / 44,972 sentences); quality report `results/minicraft/feedback_quality.md`.
- Pod choice: RTX 3090 community cloud (~$0.22/hr), 20 GB container + 40 GB volume, public IP; terminate (not stop)
  between sessions. Create only with the user's explicit go-ahead.

## MiniCraft learning setup (2026-10-21)

- First GFT smoke run collapsed. Cause: **coverage** (Def. 2.2) -- the base policy (BC on goals 0-3) never produces
  what ideal feedback for deep goals describes (iron, bridge, gem), so Stage I never sees those messages paired with
  matching behaviour. Not memorization (aux loss ~ entropy). Environment, algorithm and model need no change for this.
- Fix: **adaptive depth-gated curriculum, shared by all arms** (`Curriculum` in `gft/minicraft/learn.py`): goals 0-3
  unlocked at start; the next goal unlocks when the learner's own recent success on the newest goal >= 0.5
  (`--cur-threshold`); 70% of requests include the newest goal, 30% replay. `--curriculum off` for ablation.
- **Diagnostics** logged every GFT round: `coverage` (share of ideal-feedback claim types voiced in actual feedback
  in the last 5 rounds) and `reader_loss_seen` / `reader_loss_heldout` (pi^fb loss on Stage I episodes trained on
  vs 10% held out). Unlock log saved in the output JSON.
- **Size-independent encoder** (conv stack + agent-centred 5x5 crop + global mean/max pooling), for tests on other
  map sizes. Base cache version `v2`. CPU cost 0.27 s per 512-state training step (fine on GPU, slow on CPU).
- **Validation splits** `val_worlds` / `val_combos` (`--tune`), seed-disjoint from the test splits, for tuning.
- Damping (`--damp`) is a tuned hyperparameter; include 1.0 (undamped, the paper's update) in the sweep.
- **CPU smoke with curriculum (10k-episode budget, stopped early: GFT 7.7k, PPO 5.1k episodes):** both arms unlocked goal 4 (stone) after one round, then stalled
  at 0% on it (overall ~21%, = base). `check_reader.py`: base collects stone in 3.6% of episodes; pi^fb trained
  longer **memorizes** (train loss down, held-out up; near-unique details like step counts identify episodes) and
  under ideal feedback collects stone 3.9-6.2% vs 3.1% -- no usable Stage II signal.
- **BC capacity check (`check_bc.py`, CPU, 2k demos, 4k steps):** BC on expert demos of all goals reaches only
  ~35% (val_worlds) / ~6% (val_combos); easy goals degrade (firewood 0.60: the policy turns wood into planks as most
  demos do -- weak goal conditioning); gem 0.00. So the network/training cannot yet represent the deep goals; fix
  this before judging GFT or PPO. GPU sweep (10k demos, 20k steps; trunk depth, width, crop, channels) running.

- **Policy architecture (2026-10-22, `check_bc.py` / `check_dagger.py`, validation splits):** shallow encoders
  memorize training worlds (DAgger fits its batch to loss 0.01 but agrees with the expert on new worlds only ~72%).
  A shared 3x3 conv applied K times with stem re-injection (VIN-style) generalizes better. At about equal compute,
  shallow d256 (35 rounds) and 8-pass deep d256 (14 rounds) both reach ~0.86-0.87 new worlds / ~0.49-0.50 new combos,
  but the deep one needs ~2.5x fewer rounds (episodes) -> chosen as `POLICY_ARCH = dict(d=256, rec=8, trunk=3)`.
  12 passes add nothing. Remaining gaps: new goal combinations (~0.5), and evaluation uses sampled actions.
  New base (`BASE_VERSION = "v3"`): easy goals 0.94 on new worlds; collects stone in 9% of episodes (old: 3.6%).
- **Fairness rule:** identical deployed policy (architecture, init, budget) for all methods; training-only parts
  (PPO critic, GFT's pi^fb) may differ and are reported; size controls: PPO with a separate critic as large as
  pi^fb, and both methods with a larger policy.
- **pi^fb designs** (`FeedbackPolicy(kind=...)`, `--reader`): residual (feedback vector + clean trunk features),
  film (feedback scales/shifts the normalized clean conv map, then conv + crop + pooling), xattn (trunk features
  attend over feedback token states). Chosen by `check_reader.py` diagnostics (held-out key-step loss, Stage II stone
  rate under ideal feedback, also at the best held-out checkpoint), not by GFT test results.

## RunPod setup state

- **Live pod: `u3u9jveo4alkxb`** (`gft-minicraft-ro`, recreated several times when the host GPU was taken), RTX PRO 4000 Blackwell 24 GB, secure
  cloud EU-RO-1, $0.57/hr, CPU quota 10.2 cores. SSH `ssh -p 25991 -i ~/.ssh/id_ed25519 root@213.173.104.44`
  (changes if recreated). **Network volume `yy9bsgna2b`** (`gft-data`, 40 GB, EU-RO-1, $0.07/GB/month, ~$2.80/mo)
  mounted at `/workspace` (code, venv, caches, outputs) -- survives pod loss; any EU-RO-1 pod can attach it.
- The first pod (`m3fpf2ztd34alh`, RTX 4000 Ada, EUR-IS-1, pod volume) was stopped by the watchdog, could not restart
  (host GPU taken), and was deleted 2026-10-22. Lesson: pod volumes are tied to one machine; use the network volume.
- GPU speed: the new GPU is ~2.5x faster per training step (shallow net, batch 1024: 3.9 vs 9.7 ms); CPU-bound parts
  (env stepping, teacher, expert labelling) are not faster.
- Pod scripts: `experiments/minicraft/pod/sync.sh up|down <ip> <port>` (code + cached pools/base; results back to
  `outputs/pod/`), `setup.sh` (uv venv, torch 2.14.1+cu130 -- no cu128 wheel exists for 2.14.1). On the pod run
  `.venv/bin/python`, never `uv run` (pyproject pins CPU torch).
- **Idle watchdog** (`experiments/minicraft/pod/watchdog.sh`, started with nohup on the pod; log
  `/workspace/watchdog.log`): stops the pod (GraphQL `podStop`, pod-scoped key from `/proc/1/environ`) after 20 min
  with no `.venv` python process. Restart it after a pod restart. Bug fixed 2026-10-22: it matched the text
  anywhere in a command line, so a stale ssh shell mentioning `.venv/bin/python` kept the pod "busy" (~$0.2 idle);
  it now matches only processes whose executable is the venv python. Stopped pod still pays for the 40 GB volume
  (~$0.27/day); terminate to pay nothing.
- Speed on the pod: GFT round (512 episodes) 2.7 s, PPO round 3.0 s (CPU: ~140 s / ~215 s).
- **Speed work (2026-10-22):** expert 3x faster single-core (flat-list BFS, cached plan; identical actions, tested
  vs `expert_action_reference`) and parallel via `ExpertPool` (forkserver workers; identical, tested); per-round data
  kept on the GPU; no per-minibatch `.item()` syncs; channels-last convs on CUDA; `--compile` (1.4x). The network is
  compute-bound on the RTX 4000 Ada (~7 ms per 1024-sample step); bf16 would give 3.4x at batch 4096 but is off until
  validated against fp32.
- `runpodctl` v2.14.0 at `~/.local/bin/runpodctl`; API key in `~/.bashrc` (`bash -ic`); never print it.

## Teacher (decision 2026-10-23)

- **Default teacher: "realistic"** (`teacher.compose_realistic`): outcome first; on failure what was used up / still
  missing (sometimes as a how-to hint), how far the agent got, its most frequent mistake; on success brief praise and
  sometimes a summary; occasional chatter / step counts; never empty. Verified truthful by replay (0 false / 1,221).
  Beat "mixed" and "describe" for both GFT and GFT+PPO (graph curriculum, 1 seed): GFT+PPO masters metal at 23k
  feedbacks (mixed 33k, PPO 48k); GFT alone reaches metal at ~135k (mixed stalls at "ready to mine").

## Observations (decision 2026-10-23)

- **Default observation: pixels** (`--obs pixels`, `gft/minicraft/render.py`): 6x6-px tiles per cell (78x78 RGB),
  rendered on the GPU from stored grids; the policy sees only the image (patch conv + learned attention maps instead
  of the agent crop). Rationale: with pixels GFT+PPO masters metal at 36k feedbacks vs PPO 284k (symbolic: 23k vs
  48k), final new worlds 0.40 vs 0.26 -- perception favours the feedback-driven method (1 seed).
- Pixel runs use `learn.FWD_CHUNK = 1024` (set by train.py) to avoid GPU OOM when several jobs share a GPU.

## Next steps

1. Graph curriculum, phase 1 only (running 2026-10-22): GFT, PPO gamma 0.99, PPO gamma 1.0 -- which goals get
   mastered, and whether the discount stops PPO on long goals (iron tool, river).
2. Then the reuse experiment (docs/sequential_goals_design.md, revised): phase 1 to convergence with the graph
   curriculum, phase 2 = new request phrases built only from goals both methods master; GFT should learn each new
   request faster if its feedback grounding is reusable.
3. Later: visual (pixel) observations -- PPO must learn perception from a scalar, GFT's Stage I gives dense
   per-step supervision; may favour GFT when perception must improve during training.
4. Ask before committing (nothing committed since `fa1d6e6`).
