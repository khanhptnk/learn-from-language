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

## Pretraining v2 (2026-10-28)

Where it stands: the new game (docs/minicraft_design.md) post-trained from a language-pretrained base gave no
held-out successes (lang5: GFT 0.03 / PPO 0.00; lang6 with pretraining replay: forgetting fixed, held-out 0.00). The
diagnosis: hindsight BC grounds only descriptions of what was done; the held-out step (craft at the site with rope
and a brick) was never described-as-done, so the inferred message had nothing grounded left to say.

Decision (user, 2026-10-28): rebuild pretraining so it teaches a grounded, compositional language of the world from
seen tasks only, keeping the setting realistic -- language-rich teacher data that never reveals the post-training
recipe, a strong explorer that never shows the held-out policy, a bidirectional objective, token-level text fusion.
Plan: `~/.claude/plans/this-is-a-new-glowing-candle.md`; design section "Pretraining v2" in docs/minicraft_design.md;
code `gft/minicraft/pretrain.py`, `train.py --base grounded`, gates `experiments/minicraft/check_grounded.py`.
Built and tested on CPU (teacher narration verified truthful on 5,779 sentences; corpus audit; tiny end-to-end GFT
post-training run). Next: pod ladder `scratchpad/jobs_grounded.sh` (timing -> main base + gates -> proxy held-out
fence base + gates, the composition test); only then post-training arms. Each pod start / launch needs approval.

### 2026-10-29: first pod runs, audit, rerun

- Main base (pre-fix recipe, 20k eps / 10k steps, 55 min): BC on seen requests 0.44 (expert 1.0); composition
  probes with control (`outputs/pod/grounded/gates_main*.json`): the description sentence alone makes the named
  composite 1.7-3x more often than another composite's description (raft 0.66 vs 0.38, wall 0.39 vs 0.14, fence 0.40
  vs 0.13; ops-left deltas 4-10 SE) -- specific steering on *seen* composites, with a large nonspecific "keep
  crafting" component. The fence-held-out run was killed at step 5k (pre-fix recipe; user decision).
- Reader bug: masked words lost their position embedding (one prediction per sentence). Fixed; masked-word loss
  2.9 -> 0.47, and the trajectory is used (items 0.82 vs 0.68 without it). Station words are ~0.85 from text alone
  (the item implies the station), so that metric cannot show pixel grounding here; probes / info gain are the measures.
- Audit by three independent reviewers (model/objective; corpus/teacher/verifier; gates/wiring): all findings fixed
  (cache key ignored the architecture; span-mode BC on event hulls; craft_held spans; free-explorer hindsight not
  held at the end; probe control; diagnostic masks; several small ones). 77,600 teacher sentences true by replay.
  Full suite 1,059 passed.
- Speed: TF32 + bf16 autocast + torch.compile, merged reader forward: 278 -> 115 ms/step, same losses to 3 decimals.
- Rerun launched 2026-10-29 ~06:40 UTC (user: "launch when ready"): corrected recipe, main base + gates, then
  fence-held-out base + gates (`jobs_rerun.sh`, log `outputs/minicraft/grounded/rerun.log` on the pod). Pre-fix bases
  moved to `outputs/minicraft/base/prefix_bases/` on the pod.

### 2026-10-29 (night): corrected-recipe rerun results (`outputs/pod/grounded/rerun_gates_*.json`)

- Both bases retrained with the fixed reader / corpus / spans and the fast path: 31 min each (20k eps, 10k steps).
- Main base (well, sling held out): BC on seen requests 0.59 (was 0.44); masked-word loss 0.23, items 0.92 with
  frames vs 0.81 without; info gain +0.044. Probes with control: the nonspecific "keep crafting" effect is gone;
  oven (0.42 vs 0.06 control) and wall (0.25 vs 0.04) are steered specifically by their description under an
  unrelated request, raft and fence are not (the policy finishes "gather wood" and stops); with the request naming P
  the description helps in all four (+0.11..+0.17 made).
- **Composition test** (fence also held out, never made or requested): under "gather wood" + "You made a fence at a
  site from a plank and stone.", ops-left to a fence 6.80 -> 3.27 (vs 4.60 under another composite's description:
  a specific -1.3 +- 0.17), site visited 45% (vs 14%), plank 22% / stone 27% held at the end (vs 5% / 2%), but the
  fence itself is made in 1.4% (0.0% without the sentence). Under the new-word request "build a fence" (+ the
  sentence): 0.8% / 0.2%. The seen wall description, for comparison, yields walls 31% and fences 3.9% as a
  by-product. Verdict: the grounded words steer toward the right inputs and the right station individually; the
  conjunction (craft *that* at *that* station with *those* inputs) does not transfer zero-shot from one sentence.
  This is the "last step" gap again, now measured before any post-training.
- Pod `jg2buwcqw9fejo` left idle after the runs (watchdog stops it); pre-fix bases kept in `prefix_bases/`.

### 2026-10-29 (morning): fence composition test with controls — the steering is not specific

Rerun of the fence diagnostic (fence-held-out base, 512 episodes per condition, same maps) with two control
sentences (raft, oven descriptions) under the same "gather wood" request. Log: `outputs/pod/grounded/fence_diag_ctrl2.log`.

| under "gather wood" | steps | ops-left | fence | visited site | plank | stone | both | craft at site holding both |
|---|---|---|---|---|---|---|---|---|
| nothing added | 21 | 6.73 | 0.4% | 14% | 5% | 2% | 1% | 0.4% |
| + fence sentence | 76 | 3.22 | 1.4% | 45% | 22% | 27% | 8% | 2.5% |
| + raft sentence (control) | 94 | 4.67 | 3.3% | 47% | 30% | 22% | 10% | 4.7% |
| + oven sentence (control) | 124 | 3.58 | 2.3% | 59% | 36% | 33% | 12% | 5.3% |
| + wall sentence (seen) | 152 | 2.38 | 3.9% | 75% | 33% | 56% | 17% | 7.4% |

Reading: every direct fence-relevant signal (site visits, plank, stone, both, crafting at the site holding both,
fences made) is as high or higher under either control sentence. Any crafting sentence lengthens the episode and
makes the policy gather and craft more; the fence sentence adds nothing specific on top of that. The "specific"
verdict from the gates run rested on ops-left alone, which is a plan-length from the end inventory and depends on
the control chosen (vs raft -1.45, vs oven -0.36); it is not corroborated by the inventory-level measures.
Corrected verdict: the corrected recipe grounds words well enough to steer seen composites (oven, wall) but
shows no zero-shot compositional steering toward a never-made composite, not even toward its parts.

Pod: `jg2buwcqw9fejo` could not restart (host GPU taken); replaced by `mc3yo0sbdpalfc` (RTX PRO 4500, EU-RO-1,
$0.72/hr, ssh -p 47198 root@213.173.110.174) on the same network volume. Both grounded base checkpoints are now
also local in `outputs/pod/grounded/`.

### 2026-10-29 (afternoon): primitive-sentence diagnostic and the paradigm panel

**Primitive-sentence diagnostic** (fence-held-out base, CPU, 256 episodes/condition, scripted inventory stage switch,
so a diagnostic only; script in the session scratchpad `primitive_seq.py` / `primitive_seq_ctrl.py`): prompt
"gather wood" + one narration sentence per stage (pickaxe, stone, plank, then the craft sentence).

| programme | reached pickaxe / stone / third input | fence | wall | raft |
|---|---|---|---|---|
| fence stages, last = failed-craft sentence at a site holding planks and stone | 46 / 38 / 23% | 10.2% | 0.8% | 3.5% |
| fence stages, last = "You made a fence at a site from a plank and stone." | same | 9.4% | 0.8% | 1.6% |
| fence stages, last = wall sentence (control: same station, other inputs) | same | 9.0% | 1.6% | 1.6% |
| fence stages, last = raft sentence (control: other station) | same | 0.0% | 0 | 0.8% |
| fence stages, last = no message | same | 3.1% | 0 | 2.0% |
| fence stages, last = pickaxe sentence (control: other station) | same | 0.0% | 0 | 0.4% |
| wall (seen) stages | 46 / 38 / 15% | 0 | 7.4-8.2% | 0 |
| raft (seen) stages | 93 / 76 | 0 | 0 | 72.7% |
| no sentences | | 0 | 0.8% | 0.4% |

Reading: at the last step the policy follows the station word: of the episodes that arrive holding plank and stone,
~45% craft at the site under a sentence naming the site (fence or wall), 15% with no message, 0% under a sentence
naming the workbench (raft or pickaxe); the input words make no difference (wall and fence sentences tie). The losses are upstream: the pickaxe
sentence is followed 46% of the time and each later primitive loses 20-40%. Fence 10% vs 1.4% one-shot, but the
sequencing was done by the harness, which no paradigm may do.

**Paradigm panel** (37 agents: 7 paradigm-level proposals x 4 adversarial lenses, critic, synthesis; full texts in
`docs/panel_paradigm_synthesis.md`, `docs/panel_paradigm_critic.md`). Decisive findings, all checked:
- *Identifiability (checked from `core.RECIPES`)*: in every seen recipe the input set determines the station and the
  item determines both; no input set occurs at two stations; "rope" is never the first input after "from". So the
  station word carries zero conditional information given the inputs on the training distribution, and no general
  objective is pressured to read it. The fence proxy cannot detect this (stone occurs only at the site, a shortcut
  that is right for the fence, uninformative for the well and wrong for the sling). MiniCraft as configured cannot
  falsify any paradigm on compositional generalization; every proposal's generalization verifier predicted a null.
- *Censoring*: the drop filter removes every accidental held-out craft, so the corpus holds the (site, rope+brick)
  state with zero CRAFT presses (74 steps); every imitation-trained component is trained against the needed action.
- *Recommended paradigm* (the only operator all lenses passed): explicit-EM GFT, i.e. reweight the policy's own
  samples by a learned feedback likelihood F(f* | trajectory), imitate with replay (Lemma 4.1 in sample form,
  RAFT/best-of-N with a generative reward model); F is a conditional language model of the whole feedback given
  frames and actions, trained by MLE on hindsight narration plus verifier-labelled false sentences as negatives;
  a pre-registered identifiability condition: F can score a never-seen f* only if each content word of f* carries
  positive conditional information given the others in the training data.
- *Minimal instance fix*: at least one pair of seen recipes sharing an input set at two different stations (so the
  station word must be read), and rope occurring first; wording and drop filter unchanged.
- Cheapest tests: (0) the identifiability count above (done: zero); (1) splice test on the existing fence base:
  score the 74 (site, rope+brick) replay windows with CRAFT appended under the well sentence vs wall and near-miss
  windows; kill if AUC < 0.6.

### 2026-10-29 (evening): old curriculum recovered; span-aligned GFT design

- The old game (firewood -> gem, graph curriculum, GFT vs PPO from the random base) had been replaced in the
  working tree on 2026-10-28 without a commit. Recovered into the git worktree `../learn-from-language-old`
  (branch `old-curriculum`, from commit 24b4b52) from Claude Code's file-history snapshots of the last pre-switch
  state (learn.py of 20:08 UTC with the filter / NCE / sentence-reader / inferred-ideal arms, teacher, verify,
  infer_ideal, synth_feedback, train.py, tests). Kept the 10-goal list and the cached `v7` random pixel base of the
  Oct 24-25 runs (the 11th goal "build a bridge" of 10-28 dropped). Two fixes on top: 19 teacher words added to the
  vocabulary (the pre-switch vocab edit was not snapshotted), and `teacher.true_claims` now lists every achieved
  unrequested sub-goal (it listed only the deepest; the realistic composer voices up to two, so true messages could
  be treated as false negatives -- a pre-existing inconsistency, 84/200 ideal messages). Old pools and bases copied
  to the worktree's `outputs/`. Checks: all 83 old tests pass; CPU smokes of the discriminator-filter GFT arm (6
  rounds, weights correlate 0.76 with the requirement fraction) and PPO run end to end. Run from that directory
  with `PYTHONPATH=$PWD ../learn-from-language/.venv/bin/python`.
- Design for the next idea (user, 2026-10-29) in `../learn-from-language-old/docs/span_aligned_gft.md`: teacher
  spans as (i) an attention mask + alignment loss on a per-(step, sentence) gate in the sentence reader (BC loss on
  all steps, no loss mask), (ii) gold windows + alignment loss in a per-sentence scorer trained contrastively with
  window-local negatives from the one-fact-improved and ideal messages, (iii) Stage II = K reader rollouts under
  f* (+ K from pi), rerank by the scorer, distil the top one. CPU checks listed there; nothing implemented yet.
- Span-aligned GFT implemented in the worktree (2026-10-29 night): `SentenceReader(gate=True)` (`--reader-gate`),
  `SentenceScorer` + `scorer_loss` (window-local negatives from flip / improve / ideal messages, other-episode
  windows, alignment loss), `--stage2 rerank` (K reader + K pi rollouts per task under f*, top-1 by the scorer,
  damped weighted distillation). Tests: 89 pass (83 old + 6 new). CPU smoke (pixels, random base, 8 rounds,
  1,024 teacher calls; only "gather firewood" open, so a plumbing check): the scorer works on episodes it never saw
  (sentence among same-window negatives top-1 0.94, sentence-to-episode top-1 0.93, attention mass on the span
  ~1.0, D(x, f*) correlates 0.93 with the requirement fraction; within task rank 0.7-0.8; the kept rollout is a
  success every round); the gate finds the spans (held-out AUC 0.93-0.99). But the reader does not use the
  feedback (held-out NLL with = without feedback, every round), its rollouts are no better than pi's, and about
  half the kept rollouts come from each: Stage II is currently best-of-8 of pi under a learned language scorer.
- 2026-10-30: reader switched to a separate network with pi's architecture plus the sentence gate (user); reader
  pretraining added (user): reader and policy trained side by side from scratch on expert demonstrations of random
  requests (all goals, no held-out goals), the reader prompted with the teacher's message (success message = ideal
  feedback), the policy with a blank prompt; every arm starts pi from that policy. CPU smokes of both arms pass;
  93 tests pass in the worktree. Real pretraining needs the GPU (CPU: ~1.4 s per 128-step batch).
- 2026-10-30 audit (three independent reviewers: reader/pretraining, scorer/rerank, speed), all fixed and tested:
  (1) the gate leaked: the prompt encoder's self-attention carried switched-off sentences into the request and the
  visible sentences (~22% of a visible sentence's effect after training); now each sentence attends only to itself
  and the request (a hidden sentence has exactly zero effect, tested); (2) `true_claims` lacked "used up" claims
  (21% of those negatives were true of the episode) and truncated missing-input claims; every voiced claim is now
  in it (11,000 messages, all profiles); (3) scorer diagnostics compared D under a different f* per episode (now one
  f* per task); (4) gate AUC was dominated by whole-episode sentences (now also reported on proper spans only);
  (5) tied ranks in the AUC, random tie-break in rerank, symmetric sampling temperature, Stage II uses the message's
  own sentences, pretraining cache keyed on the code and saved atomically, pretraining no longer shifts the run's
  RNG. Speed: scorer sampling cached per round, windows and alignment vectorized (parity with the old version
  tested), gate without device syncs (parity tested). CPU timing shows no gain (contended; the gains are GPU syncs).
  95 tests pass. Pod job: experiments/minicraft/pod/jobs_span.sh in the worktree (to /workspace/gft_old, not
  /workspace/gft).
- 2026-10-29 23:33 UTC: pod `mc3yo0sbdpalfc` could not start (no free GPU on its host); new pod `uf5ne46eyzz2p6`
  (gft-minicraft-ro6, RTX PRO 4500 Blackwell, EU-RO-1, $0.72/hr, ssh -p 20408 root@213.173.109.45, network volume
  yy9bsgna2b; watchdog started). Old-curriculum code at /workspace/gft_old (shared venv /workspace/gft/.venv, run with
  PYTHONPATH=/workspace/gft_old). Launched (user "go"): jobs_span.sh, log /workspace/span.log, results
  outputs/minicraft/span/: reader+policy pretraining (20k demos, 10k steps), reader check, then GFT rerank and PPO
  in parallel, 100k teacher interactions each (`--budget-unit feedbacks`; PPO: 100k episodes). Dead pods for the
  user to delete: mc3yo0sbdpalfc and the earlier ones.
- 2026-10-30 results (outputs/pod/gft_old/span/, pulled via S3; one seed): reader pretraining (20k demos, 97%
  success, 10k steps): held-out action loss request-only 0.999, + message 0.915, + message with gold spans 0.828;
  gate AUC on proper spans 0.93. Reader check on 512 curriculum tasks: success pi (blank) 0.14, reader + request
  0.08, reader + ideal message 0.29 (gem 0). GFT rerank: stopped by me at 10.2k teacher calls (kill-early rule):
  pi's entropy rose every round (1.08 -> 2.00 nats, entropy bonus ~0) and skills from pretraining decayed outside
  the curriculum frontier (get ready to mine 0.77 -> 0.23, upgrade tools 0.55 -> 0.05) while the frontier advanced
  (3 goals mastered at 9.7k vs PPO 8.7k). Likely cause (untested): Stage II imitates sampled actions of reader
  rollouts while the reader fits pi's own noisy actions. PPO from the same pi, 100k episodes: 6 goals mastered
  (stone pickaxe at 47k), val 0.32; the earlier PPO from the old random base reached 6 goals at 59k, val 0.38 at
  100k.
- 2026-10-30 entropy probe (scratchpad entropy_probe.py, fixed Stage II data from the pretrained reader and pi, 6
  re-anchored passes, one seed): "stay" alone keeps pi's entropy flat (1.12); imitating only pi's own kept rollouts
  lowers it (1.12 -> 1.04) and builds request information (0.007 nats); any imitation of the reader's rollouts
  raises it (full 0.5/0.5 update 1.12 -> 1.18; soft targets the same). The reader conditioned on f* is broader than
  pi (1.23 vs 1.10 nats), so the rise comes from imitating its rollouts.
- 2026-10-30: pod `uf5ne46eyzz2p6` gone; new pod `59de2cn959bslr` (gft-minicraft-ro7, RTX PRO 4000 Blackwell, EU-RO-1,
  $0.57/hr, ssh -p 11794 root@213.173.102.8, volume yy9bsgna2b, watchdog started). Worktree code synced to
  /workspace/gft_old (1050 tests pass; pretrained-reader cache key 00ce745c unchanged). Launched (user "let's just
  launch gft with the new setting"): jobs_span2.sh (top-4 kept, kept-steps normalization, damping 0.5, reader
  replay 4 rounds x 1 pass, no entropy bonus), log /workspace/span2.log, results
  outputs/minicraft/span/gft_rerank2_s0.json, 100k teacher calls.
- 2026-10-30 GFT run 2 (outputs/pod/gft_old/span/gft_rerank2_s0.json, one seed): stopped by me at 35.1k teacher calls,
  stuck on "collect building stone" for ~26k calls (pi 0.3-0.5 in training, mastery needs 0.7). Mastered 4 goals
  faster than PPO (wood pickaxe at 8.7k vs 11.8k) but val 0.17 at 35k vs PPO ~0.25. Entropy plateaued ~1.8.
  Investigation (scratchpad stone_failures.py, 256 stone-only rollouts per policy): pi makes the wood pickaxe 96%
  of the time and stone-only success is 0.54 (PPO final 0.57), but it is slow and noisy: entropy 1.8 vs PPO 0.65,
  pickaxe at median step 58 vs 33, after the pickaxe 18% "use" vs 28% and 26% crafts (11% "craft iron", which
  cannot succeed there); failures mostly end with one stone when steps run out. The reader stopped helping: on the
  stone phase reader+f* success is about pi's, and f* lowers the reader's held-out loss only 1.84 -> 1.80. pi stopped
  pressing DONE (3% vs PPO 47%), so 34% of successful episodes' steps come after success and are imitated; entropy
  there equals entropy elsewhere, so this wastes updates but does not explain the entropy level.
- 2026-10-30 feedback audit on "collect building stone" (scratchpad stone_feedback.py): the realistic teacher's
  failure messages repeat the recipe (88%: "You can get stone from a rock with a wood pickaxe") that pi already
  follows (pickaxe in 96% of episodes); its real errors (40-70 failed crafts per episode, walking past rocks, one of
  two stones, time running out) are named rarely or never; the Stage II targets ("Done: collect building stone") add
  nothing to the request. Pretrained-reader check per goal: f* helps where it names intermediate steps (upgrade tools
  +0.52) and not on the collect-two-of-X goals (firewood -0.09, stone +0.04).
- 2026-10-30 process feedback in the old-curriculum worktree (user: "useful yet still easy to interpret"):
  env step events gain `adj` (neighbour cell bitmask); teacher facts gain missed contacts, done step, DONE pressed,
  timed out; new claims, each one fact, 2 templates, verified against a replay: progress ("You got only 1 stone of
  the 2 you needed."), missed ("You walked past a rock."), kept_going / stopped, out_of_time, clean ("You made no
  mistakes."); realistic messages voice them (repeated mistakes usually, also on successes); the improved message of a
  success removes one process flaw; the ideal message describes the cleanest success in the bank (was uniform);
  20% of reader demonstrations keep going after the plan is complete. Switch: GFT_PROCESS_FEEDBACK=0 restores the
  earlier teacher. 3000 generated messages: 0 false sentences; tests/test_process_feedback.py (7 tests); full suite
  passes. Pool caches keyed by FACTS_VERSION (rebuilt); the pretrained-reader cache key changes (no reuse of 00ce745c).
- 2026-10-30 CPU comparison, old vs process-feedback teacher (scratchpad process_reader_cmp.py; reader + blank pi
  pretrained side by side, 4000 demos, 2000 steps, batch 256: ~1/13 of the GPU run's samples; eval 576 pool tasks incl.
  96 single-goal firewood and 96 stone; one seed). Held-out info gain of the message (request-only minus with-message
  action loss): old 0.016 / 0.013 / 0.007 / 0.023 at steps 400 / 800 / 1200 / 2000, new 0.074 / 0.124 / 0.060 / 0.041.
  Rollouts (success; firewood; stone; failed actions per episode): old reader/req 0.177, 0.61, 0.05, 23.0 vs
  reader/f* 0.146, 0.41, 0.07, 23.5; new reader/req 0.172, 0.62, 0.03, 25.1 vs reader/f* 0.149, 0.50, 0.06, 22.4.
  At this scale the ideal message does not raise success under either teacher (the GPU-scale old reader did: 0.08 ->
  0.29). Inconclusive about the teacher; the queued no-span arm was cancelled (no effect to compare against).
- 2026-10-30: pod `59de2cn959bslr` could not restart (host GPU taken; for the user to delete). New pod
  `5kksow2tjaeb33` (gft-minicraft-ro8, RTX PRO 4500 Blackwell, EU-RO-1, $0.72/hr, ssh -p 13455 root@213.173.107.27,
  volume yy9bsgna2b, watchdog started). Launched (user "go"): experiments/minicraft/pod/jobs_process.sh -- readers
  pretrained at GPU scale (20k demos, 10k steps, batch 512) with the old teacher, the process-feedback teacher, and
  the process-feedback teacher without spans; each tested request-only vs + ideal message. Results in
  outputs/minicraft/process/ on the volume.
- 2026-10-30 GPU-scale reader comparison (outputs/pod/gft_old/process/; 20k demos, 10k steps, batch 512; 576 eval tasks
  incl. 96 single-goal firewood + 96 stone; one seed). Info gain of the message at 10k: old 0.066, new 0.126, new
  without spans 0.136. Success request-only -> + ideal message (stone; firewood; failed actions with f*): old 0.222 ->
  0.280 (0.01 -> 0.16; 0.77 -> 0.61; 23.5), new 0.210 -> 0.312 (0.06 -> 0.20; 0.71 -> 0.61; 24.4), new no spans 0.179 ->
  0.344 (0.04 -> 0.26; 0.64 -> 0.59; 34.2). Blank pi 0.16 / 0.155 / 0.144. Reading: the ideal message helps more with
  process feedback, most without spans (vs old ~2 SE); spans are not needed in pretraining. But "no mistakes / stopped"
  does not make the reader cleaner (failed actions not lower, episodes longer): the gain is from goals / steps named.
  f* lowers firewood in every arm (unexplained).
- 2026-10-30 sharpening probes (scratchpad entropy_probe.py, old reader, fixed data, 6 rounds, start 1.09): full update
  as run 1.15 -> 1.04 (this pool; earlier probe ended above start); soft targets 1.13; soft targets sharpened T=0.5
  -> 0.46; reader rollouts sampled at T=0.5 with sampled targets -> 0.84 (steady), request info 0.012; T=0.5 + pi-only
  kept 0.86; T=0.5 + sharpened targets 0.44. Recommendation: stage2_temperature 0.5, sampled targets, Stage I at T=1,
  bonus-only entropy floor ~0.8.
- 2026-10-30: pod `5kksow2tjaeb33` could not restart (host GPU taken; for the user to delete, with `59de2cn959bslr`).
  New pod `roerqtlkphsqk4` (gft-minicraft-ro9, RTX PRO 4000 Blackwell, $0.57/hr, ssh -p 28457 root@213.173.104.44,
  volume yy9bsgna2b, watchdog started). Launched (user "go"): jobs_span3.sh -- GFT (rerank, top-4 kept, damping 0.5,
  Stage II temperature 0.7, bonus-only entropy floor 0.8) and PPO (entropy target 0.8), both from the
  process-feedback no-span reader / pi (outputs/minicraft/process/new_nospan.pt, new train.py --reader-file),
  100k teacher calls each; results outputs/minicraft/span3/.
- 2026-10-30 run 3 results (outputs/pod/gft_old/span3/, one seed). PPO (100k): 7 goals (stone 30.2k, stone pickaxe 44k,
  metal 68k), val 0.36. GFT stopped by me at 40.2k calls (kill-early): 4 goals by 5.1k calls (PPO 11.3k), then stuck
  on "collect building stone" (eval 0.42 / 0.44 / 0.44 at 30 / 35 / 40k; pi's Stage I stone 0.4-0.5), val 0.18 -> 0.13.
  Entropy collapsed to 0.3 in ~13 rounds at Stage II temperature 0.7, then the bonus-only floor (alpha 0.01 -> 0.15,
  too slow for GFT's loss scale) overshot to 0.97 and settled ~0.8. Stone failure mode at 10k calls (256 stone
  tasks): success 0.07, DONE pressed before success 0.87 (start pi 0.81, PPO 0.00), no stone collected 0.64. The
  teacher has no claim for quitting early; "walked past a rock" (54%) and failed crafts (41%) are voiced.
- 2026-10-30 fixes after run 3 (old-curriculum worktree): teacher claim quit_early ("You stopped before you were done." /
  "You said done too early.") on failures that end with DONE (GFT_QUIT_EARLY); reader_corpus records the expert's own
  steps and pretrain_reader(policy_expert_only=True) trains pi only on them; GFTConfig entropy_lr / entropy_alpha0
  (train.py --entropy-lr / --entropy-alpha0). 8 process tests, full suite pass. Pod `roerqtlkphsqk4` restarted (ssh -p
  14507 root@213.173.104.44, watchdog started). Launched (user "go"): jobs_span4.sh -- pretraining (20k demos, 10k
  steps, no spans, pi on expert actions), then GFT (Stage II T=0.7, entropy floor 0.8 with lr 0.1 / alpha0 0.1), GFT
  reader-only (--rerank-no-pi) and PPO from that pi, 100k teacher calls each; results outputs/minicraft/span4/.
- 2026-10-30 run 4 stopped by the user right after pretraining ("let's just stop the pod"); pod `roerqtlkphsqk4`
  stopped. Pretraining result (outputs/pod/gft_old/span4/, 20k demos, 10k steps, no spans, pi on expert actions,
  quit_early claim): info gain 0.173; success pi 0.141, reader/req 0.149, reader/f* 0.339 (stone 0.12 -> 0.25; get ready
  to mine 0.17 -> 0.95); pi still ~25 failed actions per episode. GFT / GFT reader-only / PPO had only just started
  (partial files only). Open concern (user): pretraining covers every goal and the ideal message names them, so the
  reader is already a goal-conditioned policy; testing learning from feedback needs goals held out of pretraining.
- 2026-10-30 run 5 launched (user "go"; RunPod balance $4.81 before): jobs_span5.sh on pod `roerqtlkphsqk4` (ssh -p
  39027 root@213.173.104.44, watchdog started) -- no reader pretraining: GFT, GFT reader-only and PPO all from the
  text-free random base (base_v7_random_pixels), GFT's reader a gated copy of it learning language only from Stage I
  feedback; process teacher with quit_early; Stage II T=0.7; entropy floor 0.8 (lr 0.1, alpha0 0.1); 100k teacher
  calls each; results outputs/minicraft/span5/.
- 2026-10-30 run 5, added (user request): PPO with binary reward (--reward bin: 1 iff the whole request is met), same
  base and settings, 100k episodes, outputs/minicraft/span5/ppo_bin_s0.json. PPO (frac reward) finished: 100k
  episodes, 7 goals, val 0.33.
- 2026-10-30 run 5 GFT arms stopped by me at 30.2k calls (kill-early): both 4 goals (firewood 2.3k, then by 20k), val
  0.14-0.15, stuck since ~20k; PPO (frac) 100k: 7 goals, val 0.33. Reader feedback gain (no feedback minus feedback
  held-out loss) stayed ~0.0-0.04 nats; reader-with-f* minus pi success in Stage II ~+0.00-0.03, the same in runs 1-3
  (pretrained readers start +0.11 and lose it): the reader does not learn to use feedback. Likely cause: in Stage I the
  reader also gets the request pi acted on, so the message adds little. NB: the run 5 reader-only arm kept 4 of 4
  rollouts per task (no reranking at all).
- 2026-10-30 new Stage II option (user design): --rerank-target improved (reader conditioned on the one-step-better
  message of a Stage I episode, in its world/request), --rerank-reader-imitate (the reader also imitates kept rollouts
  scored above the Stage I episode they improve on; Stage I training stays as the anchor), --rerank-threshold-pi
  (pi imitates only kept rollouts that pass the same threshold). Tests: 23 pass (span-GFT + process).
- 2026-10-30 run 6 launched (user "go"): jobs_span6.sh on pod `roerqtlkphsqk4` (ssh -p 39027), next to the binary
  PPO -- reader-only Stage II on improved messages (8 rollouts per task, top 4), threshold for reader and pi, reader
  imitation; plain random base; results outputs/minicraft/span6/.
- 2026-10-30 binary-reward PPO (span5/ppo_bin_s0, plain random base, 100k episodes): 5 goals, val 0.18 (frac-reward
  PPO: 7 goals, val 0.33). Run 6 at 10.2k calls: 2 goals, val 0.13 (run 5 GFT ~0.09); Stage II: source episodes 0.54,
  reader on improved message 0.66, kept 0.83, threshold passed 0.53, ~93 reader-imitated rollouts per round.
- 2026-10-30 run 6 extra arm (user "do 0.9"): same as run 6 with --damp 0.9 (pi's Stage II loss 0.1 stay + 0.9 push),
  outputs/minicraft/span6/gft_damp09_s0.json. A --damp 1.0 arm started by mistake (the user had rejected that command;
  it ran before the rejection) was killed within 2 minutes and its files removed.
- 2026-10-30 run 6 (damping 0.5) stopped by me at 55k calls (kill-early): 4 goals since ~14.6k (stone stuck ~40k
  calls), val 0.19 -> 0.13; Stage II last rounds: pi's Stage I episodes 0.35-0.42, reader on improved message ~0.58,
  kept ~0.74-0.77, threshold passed 0.53-0.61, ~95 reader-imitated rollouts per round; reader feedback gain ~0.015.
  pi lags its imitation targets by ~0.3 (slow distillation). Damping-0.9 arm continues (at 5k calls: pi's Stage I
  0.46 vs 0.32 with damping 0.5 at the same point).
- 2026-10-30 damping-1.0 arm launched (user request): as run 6 with --damp 1.0 (no stay term),
  outputs/minicraft/span6/gft_damp10_s0.json. Damping 0.9 at 10.2k calls: 4 goals (damping 0.5: 2; PPO frac: 3), val
  0.16; at 15.1k: val 0.22, now on stone.
- 2026-10-30 damping-0.9 arm stopped by me at 45k calls (kill-early): 4 goals by 10.2k (fastest GFT start), then stone:
  pi's Stage I stone 0.01 -> 0.24 in ~15k calls (damping 0.5: ~16k), plateau ~0.39 then 0.30; reader ~0.5, kept ~0.66;
  val 0.22 at 15k, then 0.09-0.18. All damping values (0.5 / 0.9 / 1.0) stall at pi ~0.4 on stone with kept ~0.66-0.74:
  damping changes the speed, not the ceiling. RunPod balance $2.91 (damping-1.0 arm still running, $0.58/hr).
- 2026-10-30 damping-1.0 arm stopped by me at 40k calls (kill-early): 4 goals, val 0.16, pi's Stage I stone ~0.38-0.40
  (same ceiling). Stone failure analysis (scratchpad stone_traj6.py, 256 stone-only tasks): GFT pi (damping 0.5 / 0.9)
  success 0.33 / 0.00, presses DONE before success 0.62 / 0.77 -- after the wood pickaxe it repeats the wood/plank routine
  and quits (the "get ready to mine" habit), "use" only 6-10% of actions, few rock contacts; PPO frac 0.41, never quits,
  spams "use" (54%, ~70 failed uses) until time runs out. Reading: pi does not separate the stone request from "get
  ready to mine" (request info ~0.01 nats); kept failures that quit early may teach quitting (not logged).
- 2026-10-31 teacher audit by reading cases (user: "read, case by case"; scratchpad cases.py, 60 episodes from the base,
  GFT run-6 pi, PPO and a 0.7 noisy expert, with full event timelines). Problems found and fixed (realistic teacher,
  PROCESS_FEEDBACK on): a requested item made then consumed below the count was blamed on missing inputs (used_up now
  fires whenever some was obtained and not kept); deep goals were diagnosed by the next raw resource ("walked past a
  tree" for the gem) -> the next milestone tool ("You still needed a stone pickaxe." / its recipe when part is held);
  workbench complaints aimed at other items' crafts -> "You tried to make a stone pickaxe away from a workbench."
  (per-item station failures); circular "Without a plank..." for the planks goal -> "You had what you needed for a
  plank but did not make it."; the cause was dropped at random -> one cause per unmet goal, always said, in a fixed
  priority (used up, partial, river step, ready craft, milestone, never went to / walked past the resource), goals
  closest to done first, one remark per item; repeated failed hits counted ("You hit the ore 80 times without a stone
  pickaxe."); new "You had a wood pickaxe but never went to a rock.", "You gathered 5 wood you did not need.";
  ordering advice ("Next time get ore before anything else.") removed; "You never managed to X" -> "Not done / Still
  to do" when it holds some; "an ore" -> "the ore"; "used empty ground" -> "pressed use on nothing"; success messages
  state kept-going / habits reliably. Verified: 3000 generated messages 0 false sentences; 1059 tests pass. The
  improved (one-step-better) messages are still to be fixed (user: later).

### 2026-10-30 — Teacher audit, round 2 (case by case, old-curriculum worktree)

Read 60 new episodes (seed 23: base, GFT run 6, PPO, expert 0.7), then about 45 more after the fixes changed the random stream.
Fixes in `teacher._cause_part` and `compose_realistic`:
- **Priority order:**
  - "You had a <tool> but never went to <cell>" now comes before the milestone or recipe hints.
    - Before: "A stone pickaxe is made from…" for an agent holding a wood pickaxe that never visited a rock.
    - Before: "A bridge was still missing" for one holding a stone pickaxe that never visited ore.
  - For river goals, the bridge now counts as the goal item, not as a milestone.
  - When no milestone tool is missing and it holds part of the goal's inputs, the goal's own recipe now comes before deeper
    crafts. Before: "You had what you needed for a plank" for the request "upgrade your tools" while holding stone.
- **Wording:**
  - When it still holds some of a used-up item: "You used some of the <item> you needed", not "You used it up".
  - "You did not / never managed to <goal>" is no longer used when the item was obtained at some point.
  - A step count is no longer given after "ran out of time".
  - "Okay." / "Thanks for trying." no longer follow a success.
- **Habits:** a mistake repeated 20 or more times is always reported (one PPO episode had 167 ore hits that went unmentioned).
- **Checks:** 0 false sentences in 3000 messages; 1059 tests pass.
- **Not fixed:**
  - Messages for 3-goal requests run to 7–8 sentences.
  - The improved messages are untouched.

### 2026-10-30 — Teacher audit, round 3, and no numbers in the text

Read 60 new episodes (seed 41) one by one. At the user's request (too little data to learn counts), the realistic teacher no longer says numbers:
- **progress:** "You did not have enough stone." / "You needed more planks."
- **unneeded:** "You gathered more wood than you needed."
- **use_fail_n:** "You kept hitting the ore without a stone pickaxe."
- **wasted:** "You kept pressing use on nothing", only for a habit; otherwise "You pressed use on nothing."
- **station_fail:** "Crafting failed because no workbench was next to you."
- **Recipe hints:** "A bridge is made from iron and planks at a workbench."
- **Dropped from the realistic teacher:** "k of m done" and "That run lasted N steps".

The claims keep their counts internally. The verifier checks the wording: "kept …" requires at least REPEAT_MIN, "more than needed" requires at least 1, "not enough" requires 0 < held < needed. Six words were added to the vocabulary.

Also fixed:
- **Circular "Without a gem this was not going to work".** It reached the gem on the last step; nothing else was missing. Now no cause is given when only the goal's own item is left.
- **Successes** now always confirm the deepest requested goal (it was 90%).

Checks: no digits in the realistic messages of the batch; 0 false sentences in 3000 messages; 1059 tests pass. Any pretrained reader now needs retraining (wording and vocabulary changed).

### 2026-10-30 — Teacher audit, round 4

Read 60 new episodes (seed 59) one by one. None were false or misleading.
- **Weaker messages in 7 of them:** the agent held the right pickaxe for a resource the plan needs, but the teacher gave a recipe hint or "Without a stone pickaxe…". The cause: the "had the tool" diagnosis only looked at the plan's first step.
- **Fix, `teacher.reach_op`:** the diagnosis now looks at the first resource in the plan, before any river step, whose tool the agent holds at the end. "You walked past <cell>" for such a resource now ranks with "You had <tool> but never went to <cell>", ahead of the milestone and recipe hints. The verifier uses the same rule.
- **Examples:**
  - #34: "Without a stone pickaxe this was not going to work" became "You had a wood pickaxe but never went to a rock". It had a wood pickaxe from step 82 and no rock contact after.
  - #37: the iron-pickaxe recipe became "You walked past the ore". It had a stone pickaxe from step 162 and stood by ore at step 164.
- **Checks:** 0 false sentences in 3000 messages; 1059 tests pass.

### 2026-10-30 — Sentence spans, read one by one

Printed each realistic sentence with its span (`teacher.claim_steps`) next to the full event timeline, over 48 episodes (seed 71), and read them.
- **Whole-episode spans:** most of the new diagnosis claims had no case in `claim_steps`, so they covered the whole episode.
- **Fixed spans:**
  - kept hitting X: the failed uses.
  - pressed use on nothing: those steps.
  - tried to make X away from Y: those failed crafts.
  - had the tool but never went to X: from getting the tool to the end.
  - had what you needed for X: from when it last came to hold every input to the end.
  - used up / used some: the crafts that consumed it.
  - bridge stayed in your pack: from making the bridge to the end.
  - bridged but stayed on this side: from laying the bridge to the end.
- **Goal and sub-goal spans:**
  - "Along the way you managed to X": the part events up to when X was first obtained in total. Before, it ran to the end, e.g. "collect building stone" covered 0–83 when the stone came at 50.
  - "Done: X": the build-up of what it ended with, from the last time it used up all of the item. For gathered items, it is the gains that make up the final amount. Before, it was the whole episode.
- **Unneeded gains:** "more X than you needed" now covers only the gains the plan did not need. The steps travel in the claim as `c[3]`, like "walked past" does.
- **Checks:** no empty spans; 0 false sentences in 3000 messages; 1059 tests pass.

### 2026-10-30 — Teacher passes until clean (messages and spans, read one by one)

Made four passes of 48 episodes each (seeds 83, 97, 113, 131), reading every message and span against the timeline. The last pass found nothing to fix.
- **Pass 1:**
  - "Walked past" missed a contact that began before the agent had the tool. It now counts from the step the tool is held.
  - A contact still going when the agent says done now counts.
  - "Done: get across the river" / "You managed to get across the river" spanned the whole episode. They now span laying the bridge and crossing.
- **Pass 2:**
  - River-only requests never got "walked past": the check ignored river goals, though the bridge needs wood and ore. Fixed.
  - "You kept trying to make X without Y" was said after a single attempt. It now needs at least `KEPT_MIN`=3 tries; the verifier checks the count.
- **Pass 3:** "You tried to make a stone pickaxe away from a workbench" was said about an attempt with no inputs, while the real cause was that it held everything at the end and quit. Station remarks now count only attempts made while holding every input (`craft_fail_ready`; the span uses the same steps).
- **Pass 4:** clean.
- **Checks:**
  - Sweep over the 582 spans in the four passes: none empty, none out of range.
  - 0 false sentences in 3000 messages; 1059 tests pass.

### 2026-10-31 — Improved message from the feedback text alone (old curriculum)

Requirements: the improved (one-step-better) message is inferred purely from the received feedback, describes an improved trajectory, and must be easy to generalize to. The old `teacher.improve_feedback` read the episode's facts and the task graph; it is replaced by `gft/minicraft/improve.py`.
- **New teacher sentences** (option B): the realistic teacher now narrates, on half of failures, the run's last gain in the item and place words its causes use: "You got a stone pickaxe." / "You got more stone." / "You used a rock." (also "obtained", "went on to get more", "harvested"). The span is that gain plus the approach to it. A tool is never "more" ("got more pickaxes"). The verifier checks them: got ≥ 1, more ≥ 2, collected from that cell.
- **The rewrite:**
  - The first cause sentence becomes its narration counterpart by copying its word.
    - never went to / walked past C → "You used C."
    - still needed / hint / had what you needed for / away from a station X → "You got X."
    - used up / needed more X → "You got more X."
    - bridge not laid → "You bridged the river."
    - bridged but not crossed → "Done: get across the river".
    - With no cause, a "tried to make X without Y" / "kept hitting C without T" remark is fixed instead.
  - It keeps the done verdicts, sub-goals and bridge.
  - It drops the "Not done" verdicts, the other cause, how the run ended, mistakes, unneeded work, the old last gain and chatter.
  - A success loses one process flaw: "kept going" becomes "stopped as soon as you were done", or a mistake remark is dropped.
- **Why the "Not done" verdicts go:** keeping them gave "Still to do: upgrade your tools. You got a stone pickaxe.", a pair the teacher never says. Learning which gain finishes which goal from co-occurrence in received messages was tried and rejected. At 3000 messages it learned 18 links, several false: "got iron → get some metal" was wrong in 12% of fresh episodes, and "got more wood → get ready to mine" is an expert confound. It also missed the needed "got a stone pickaxe → upgrade your tools". The request is in pi^fb's prompt anyway.
- **Reading:** 4 batches of 48 episodes (seeds 201, 307, 523, 619) were read case by case.
  - Every failure in them got an improved message (122 of 122).
  - Fixes from reading: "You obtained sticks" for one stick became "a stick"; another goal's cause is dropped (it could be upstream: "had what you needed for a plank" next to "got a wood pickaxe"); the verdict contradiction above; "more pickaxes".
- **Wired in:** `stage2_target="improved"` and `rerank_target="improved"` call `improve_feedback(parts, rng)`. Failures without a stated cause are left out of Stage II.
- **Checks:** 0 false sentences in 3000 teacher messages (replay); 1074 tests pass. A test checks that improved sentences are all forms the teacher voices.
- **Caveats:**
  - The reader must be retrained: there are new sentence types and words.
  - Improved messages are often short ("You used the ore."), and "You got more X" is vague about amounts.
  - Not yet run in GFT.

### 2026-10-31 — Improved messages made to sound like received feedback (passes)

User question: do the corrected sentences sound like anything the learner has seen in received feedback? Measured with `scratchpad/seen_check2.py`: received = teacher messages on 1536–4800 episodes; improved = rewrites on fresh failures.
- **Pass 1 (teacher):** a repeat gain is narrated "You got X" half the time, not always "You got more X" (both true). Before, "You got a plank" was never received and "You got a wood pickaxe" twice in 4800 messages. Still 0 false in 3000 (replay).
- **Pass 1 (rewrite):** sentences now stay next to the fix only if received together with it. This brought "not done" verdicts back where they fit, e.g. "Not done: get some metal. … You used a rock.". "Still to do: upgrade your tools" next to "You got a stone pickaxe" is never received, so it is dropped.
- **Pass 2:** whole messages, not just pairs. Sentences are added in order while some received message said all of them with the fix (`Received`: claim sets indexed by claim). Every improved message is what is left of a received message after dropping sentences, which is what Stage I sentence dropout shows the reader. Before this, 64% were.
- **Pass 3:** the fix sentence takes a wording received before (the other template; "You got X" for "You got more X").
- **The received store in training:** `GFT._improved` feeds it every Stage I message, and it is saved in checkpoints.

Results with 1536 received messages from each source:

| Received stream | Improved sentences received before | Fix sentences received ≥20× | Inside a received message | Keeps a "not done" |
|---|---|---|---|---|
| base policy | 100% | 75% | 100% | 85% |
| GFT run-6 policy | 99% | 55% | 98% | 88% |
| noisy experts | 100% | 29% (71% ≥5×) | 100% | 86% |

- **Never received** (GFT run-6 stream only): "You used / harvested the ore", "You obtained iron". That policy never mined ore, so no message ever said it. Such a fix comes out alone.
- **No improved message has exactly a received message's claim set:** received failure messages always carry how the run ended and a cause, which the rewrite drops.
- **Reading:** about 120 examples across the three streams were read case by case. No contradictions; verdicts that the fix may have finished are dropped.
- **Checks:** 1075 tests pass (the 107 touching improved messages re-run after pass 3).

### 2026-10-31 — Run 7 launched (user "go")

- **Pod:** ro9 could not start (no free GPU on its host). New pod `wl4uq8pacgsuwb` (gft-minicraft-ro10): RTX PRO 4000 Blackwell, EU-RO-1, $0.57/hr, ssh -p 36805 root@213.173.102.12, network volume yy9bsgna2b. Watchdog running. RunPod balance $2.53 before launch.
- **Code:** worktree synced to /workspace/gft_old. `FACTS_VERSION` is now 3 (Facts changed), so the pool was rebuilt locally and synced. 25 improve/process tests pass on the pod.
- **Job:** `experiments/minicraft/pod/jobs_span7.sh`, log /workspace/span7.log, results outputs/minicraft/span7/.
- **Recipe:** run 6b (reader-only Stage II on improved messages, top 4 of 8, threshold, reader imitation, damping 0.9, Stage II T=0.7, entropy floor 0.8, random base, no pretraining) with three changes:
  - the audited teacher;
  - improved messages built from received feedback only;
  - `--epochs-clean 2`: about half the reader's updates per round. In run 6a, pi lagged the reader by 0.13–0.31 on stone for about 70 rounds.
- **Budget and comparison:** 100k teacher calls, evaluation every 5k. The comparison is span5 PPO (same base): 7 goals, val 0.33.
- **Speed:** 2,560 calls in about 2 minutes, so roughly 80 minutes for 100k.
- **Local CPU smoke** of the same command: 2 rounds and 1 evaluation with no errors; stopped once the pod run was verified.
- **Run 7, first try, stopped by me at about 10k calls** (outputs kept in span7_notdone/). The improved messages had brought back "Not done" verdicts, e.g. "Not done: gather firewood. You used a tree.", so the scorer ranked rollouts that finished the goal below ones that stopped a step short.
  - Kept rollouts were worse than the raw reader in early rounds: 0.14–0.20 vs 0.33–0.41. Run 6b had 0.73 vs 0.35.
  - Firewood took about 9.5k calls to master; run 5 took about 2.3k.
  - Fix: improved messages always drop "Not done" verdicts (still inside a received message). Synced; tests pass on the pod.
- **Run 7 relaunched** (user "go") with the fix, same recipe, outputs/minicraft/span7/.
- **Run 7 (fixed) stopped by me at 7.7k calls** (kill-early). It is still on its first goal: policy firewood 0.26–0.50, while run 5 mastered firewood at 2.3k calls and run 6b had 4 goals by 10.2k.
  - Stage II adds nothing. The reader (0.19–0.50) is no better than the policy, and kept rollouts are no better than the reader: mean kept minus reader −0.04 over 30 rounds, −0.007 over the last 10.
  - Reading: the text-only improved message for a firewood failure ("You used a tree." / "You got more wood.") does not separate a finished attempt from a partial one, so reranking by match is close to random with respect to success. Run 6's facts-based message could say "Done: gather firewood" when that was the next step.
  - Removing "Not done" was necessary but not sufficient. Pod idle; the watchdog stops it after 20 minutes.

### 2026-10-31 — Run 8 prepared: no spans, no hidden truth (not launched)

- **Scorer** (`--scorer-mode message`): whole-message pairs from received feedback only.
  - Positive: (x_e, f_e).
  - Negatives: (x_e, improved message of e); (x_e, a received success message for the same request) if e failed; other episodes' messages.
  - Trajectory direction: (x_j, f_e).
  - D is the mean of the sentence scores. No spans, no alignment, no `true_claims`.
- **Stage II:** keep x′ if D(x′, f̂) > D(x′, f), up to 8 of 16 at random. Each round logs the pass rate for finished and unfinished attempts.
- **Reader:** ungated, with 10% whole-sentence dropout (`--text-sentence-dropout`).
- **Audit** (two independent reviewers), all fixed:
  - "You got more X" was already true of the source in 114 of 129 used-up cases. Now it flips to "Done: G" for the goal whose item it is: the one unmet goal, or one received feedback tied to the item from single-goal messages.
  - Unchanged successes gave ties.
  - `succ_msgs` was not checkpointed.
  - The reader never saw messages with sentences dropped.
  - Soft leaks, now read from the text: wordings never received, the claim-tuple filter, success read from the environment.
- **Rule from the user:** the improved message changes only the stated cause, with its explicit counterpart. Nothing is dropped to hint at a change.
  - Verdicts always stay.
  - A "not done" verdict never received together with the new sentence means no improved message: the step may have finished that goal.
  - Successes are improved only by the explicit "kept going" → "stopped" swap.
- **Coverage** (1536 received messages): 86% of failures on the base-policy stream, 65% on the expert stream. Every sentence was received before. Whole message inside a received message: 25–47% (verdicts are always kept).
- **Leak check:** with call counters, there are no calls to true_claims, claim_steps, flip_toward_success, expert actions or ideal facts during rounds.
- **Disclose:**
  - The parser and the cause→fix rules are hand-built from the teacher's templates.
  - The curriculum uses success (same as PPO).
  - Stage II attempts are not counted in the budget (about 3× PPO's environment episodes per teacher call).
- **Tests:** 1077 pass.
- **Run 8 launched** (user "go"). New pod `mz0eu7buz750nu` (gft-minicraft-ro11): RTX PRO 4000 Blackwell, EU-RO-1, $0.57/hr, ssh -p 39229 root@213.173.104.71, volume yy9bsgna2b, watchdog running. ro10 (`wl4uq8pacgsuwb`) is stopped; its host has no free GPU. Job `jobs_span8.sh`, results outputs/minicraft/span8/. 35 tests pass on the pod. Balance $1.99 before launch.
- **Run 8 stopped by me at 17.6k calls** (kill-early). Outputs pulled to outputs/pod/gft_old/span8/.
  - **Rounds 0–33 (firewood, planks):** the new filter worked. Pass rate 0.8–1.0 for finished attempts vs 0.2–0.4 for unfinished; kept success above the raw reader (e.g. 0.70 vs 0.39). Firewood was mastered by ~1k calls.
  - **From round 34, when "make a handle" unlocked:** no separation (finished ~0.5 vs unfinished ~0.5), kept ≈ reader, for 30 rounds without recovery. Still 3 goals at 17.6k (run 6b had 4 by 10.2k); val 0.09–0.13.
  - Next: diagnose offline why the scorer cannot judge the new goal's improved messages.

### 2026-10-31 — Success needs DONE (user, option 1); run 8 restarting with both PPOs

- **Diagnosis of run 8's collapse** (offline with its checkpoint; scratchpad diag8.py, diag8b.py): the scorer was fine. Stage II drew source episodes from all of Stage I, and once the policy succeeded about 45% of the time (source success 0.00 at round 20, 0.35–0.48 from round 36), nearly half the sources were successes. Their improved message only swaps "kept going" for "stopped" and keeps "Done: G", so the filter judged stopping, not finishing. The reader also imitated unfinished attempts labelled "Done: G … stopped". With failure sources only, the end checkpoint separates well: reader attempts on handle 0.93 vs 0.14, planks 0.81 vs 0.16, firewood 0.91 vs 0.00.
- **Fix:** in message mode, Stage II sources are failures only, read from the text (a "not done" verdict).
- **New rule** (`--success-needs-done`, WorldConfig.success_needs_done): a request counts as met only if it holds at the end and the agent took DONE. It is applied in run_episodes, so it covers PPO's reward, GFT, the curriculum and evaluation alike.
  - Teacher: when every goal holds but DONE was never taken, the verdicts say "Not done" with "You finished but never said done." / "You had it all but never said done."
  - The verifier uses the same rule. The improved message flips every goal to done (the success form).
  - 0 false sentences in 3000 messages with the rule on. Tests: 1078 pass, including a new end-to-end test.
- **Pool:** rebuilt locally for the new config (pool_dbb76bbb86767fdb.pkl). Local PPO smoke run passed.
- **Jobs:** `jobs_span8.sh` (GFT, now with --success-needs-done) and `jobs_span8_ppo.sh` (PPO with frac and with bin reward, same rule). RunPod balance $11.81 after the user's top-up.
- **ro11** (`mz0eu7buz750nu`) could not start (no free GPU on its host). Creating a replacement pod was blocked pending the user's approval.
- **Launched** (user "go") on new pod `kvqp8uscqlt5s7` (gft-minicraft-ro12, RTX PRO 4000 Blackwell, EU-RO-1, $0.57/hr, ssh -p 17354 root@213.173.104.29, volume yy9bsgna2b, watchdog running): GFT run 8 (failure sources, success needs DONE) plus PPO frac and PPO bin with the same rule, in parallel. Results in outputs/minicraft/span8/. The earlier run 8's outputs were moved to span8_allsrc (on the pod and locally). Stopped pods for the user to delete: ro10 `wl4uq8pacgsuwb`, ro11 `mz0eu7buz750nu`.
- **Run 8b launched** (user "1", one pod only) on ro12, next to run 8 and both PPOs. Same recipe as run 8, but the reader makes as many Stage II imitation updates per round as the policy's distillation (`--reader-imitate-match-pi`: ⌈Stage II steps / batch⌉ × epochs_clean ≈ 90, batches resampled from its kept steps; before, 1 pass ≈ 10). A test checks the counts are equal. Job `jobs_span8b.sh`, results outputs/minicraft/span8/gft_rmatch_s0.*.
- **Run 8 at 12.3k calls:** 5 goals open, val 0.109. On the newly opened goals pi is 0.14, reader 0.05, kept 0.09; filter pass rate finished 0.35 vs unfinished 0.21.
- **PPO:** frac 90k episodes, 6 goals, val 0.238; bin 75k episodes, 5 goals, val 0.223.
- **Run 8 stopped by me at 14k calls:** the scorer collapsed at the round-49 reset. Its trajectory features barely vary across episodes (std 0.04), a message scores the same on every episode (std 0.003), the loss is stuck at ln 5, and nothing is kept from round 49 on. The weights were finite. Earlier resets (rounds 9, 19, 29, 39) dipped accuracy (0.6–0.7 → 0.36–0.54) and recovered. Before the collapse, run 8 had mastered 4 goals by 10.5k calls (firewood 1.0k, planks 5.9k, handle 9.5k, ready to mine 10.5k); stone stayed at 0–0.02.
- **Runs 9 and 9b** (user "restart both"): runs 8 and 8b with no scorer resets (`--disc-reset-every 0`). Jobs `jobs_span9.sh` / `jobs_span9b.sh`, results outputs/minicraft/span9/. Run 8b was stopped at about 2k calls to make room.
- **PPO with success needing DONE, 100k episodes** (outputs/minicraft/span8/ppo_*):
  - frac: 6 goals, final val 0.230 (0.19–0.26 over the last evals).
  - bin: 6 goals, final val 0.172 (0.18–0.25).

### 2026-10-31 — Audit of leftovers from older designs; run 10 prepared (not launched)

Three independent reviewers (config defaults, training mechanics, environment/eval/fairness), verified by me.
- **Fixed in code or flags (run 10):**
  - Stage II at temperature 1 (was 0.7, imitated as if from T=1 while the entropy controller pushed back). The reader no longer self-sharpens on T=0.7 samples.
  - `--fb-word-dropout 0`: word dropout blanked "not", turning "Not done: X" into "Done: X". Whole-sentence dropout is kept.
  - `--exact-vocab`: pi and the reader used 4096 hashed buckets, with collisions (bridge/request, get/up, stick/can, crafting/is, okay/wall, to/workshop).
  - `--push-mode kept`: pi's updates now scale with the kept steps (2 passes, no resampling). It was ~90 updates resampled from as few as 7 kept steps.
  - `--stay-stage1`: half of pi's stay term is on its own Stage I states (was the reader's states only).
  - `--reader-imitate-epochs 2`: the reader imitates the kept steps at pi's rate.
  - `--heldout-frac 0`.
  - `--cur-review 0.5 --cur-confirm`: regressed mastered goals return to training, and mastery needs two consecutive windows. Mastered goals had regressed: run 9's mastered goals were at 0.41–0.61; PPO-bin's at 0.09–0.25.
  - `--eval-on-spent`: evaluate on the budget unit and at the end. GFT was evaluated every ~1.8k teacher calls because Stage II rollouts count as episodes.
  - A `spent` field in every eval row.
  - Speed: no discarded pi forward pass in reader-only Stage II; no sentence-level score in message mode.
- **Kept as is (user: (a)):** kept attempts that fail. In run 8, 34% of kept attempts failed, supplying 42% of pushed steps.
- **To disclose:**
  - GFT uses ~290k environment episodes per 100k teacher calls.
  - val_combos is ~0 by construction (every held-out pair needs goal ≥6), and val_worlds is capped at ~32% by curriculum depth. Report per-goal success.
- **Runs 9 / 9b at ~27k calls:** 4 goals mastered (firewood ~1.0k, planks ~7.7k, handle ~9.0k, ready to mine ~9.2k; run 9b's last at ~15.9k). Stone has stayed at 0.04–0.06 for ~14k calls. Val 0.08–0.15.
- **Checks:** 1081 tests pass. CPU smoke of GFT and PPO with all new flags passed; evals landed at spent 0/256/512, including the final one.
- **Jobs:** `jobs_span10.sh` (GFT) and `jobs_span10_ppo.sh` (PPO frac + bin with the same environment-side flags), results outputs/minicraft/span10/.
- **Run 9 stone investigation** (run 9 checkpoint, 24 single-goal stone tasks, read case by case; scratchpad stone9.py, proc/stone9.txt). Outcomes: 3 successes, 7 with 1 stone then quit or timeout, 11 with a pickaxe that never mined, 3 without a pickaxe.
  - The feedback is accurate and names the real mistakes: "not enough stone" + "stopped before you were done"; "walked past a rock"; "still needed a wood pickaxe"; plus habits (hitting ore without a stone pickaxe, extra wood, use on nothing).
  - The improved messages go one step further: "You used a rock." / "Great job! Done: collect building stone." / "You got a wood pickaxe."
  - The bottleneck is the reader: its stone success under these messages stayed at 0.00–0.02 in training. A steering check (reader under improved vs original vs request only) is proposed.
- **Runs 9 and 9b stopped** (plateau: 4 goals, stone ~0.05 for ~20k calls, val ~0.12). Outputs pulled to outputs/pod/gft_old/span9.
- **Run 10 launched** (user "go"; PPO with frac reward only, user choice) on ro12: `jobs_span10.sh` (GFT with all audit fixes) and `jobs_span10_ppo.sh` (PPO frac, same environment-side fixes). Results in outputs/minicraft/span10/.
- **Run 10 GFT stopped by me at 40k calls** (kill-early, declining). Mastered firewood 2.3k, planks 7.7k, handle 12.3k; then stuck at 4 goals. Policy went 0.47 → 0.27, val 0.09–0.13, and the filter's separation eroded (0.86 vs 0.61). Its policy entropy is ~1.4 nats on its training states (target 0.8). On stone tasks (not yet open for it) actions are near-uniform (entropy 1.4–1.6, use 0.16 when facing a rock with a pickaxe). Reading: temperature 1 plus proportional push left pi too random.
  - PPO frac (run-10 settings) finished: 100k episodes, 6 goals, final val 0.203.
  - Run 9's policy on stone, top actions per situation: with 2+ stones DONE is not in its top 4 (down 0.42 / up 0.37, entropy 0.12); facing a rock with a pickaxe it mines (use 0.64), but that is only 0.8% of steps.
- **Run 11 prepared:** advantage-weighted replay of Stage II attempts, rescored each round. A = D(x', f_hat) − D(x', f) ≈ log P(f_hat | x') / P(f | x') under the InfoNCE scorer; scale capped at 100 (CLIP); Stage II temperature back to 0.7. 1082 tests pass. Open: raw vs cosine-unit advantage (proposed raw, τ = 3); a set of improved messages per source (log-sum-exp over one-step fixes) so other improvements are credited.

### 2026-10-31 — Runs 11 / 11b prepared (not launched)

- **Advantage** (user-agreed): r(x) = D(x, f_hat) − D(x, f) ≈ log P(f_hat | x) / P(f | x) under the InfoNCE scorer, and A(x') = r(x') − r(x_source). The message priors cancel, shared verdicts cancel inside r (no overshoot penalty), and the baseline is the current policy's episode. A < 0 is dropped. The scorer scale is capped at 100 (CLIP). The replay buffer was dropped (user); only the current round's attempts are used (`--stage2-replay 1`).
- **Run 11** (`jobs_span11.sh`): run 10 plus advantage-weighted imitation (keep A > 0, top 8 per task, weights exp(A/3) split evenly over each attempt's steps), Stage II temperature back to 0.7.
- **Run 11b** (`jobs_span11b.sh`, user): pi is updated by off-policy PPO on the reader's attempts instead of imitation. Reward r(x'), advantage r(x') − r(x_source) standardized, ratio pi / reader at the sampling temperature (behaviour log-probs recorded during Stage II), clip 0.2. The reader still imitates the kept attempts.
- **Research note:** CLIP uses cosine × learned scale (init 1/0.07, cap 100); SigLIP uses sigmoid(t·cos + b) (t = 10, b = −10); VLM-RM uses raw CLIP cosine as an RL reward, with goal-baseline regularization. That is the analogue of our D(f_hat) − D(f).
- **Tests:** 1083 pass.
- **Audit of the run-11 code** (independent reviewer, verified):
  - Fixed: tasks with the same pool id in one round were judged against each other's messages, with A_now = max over both (confirmed by running; it happened every round). Attempts are now judged only within their own round task.
  - Fixed: stale replay stats in rounds with nothing kept.
  - Fixed: mastery review now uses the recent window only.
  - Checked correct: behaviour log-probs (match recomputation to 5e-7), source-episode features, pool weights, distillation from the pool, PPO ratio/clip, reader prompts, CUDA devices, empty cases.
  - Risk to watch: exp(A/3) with a scale up to 100 can let one attempt dominate.
  - 1084 tests pass.
- **Runs 11 and 11b launched** (user "audit code and go") on ro12 (restarted; ssh -p 10845 root@213.173.104.29), watchdog running. Results in outputs/minicraft/span11/ (gft_s0 = run 11, gft_ppo_s0 = run 11b). RunPod balance $9.48 before launch.
- **Runs 11 and 11b stopped by me** (kill-early) at 15.1k and 10.2k calls: still 2 goals, val 0.051 / 0.016 (start 0.082; runs 9 and 10 had 5 and 4 goals by then).
  - The keep rule A = r(x') − r(x_source) > 0 is far too permissive. The source is a failure with low r (it fits f), so nearly every attempt beats it: unfinished attempts pass 0.74–0.86 and ~240 of 256 attempts are kept per round, so pi imitates mostly failures.
  - The mean advantage grew 8 → 13 (scale drift), making exp(A/3) weights extreme. Run 11b's PPO clipped fraction was 0.36–0.37.
  - Proposed fix: keep only if A > 0 AND r(x') > 0 (the old filter, which separated finished from unfinished 0.9 vs 0.3), with a larger τ (~10).
- **Run 12 launched** (pod ro12, `experiments/minicraft/pod/jobs_span12.sh`, outputs/minicraft/span12): run 11 with the keep rule r(x') > 0 (the attempt is described better by the improved message than by the original) instead of A > 0; top 8 per task ranked by A = r(x') − r(x_source); weights exp(A/10); weighted filtered BC only, no PPO variant.
- **Run 12 stopped by me** (kill-early) at 15.1k calls: 2 goals, val 0.035 (start 0.082; runs 9 and 10 had 4–5 goals by then).
  - The r(x') > 0 filter starts selective (unfinished attempts pass 0.16 in rounds 0–10) but stops separating: 0.35 in rounds 10–30, 0.92 in rounds 30–60; kept-attempt success falls 0.68 → 0.15.
  - Run 10 used the same filter and stayed at 0.09 / 0.31 / 0.41 over the same rounds. Scorer message accuracy is about the same (run 12 0.54 / 0.60 / 0.72 vs run 10 0.61 / 0.67 / 0.71), so the scorer is not broken by its own metric; what passes it changed.
  - Differences from run 10: Stage II temperature 0.7 (run 10: 1), top 8 ranked by A (run 10: random among passers), exp(A/10) weights, scorer scale capped at 100. Not yet isolated which one causes it.
- **Run 12 diagnosis** (replay of Stage II on planks from run 12's checkpoint): a bug of mine in the new Stage II path (runs 11, 12), not the filter or the teacher. Each kept attempt's weight was split over its steps (1/length per step), so short attempts dominated: 80% of pi's push weight on attempts of ≤10 steps and 27% on DONE actions (run 10's 1-per-step weighting: 10% / 3%); kept attempts had median length 4, 89% ending in DONE. Fits firewood 0.94 → 0.31 and planks never starting.
- **Run 13 launched** (pod ro12, `jobs_span13.sh`, outputs/minicraft/span13): run 12 with every kept step weighted exp(A/10) (not divided by length; τ → ∞ gives run 10's weighting). New test `test_stage2_select_weights_each_step_by_advantage` (fails on the old weighting).
- **Run 13 stopped by me** (kill-early) at 30.2k calls: 2 goals for 60+ rounds, val 0.035–0.066 (start 0.082). Better than run 12 (planks 0.35–0.58 from round 42 on, firewood 0.35–0.80) but well behind run 10 (4 goals by round 110, val 0.11–0.13 at 15–35k). Unfinished attempts pass r > 0 at 0.4–0.7 and kept attempts stay near the cap (213–243 of 256). Untested differences from run 10: Stage II temperature 0.7 (run 10: 1), top 8 ranked by A (run 10: random among passers), exp(A/10) weights, scorer scale cap 100.
- **Run 14 launched** (user "ok you can try those variants"; pod ro12, `jobs_span14.sh`, outputs/minicraft/span14), two arms in parallel: `gft_rw_s0` = run 13 with kept steps weighted by r(x') (normalized to mean 1; `--adv-weight r`) instead of exp(A/10); `gft_run10_s0` = run 10's exact flags on the current code (sanity check of the old path and the scorer cap).
- **Run 14 at 15.1k calls:** the run-10 replica (`gft_run10_s0`) reproduces run 10: planks 0.37 by round 24, 4 goals by round 48 (run 10: 4 by ~48), val 0.074–0.082 (run 10: 0.07–0.12). So the old path and the scorer cap are fine. The r-weighted arm (`gft_rw_s0`) is stuck like runs 12–13: planks 0.00–0.01 for 40 rounds, 2 goals, val 0.04–0.06; stopped by me. The weighting is not the cause; what remains is the new path's selection (top 8 by r among r > 0 vs random 8 among passers) and the reader temperature 0.7 vs 1.0.
- **Run 14b launched** (user "go"; `jobs_span14b.sh`, outputs/minicraft/span14) next to the run-10 replica: `gft_rw_t10_s0` = new path (r > 0, top 8 by r, r weights) at Stage II temperature 1.0; `gft_run10_t07_s0` = run 10's flags (old path) at 0.7. Completes the path × temperature 2×2.
- **Run-10 replica finished** (span14/gft_run10_s0, 100k teacher calls, 292k episodes): mastered firewood 2.3k, planks 9.5k, handle 12.3k, pickaxe 36.9k; stone open (0.50) at the end; val 0.168 final (0.15–0.19 over the last 25k). PPO with the same success rule, 100k episodes: frac 6 goals, val 0.20–0.23; bin 6 goals, val 0.17. (The original run 10 was stopped at 40k.)
- **Run-10 replica, stone** (round history + final checkpoint, 160 single-goal stone episodes; scratchpad stone14.py):
  - Not stuck but slow: stone ~0.00–0.02 for ~80 rounds after it opened (≈20k calls), then a steady climb 0.04 → 0.52, still rising at 100k.
  - Final policy: success 0.47. Failures: 1 stone then timeout 29%, 1 stone then DONE 9%, pickaxe but never mined 9%, 2+ stones without DONE 4%.
  - Navigation is a random walk: whenever it is not facing a rock (~90% of steps) the four moves are ~0.2 each (entropy 1.4–1.7). Local actions are right: facing a rock with a pickaxe → use 0.45–0.51. With 2+ stones DONE is not in the top 4. Finished episodes take a median 130 of 200 steps.
  - The entropy bonus is off from round 40 (alpha 1e-4), yet pi's entropy climbs 1.31 → 1.57 (target 0.8): Stage II itself keeps pi random (pi imitates the reader's actions sampled at T = 1; the trajectory-level filter says little about which steps mattered).
  - Why the first ~20k calls on stone were flat cannot be read from the final checkpoint (no earlier one saved).
- **PPO frac (run-10 settings, 100k episodes) on the same 160 stone episodes:** success 0.54 (GFT replica 0.47). It does not head to rocks either: it sweeps sideways (no pickaxe: left 0.37 / right 0.35, entropy 0.39; with a pickaxe ~1.0), a lower-entropy pattern than GFT's uniform walk (1.4–1.7). Its local choices are sharper: facing a rock with a pickaxe use 0.78 (0 stones) / 0.94 (1 stone) vs GFT 0.51 / 0.45; with 2+ stones DONE 0.22 (2nd) vs not in GFT's top 4. Its failures are earlier in the chain: no pickaxe by the end 14%, pickaxe but never mined 16%, 1 stone 14% (GFT: 1%, 9%, 38%).
- **Run 14b at 15.1k calls — path × temperature 2×2:**
  - old path, T 1.0 (replica): planks 0.37 by round 24, 3 goals by round 40; val 0.078.
  - new path, T 1.0 (`gft_rw_t10_s0`): planks 0.65 by round 32, 3 goals by round 40, firewood held 0.8–0.9; val 0.098 (best at 15k of all fair runs).
  - old path, T 0.7 (`gft_run10_t07_s0`): planks 0.28–0.41, 2 goals; val 0.074. Stopped by me.
  - new path, T 0.7 (`gft_rw_s0`): planks ~0 for 40 rounds; val 0.062 (stopped earlier).
  - Reading: Stage II temperature 0.7 is what slowed runs 11–13 (it slows the old path and stalls the new one); at 1.0 the new path (r > 0, top 8 by r, r weights) works at least as well as run 10's. `gft_rw_t10_s0` continues to 100k.
- **Entropy on the same states** (final checkpoints, 160 single-goal tasks of goals 0–4, both policies scored on each one's rollout states; scratchpad ent_same.py): on GFT's states GFT 1.52 vs PPO 0.76; on PPO's states GFT 1.32 vs PPO 0.56. Success on these tasks: GFT 0.67, PPO 0.86. GFT's pi is about twice as random as PPO's on identical states. (The training-time entropy curves are on different states: PPO 0.4–1.3 oscillating, GFT 1.2 → 1.57 rising; only final checkpoints exist for the same-state check.)
- **`gft_rw_t10_s0` (new path, T 1.0) stopped by me at 60k calls** (kill-early): mastered firewood 2.3k, planks 10.5k, handle 17.9k, then "get ready to mine" at 0.00 for 160 rounds (~40k calls); val 0.08–0.12 flat since 15k; unfinished attempts pass r > 0 at 0.75–0.88 from round 120. The replica (old path, T 1.0) mastered the pickaxe at 36.9k. So at T 1.0 the new path is faster early but stalls at the pickaxe; the old path is the better recipe so far.
- **Why the new path stalls at the pickaxe** (`gft_rw_t10_s0` checkpoint, 44 failed pickaxe worlds × 8 reader attempts at T 1; scratchpad pick14.py):
  - pi: 0/44; 25 stop after a stick, 18 after a plank. Reader under f̂: 0/352 make the pickaxe.
  - Improved messages flip the agent between the two steps: a source that made a stick gets "…You got a plank." (its "You got a stick." goes: the teacher narrates only the last gain), and the reader then makes only a plank (194/200, all kept); a source that made a plank gets "…You got a stick." and the reader makes a stick (121/144, all kept).
  - Ranking by r selects against finishing: the replica's pi on the same tasks (success 0.84), scored by this arm's scorer against the same f / f̂, gets r −0.69 when it finishes (pass 0.43) vs +10.9 / +10.5 for the reader's plank-only / stick-only attempts; a finished episode ranks below 83% of the reader attempts on its task. Top 8 by r (and r weights) keep the partial steps the improved message describes and drop overshoots; run 10's random 8 among passers keeps the 43% of finishers that pass, at weight 1.
  - Correction to the run-12 planks replay: its "made planks" category read collected (harvested) counts, so plank-making attempts were never detected (wood counts and success were right; the length-weighting diagnosis does not depend on it).
- **Local disk cleanup (2026-11-01, user OK):** deleted local copies that match the RunPod volume by md5 (outputs/pod/reader_bench/*.pt → /workspace/gft/outputs/minicraft/reader_bench/; outputs/pod/gft_old/span7, span7_notdone, span8, span8_allsrc, span12, span13 *.ckpt and span14/gft_run10_s0.ckpt → /workspace/gft_old/outputs/minicraft/). Kept span9 and span10 gft_s0.ckpt (local snapshots differ from the volume's). Also removed my leftover checkpoint copies in /tmp.
- **Run 15 launched** (user request; pod ro12, `jobs_span15.sh`, outputs/minicraft/span15/gft_noimit_s0): run 10 without the reader imitating the kept Stage II attempts (no `--rerank-reader-imitate`); the reader learns only from the teacher's messages on Stage I episodes, pi still distils from the kept attempts.
- **Run 15b launched** (user request; `jobs_span15b.sh`, outputs/minicraft/span15/gft_noreplay_s0), next to run 15: run 10 without the reader's replay buffer (`--replay-rounds 1`: this round's Stage I data only; run 10 used 4). The scorer's own replay (6 rounds, default) is unchanged.
- **Run 15c launched** (user request; `jobs_span15c.sh`, outputs/minicraft/span15/gft_noreplay2_s0), next to runs 15 and 15b: run 10 without either replay buffer (`--replay-rounds 1` for the reader, new flag `--disc-replay 1` for the scorer; run 10: 4 and 6).
- **Runs 15/15b/15c at ~15k calls** (replica at 15k: firewood 2.3k, planks 9.5k, handle 12.3k mastered; val 0.078):
  - 15 (no reader imitation): firewood 2.3k, planks 10.5k, handle at 0.68; val 0.055 / 0.102 / 0.086 at 5/10/15k. On par with the replica; continues.
  - 15b (no reader replay; 10k so far, slower per call): firewood 1.5k, planks 0.62; val 0.062 / 0.059 at 5/10k. A bit behind; continues.
  - 15c (no reader or scorer replay): firewood 1.5k, then planks 0.00 for 40 rounds; unfinished attempts pass r > 0 at 0.88–0.96 (the scorer trained on one round does not separate); val 0.023–0.035, below the start. Stopped by me (not promising).
- **Run 16 launched** (user "do 1"; `jobs_span16.sh`, outputs/minicraft/span16/gft_sharp_s0), next to runs 15 and 15b: run 10 with pi's push target on kept steps = the reader's distribution sharpened to T 0.7 (new `--push-target-temp 0.7`), sampling still at 1.0; the reader still imitates the kept actions. Test `test_push_target_is_the_sharpened_reader_distribution`.
- **Can the scorer rank by an ideal message?** (run-10 replica checkpoint, stone; 24 failed worlds × 8 reader attempts at T 1; "ideal" = the 16 success messages the teacher gave pi's own successful stone episodes in the same sample, i.e. received feedback; score = mean D over them; scratchpad ideal14.py). Mean scores by attempt: finished 11.3, 2+ stones no DONE 4.3, 1 stone −4.8, pickaxe no stone −5.2, no pickaxe −12.1. Finished vs unfinished AUC: D(x, ideal) 1.00, D(x, ideal) − D(x, f) 1.00, r (improved) 0.95. Within-task Spearman with progress: D(x, ideal) 0.79 (0.49 among unfinished), r 0.60 (−0.12 among unfinished). On a partly solved goal the ideal message ranks better than the improved one, partial progress included. Only possible once the goal has received success messages (stone was at 0.40 here); one checkpoint, one goal.
- **Check (runs 15/15b at 35k, run 16 at 15k calls;** replica: firewood 2.3k, planks 9.5k, handle 12.3k, pickaxe 36.9k; val 0.078 / 0.113 / 0.141 / 0.156 / 0.137 at 15/20/25/30/35k; pi entropy 1.2–1.4):
  - 15 (no reader imitation): firewood 2.3k, planks 10.5k, handle 16.4k; val 0.098 / 0.121 / 0.121 / 0.121 at 20–35k; entropy 1.27 → 1.57. Slightly behind; continues.
  - 15b (no reader replay): firewood 1.5k, planks 12.8k, handle 22.0k; val 0.125 / 0.121 / 0.109 / 0.117 at 20–35k. Slightly behind; continues.
  - 16 (sharpened push target): entropy 0.83–0.92 (down from run 10's 1.2–1.3, as intended); firewood mastered 4.1k (run 10: 2.3k), planks 0.35–0.55 at rounds 40–58 not yet mastered (run 10: 9.5k); val 0.066 / 0.074 / 0.082 at 5/10/15k (run 10: 0.074 / 0.082 / 0.078). Slower mastery, val on par; continues.
- **Runs 15/15b at 40k:** 15 val 0.117, 3 goals (no pickaxe yet; replica had it at 36.9k, val 0.137 at 40k); 15b val 0.113, 3 goals. Both continue.
- **Ideal-message filter variants prepared (run 17, not launched):** offline pass rates on the replica's stone attempts (finished / 2+ stones / 1 stone / pickaxe only / no pickaxe; kept-set success): improved (run 10) 0.98 / 0.90 / 0.16 / 0.11 / 0.00 (0.79); ideal D(x',ideal) > D(x',f) 1.00 / 0.20 / 0.04 / 0.06 / 0.00 (0.94); ideal vs the source D(x',ideal) > D(xsrc,ideal) 1.00 / 0.90 / 0.28 / 0.22 / 0.00 (0.72); union improved ∪ ideal 1.00 / 0.90 / 0.18 / 0.17 (0.78); weighted 0.5/0.5 0.98 / 0.40 / 0.05 / 0.06 (0.91). Implemented `--rerank-ideal {ideal,union,weighted}` (old path; ideal = mean D over the last 8 received success messages of the request, success read from the text; the improved test where none was received; `--ideal-k`, `--ideal-weight`); tests added (47 pass). `jobs_span17.sh`: three arms on run 10's flags.
- **Run 16 stopped by me at 25k calls** (not promising): entropy held at 0.70–0.84 (the controller's bonus came back on, alpha 0.15–0.17), but only firewood mastered (4.1k); planks peaked at 0.52 (round 68) and fell to 0.19–0.25; unfinished attempts pass r > 0 at 0.71–0.93 from round 74; val 0.082 → 0.051 → 0.020 at 15/20/25k (replica 0.078 / 0.113 / 0.141). The sharper pi explores less in Stage I (user's concern) and the run degrades.
- **Runs 15/15b at 45k:** val 0.137 / 0.129 (replica 0.145 at 45k), still 3 goals each (replica had the pickaxe at 36.9k). Continue.
- **Run 17 launched** (user "go"; pod ro12, `jobs_span17.sh`, outputs/minicraft/span17): ideal / union / weighted filter arms on run 10's flags, next to runs 15 and 15b.
- **Runs 15 and 15b stopped** (user) at 55k / 60k calls: both 3 goals (no pickaxe; run 10 had it at 36.9k), val 0.145 / 0.137 (run 10: 0.109 at 55k, 0.152 at 60k). Without the reader's imitation of kept attempts, or without its replay buffer, val stays near run 10 but mastery is slower; neither component is essential, both help a little.
- **Run 17 at 15k calls** (replica: firewood 2.3k, planks 9.5k, handle 12.3k; val 0.074 / 0.082 / 0.078 at 5/10/15k): ideal arm firewood 2.3k, planks 8.2k, handle 11.0k, val 0.062 / 0.105 / 0.125; weighted 2.6k / 9.2k / 14.1k, val 0.062 / 0.074 / 0.102; union 2.3k / 8.7k, handle not yet, val 0.082 / 0.078 / 0.098. The ideal test applies on 0.89–1.00 of Stage II attempts (success messages already received for the request); its pass rate 0.45–0.56 vs the improved test's 0.49–0.69.
- **Run 17 at 20k calls:** ideal arm mastered the pickaxe at 18.2k (replica 36.9k), stone open at 0.07, val 0.152 (replica 0.113 at 20k). Weighted: 3 goals (handle 14.1k), pickaxe 0.47, val 0.113. Union: still 2 goals (handle 0.75), val 0.078; its filter passes 0.56 of unfinished attempts (ideal arm 0.22).
- **Run 17b launched** (user request; `jobs_span17b.sh`, outputs/minicraft/span17/gft_ideal_noimp_s0): run 17's ideal arm with no improved message anywhere (new `--no-improved`): Stage II reader prompted with a received success message of the request (no task for requests without one), ideal filter, scorer negatives without improved messages. Test `test_no_improved_uses_received_success_messages_only`; 48 tests pass.
- **Run 17 at 25k calls:** ideal arm 4 goals, stone already at 0.20 (run 10: stone ~0.00–0.02 for ~20k calls after it opened), val 0.148 (replica 0.141 at 25k). Weighted: pickaxe mastered at 24.8k, val 0.148. Union: handle at 21.0k, val 0.133.
- **Run 17b at 10k calls:** firewood 2.3k, planks 6.1k, handle 8.2k (the fastest GFT run so far; replica 2.3k / 9.5k / 12.3k), val 0.094 / 0.148 at 5/10k (replica 0.074 / 0.082), pi entropy 1.04–1.07. Run 17 at 35k: ideal stone 0.37, val 0.141; weighted val 0.129; union val 0.137 (replica 0.137 at 35k).
- **Fairness audit (runs 17/17b code paths, user request):** no hidden information found. Stage II keeps attempts by the learned scorer on text only (Stage II env success is logged, never used); success messages are read from the text (no "not done" verdict) and only received ones are used; truth / ideal-facts code paths are not reached (message-mode scorer, improved target); the reader trains on request + teacher text → actions (no reader gate, no masked claims, no sentence reader: no spans); validation worlds use disjoint seeds; one teacher call per Stage I episode, nothing else gets a message. Caveats: (1) the budget axis counts feedback, not episodes — each teacher call comes with ~2 Stage II rollouts on the same world without feedback; per environment episode 17b mastered the handle at ~24k episodes vs PPO's 9k; (2) the curriculum reads env success of Stage I episodes (same information as the teacher's verdicts; same mechanism as PPO); (3) hand-written language knowledge: a parser of the teacher's sentences reads verdicts (17 and 17b) and run 17's arms use the improved-message rewrite rules (17b does not); (4) the inventory is a symbolic input (log counts) next to the rendered map, for GFT and PPO alike.
- **Status (17 at 40k, 17b at 15k):** 17b pickaxe 12.5k (PPO bin 14.3k, frac 28.2k), stone 0.21, val 0.180 at 15k (replica 0.078; PPO frac final 0.20). 17 ideal stone 0.43, val 0.168 at 40k (replica 0.137). Weighted val 0.121, stone 0.03. Union 3 goals, val 0.148.
- **Status (17 at 50k, 17b at 25k):** 17b val 0.203 at 25k (= PPO frac's final 0.20 at 100k episodes; run 10 0.141 at 25k), stone 0.50, pickaxe 0.85. 17 ideal stone 0.48, val 0.152; union 3 goals, val 0.160; weighted stone 0.10, val 0.148.
- **Status (17 at 55k, 17b at 30k):** 17b val 0.215 at 30k (above PPO frac's final 0.20), stone 0.66. 17 ideal val 0.184, stone 0.55. Union pickaxe at 51.2k, val 0.129. Weighted stone 0.10, val 0.129.
- **Status (17 at 60k, 17b at 35k):** 17b val 0.203, stone 0.69 (not yet confirmed as mastered: needs 0.7 on two windows). 17 ideal val 0.203, stone 0.52. Weighted val 0.156, stone 0.16; union val 0.148, stone 0.01.
- **Status (17 at 65k, 17b at 45k):** 17b mastered stone at 37.6k calls (first GFT run with 5 goals; PPO frac 63.0k / bin 82.4k episodes), now on 'upgrade your tools' (0.01); val 0.223 at 40k, 0.180 at 45k. 17 ideal val 0.203, stone 0.51. Weighted val 0.164, stone 0.27. Union val 0.141, stone 0.00.
- **Status (17 at 75k, 17b at 50k):** 17b val 0.230 (best of any run), upgrade-tools 0.18 (no run has mastered it). 17 ideal val 0.20, stone 0.65. Weighted val 0.152, stone 0.53. Union val 0.141, stone 0.00.
- **Status (17 at 80k, 17b at 55k):** 17b upgrade-tools 0.44 (stone pickaxe; no other run above 0.0), val 0.188. 17 ideal val 0.207, stone 0.65. Weighted val 0.199, stone 0.38. Union val 0.141, stone 0.00.
- **Status (17 at 90k, 17b at 65k):** 17b val 0.262 / 0.258 at 60/65k (best of any run; PPO frac final 0.20), upgrade-tools 0.68 (about to be the 6th goal). 17 ideal val 0.199, stone 0.67. Weighted val 0.18, stone 0.50. Union val 0.121, stone 0.00.
- **Run 17b validity audit:** flags as intended (no_improved, ideal filter, success needs DONE, pixels, realistic teacher, random base, seed 0); evaluation identical to PPO (pi on the request alone, no feedback, sampled with a fixed seed, same 256 val tasks — per-goal counts match exactly); Stage II active throughout (26–32 tasks per round, ideal bank available for 0.99–1.00 of them). 179.8k episodes at 65k teacher calls (2.8 per call).
  - **Correction:** "17b val 0.26 well above PPO frac's 0.20" compared 17b's high point with PPO's final eval, which was a dip. Mean of the last 4 evals: 17b 0.234 (50–65k calls) vs PPO frac 0.221 (85–100k episodes), PPO bin 0.189, run 10 0.173. Paired by val task, 17b − PPO frac = +0.014, 95% bootstrap CI [−0.016, +0.043]: not distinguishable from PPO frac on val yet. The clearer differences are mastery speed (pickaxe 12.5k calls vs 14.3k / 28.2k episodes; stone 37.6k vs 63.0k / 82.4k) and upgrade-tools progress (0.68 vs 0.0) — one seed each.
- **Status (17 at 95k, 17b at 70k):** 17b val 0.266 at 70k (last 3 evals 0.262 / 0.258 / 0.266), upgrade-tools 0.71. 17 ideal val 0.203, stone 0.70; weighted 0.16; union 0.168.
- **Run 17 finished (100k calls):** ideal 5 goals (stone 98.3k), final val 0.172 (last 4: 0.184 / 0.199 / 0.203 / 0.172); weighted 4 goals, 0.191; union 4 goals, 0.148. **Run 17b at 80k:** 6 goals (upgrade tools 71.4k — no other run got above 0.0 on it), val 0.258 / 0.266 / 0.285 / 0.258 at 65–80k; now on "get some metal" (0.00).
- **Run 17b finished (100k calls, ~277k episodes):** 6 goals (firewood 2.3k, planks 6.1k, handle 8.2k, pickaxe 12.5k, stone 37.6k, upgrade tools 71.4k); "get some metal" 0.00. Val last 4 evals 0.184 / 0.246 / 0.250 / 0.219, final 0.219. Paired by val task (mean of last 4 evals each): vs PPO frac +0.004 [−0.029, +0.037] (tie); vs PPO bin +0.035 [+0.003, +0.068]; vs run 10 +0.052 [+0.022, +0.083]; vs 17 ideal +0.035 [+0.004, +0.066]. Last 8 evals vs PPO frac +0.044 [+0.017, +0.073]. One seed; ~2.8× PPO's episodes. Pod idle (watchdog stops it).
- **Run 17c launched** (user "yep"; `jobs_span17c.sh`, outputs/minicraft/span17/gft_synth_s0): 17b + new `--synth-success`: a request with no received success message gets a Stage II task under its source's verdicts all said as done in received wording (e.g. "Not done: get some metal. You hit the ore…" → "Great job! You did get some metal."); never added to the bank. Test `test_synth_success_for_requests_never_seen_to_succeed`; 49 tests pass. No more 17b seeds (user).
- **Run 17c at 40k:** 4 goals (pickaxe 12.5k, as 17b), stone 0.64 not yet mastered (17b: stone 37.6k), val 0.176 / 0.188 / 0.184 / 0.203 / 0.195 at 20–40k (17b 0.172 / 0.203 / 0.215 / 0.203 / 0.223).
- **Run 17c stopped by me at 65k calls** (not promising): stone mastered at 42.5k (17b 37.6k), then "upgrade your tools" at 0.00 for 88 rounds (17b got its first successes 24 rounds after the goal opened and mastered it at 71.4k); val 0.195 → 0.152 over 45–65k (17b 0.180–0.262). Cause: Stage II tasks under synthesized success messages ("Great job! You did upgrade your tools.") keep mostly failed attempts — kept-attempt success 0.11–0.34 vs 17b's 0.39–0.81 in the same phase. The scorer has never seen that success, so D(x', synthesized) > D(x', f) passes failures, and pi imitates them. In 17b the goal got no Stage II tasks until a real success was received, which avoided this. So built success messages do not solve the cold start; the frontier goal needs a signal the scorer can judge (e.g. the improved message's one step) until the first real success.
- **Run 17d launched** (user request; `jobs_span17d.sh`, outputs/minicraft/span17/gft_hybrid_s0): hybrid — Stage II reader under the improved message until the request has a received success message, then under a received success message (new `--success-prompt`); filter per task by its own message (`--rerank-ideal ideal`: improved test before, ideal test after); scorer with improved messages (run 10). Test `test_success_prompt_uses_improved_until_a_success_is_received`; 50 tests pass.
- **Run 17d at 30k:** 4 goals (2.3k / 6.7k / 8.7k / pickaxe 12.8k; 17b 2.3k / 6.1k / 8.2k / 12.5k), stone 0.63 (17b 0.66 at 30k), val 0.164 / 0.168 / 0.160 / 0.199 / 0.199 at 10–30k (17b 0.148 / 0.180 / 0.172 / 0.203 / 0.215), kept-attempt success 0.77.
- **Run 17d at 55k:** stone mastered 39.7k (17b 37.6k); "upgrade your tools" 0.00 for 60 rounds since it opened (17b: first successes at +24, 0.18 at +48); stone-pickaxe coverage 0.000–0.003; stone collection 0.57 → 0.44 (slower decline than 17c's 0.64 → 0.30; 17b held 0.52); kept-attempt success 0.18–0.30 (17b 0.39–0.81); improved-message fallback on 44% of tasks; val 0.211 / 0.164 / 0.199 / 0.203 / 0.172 at 35–55k (17b 0.203 / 0.223 / 0.180 / 0.230 / 0.188). So far the fallback tasks behave like 17c's built messages: they keep failed attempts and the frontier goal does not start.
- **Why 17d stalls on "upgrade your tools"** (17d checkpoint, 13 failed tasks containing the goal × 8 reader attempts; scratchpad upg17d.py): kept attempts do progress — 92% reach stone + stick vs 62% of sources; 44% get further than their source vs 29% of all attempts — but none makes the stone pickaxe. Only 13 of 64 failed sources got an improved message: for sources holding stone + stick ("You had what you needed for a stone pickaxe but did not make it.") the rewrite refused "You got a stone pickaxe." because a "not done" verdict for the goal was never received with it (getting it finishes the goal). So the finishing step had no message.
- **Fix (improve.py):** when the new sentence was never received with the goal unmet but was received with it achieved (done verdict or subgoal remark), it finishes the goal: the verdict flips to done (a done wording received for another goal if this goal's was never received), and with every goal done the message takes the success form. On 17d's received feedback: 28 of 64 sources now get an improved message, e.g. "…had what you needed for a stone pickaxe…" → "Well done! Done: upgrade your tools. You got a stone pickaxe."; multi-goal requests whose other goal was never said with it still get none. Tests updated / added (51 pass).
- **Run 17e launched** (user "ok"; \`jobs_span17e.sh\`, outputs/minicraft/span17/gft_hybrid2_s0): 17d with this rule. 17d keeps running (to ~70k).
- **Run 17d finished (100k; I meant to stop it at ~70k but it completed between checks):** 5 goals (stone 39.7k), "upgrade your tools" 0.00 to the end (stone pickaxe made in 0.1% of Stage I episodes, stone collected 0.43, kept success 0.11); val last 4 0.223 / 0.199 / 0.223 / 0.191.
- **Run 17e at 15k:** 4 goals (2.3k / 5.9k / 9.2k / pickaxe 13.1k), val 0.086 / 0.113 / 0.180 at 5–15k; stone pickaxe already made in 6.3% of Stage I episodes (17d: ~0.1% throughout), stone 0.09 (just opened).
- **17e filter favours regressions on the finishing step** (17e checkpoint; 15 upgrade tasks where f says "You had what you needed for a stone pickaxe but did not make it." and f̂ swaps in "You got a stone pickaxe."; 120 reader attempts, none made a stone pickaxe): by attempt stage — no wood pickaxe (n=14): r +1.79, pass 0.79, D(x, cause) −16.2, D(x, fix) −11.5; wood pickaxe only (n=18): r −2.94, pass 0.22, D(cause) −14.5, D(fix) −20.9; stone + stick (n=88): r −6.43, pass 0.09, D(cause) +11.6, D(fix) −15.6. The cause sentence is scored as true exactly for the attempts that got furthest, so they fail the filter; the scorer also rates "You got a stone pickaxe." slightly higher for attempts with nothing than for attempts one craft away (it has no grounding for it). The filter keeps the regressions.
- **17c kept attempts regressed (verified after the fact; 17c checkpoint at 65k, 64 failed upgrade tasks × 8 attempts under the constructed messages; scratchpad upg17c.py):** kept attempts were behind their source 43% of the time vs further 28% (all attempts: 31% / 20%) — the filter picked regressions more often than chance. Pass rate by stage: no wood pickaxe 0.52, wood pickaxe only 0.47, stone + stick 0.41. D(x, constructed) barely separates the stages (−0.25 / +1.99 / +1.65), so r = D(x, constructed) − D(x, f) is decided by how true f is, lowest for the furthest attempts (−1.59). The attempt-vs-source comparison D(x, c) − D(x_src, c) does point the right way (−1.79 / +0.35 / +1.45). Constructed messages for multi-goal requests also declare every other goal done ("That worked. Done: fetch the gem. Done: upgrade your tools.").
- **Cheap ensemble test (offline, 17e checkpoint; scratchpad ens17e_b.py):** frozen scorer trunk, 4 heads replacing the final projections (anchored at the trained weights, + fixed random prior ×0.3, each on a bootstrap 70% of the replay, 300 steps). Heads fit (msg accuracy 0.89–0.92 vs the scorer's 0.90). Disagreement std_k D_k(x, m*) on 17e's upgrade attempts: attempts that tried to craft a stone pickaxe (n=4) 0.39 vs others 0.46; by stage: no wood pickaxe 0.52, wood pickaxe only 0.56, stone + stick 0.43. So the head ensemble is not more uncertain about the novel goal-directed attempts (if anything the reverse); the first try (fresh heads, prior ×1.0, 150 steps) underfit (accuracy 0.11–0.29) and is not informative. Likely reason: the frozen shared trunk maps novel trajectories onto familiar features, so heads on top cannot see the novelty.
- **Run 17e final (100k):** 5 goals (2.3k / 5.9k / 9.2k / 13.1k / stone 33.8k), val 0.207; upgrade tools 0.00 to the end.
- **Run 17f** (\`--rerank-s\`, \`jobs_span17f.sh\`, span17/gft_srank_s0): 17e with improved-message tasks (no received success yet) keeping the top 8 attempts by s = D(x', f̂) − D(x_src, f̂) > 0 instead of 8 random ones with r > 0. 5 goals (1.5k / 6.4k / 6.9k / 12.8k / stone 34.6k), val 0.19 at 60k; upgrade tools 0.00 in Stage I and Stage II from 35k to 60k. **Stopped at 60k** (user "yes"). Why (offline, same 29 failed upgrade worlds × 16 reader attempts at T 1.0): kept by stage (no wood pickaxe / wood pickaxe only / stone + stick / stone pickaxe made) 0.06 / 0.15 / 0.79 / 0 — the 17e plateau (17e final, r rule: 0.10 / 0.18 / 0.72 / 0); the reader made a stone pickaxe in 1–2 of 464 attempts and s rated them below their source (pass 0.00). On this checkpoint s keeps fewer attempts that get further than their source than r does (0.28 vs 0.68). The earlier 17e hint that s favours craft attempts (n≈4) did not hold. Bottleneck: the reader almost never crafts, and the scorer does not read the craft as progress.
- **Runs 17g / 17h launched** (user "launch a variant with s as ideal filter as well" / "with that off"): 17g (\`--rerank-s --rerank-s-ideal\`, span17/gft_sall_s0) ranks every task by s, ideal-message tasks by D(x', ideal) − D(x_src, ideal) (ideal = mean over the last 8 received success messages); 17h (span17/gft_sall_noimit_s0) = 17g without \`--rerank-reader-imitate\`.
- **Runs 17g / 17h final (100k)** — both finished before the account ran dry (pod ro12 was then removed by RunPod at balance −$0.02; network volume intact; ro13 \`byscojmvky3y0u\` created 2026-11-02 to recover, nothing to resume). 17g (s on every task): 5 goals (2.3k / 6.4k / 10.0k / 14.3k / stone 35.6k), val 0.219, upgrade 0.00. 17h (17g, reader does not imitate): 5 goals (2.3k / 6.4k / 8.4k / 13.6k / stone 47.1k), val 0.223, upgrade 0.00. Neither breaks the upgrade-tools stall; 17b (success messages only, r-type ideal test) remains the only run with 6 goals (val 0.219, upgrade 0.38). Single final evaluations, one seed: the val differences (0.207–0.223) are within noise.
- **Pods deleted 2026-11-02** (user "delete pods then"): ro13 `byscojmvky3y0u`, ro11 `mz0eu7buz750nu`, ro10 `wl4uq8pacgsuwb` (all network-volume only). No pods left; network volume `yy9bsgna2b` kept (all outputs and checkpoints; S3 access via `s3_pull.sh`). Balance $9.98.
- **Default GFT = run 17g** (user 2026-11-02 "use 17g as default gft"): `train.py --arm gft --preset 17g --budget ... --out ...` applies 17g's flags (explicit flags win); argparse defaults left unchanged so old job scripts reproduce. tests/test_train_preset.py checks the preset parses to exactly jobs_span17g.sh's arguments.
- **Audit of the 17g path (2026-11-02, two independent reviewers + spot checks).**
  - *Leaks:* none found. Success for the bank is read from the message text (no "not done" verdict, learn.py:3748); Stage II `ok`/`frac` (env success) feed only the logged rank correlation; every teacher message is counted; Stage II makes no teacher calls. Caveats (shared with PPO unless noted): the curriculum's mastery/unlock uses env success (`cur.update(..., sat, ...)`); the prerequisite graph comes from the expert planner; the base policy and the scorer's frozen features come from BC on expert runs; GFT reads the teacher's text through an exact template parser (`verify.parse`, `improve.py`), not a learned one (GFT only).
  - *Bug fixed:* the ideal score averaged bank messages by cycling them to a common length, and D(x', ideal) and D(x_src, ideal) used different cycle lengths when 512 Stage II rows split into two 256-row chunks — the two sides of s then weighted the same messages differently (a per-task offset that shifts the s > 0 cutoff, not the order within a task). Both now use `bank_mean` (plain mean over each bank's own messages). This also changes r_ideal slightly for 17b/17d/17e-style runs (old runs are not bit-reproducible on the new code). Stale `s_ideal_pass` logging fixed.
  - *Tests strengthened:* the s tests now include attempts with s ≤ 0 (only s > 0 kept, fewer than topk); `bank_mean` checked against brute force with unequal banks over several chunk sizes. 1099 tests pass.
  - *Prompt/judge aligned (user "ok yeah do it"):* with `--success-prompt` the reader's prompt is now drawn from the last `ideal_k` (8) received success messages of the request, the same ones the ideal averages over (was: any of the last 32). Test checks older messages are never used. 1099 tests pass.
  - *Logging caveats:* under s, `ideal_pass` / `imp_pass` / `rerank_rank_frac` describe r, not the rule in use.
- **Where the scorer attends (17e checkpoint, ~65k; scratchpad att17e.py / att17e_b.py).** 27 expert episodes on upgrade-only worlds (all craft a stone pickaxe) vs 17 pi near misses (end with stone + stick, no craft). "You got a stone pickaxe.": attention peak on making planks (−wood +plank) in 22/27 expert episodes, ~18 steps before the craft; mass within ±2 steps of the craft 0.07 (uniform 0.15); scores real crafts −8.3 vs near misses −14.6 (right order, wrong feature). "You had what you needed for a stone pickaxe but did not make it.": attention on the craft step (mass 1.00, argmax +1) in expert episodes, on the episode end in near misses; scores −15.2 vs +24.2. Without span supervision the attention finds the right step for a sentence with many positives (the cause) and not for one with a few early ones (the fix). Also (from stage1_coverage): the base policy makes a stone pickaxe in 33% of round-0 episodes (whatever the request); 17e / 17g lose this by ~30k, before upgrade unlocks (~34k); 17b keeps it (0.13–0.32 of episodes at 51–82k) and is the only run to master upgrade (71k).
- **Scorer and improved messages:** f̂ is used only as a negative (D(x_e, f_e) vs D(x_e, f̂_e)); since f̂ swaps one cause sentence for one fix sentence, the contrast is effectively d(x, cause) up vs d(x, fix) down, and the fix sentence's negatives fall on near misses only — d(·, fix) is pushed down exactly where the attempt got closest.
- **Run 18 launched 2026-11-02** (user "go"; pod ro14 `upg24w69bq7p32`, RTX PRO 4000, ssh `-p 12028 root@213.173.104.20`; `jobs_span18.sh`, outputs/minicraft/span18): GFT `--preset 17g` and PPO fraction reward, both from `--base scratch` (new: an untrained, seeded random-init policy; `--base random` is BC on expert runs for random goals and already makes a stone pickaxe in ~1/3 of its first episodes). Smoke (CPU, 2 rounds): from scratch pi gathers firewood in ~2% of Stage I episodes. Code synced includes the bank_mean fix and the prompt/judge alignment.
- **Auto-shutdown (user 2026-11-02):** `pod/watchdog.sh` gained `ACTION=terminate`; started on ro14 with `IDLE_MIN=5 ACTION=terminate`: the pod deletes itself 5 min after its last job exits. Results stay on the network volume; pull with `bash -ic "experiments/minicraft/pod/s3_pull.sh gft_old span18"`.
- **Pretraining plan revised (user 2026-11-02, "right"):** v3 in docs/minicraft_design.md -- pretrain only a diverse behaviour prior (random courses of scripted primitives: go-to, use, select/put/take, craft, wander, done; unconditional BC, no requests, no text); all language from post-training feedback. Planned mechanics change: craft makes exactly what is put into a station's slot (generic select/put/take/craft). Supersedes v2 (expert-plan BC + narration captions). Motivation: old-curriculum BC base already demonstrated every goal; from scratch (run 18) both arms stalled at goal 1.
- **Run 18 final:** PPO from scratch 100k: 1 goal (firewood 8k), val 0.035, planks 0.00 throughout. GFT from scratch stopped at 60k (user "yeah stop"): 1 goal (firewood 10.2k), val 0.039; firewood val 0.91 at 20k then oscillating 0.17-0.66 after planks opened; planks 0.00 (pi gathers wood in 97% of planks episodes but presses craft plank away from a workbench; actions near uniform, entropy 2.4 of 2.56). Pod ro14 left to the watchdog (terminate 5 min after the last job).
- **v3 step 1 done (2026-11-02): slot crafting implemented** in the new game (core.py: select / put / take / craft, exact match per station, `agent_state()` = inventory + cursor + adjacent slot (42); expert plans with slot contents and does take / select / put / craft (56 steps mean, 98.8% at 20% noise); teacher `craft_held` = what the slot held, replayed from events (`slot_steps`), templates reworded; snapshots and policy inputs use agent_state, inventory-only consumers slice the first 14 columns). 1060 tests pass. Next: the primitive explorer (v3 data policy) and its coverage stats.
- **v3 step 2: primitive explorer** (`gft/minicraft/explore.py`, tests/test_minicraft_explore.py): per-world random courses of gather / craft-attempt (go to station, put 1-3 held units, craft, maybe take) / wander / done, curiosity biases (unheld resource kinds, least-visited stations, kinds not put yet, untried slot contents); the behaviour never reads recipes, goals or the expert plan (AST test). 2,048 episodes (~25 s CPU): no held-out drops, mean 130 steps, ~16 events/episode, ~130 distinct (station, slot kinds) craft attempts, craft success ~7%; by chance only first-tier crafts (plank / rope / brick ~0.12 per episode each), no pickaxe, hence no stone and no composites. Repeating successful options (p 0.5 / 0.8) and a corpus-wide novelty memory did not raise depth.
- **Decision (user 2026-11-02 "ok accept it"):** the v3 explorer stays recipe-agnostic with no random starting inventories; its prior only makes first-tier items by chance, and deeper crafts must come from post-training feedback.
- **v3 step 3: unconditional BC on the explorer corpus** (`explore.pretrain_explorer`, `policy_coverage`). CPU smoke: 512 episodes, 500 steps (422 s), loss 2.30 -> 1.34; the clone keeps the explorer's diversity on 256 fresh worlds: events/episode 16.5 vs 16.8, distinct craft attempts 72 vs 81, craft success 0.056 vs 0.056, first-tier crafts ~0.1/episode each, action mix within ~2 points per action; it made a pickaxe once (0.004/episode). Next: wire `--base explore` into train.py and post-train GFT / PPO from it (GPU).
- **v3 step 4: `--base explore` in train.py** (20k explorer episodes, 10k BC steps; held-out items from --heldout-goals, else none). CPU smoke: PPO from a tiny explore base runs end to end (pixels). GFT: the new game's learn.py has none of the old-curriculum 17g machinery (message scorer, Stage II rerank of reader attempts, success banks, s filter); porting it is the open item.
- **v3 step 5: run 17g ported to the new game** (user "1"): `gft/minicraft/gft_message.py` (MessageGFT: message-level scorer, Stage II reader attempts under a received success message of the request (last ideal_k) or the improved message, top 8 by s > 0, damped distillation with stay_stage1 and the entropy controller, reader imitation of kept attempts; only 17g's path) and `gft/minicraft/improve.py` (improved message for this teacher: needed / recipe hint / source hint for X -> made or collected X; tool use failure -> made the tool; used up -> the goal flips; finishing step flips a goal only if received feedback said the fix only with it achieved). Infrastructure: curriculum `confirm` (two-window mastery), `WorldConfig.success_needs_done` (run_episodes and the verifier), train.py `--recipe message`, `--cur-confirm`, `--success-needs-done`, `--eval-on-spent`, `--preset 17g`. Tests: tests/test_improve.py (10), tests/test_gft_message.py (4: a round runs and resumes exactly; tasks are failures under a recent success or the improved message; top-by-s with only s > 0; bank_mean). Deviation: a round whose failures state no cause and have no received success skips Stage II (the old code fell back to the source's own message).
- **17g on the new game, CPU smoke** (tiny explore base: 128 episodes / 30 BC steps; 512 teacher calls, 64 per round): 8 rounds run end to end; Stage II on 4 tasks per round, improved-message tasks (s pass 0.50, round 1) then received-success tasks (s pass 0.36-0.67); scorer message accuracy 0.20 -> 0.43; success 0 (toy base). 1078 tests pass. Next: on a pod, the full explore base (20k episodes, 10k BC steps), then GFT `--preset 17g` and PPO (fraction reward) from it, 100k budget, one seed.
- **Audit of today's new-game code (2026-11-02; three independent reviewers: mechanics, 17g port fidelity, leaks / budget).**
  - *Leaks:* none. Banks, improved messages, Stage II targets, scorer data and the s filter read message text only; env success only in the curriculum (shared with PPO; the realistic teacher's text matched it in 1024/1024 episodes) and logs; Stage II never calls the teacher; success_needs_done applied once in run_episodes (teacher facts, curriculum, PPO reward, evaluation, verifier agree). Caveats: exact template parser / improve.py use the teacher's language; the pool keeps only expert-solvable tasks; prerequisite graph from the expert's plan (both arms).
  - *Fixed:* (1) expert counted slot contents as held for goals and tools (said done early, hit rocks forever without its pickaxe; 1-3% of noisy-expert episodes) -> takes stranded items back before any non-craft step (fast and reference agree; 0 premature dones in 800 noisy episodes); (2) improve.py lost the old guard rejecting a used-up flip when another "not done" goal was never received with it (~5% of improved messages); (3) the inherited finishing-step flip (co-occurrence) made false "Done" claims (old rule: 22 of 36 improved messages flipped a goal, all false by item check) -> a goal flips only if received feedback tied the fixed item to it ("used up X" with it the one goal left); now 11 improved messages per 81 noisy-expert failures, 0 false flips; (4) failed-craft narration now says counts ("with only 3 planks in it"; 2 planks make a pickaxe); (5) explore drops episodes that pick up a held-out item too; (6) exact-vocabulary policy text (`--exact-vocab`, in the preset, as old 17g); (7) MessageGFT forward chunks follow learn.FWD_CHUNK (pixel memory); (8) final evaluation at the end of the budget with --eval-on-spent; one `examples` key; (9) BASE_VERSION v10 and pool tags v2 (caches from before slot crafting are not reused). Tests: expert with stranded items, success_needs_done, improve flips; 1081 pass.
  - *Open:* `--heldout-goals` forces `--curriculum off`, so a curriculum run from the explore base has no held-out items (design decision for the v3 experiment); the teacher names a later missing item half the time, so a "fix" may skip a prerequisite (teacher design); TAKE counts as a gain in keyframes / cumulative_gains / trajectory_facts (frames, sentence-reader, discriminator arms only; unused by 17g); lang_corpus labels ignore success_needs_done (v1 lang base only); launch.py / plot_mastery.py compare budgets by episodes (use `spent`).
- **`--exclude-goals` (2026-11-02, user "ok"):** goals never requested in training (pool filter, so the curriculum never samples them; `--base explore` drops episodes that make or pick up their items; other bases except easy refused) and evaluated on their own split (`val_unseen` / `new_unseen`: requests containing one; val_worlds / val_combos leave them out). Intended for the v3 run: `--exclude-goals "build a well,make a sling"`. tests/test_train_options.py (preset 17g flags; splits).
- **v3 run launched (2026-11-02):** pod `1722vfqnt7uva3` (`gft-minicraft-v3`, RTX PRO 4000, EU-RO-1, $0.57/h, ssh `-p 33936 root@213.173.102.10`), `experiments/minicraft/pod/jobs_v3.sh`: explore base (20k episodes, 10k BC steps) then GFT `--preset 17g` and PPO (frac) from it, `--exclude-goals "build a well,make a sling"`, 100k feedbacks, 1 seed; outputs `outputs/minicraft/v3`. Watchdog `IDLE_MIN=5 ACTION=terminate`; pull with `bash -ic "experiments/minicraft/pod/s3_pull.sh gft v3"`. CPU smoke of both arms passed first (tiny base).
- **v3 run result (2026-11-02, 1 seed, stopped):** from the explore base (round 0: train 4.3%, val_worlds 5.5%) both arms
  mastered only gather wood / gather fiber / dig clay (GFT by 6.7k feedbacks, PPO by 9.2k) and then stalled on the
  first crafting tier (rope, bricks, planks). PPO 100k: train 9.8%, val_worlds 9.8%, val_combos 0, val_unseen 0. GFT
  stopped at 25k (flat, train 9.8%, val_worlds 8.6%) to optimise speed first. Results: `outputs/pod/gft/v3/`.
- **GFT speed benchmark (2026-11-02, RTX PRO 4000, `pod/jobs_bench_gft.sh`, 10 rounds each, steady state = rounds
  5-9):** fp32 14.8 s/round (256 feedbacks): scorer training 4.8 (72 steps x ~67 ms GPU, compute-bound, fp32 SIMT
  matmuls), reader on 4 replayed rounds 2.7, distillation 2.4, rollouts 2.2, reader imitation 1.0, features 0.9,
  old probs + teacher 0.6. `--compile` 12.8 s (1.15x; first round +20 s). `--bf16 --compile` ~9 s (1.6x, losses
  normal) but plain `--bf16` gave NaN policy logits in round 0 (reader acting under the feedback prompt): bf16 is not
  trusted until that is root-caused. TF32 matmuls: scorer step 82 -> 71 ms (`experiments/minicraft/bench_scorer.py`).
  The v3 run's 23 s/round was GPU sharing with PPO; alone GFT did ~15 s/round. Exact code changes: scorer loss
  builds its group matrix with one transfer (identical loss and gradients), scores cast to float (autocast-safe);
  per-phase timers (GPU-synchronised) in `round()["time"]`.
- **Can BC master the tasks? (2026-11-02, `experiments/minicraft/check_bc.py`, RTX 4090 pod, 1 seed):**
  request-conditioned BC (prompt = request) on 10k noise-free expert episodes of all requests without well/sling,
  pixels, success needs DONE; single-goal requests on fresh worlds, 64 per goal (sampled / greedy; expert 1.00 on all).
  10k steps (loss 0.30): mixed 1-3 goal requests 0.24; wood 0.97, planks/pickaxe 0.69, bricks 0.55, fiber 0.58,
  clay 0.39, rope 0.27, raft 0.38, oven 0.28, stone 0.22, wall 0.02, fence 0.05. 50k steps (loss 0.296 -> 0.229 ->
  0.209 -> 0.192 -> 0.181, still falling): mixed 0.49; wood 0.98/0.83, pickaxe 0.94/0.81, planks 0.91/0.73, fiber
  0.89/0.62, clay 0.88/0.67, bricks 0.80/0.73, rope 0.69/0.41, stone 0.62/0.64, oven 0.58/0.48, raft 0.55/0.34, fence
  0.33/0.27, wall 0.27/0.14. Reading: the first crafting tier is learnable from pixels with demonstrations (the v3
  arms' stall there is a learning-signal problem, not capacity/perception); deeper chains stay hard even for BC.
- **Feedback audit and fix (2026-11-02):** on the v3 GFT policy (48 single-goal episodes per goal) the teacher was
  accurate (0 false of ~5.7k sentences, also with DONE required) but two causes went unexplained: (1) "make planks"
  takes 2 planks and an agent holding 1 heard "You still needed a plank." -- its improved message added "You made a
  plank." (already true), so GFT had no target toward the second plank (planks 0.02, BC 0.91); (2) episodes that held
  everything but did not end by DONE (2-10%, mostly gathering goals) got "Not done: G." with no cause (no DONE wording
  existed). Fix: teacher claims `short` ("You had 1 of the 2 planks needed to make planks." / "You needed 2 planks to
  make planks but had only 1.") and `no_done` ("You had what you needed to G but never said you were done." / "You
  never said you were done, so G did not count."), both naming the goal; neutral "Not done:" when part of the item is
  held; verifier parses/checks both (still 0 false; plain-English misreadings 7% -> 5% of failures).
  Improved message (improve.py `_finish`): the named goal flips to done ("Great job! Done: make planks. ... You made 2
  planks."; a missing DONE flips every goal said ready). On the v3 policy's mixed-request rollouts 41/171 planks
  failures carry `short`, all 41 now flip to done. Vocab: appended what/said/were/count; `PromptEncoder` loads exact
  checkpoints saved before words were appended (new rows inserted, old word/number/bucket rows kept in place;
  checked on the v3 policy). `infer_ideal` reads `short` as a goal requirement.
- **v3b launched (2026-11-02):** pod `c81lv4ocpv961o` (RTX PRO 4000, $0.57/h), `pod/jobs_v3b.sh`: GFT `--preset 17g --base explore --exclude-goals "build a well,make a sling" --compile`, 100k feedbacks, 1 seed, after the feedback fix; outputs `outputs/minicraft/v3b`; watchdog terminate.
- **v3b result (2026-11-02, 1 seed, 100k feedbacks, ~100 min, `results/minicraft/v3_frontier.png`):** mastered the
  same 3 gathering goals as v3 (wood 5.1k, fiber 6.4k, clay 16.4k -- clay later than v3's 6.7k); nothing on the
  crafting tier reached 0.7. Own-episode success, last 10 rounds: planks 0.24 (peak 0.33 at ~95k; v3 never above
  0.03), rope 0.27, bricks 0.31; planks climbed 0.01 (35k) -> 0.08 (50k) -> 0.17 (80k) -> 0.33 (95k). Final eval
  train 0.098, val_worlds 0.121, val_combos 0, val_unseen 0 (v3 PPO at 100k: 0.098 / 0.098 / 0 / 0). Rope and bricks
  plateau near 0.25-0.3 from ~35k. `--compile` speed: ~285 s per 5k feedbacks.
- **Teacher helpfulness audit on v3b rollouts (2026-11-03).** 576 rollouts of the final v3b policy on single
  planks / rope / bricks requests. Dominant failure: the right input crafted at the wrong station, often 20-40 times
  (69-82% of failures). Feedback was true but often unhelpful: circular "needed" of the goal's own item ("Next time
  get rope before anything else.": 18% / 36% / 20% of failures), the right station named in only 35-44% (upper bound,
  narration included), narration of unrelated events, ~6-11% of praise without the goal. Also seen: Stage II success
  on the improved message never above Stage I (the message pathway does not steer), and many kept Stage II episodes
  were failures (planks: 14 of 14 in the logged sample).
- **Teacher fix** (`compose_realistic`, `narrate`): an unmet goal's own item gets the recipe/source hint, never
  "needed"; a success always confirms its deepest goal; narration events about none of the request's items weigh
  0.2x (`OFF_FOCUS`). On the same rollout set: circular 0 in all three, right station named 83% / 98% / 94% of
  failures, praise names the goal 100%. check_teacher_text (2,048 msgs): 0 false sentences; literal
  needed_but_holds 2.7%, managed_but_needed 0.8% (pre-existing kinds, multi-goal requests). New test in
  tests/test_minicraft.py; full suite 1086 + 1 passed.
- **Scorer check (v3c checkpoint at 45k, offline, 64 Stage II tasks x 16 attempts on single planks / rope / bricks
  requests, targets all from the success bank).** Progress level per episode: 0 none, 1 input collected, 2 goal item
  crafted, 3 enough held, 4 success. Attempts vs their source: 27% better, 55% same, 18% worse. The scorer ranks
  well: within-task AUC of s, better vs not 0.98, success vs failure 0.99; P(kept | better) 0.88, success 0.98, same
  0.27, worse 0.06. But the keep rule (top 8 of 16 among s > 0) fills up with same-level attempts: of 410 kept, 159
  succeed, 87 are better failures, 153 same, 11 worse. Keeping only the better ones (oracle) would make kept success
  0.63 vs 0.39. Caveat: "same" is same coarse level, possibly finer progress. Diagnostic hook:
  `MessageGFT.last_stage2`.
- **v3c killed at 45k** (user) for v3d: v3c + `--msg-target ideal` (constructed ideal feedback: teacher's message
  about a banked verified success of the same task). v3d at 20k: 3 gathering goals (wood / clay 10.0k, fiber 10.5k),
  planks 0.00, rope 0.01, bricks 0.01, kept_success 0.07 (v3c at 25k: 0.06 / 0.14 / 0.16).
- **v3d stopped at 35k** (not promising): planks 0.01, rope 0.08, bricks 0.03 (v3c at 40k: 0.11 / 0.18 / 0.18);
  results in outputs/pod/gft/v3d. The constructed same-world ideal target did worse than received successes.
- **Temperature sweep (v3c policy at 45k, 96 single requests each):** success planks / rope / bricks at T 1.0
  0.21 / 0.25 / 0.25, 0.7 0.12 / 0.22 / 0.26, 0.5 0.17 / 0.09 / 0.19, 0.3 0.01 / 0.02 / 0.07, greedy 0 / 0 / 0
  (mean steps 152 -> 200: lower temperature loops until the step limit). Entropy is not the brake; the randomness is
  what gets the policy out of loops. Policy entropy stayed ~1.4 nats/step in v3b (target 0.8, alpha at its floor)
  and distillation loss = entropy throughout.
- **v3e prepared** (user): v3c + `--msg-keep-frac 0.5` (top half of each Stage II task's attempts with s > 0, not
  the top 8) + `--msg-epochs-clean 4` (distillation passes, not 2). New MessageGFTConfig.keep_frac; test
  parametrized; full suite 1089 passed.
- **v3e result (100k feedbacks, 1 seed; v3c + keep top 50% of s > 0 + 4 distillation passes):** gathering mastered
  at 4.9k / 5.4k / 5.6k; no crafting goal reached 0.7 (own episodes, 10-round mean at 100k: planks 0.65, rope 0.54,
  bricks 0.64; plateau ~0.65 from ~70k). Final eval train 0.207, val_worlds 0.215, val_combos 0, val_unseen 0
  (v3b 0.098 / 0.121). Eval per-goal (train split, ~40 eps each): planks 0.57, rope 0.30, bricks 0.67. Kept success
  ~0.49 at 25k (v3c 0.28); policy entropy fell to ~1.0 nats/step (v3b/v3c ~1.4). Biggest gain of any change so far.
- **PPO sweep (one change each from the v3 PPO baseline, 50k rewards, running):** at 45k only P4 (group advantage,
  no critic) moves the crafting tier: eval planks 0.02, rope 0.45, bricks 0.43, train 0.168, val_worlds 0.188
  (v3e at 45k: 0.45 / 0.32 / 0.52, train 0.164). P1 (4 epochs), P2 (256 episodes / round), P3 (lr 1e-3): 0 on the
  crafting tier at 30-45k.
- **PPO sweep result (50k rewards):** eval at 50k (train split per goal planks / rope / bricks; train, val_worlds):
  P1 epochs 4: 0 / 0 / 0; 0.074, 0.105. P2 256 episodes: 0 / 0 / 0; 0.102, 0.145. P3 lr 1e-3: 0 / 0 / 0; 0.094,
  0.105. P4 group advantage: 0.00 / 0.25 / 0.52; 0.152, 0.188 (v3e at 50k: 0.48 / 0.23 / 0.50; 0.152, -). All
  mastered only the 3 gathering goals. Results in outputs/pod/gft/ppo_sweep; checkpoints on the volume.
- **PPO P4 (group advantage) resumed to 100k:** eval at 100k train 0.184, val_worlds 0.215, combos 0, unseen 0;
  planks 0.20, rope 0.39, bricks 0.62; gathering mastered at 5.6k / 6.7k / 7.2k, nothing else. GFT v3e at 100k:
  train 0.207, val_worlds 0.215; planks 0.57, rope 0.30, bricks 0.67. The arms tie on rope / bricks and validation;
  GFT leads on planks (the count-2 goal). Plot: results/minicraft/v3_frontier.png (plot_v3.py now v3e vs P4).
- **Reader check (v3e checkpoint, local):** single planks / rope / bricks requests, Stage I success 0.57 / 0.72 /
  0.75 (higher than the per-goal numbers from mixed requests). The 92 failed tasks retried 8x each in the same world
  (736 attempts per condition, SE ~0.02): policy + request 0.515; reader + request 0.447; reader + request + m* (a
  received success message of the request, as Stage II) 0.507; reader + request + a success message of another
  tier-2 goal 0.512. Any success-form message lifts the reader by ~0.06 equally: the reader does not use the
  message's content (consistent with the earlier blind-reader audit). Failed tasks are only slightly harder than
  average (0.52 on retry): failures are mostly sampling luck.
- **Unlock-0.5 runs launched (user):** GFT v3e settings and PPO P4 settings with --mastery 0.5, one RTX 4090 pod,
  `pod/jobs_unlock05.sh`, outputs/minicraft/unlock05. Mastery still reported at 0.7 from the evaluation.
- **Unlock-0.5 runs killed (user) and pod terminated:** GFT at 10k feedbacks, PPO at 65k rewards; neither had
  unlocked a tier-2 goal at the 0.5 bar. Partial results in outputs/pod/gft/unlock05. RunPod balance $2.71.
- **Single-task requests (user 2026-11-03):** `train.py --max-goals` now defaults to 1: training pool, curriculum,
  evaluation and the unseen-goal split (fixed: it ignored max_goals) all use single-goal requests; val_combos is
  dropped. Earlier runs (v3-v3e, PPO sweep) used 1-3 goals per request. New pool cache key (max_goals in kwargs).
  Test: tests/test_train_options.py.
- **Reader-free GFT modes (user 2026-11-03):** `--msg-mode policy` (Stage II attempts are pi's own retries under the
  request, scored and kept as before, no reader training) and `--msg-mode stage1` (no Stage II rollouts: the scorer
  ranks each request's Stage I episodes by D(x, ideal) and the top `--msg-keep-frac` (default half) are distilled).
  Tests in tests/test_gft_message.py. Job script `pod/jobs_lite.sh` (100k each, single tasks) ready; waiting for a
  top-up (balance $1.75).
- **Single-task results (100k feedbacks, 1 seed each, explore base, well/sling excluded, mastery 0.7):**
  GFT (v3e settings): mastered gathering at 5.1k / 5.4k / 5.9k, then rope 75.8k, bricks 77.8k, planks 78.8k (first
  run to master planks); tier 3 (pickaxe / raft / oven) unlocked but 0 by 100k. Final eval train 0.297, val_worlds
  0.348, val_unseen 0; eval per goal planks 0.42, rope 0.54, bricks 0.59 (~40 episodes each, noisy; peak 0.58 /
  0.69 / 0.82 at 65k). PPO GRPO (group advantage): gathering 6.7k / 7.2k / 7.7k, bricks 52.7k, rope 56.8k, never
  planks (eval 0.00 throughout), so tier 3 never unlocked. Final train 0.273, val_worlds 0.281, val_unseen 0.
  Results in outputs/pod/gft/single. Reader-free GFT modes (policy / stage1) launched on the same pod.
- **Reader-free GFT stopped (not promising):** `policy` at 40k: eval planks 0.00, rope 0.27, bricks 0.06, train
  0.176, kept_success ~0.24; `stage1` at 65k: 0 / 0 / 0, train 0.113 and falling (gathering eval regressing),
  kept_success ~0.31. Full GFT (reader) at 40k: 0.21 / 0.54 / 0.59, train 0.297, kept_success 0.81. So the reader
  matters: it is trained by hindsight on (request + message) and, prompted with a success message, acts as a
  success-conditioned policy -- it reads the verdict (success vs failure), not the message's details (reader check:
  right vs wrong success message equal). stage1 self-imitation of the top half of its own episodes degrades.
- **Reader verdict check (single-task GFT checkpoint at 100k, `experiments/minicraft/check_reader_verdict.py`, on a
  pod):** 152 failed tier-2 tasks x 8 attempts (SE ~0.014): policy 0.399; reader alone 0.404; reader + right success
  message 0.404; reader + own failure feedback 0.406; reader + "Thanks!" 0.386; reader + another goal's success
  message 0.312 (rope 0.24 vs 0.41, bricks 0.25 vs 0.30). Tier 3 (pickaxe / raft / oven, 144 tasks x 6, right
  message = teacher on an expert success): 0 under every condition. So the reader ignores the verdict and the how-to
  details; the only effect is that a message naming another goal pulls it off its task. The "reader reads the
  verdict" explanation of the reader-free result is NOT supported. Open: why reader-free GFT did worse; one
  difference in the logs is entropy (policy mode 0.66 at 10k vs 1.22 with the reader): self-imitation collapsed it.
- **Cause runs launched (user "go"):** `pod/jobs_cause.sh`, single tasks, v3e settings, 40k each, one pod:
  control (+ `--msg-shadow`: pi also plays every Stage II task, logged), h1 (`--msg-mode policy --stage2-temperature
  1.3`), h2 (`--msg-mode lagged --msg-lag-rounds 4`), h3 (`--msg-reader-request-only`). New options in
  MessageGFTConfig (mode "lagged", lag_rounds, reader_text, shadow); tests in tests/test_gft_message.py.
- **Cause runs result (single tasks, v3e settings, 40k feedbacks, 1 seed each; eval planks / rope / bricks, train,
  val_worlds):** control (full GFT) 0.79 / 0.69 / 0.76, 0.355, 0.371; h2_lag (Stage II from a copy of pi refreshed
  every 4 rounds, no reader) 0.16 / 0.42 / 0.47, 0.273, 0.289; h3_noms (reader trained and prompted with the request
  only) 0.00 / 0.54 / 0.41, 0.281, 0.254; h1_temp (pi's retries at T 1.3) stopped at 15k (entropy 2.16, collapse).
  Same full-GFT config in the earlier single run at 40k: 0.21 / 0.54 / 0.59, 0.297, 0.266 -- so seed-to-seed spread
  is large (planks 0.21 vs 0.79) and h2 / h3 fall inside it; reader-free `policy` mode (0.00 / 0.27 / 0.06, 0.176) is
  below it. Shadow (control): reader vs pi on the same Stage II tasks 0.05 / 0.05 (10k), 0.24 / 0.25 (20k), 0.44 /
  0.45 (30k), 0.54 / 0.56 (40k): the reader is never a better sampler. Reading: language in the reader is not
  needed (h3), a second lagged network works (h2); what fails is sampling Stage II from the very policy being
  trained (self-imitation collapse, entropy 0.66). Needs 2-3 seeds per arm to confirm.
- **Reader loss vs policy entropy:** the reader's training loss equals pi's entropy all run (single-task v3e 1.283 /
  1.284 at 0k ... 0.752 / 0.744 at 98k; control within 0.016 nats throughout): it never predicts pi's actions better
  than pi's own uncertainty, i.e. the messages add no usable information about the actions.
- **Message effect per state (`experiments/minicraft/check_reader_kl.py`, single-task v3e checkpoint, 4,000 states
  from 120 of pi's episodes):** KL from the reader with the right success message (entropy 0.76 nats) to: no message
  0.030 (p95 0.12), own feedback 0.018, another goal's success 0.044, "Thanks!" 0.024; to pi (request only) 0.035.
  The sampled action differs at 5-6.5% of steps (7% vs pi). Messages do change trajectories a little (~8-10 of ~150
  steps), about as much as reader vs pi differ, but not toward success (success unchanged; check_reader_verdict.py).
- **Reader fit test (`experiments/minicraft/check_reader_fit.py`, single-task v3e checkpoint, on a pod):** 6,000
  train / 1,000 held-out episodes of pi (734k / 123k steps, success 0.63; goals wood, fiber, clay, planks, rope,
  bricks); readers from the checkpoint's reader, 10k Adam steps (batch 1024, lr 3e-4) per prompt. Floor (pi's own
  NLL on held-out actions) 0.713. Held-out NLL, best -> at 10k: request 0.726 -> 0.732; realistic 0.731 -> 0.789;
  complete (item counts) 0.727 -> 0.746; narrated (every event with places, ~47 words) 0.729 -> 0.773. Train NLL at
  10k: request 0.697, realistic 0.635, complete 0.675, narrated 0.648. So the text pathway works (feedback lowers
  train NLL) but only to memorise episodes: no feedback ever lowers held-out NLL below the request-only reader, even
  a full narration of the episode, and all overfit. Memorisation test (48 episodes) uninformative: ~0.30 for all,
  the states already identify the episode.
- **Seed runs launched (user "launch a couple more seeds"):** `pod/jobs_seeds.sh` on pod `5lwr2sknpr4401` (RTX PRO
  4000, $0.57/h, watchdog terminate): seeds 1 and 2 of control (full GFT), policy (reader-free, pi's own retries) and
  h2_lag (copy of pi refreshed every 4 rounds), single tasks, v3e settings, 40k each; base is seed 0 in all runs.
  Question: is the reader's value only being a second, lagging sampler (control ~ h2_lag > policy across seeds)?
- **GRPO seeds queued (user: "run more grpo seeds to make sure it can't learn planks"):** `pod/jobs_grpo_seeds.sh`
  on the same pod, starts when the seed runs exit: PPO group advantage, seeds 1-2 (cut from 1-3 at user request), same settings as
  single/ppo_grpo (seed 0: 5 goals, no planks), 100k each, outputs in outputs/minicraft/single/ppo_grpo_s{1,2}.
- **Seed runs incident (2026-11-04):** the network volume hit its 40 GB quota; policy_s1 / policy_s2 died writing
  checkpoints, the other jobs were paused (SIGSTOP) until the user chose to grow the volume to 60 GB (~6 h paused,
  ~$3.5 of pod time lost; balance $5.88). Resuming then exposed a bug: MessageGFT.load_state_dict left the restored
  replays on the GPU (torch.load map_location) while new entries are on CPU -> torch.cat device error. Fixed (both
  replays moved to CPU on load); policy runs resumed from their checkpoints (s1 at 5k, s2 at 10k).
- **Does feedback steer the reader? (single/gft_v3e reader, CPU, scratchpad scripts):** (1) On its own training
  replay, action NLL under its own message vs another episode's message (same request) vs feedback removed: 0.698 /
  0.713 / 0.740 (oldest round, seen 4x) ... 0.730 / 0.736 / 0.747 (newest, seen 1x): seen feedback steers a little
  (memorisation, grows with exposure); most of the effect is "some feedback present". (2) Same 96 worlds and noise,
  teacher sentences: quarry stone / make a pickaxe -> success 0 and pickaxe 0 under every sentence ("You still
  needed a pickaxe.", recipe hint, use_fail, verdicts, success claims); sentences naming planks raise plank-making
  (stone task 0.48 -> 0.68 with the recipe hint; pickaxe task 0.60 -> 0.70-0.73); make planks unaffected (success
  0.49-0.56). Reading: the reader reacts to item words, not instructions, and cannot be steered to a behaviour it
  never produces (pickaxe), so Stage II cannot select it.
- **Seed runs result (40k, single tasks, v3e settings, seeds 0/1/2; plot results/minicraft/seeds_reader.png via
  experiments/minicraft/plot_seeds.py):** val_worlds full GFT 0.371 / 0.281 / 0.312 (mean 0.32), h2 lagged copy
  0.289 / 0.238 / 0.234 (0.25), pi's own retries 0.172 / 0.188 / 0.105 (0.16); train 0.30 / 0.26 / 0.15 means;
  planks full GFT 0.79 / 0.00 / 0.63, h2 0.16 / 0.53 / 0.00, policy <= 0.11. All 3 goals mastered (tier 1) in every
  run. Reading: pi sampling its own retries is worst on every seed; a lagged copy recovers ~60% of the gap to full
  GFT; the reader adds a smaller, consistent extra (val means 0.32 vs 0.25; ranges barely touch). Cause of the extra
  unknown (replay mixture, kept-retry imitation, or text; h3 request-only reader is 1 seed: 0.254). Caveat: policy
  s1/s2 resumed from 5k checkpoints after the disk-full crash. GRPO seeds 1-2 started directly (the queue waiter
  never fired; cause not found).
- **GRPO seeds (100k, single tasks):** s1 train 0.352 / val 0.355; s2 at 90k train 0.316 / val 0.367 (planks being
  learned on s2). Seed 0: 0.273 / 0.281.
- **Writer (user idea, 2026-11-04: "train the reader to generate feedback before generating trajectories; use it to
  draw Stage II samples"; whole-message version chosen):** `MessageWriter` in gft_message.py (GRU decoder over the
  prompt vocabulary, conditioned on the request, the reader's encoding of the first frame (detached) and the verdict;
  trained each round on the last replay_rounds rounds of Stage I (request, first frame, success, teacher message)).
  `--msg-writer`: each Stage II attempt is played by the reader under a message the writer samples with verdict =
  success (temperature `--msg-writer-temperature`), instead of m*; scoring against m* and the rest unchanged.
  Logged: writer_loss, written_distinct (distinct messages per task / K), written_words, written examples. Tests in
  tests/test_gft_message.py. Known risk: the reader barely follows message content on new worlds (steering tests
  above), so written messages may not change the attempts. Job: pod/jobs_writer.sh (not launched).
- **Writer audit (independent reviewer + me):** no crash or semantic bug for the planned run. Fixed: writer vocabulary
  now has numbers 0-255 (was 0-31: step counts lost) and the four teacher words missing from vocab.WORDS (in, makes,
  put, which); messages up to 64 words (48 cut ~5%); requests up to 16 words; verdict read from the message, not the
  environment; writer refused with mode policy/lagged/stage1 or reader_text=False; one device sync per written word;
  `writer_warmup` (10 rounds): Stage II uses m* until the writer has trained (an untrained writer writes ~45 random
  words). Tests: 13 pass in tests/test_gft_message.py; full suite 1096 passed (before these fixes).
- **All-tasks variant launched (user: "not assume the knowledge of task success ... launch a variant with all
  tasks"):** `--msg-stage2-tasks all` (MessageGFTConfig.stage2_tasks): Stage II tasks drawn from all Stage I episodes,
  not only those the message says failed; same 512-retry budget. pod/jobs_batch5.sh RUNS=alltasks on pod
  `i1q72o0g57wjlo` (RTX PRO 4000, $0.57/h, watchdog terminate), 40k, seed 0, out outputs/minicraft/batch5/.
  Remaining uses of success knowledge: m* from the bank of messages the teacher marked success; the curriculum's
  mastery (environment success). writer / passes8 / passes16 written, not launched.
- **All-tasks result (40k, seed 0):** train 0.309, val_worlds 0.340, planks / rope / bricks 0.32 / 0.46 / 0.53, 3
  goals mastered -- inside full GFT's seed range (val 0.28-0.37): choosing Stage II tasks by the "not done" verdict
  is not needed. Discussion (user): reading the verdict from the message is legitimate learning from language; the
  curriculum (which reads environment success) is the teacher's side (task choice), shared with PPO.
- **New base (user, 2026-11-04): v3e + Stage II tasks from all Stage I episodes.** `stage2_tasks` now defaults to
  "all" (MessageGFTConfig and `--msg-stage2-tasks`); "failed" reproduces 17g / v3e. Base reference: batch5/alltasks
  (40k, seed 0: train 0.309, val 0.340). Next experiments build on it: pod/jobs_batch5.sh now runs writer / passes8 /
  passes16 on the base (not launched).
- **Pass-ratio runs launched (user "go"):** pod/jobs_ratio.sh on pod `ol58ecj7gw8pc6` (RTX PRO 4000, $0.57/h,
  watchdog terminate). Four pass counts per round (pi distillation, reader on Stage I replay, reader on kept
  attempts, scorer), pi held at 4; base (4,1,2,3) = batch5/alltasks val 0.340. R1 (4,1,0,3) no reader
  self-imitation; R2 (4,1,4,3); R3 (4,1,2,1) lighter scorer; R4 (4,4,4,4). All with --msg-shadow; new logs
  kept_nll_first / kept_nll_last (pi's NLL of kept actions on its first / last pass). New flags --msg-reader-passes,
  --msg-reader-kept-passes, --msg-scorer-passes, --msg-reader-window, --msg-passes. 40k, seed 0. Next: scale the best
  ratio x0.5 / x2. (An earlier pi-passes x reader-window grid was started by mistake and killed within a minute.)
- **Pass-ratio results (40k, seed 0, base recipe; (pi, reader Stage I, reader kept, scorer) passes):** base (4,1,2,3)
  train 0.309 / val 0.340; R1 (4,1,0,3) 0.289 / 0.336; R2 (4,1,4,3) see below; **R3 (4,1,2,1) 0.375 / 0.406**
  (planks / rope / bricks 0.53 / 0.77 / 0.65; best 40k GFT so far); R4 (4,4,4,4) ~2.5x slower, 0.27 / 0.25 at 20k.
  Reader = pi on the same Stage II tasks in every run (no lag gap); pi's NLL of kept actions falls only ~0.03-0.04
  per round and stays at its entropy. Reading: a lighter scorer (1 pass) helps (one seed, +0.07 val); the reader's
  self-imitation does not matter.
- **More ratios launched (user: "2121 and 4221"):** pod/jobs_ratio2.sh on the same pod: R5 (2,1,2,1) and R6
  (4,2,2,1), 40k, seed 0, --msg-shadow. R4 still running.
- **R4 (4,4,4,4) stopped at 35k** (val 0.26 vs R3 0.39 and base 0.34 at the same point; ~2.5x slower): equal passes
  for every learner is not the simple recipe.
- **R5 (2,1,2,1) final 40k:** train 0.281 / val 0.301 (R3 0.375 / 0.406): halving pi's passes hurts. R6 (4,2,2,1)
  at 25k: val 0.27 vs R3 0.32 at the same point.
- **Scale-up launched (user "ok scale up"):** pod/jobs_scale.sh S2 = R3 x2 = (8,2,4,2), 40k, seed 0, --msg-shadow,
  same pod. x4 (16,4,8,4) written, not launched (budget).
- **R6 (4,2,2,1) final 40k:** train 0.30 / val 0.32 (R3 0.38 / 0.41): more reader passes on Stage I hurt. R3
  (4,1,2,1) remains the best ratio; S2 (x2) running.
- **S2 = R3 x2 (8,2,4,2) final 40k:** train 0.199 / val 0.148 (R3 0.375 / 0.406): scaling every learner's passes up
  over-sharpens pi early (entropy 0.78 by round 40 vs 1.21 for R3) and fails. With R4 (4,4,4,4) also worse, R3's
  scale (4,1,2,1) looks about right. Ratio search, 40k val: R3 0.41 > R2 0.36 > base 0.34 = R1 0.34 > R6 0.32 >
  R5 0.30 > R4 0.26 (stopped 35k) > S2 0.15. All seed 0.
- **New defaults (user, 2026-11-04: "make r3 new default"):** MessageGFTConfig keep_frac 0.5, epochs_clean 4,
  disc_epochs 1 (reader 1 pass on Stage I, 2 on kept attempts, unchanged), stage2_tasks "all": a plain
  `--preset 17g` run is now R3 on the base (40k seed 0: train 0.375, val 0.406). 17g / v3e need explicit flags now:
  v3e = --msg-stage2-tasks failed --msg-scorer-passes 3; 17g additionally --msg-keep-frac 0 --msg-epochs-clean 2
  (fixed: train.py ignored 0 for these). Unconfirmed beyond seed 0.
- **R3 100k launched (user "let run r3 for 100k"):** pod/jobs_r3_100k.sh (all defaults = R3), pod `2fqxgscs7ld3b7`
  (RTX 4000 Ada, $0.28/h, watchdog terminate), seed 0, out outputs/minicraft/r3/r3_100k.json.
- **R3 100k (default recipe, seed 0):** train 0.359 / val 0.379, 6 goals (all of tier 2; tier 3 still 0). Compare
  at 100k: GFT v3e 0.297 / 0.348 (6 goals), GRPO s0 / s1 / s2 val 0.281 / 0.355 / 0.336 (5 / 6 / 3 goals). Best
  100k run so far, by a margin within noise of GRPO s1. Plot results/minicraft/grpo_seeds.png.
- **Pace teacher (user: "changing the teacher is the right fix"), `--msg-pace`, not launched:** teacher.Pace keeps
  each request's last 64 success lengths; a success is "That was quicker than usual." / "You were faster than usual
  this time." when >= 75% of them took strictly longer, "That took longer than usual." / "You were slower than usual
  this time." when >= 75% took strictly fewer steps (ties neither). Scorer keeps these sentences; a slow success's
  improved message says quick (improve.py), a quick success gets the slow-swapped message as the scorer's contrast
  (pace_swapped); Stage II m* prefers quick success messages from the bank (MessageGFT._bank). Six words appended to
  vocab.WORDS. Audit (independent reviewer): no bug on the pace path; fixed tie handling and the one-sided contrast.
  Caveats: (1) the vocab growth changes the random init, so pace-off runs are no longer bit-identical to older
  ones -- compare against a fresh R3 run; (2) resuming a .ckpt made before the vocab change crashes (optimizer
  state rows) -- use fresh output paths; (3) pace is per request, not per world (easy layouts look "quick").
- **Temperature check (final policies, 64 fresh worlds per goal, CPU, check_temperature.py):** lower temperature is
  worse for every policy (v3e all-goal success 0.44 at T=1 -> 0.00 greedy; GRPO s0 0.32 -> 0.00; s1 0.49 -> 0.02;
  greedy runs to the 200-step limit): the policies rely on sampling noise to avoid loops; the entropy floor is not
  what caps success.
- **Pace runs launched (user "go"):** pod/jobs_pace.sh on pod `tmfxnsqabn57o8` (RTX 4000 Ada, $0.28/h, watchdog
  terminate): r3 (all defaults, current code) and pace (r3 + --msg-pace), 40k, seed 0, outputs/minicraft/pace/.
- **Pace run stopped at ~20k (failed):** val 0.06 / 0.08 / 0.11 / 0.14 at 5-20k (fresh r3 0.25 / 0.23 at 5-10k);
  kept-retry success 0.32 at rounds 10-20 (r3 0.87); pi entropy 0.68 by round 60 (r3 1.17 at round 40); rounds run
  ~3x faster (reader / distillation time per round 1.7 s vs 6.5 s) = far fewer steps per episode. Reading: the
  quick target made the scorer rank short episodes high -- including short failures (the agent quitting early) --
  so pi learned to stop early and narrowed. Likely cause: D is the MEAN of sentence scores, so a short failure that
  fits "quicker than usual" well can outscore a long success that fits "You did X" well; the pace remark should only
  count among successes.
- **Fresh r3 baseline stopped (user: "we already have baseline")** at ~15k. Gated pace fix implemented (Stage II
  keeps an attempt only if s without the pace sentences > 0; pace orders the qualified ones; MessageGFT._s /
  _is_pace; test test_pace_ranks_only_attempts_that_qualify_without_the_pace_sentence), not launched.
- **Gated pace run launched (user "go"):** pod/jobs_pace_gated.sh on pod `bjp9kk0ira94xc` (RTX PRO 4000, $0.57/h,
  watchdog terminate), R3 + --msg-pace (gated), 40k, seed 0, out outputs/minicraft/pace/pace_gated.json.
- **Gated pace run stopped at 5k (failed the same way):** val 0.07, eval mean episode length 116 -> 37 steps (R3
  runs 106-133 at 5k); a kept Stage II example was a failure on "dig clay" with s 2.13. The gate (qualify by s without
  pace) does not help early: the scorer is weak in the first rounds (R3's kept success ~0.5 there), so about half the
  qualified attempts are failures, and ordering them by pace picks the shortest -- mostly agents quitting early.
- **Pace shelved (2026-11-06, user "let's always switching off this pacing"):** six single-seed pace variants (pace
  alone, + agent "took longer" on failures, + ideal scorer pairs, pace from round 1, success-rate gate, scorer-only
  warm-up) all collapsed to short early-DONE episodes, val ~0.15-0.22 vs R3 0.38 at 100k. A scorer-only warm-up showed
  the scorer learns length from the pace contrasts themselves, not from policy drift.
- **Larger policy, 100k feedbacks, seed 0 (--arch d=512,ch=64,trunk=4, 2.9M vs 1.05M params; val success):** GFT
  2.9M 0.33 (planks 0.00 all run) vs GFT 1.05M 0.38; GRPO 2.9M 0.36 (planks mastered at 62k) vs GRPO 1.05M 0.28.
  Plots results/minicraft/big.png, big_tasks.png. Single seeds: no gap is solid.
- **GFT merged into one implementation (2026-11-06, user "merge gft into a single best implementation and
  setting"):** `gft/minicraft/gft_message.py` (`GFT`, `GFTConfig`; was MessageGFT) is the only GFT: R3 passes
  (4, 1, 2, 1), Stage II tasks from all Stage I episodes, keep_frac 0.5, plus **symmetric scorer pairs** (a success's
  message group also gets a received failure message of its request as a negative; failures keep a success message as
  before). The symmetric pairs were only ever run together with pace -- **R3 + symmetric pairs without pace is
  untested**. Removed: classic learn.GFT, gftppo, fbrl, the reader arm (reader_gft.py, infer_ideal.py,
  synth_feedback.py), MessageGFT's modes (policy / lagged / stage1), ideal target, writer, shadow, failed-only tasks,
  request-only reader, all pace code (teacher.Pace, pace templates, improve.pace_swapped, took_longer), and ~2000 lines
  of now-unused learn.py helpers. train.py: `--arm gft|grpo|ppo`; the 17g run setting is the default (graph curriculum
  0.7 + confirm + review 0.5, success needs DONE, exact vocab, pixels, explore base, well/sling excluded, budget 100k
  feedbacks with evaluation on it, validation splits); `--preset`, `--recipe`, `--msg-*` and the classic-GFT flags
  are gone; `--arm grpo` = PPO with group advantage, gamma 1, entropy target 0.8, frac reward. Pod jobs: one
  `pod/jobs_main.sh` (both arms). Scripts that needed removed code were deleted (several were already broken by the
  game change). The vocab keeps the pace words (old checkpoints still load). Pre-merge source:
  outputs/backup/pre_gft_merge_2026-11-06.tar.gz.
- **Merged GFT 100k (R3 + symmetric scorer pairs, seed 0, 2026-11-06):** pod `6ks5pcj11smsyo` (RTX 5090, 27 min),
  out outputs/pod/gft/main/gft_merged.json, plot results/minicraft/merged.png (experiments/minicraft/plot_merged.py).
  Final train 0.26 / val 0.30 (R3 0.36 / 0.38; GRPO 0.28), 5 goals mastered (R3 6). Tracked R3 to ~45k, then fell
  behind (val 0.30-0.35 vs 0.36-0.43); training success plateaued at ~0.5 vs R3's ~0.7. Single seed: the symmetric
  pairs do not help and may hurt. Watchdog now supports `ACTION=terminate` (podTerminate).
- **Why merged GFT never learned planks (2026-11-07):** its only planks success for 320 rounds (round 41) got the
  message "That worked. Done: make planks. First, you got fiber from the reeds. Next, you got 4 wood from a tree.
  After that, you got 4 clay from a clay pit. You took 139 steps." -- planks only in the verdict, no plank-making
  step: of its 5 scored sentences, 3 narrate gathering (fiber and clay irrelevant). "Done: make planks" is grounded
  by planks successes only (none), "you made N planks at a workbench" also by partial progress. A received success
  message overrides the improved message as m*, so every planks Stage II task from round 41 to 364 steered toward
  gathering fiber/wood/clay (logged Stage II targets at rounds 78-351 are all this message). R3's first planks
  success message said "your wood went into a plank. Then you made 2 planks at a workbench", and planks took off
  around rounds 120-240. The final merged policy makes one plank in 31% of planks episodes but rarely a second
  (8%) and rarely says DONE (success 0.8%). So the merged-vs-R3 gap is largely which success wording came first,
  not the symmetric pairs. Design issue: one uninformative success message locks m* while the bank is not refreshed.
- **Built success message + two-part target (2026-11-07, user-approved design; not yet run on GPU):** the built
  success message of a request = praise + a "done" verdict per goal, in wordings received before (for any goal); no
  recipe knowledge. Scorer: a failure's message group gets it as one more negative (with its improved message and a
  received success message). Stage II: each task has two targets, the source's improved message and the ideal set
  (built message + last ideal_k received successes, unstripped);
  s = 1/2 [D(x', imp) - D(x_src, imp)] + 1/2 [D(x', ideal) - D(x_src, ideal)] (a missing target drops its half);
  the reader's prompt is the improved message w.p. 1/2, else a member of the ideal set. Replaces "received success
  if any, else improved", which let one narration-only success lock planks' target. Tests 58 pass.
- **Built-message GFT 100k (seed 0, 2026-11-07):** pod `wh70bzck3m8kg9` (RTX 5090, 28 min), out
  outputs/pod/gft/main/gft_built.json, plot results/minicraft/merged.png. Final train 0.34 / val 0.375 (R3 0.36 / 0.379,
  merged 0.26 / 0.30). Planks learned earlier than R3: first Stage I successes round 37, steady from ~66; planks val
  0.35 at 50k (R3 0.27), 0.65 at 90k, 0.54 at 100k (R3 0.54). Only 3 goals mastered (gathering): rope/bricks/planks
  at 0.6-0.7 on own episodes but below the 0.7 mastery bar (R3 mastered 6). Single seed.
- **Reader writes its own messages (2026-11-07, user-approved design; implemented, not yet run on GPU):**
  `MessageWriter` (gft_message.py), the reader's message head p(f | request, first frame): writes sentence by sentence,
  each chosen among the sentences the teacher has said (TextTable), GRU over written sentences, at most 8.
  Stage I: it learns to write each episode's teacher message (claim sentences) on the reader's replay. Stage II: a
  `gen_frac` (0.5) share of each task's attempts is prompted with a written message (one per attempt), the rest with
  m* as before; keep by s as before. The reader imitates every kept attempt under its prompt, and the message head
  imitates every kept attempt's prompt (written or m*; no "describes" check -- user 2026-11-07: an audit found the
  top-half check passed 96% of kept attempts anyway). Collapse guard: the written share is fixed. `gen_frac 0` = the built-message version. Logs: gen_kept /
  tgt_kept, gen_success / tgt_success, writer_loss, gen_examples. Tests 62 pass.
- **Warm-up (user 2026-11-07):** `GFTConfig.warmup_rounds` (default 10, ~2.6k feedbacks): the first rounds are Stage I
  only -- the reader's heads and the scorer train on pi's episodes and the teacher's messages; no Stage II, pi not
  updated; the rounds count toward the feedback budget. Test added (62 pass).
- **Language ablation (verdict-only teacher, built-message GFT, 100k, seed 0, 2026-11-07):** L4 pod
  `9gtoshmue8pa44`, outputs/pod/gft/main/gft_verdict.json, plot results/minicraft/merged.png. Final val 0.297 (mean
  of last 4 evals 0.312) vs full language 0.375 (0.380); flat ~0.2 from 5k to 55k, then rose with rope/bricks (mastered
  79k / 85k; 5 goals vs full language's 3). Planks never learned: val 0 throughout, stage I planks 0 in the last 40
  rounds (full language: val 0.35 at 50k, 0.54 at 100k). Single seed each: the language content is what got planks.
- **Writer GFT 100k (reader writes its own messages, warm-up 10, seed 0, 2026-11-07):** L4 pod `9gtoshmue8pa44`
  (2 h 5 min; pod terminated), outputs/pod/gft/main/gft_writer.json, plot results/minicraft/merged.png. Final val
  0.371, mean of the 50-100k evals 0.380 (built 0.358, R3 0.378, GRPO s0/s1/s2 0.275/0.368/0.348); peak 0.414 at 70k;
  5 goals mastered (rope 75k, bricks 82k; planks 0.62-0.65 on own episodes, under the 0.7 bar); planks val 0.54 at
  100k. Planks took off earliest of all GFT runs (5-round mean >= 0.1 at round 115; built 151, R3 168). Writer:
  message loss 4.55 -> 2.38; written messages with a claim combination the teacher never used 48% (rounds 10-59) ->
  12% (late); wrong-goal verdicts 30% -> 1%. Written prompts kept slightly less often than target prompts.
- **Three-kind prompt split (2026-11-07, user; stopped):** each task's 16 Stage II attempts split evenly over
  improved / ideal (member drawn per attempt) / written prompts. Seed 0 (RTX PRO 4500, stopped at ~77k by the user):
  val 0.164 at 20k, flat 0.24-0.26 from 25k to 50k, 0.312 at 70k, 0.328 at 75k (writer run 0.414 / 0.406). Matched the writer run to
  round 59, then converted the first rope / bricks successes ~3x slower (own-episode success rounds 80-119: rope
  0.11 vs 0.38, bricks 0.12 vs 0.28) and found planks only at round 118 (writer 44). Seed-1 runs of both splits were
  started and stopped at once (user). Removed: the writer run's split (half written, half one m* per task) is the
  only one again (user: keep the best version). Tests 63 pass.
- **Current best GFT (2026-11-07) = the writer run's setting, all defaults:** R3 passes, symmetric scorer pairs,
  built success message, s = 1/2 improved + 1/2 ideal set, reader message head with half of each task's attempts
  under written messages and half under one m*, 10 warm-up rounds. 100k seed 0: mean val over 50-100k 0.380 (R3
  0.378, built 0.358, GRPO seeds 0.275 / 0.368 / 0.348), peak 0.414; planks took off earliest. Single seed.
