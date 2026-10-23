# Experiment: does GFT learn new tasks faster as its language understanding grows?

Status: proposed (2026-10-22), to run after the current tuning sweep.

## Hypothesis

GFT's pi^fb learns a request-independent mapping from feedback words to behaviour ("you crossed the river" ->
crossing). That mapping accumulates across tasks, so for a *new* task GFT only needs the task's ideal feedback: if its
words are already grounded, Stage II produces good behaviour at once and pi distils it. PPO gets only a scalar per
episode and must find each new task's behaviour by exploration.

Predictions:
1. Introducing goals in blocks, GFT's cost to learn each new block (teacher feedbacks to a threshold) shrinks over
   successive blocks; PPO's stays roughly flat.
2. pi^fb's steering strength (one-sentence probe) and its Stage II success on a newly introduced block rise before
   pi's success does.
3. This holds only when a new block's ideal feedback uses already-grounded words (new combinations of familiar
   events, not new events).

## Setup

- Environment as now: MiniCraft with DONE, random base (v6), validation splits for tuning, best settings from the
  current sweep for each method.
- Blocks of 2 goals in depth order: {0,1}, {2,3}, {4,5}, {6,7}, {8,9}. Block k is introduced after block k-1's
  budget; tasks are drawn from the pool restricted to goals of introduced blocks, 70% containing a goal of the newest
  block (the rest replay earlier blocks).
- Budget per block: equal **teacher feedbacks** for both methods (primary axis), e.g. 10k (GFT: 20k episodes, as half
  its episodes are Stage II rollouts; PPO: 10k episodes). Also report episodes and wall time.
- 3 seeds per method.

## Metrics

- Per block: success on the block's validation tasks over its first 2k / 5k / 10k feedbacks; feedbacks to reach 50%
  of the best success either method reaches on that block.
- Trend across blocks: slope of feedbacks-to-threshold vs block index (GFT predicted negative, PPO ~0).
- Diagnostics per block: steering probe on the block's events, Stage II success on its tasks.
- Forgetting: success on earlier blocks while later blocks train.

## Fairness and caveats

- Same base, schedule, task pool and feedback budget for both methods.
- GFT needs ideal feedback for every new task (from the success bank of verified runs). That is an extra assumption
  -- someone must be able to describe success in low-level terms -- and is stated with the results.
- Blocks in depth order mix two effects (new words vs harder tasks); a second order (shuffled blocks) separates them
  if needed.

## Implementation needed

- Curriculum mode "blocks": fixed block order and per-block budget instead of the success-gated unlock.
- Per-block validation metrics (from the existing per-goal evaluation) and per-block diagnostics in the round log.

## Cost estimate

5 blocks x 10k feedbacks = 50k feedbacks per run (GFT 100k episodes, PPO 50k). About 1.5-2 h for 6 runs on one
RTX PRO 4000 with 4 in parallel, roughly $1.
