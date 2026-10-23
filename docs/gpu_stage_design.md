# GPU-stage design: environments, task, feedback, protocol

Status: proposed (2026-10-21), not yet implemented.

**Scope update (2026-10-21, user decision): no LLM agents.** Policies are small neural networks trained by us;
an LLM may serve as the *teacher* (feedback writer). Consequences: the Crafter stage uses **Craftax** (JAX,
GPU-vectorized, symbolic observations) instead of text Crafter; LLM-agent rows below (Qwen policy, verl, A100)
are superseded. LLM teacher: local open model on the same GPU or Claude API, called once per unique episode
summary (cached). Method under test: Batch GFT (Algorithm 2 of
`GFT_revised_2026-10-20.pdf`), implemented incrementally (a few SGD steps per fit).

## Requirements for the environment

| ID | Requirement | Why |
|---|---|---|
| R1 | Deterministic given the request (seed can be part of the request) | GFT theory assumes deterministic dynamics |
| R2 | Compositional tasks with checkable requirements; several correct solutions | partial-audit structure of the paper; ground truth for feedback and RL baselines |
| R3 | Long horizon (100–1,000 low-level steps) with dependencies between subgoals | credit assignment is where GFT beat RL in the tabular stage |
| R4 | Low base competence; mechanics the starting policy does not know | GFT's advantage regime (low q in L1) |
| R5 | Privileged state for a simulated teacher (what happened, why it failed, how to fix it) | the full range of feedback intents |
| R6 | Watchable replays | real human feedback later |
| R7 | Symbolic observations for small networks and text for LLM agents | L2 (small neural) and L3/L4 (LLM agent) on the same tasks |
| R8 | Fast simulation, pip-installable | iteration speed, low engineering risk |
| R9 | Known RL baselines and literature | credibility |

## Candidates (measured 2026-10-21 on this machine, 1 CPU core unless noted)

| | BabyAI (minigrid 3.2) | Crafter | Craftax (JAX) | MiniHack / NetHack | ALFWorld / TextWorld |
|---|---|---|---|---|---|
| Deterministic given seed | yes (checked) | yes (checked) | yes | yes | yes |
| Compositional, checkable | instruction trees (after/and over go-to, pick-up, open, put-next) | 22 achievements + inventory | same as Crafter | partial | yes |
| Horizon | expert bot: mean 74, max 210 steps (BossLevel); max_steps 2,880 | crafting chains of 100s–1,000s of steps; episode length 10,000 | same | long | ~30–50 |
| Base competence | moderate | low: random play rarely passes "place table" (wood 35%, table 10% of episodes) | low | very low | low–moderate |
| Feedback richness | objects, doors, order | inventory, crafting prerequisites, survival, threats | same | arcane | text state |
| Speed | 3–5k steps/s | 155 steps/s (renders pixels every step) | built for GPU; ~340 steps/s on CPU with 1,024 envs | heavy build | slow |
| Expert | BabyAIBot solves 20/20 | none | none | none | handcoded |
| Privileged info | instruction objects with verifiers | `info`: inventory, achievements, 64x64 semantic map, player position | full state | yes | text |

## Decision: two environments, one shared feedback framework

1. **BabyAI-Req** -- development and the small-neural stage (L2). BossLevel / SynthSeq instructions recast as
   requirement sets (leaf instructions + ordering constraints). Fast, deterministic, expert bot gives exact optimal
   executions (calibration checks) and base policies with controlled competence q (noisy-bot BC, as the q knob in
   L1). Native hindsight-instruction-relabeling baseline.
2. **Crafter-Goals** -- headline for LLM agents and human feedback (L3/L4). Our task layer on Crafter:
   requests are compositional goals over the tech tree (e.g. "make a stone pickaxe and place a furnace, and drink
   water before the end"), m = 2–5 requirements, 3 tiers by tech-tree depth, step budget 300–1,000; requirements
   checked from achievements, inventory and an event log. Unfamiliar mechanics for small LLMs, genuinely long
   horizon, naturally descriptive failures. Simulator speed is irrelevant here (LLM inference dominates).

Rejected: MiniHack/NetHack (arcane for human feedback, heavy build), ALFWorld (text-only, short), Craftax as main
environment (no text interface; keep as an option for fast RL-from-scratch baselines).

## Shared feedback framework

- Teacher reads (request, event log, final state) and emits messages mixing the L1 intents: descriptive,
  diagnostic, process, corrective, hint, evaluative, vague, chatter; mention bias, partial audits, teacher profiles.
- Language: template grammar → LLM paraphrase checked for faithfulness → real human feedback via a small web page
  showing the request and a replay.
- Ideal feedback G*: (i) same teacher on expert/successful executions (calibrated); (ii) request-only verbalization
  of all requirements as satisfied (paper's partial-audit construction); (iii) for humans: human feedback on
  verified successes (empirically calibrated).

## Protocol

- Arms: GFT (Batch GFT) and one-round GFT; FCP (report its test-time feedback cost); SDPO; PPO / GRPO / RLOO with
  verifier reward and with reward parsed from the same feedback; filtered BC (RFT); hindsight instruction relabeling.
- Matched episodes and teacher calls; G* cost reported explicitly (paper, audit note on ideal-feedback access).
- Metrics: success (all requirements), fraction satisfied, episodes and teacher calls to X% success,
  generalization to held-out goal combinations and world seeds.

## Stages and compute

| Stage | What | Hardware |
|---|---|---|
| L2a | Build BabyAI-Req, teacher, small transformer, GFT and PPO; test tiny | CPU (now) |
| L2b | BabyAI-Req full runs, small net from a BC base with controlled q | 1 x 24 GB GPU, hours |
| L3 | Crafter-Goals on Craftax with small networks; template + LLM-teacher feedback; GFT vs GRPO/PPO | 1 x 24 GB GPU (3090/4090), ~$10–30 |
| L4 | Optional: human feedback on Crafter-Goals replays (collection page, human G*) | same + annotation time |

## Risks

- Crafter has no expert: scripted planner on the semantic map (tasks are tier-limited) or request-only G*.
- LLM rollout cost: sliding observation window + inventory summary, ~1–2k tokens per step.
- Determinism: seed is part of the request; Crafter mobs are seed-deterministic (checked).
