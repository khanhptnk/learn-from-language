# MiniCraft: small crafting world for GFT experiments

A deterministic grid world with four resource branches that combine into composite things at four stations. The
design goal (2026-10-28 redesign): every task is a combination of a few independent skills, and no task needs an
action of its own, so a held-out task is new only in which known skills it combines. The previous world was one
linear chain (wood → ... → gem), where any held-out task sat at the top and needed an action no demonstrator had taken.

## World

- 13 x 13 grid, walled border. Cells: empty, wall, tree, reeds, clay pit, rock, workbench, loom, kiln, site.
- Per world: the four stations, 8 trees, 6 reed patches, 5 clay pits, 4 rocks, and the agent, placed at random so
  that every object has a reachable neighbour cell and no cell touches two stations (a craft is then unambiguous
  about where it happens). Worlds are a deterministic function of (config, seed).
- Pixel observations: each cell is a 6 x 6 tile (`render.py`); the agent is a yellow square with a notch on the side
  it faces.

## Actions (10)

up, right, down, left (a blocked move only turns), use (the faced cell), select, put, take, craft, done (ends the
episode; success is checked then).

## Mechanics

- **Collecting:** use on a tree / reeds / clay pit gives wood / fiber / clay; use on a rock gives stone and needs the
  pickaxe. The cell becomes empty.
- **Crafting makes exactly what you put in (2026-11-02).** Each station has an input slot. select moves a cursor to
  the next item kind held; put moves one of the selected item into the adjacent station's slot; take returns the
  slot's contents; craft makes the recipe of that station whose inputs equal the slot's contents exactly (counts
  included), else nothing. No two recipes of a station share inputs (asserted), so craft is never ambiguous; several
  recipes may make the same item. Failures: no station adjacent, or the slot matches no recipe. Slots keep their
  contents when the agent walks away.
- **What the agent sees besides the map** (`MiniCraft.agent_state`, 42 numbers): inventory counts, the cursor
  (one-hot) and the adjacent station's slot. Per-step snapshots store it; its first 14 columns are the inventory.

| output | inputs | station |
|---|---|---|
| plank | wood | workbench |
| rope | fiber | loom |
| brick | clay | kiln |
| pickaxe | 2 planks | workbench |
| raft | plank + rope | workbench |
| sling | rope + stone | workbench |
| oven | plank + brick | kiln |
| well | rope + brick | site |
| wall | brick + stone | site |
| fence | plank + stone | site |

## Requests and hidden requirements

A request is 1-3 goals joined by "and", deepest first (since 2026-11-03 the experiments prompt the agent with a
single goal: `--max-goals` defaults to 1, and the held-out-combination split is dropped). Goals name the thing,
never the recipe:

| goal | checks | depth (expert ops) |
|---|---|---|
| gather wood / gather fiber / dig clay | 2 of the item | 2 |
| make planks | 2 planks | 4 |
| make rope / make bricks | 1 | 2 |
| make a pickaxe | 1 | 5 |
| quarry stone | 1 stone | 6 |
| build a raft / build an oven / build a well | 1 | 5 |
| make a sling / build a wall / build a fence | 1 | 9 |

Aliases (phase 2 of the alias experiments) are new phrases for the held-out goal pairs: "set up camp" (wall + wood),
"prepare the harbour" (raft + clay), "start the garden" (fence + rope), "open the kitchen" (oven + stone).

## Expert

Recipe expansion from the inventory plus the items in station slots, deepest goal first; a recipe's inputs are
gathered deepest branch first. Navigation by BFS to a cell adjacent to the target. At the recipe's station it takes
back a slot holding anything extra, then selects and puts the missing inputs one at a time and crafts when the slot
holds exactly the inputs; an input left in another station's slot is fetched (take) first. Solves 400 random requests
in 56 steps on average (max 152; 43 before slot crafting); with 20% random actions 98.8% succeed.

## Feedback (teacher)

`teacher.py`: event log → facts → claims → sentences from templates; `verify.py` parses sentences back and checks
them against a replay of the episode. The realistic profile: outcome first (every unmet goal named; on success praise
and what was done: the goals, the sub-goals reached on the way), then for up to two unmet goals one remark -- what was
used up, or one of the things still missing (usually the next, sometimes a later one, in one of several wordings, or
as a recipe / source hint) -- the deepest sub-goal reached, the most frequent mistake (crafting away from a station,
or at a station with nothing to make), occasionally the step count and chatter.

**Narration (2026-10-28).** Every message also tells 1-3 notable events of the episode the way a watcher would, with
place and inputs: "You made a raft at a workbench from a plank and rope.", "You picked up 3 clay at a clay pit.",
"You had rope but used it on a raft.", "You pressed craft at a loom with only wood in it, so nothing happened."
(`teacher.narrate`; sentence kinds collected / crafted with place variants, used_on, craft_held), in time order,
sometimes with sequence words ("First, ... Then ..."). Deep crafts are told more often. Each narration sentence
carries the time span of its event (`Facts.events`, `claim_steps`), so pretraining can pair a sentence with the steps
it describes. The wording rules are the same for every craft and do not depend on which tasks are held out. "Well
done!" was dropped from the praise set because "well" is a held-out item's name.

## Pretraining v3 (planned 2026-11-02): a diverse behaviour prior, no language

Supersedes v2 below. Principle: pretraining gives the agent diverse *primitive* behaviour and nothing else -- no
expert task plans, no requests, no text. All language is grounded in post-training, from the teacher's feedback on
the agent's own episodes: diverse behaviour -> diverse feedback -> grounding. Everything the agent knows about words
comes from the teacher it learns from.

Why (old curriculum, docs/STATUS.md 2026-11-02): the BC base imitated the expert on every goal, so post-training
mostly selected and retained demonstrated skills (it made a stone pickaxe in 1/3 of its first episodes; runs that
forgot this never mastered "upgrade your tools"); from an untrained policy (run 18) both GFT and PPO stopped at the
first goal -- pi never found a workbench, and the feedback ("away from a workbench") was the same few sentences.

- **Crafting: craft makes exactly what you put in** (implemented 2026-11-02, see Mechanics).
- **Data policy: random courses of primitives**, each a short scripted motor skill that knows no recipe or goal:
  go-to(cell kind) (BFS), use, select / put / take (a random subset of what is held), craft, wander(n), done.
  Courses come from a simple stochastic grammar (go somewhere, maybe use it, maybe go to a station, put some held
  things, craft) with light novelty biases (unvisited stations, untried (station, slot) pairs). Recipes succeed only
  by chance, cheap ones often and deep ones rarely. Episodes that make a held-out item are dropped (rate reported).
- **Pretraining objective:** unconditional BC on the courses (pixels -> action; no prompt), optionally
  self-supervised auxiliaries without text (inverse dynamics). Requests appear for the first time in post-training.
- **Post-training:** GFT and PPO from the same base. The teacher (verdicts, remarks, narration of notable events)
  is the only source of language; the reader and scorer ground words from (episode, message) pairs as now.
- **Measured:** coverage of (station, slot contents) pairs, chance success per recipe, distinct events per episode;
  in post-training, feedback calls until a sentence kind is grounded (masked-word / scorer accuracy per kind) and
  until a held-out item (well, sling) is made under a request.
- **Ablations:** BC on expert task plans (v2-style) vs the primitive explorer vs no pretraining; narration in
  pretraining as captions (v2) vs none.

## Pretraining v2 (superseded by v3): a grounded language of the world (`gft/minicraft/pretrain.py`)

Plan: `~/.claude/plans/this-is-a-new-glowing-candle.md`. Principle: pretraining teaches a grounded, compositional
language built from seen tasks only; the held-out tasks (well, sling) exist only as an absence.

- **Corpus** (`build_corpus`): requests of 1-3 *seen* goals; explorer mixture per episode -- 50% noisy expert
  (skill 0.7-1.0), 30% expert with wander segments (random walks that pick things up and press craft), 20% free
  explorer (pursues random seen goals, ignores the request; hindsight request = goals it reached). Held-out goals are
  never requested; episodes that make a held-out item are dropped (drop rate reported, ~2-5%); `audit` asserts no
  held-out word or inventory entry anywhere. Coverage stats: episodes and success per mode, items per episode,
  distinct craft situations, sentence kinds, distinct sentences.
- **Model** (`GroundedPolicy`): `FusionEncoder` (cell tokens from a patch conv + VIN-style recurrent conv, gated
  cross-attention to the word states, [ACT] readout; token-level text fusion instead of a pooled prompt vector),
  exact-vocabulary `PromptEncoder`, action and value heads, and a `TrajectoryReader` (bidirectional transformer over
  [words ; frame summaries ; action tokens], word head tied to the embeddings). Same calling convention as
  `CleanPolicy`, so GFT / PPO post-training is unchanged.
- **Objective** (`pretrain_grounded`), one encoder, per step: causal BC (request mode on episodes with a hindsight
  label, message mode, span mode = request + one narration sentence with the loss on that sentence's steps) +
  masked words given the frames (content words masked w.p. 0.5) + masked actions given the words and the frames,
  on whole episodes and on narration spans.
- **Diagnostics** on held-out episodes: CE per BC mode, message information gain, masked-word accuracy per word
  class (station / item / number / other) with vs. without the frames -- the gap is the grounding measure --
  masked-action accuracy.
- **Gates** (`experiments/minicraft/check_grounded.py`): data audit and coverage; request-mode success on seen
  requests vs a 0.9 expert; the grounding gap; composition probes: does "Great job! You made P at <station> from
  <inputs>." appended to an unrelated request steer the policy toward P (ops left, paired)? With a seen composite
  (e.g. the fence) additionally held out in a second pretraining, that probe is the test of the hypothesis itself:
  a sentence of grounded words about a thing never made or asked for. `train.py --base grounded`.

## Pretraining / held-out tasks (the language-pretraining experiments)

`learn.lang_corpus` / `pretrain_lang`: noisy-expert episodes with the teacher's message, trained in a request mode
(hindsight requests) and a message mode (request + message). Held-out goals (default: build a well, make a sling)
are requested in the corpus but never achieved: their words are seen in requests and feedback, their inputs and
stations are used by seen recipes, and only the combination is new. Gates: `experiments/minicraft/check_lang_pretrain.py`.
