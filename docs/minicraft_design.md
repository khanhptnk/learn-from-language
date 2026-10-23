# MiniCraft: small crafting world for GFT experiments

Replaces BabyAI-Req and Crafter-Goals as the main GPU-stage environment (decision 2026-10-21). Purpose: show that
GFT learns when the **request is a high-level goal** and the **feedback talks about low-level events and mechanics
in a different vocabulary** -- feedback is evidence about what happened, not an instruction to execute. Design
lineage: "Craft" (Andreas et al. 2017, policy sketches) and "Mining" (Sohn et al. 2018, subtask graphs).

## World

- Grid 13x13 with a wall border, deterministic given the seed (the seed is part of the request). West of the
  river: 12 trees, 4 rocks, 3 ore, a workbench, a furnace (11x11 was too crowded: random placement needed ~10^4
  resamples per world; 13x13 needs ~1.5, 1.3 ms/world).
- A one-cell-wide river (water) runs north-south near the east side; the **gem** lies beyond it, never on the bank.
- West side: trees, rocks, iron ore, one workbench, one furnace, the agent's start.
- Agent has a position and a facing direction (last move direction). Moving into a non-passable cell only turns.

Cells: empty, wall, tree, rock, ore, water, bridge (passable), workbench, furnace, gem.

## Actions (12)

`up, right, down, left` (move/face), `use` (act on the faced cell), and seven crafts:
`plank, stick, wood_pickaxe, stone_pickaxe, smelt_iron, iron_pickaxe, bridge`.

## Mechanics (default recipe book)

| Use on | Needs | Gives |
|---|---|---|
| tree | -- | +1 wood (tree removed) |
| rock | wood pickaxe | +1 stone (rock removed) |
| ore | stone pickaxe | +1 ore (ore removed) |
| water | a bridge item | water becomes a passable bridge |
| gem | iron pickaxe | +1 gem |

| Craft | Inputs | Station |
|---|---|---|
| plank | 1 wood | workbench |
| stick | 1 plank | -- |
| wood pickaxe | 1 plank + 1 stick | workbench |
| stone pickaxe | 1 stone + 1 stick | workbench |
| smelt iron | 1 ore + 1 wood (fuel) | furnace |
| iron pickaxe | 1 iron + 1 stick | workbench |
| bridge | 2 plank + 1 iron | workbench |

Tools are not consumed. "Station" means the station is in one of the 4 cells next to the agent.
Optional (later): per-world recipe variants hidden in the seed (e.g. bridges need stone), so rules are unknown to
the agent and only learnable from feedback.

## Requests (what the agent sees) and hidden requirements (what is checked)

The agent sees only a short goal phrase; the requirements are never shown. Goals by depth:

| Goal phrase | Hidden requirement (end of episode) |
|---|---|
| gather firewood | wood >= 2 |
| make some planks | plank >= 2 |
| make a handle | stick >= 1 |
| get ready to mine | has wood pickaxe |
| collect building stone | stone >= 2 |
| upgrade your tools | has stone pickaxe |
| get some metal | iron >= 1 |
| make an iron tool | has iron pickaxe |
| get across the river | agent is east of the river |
| fetch the gem | gem >= 1 |

A request is 1–3 goals joined by "and"; requirements = one per goal. Episodes end when all requirements hold or
after a step budget.

## Feedback (teacher)

Facts come from a per-step event log (moves, blocked moves, collects, failed uses with reason, crafts, failed
crafts with reason, bridge placement), requirement outcomes, and process statistics. A teacher profile chooses
what to mention and with which intents -- descriptive, confirm, diagnostic, process, corrective, hint, evaluative,
vague, chatter -- and the template grammar (later an LLM) verbalizes it. Example for request "fetch the gem":

- "You chopped 5 trees and made 3 planks, but you never got a wooden pickaxe, so the rocks wouldn't budge."
- "Crafting failed twice -- there was no workbench next to you."

Ideal feedback G*: the same teacher on successful executions of the expert planner (calibrated).

## Expert

Greedy replanning: expand the recipe tree for the unmet requirements, then navigate (BFS) to the nearest needed
resource or station, face it, and use/craft. Solves every goal on every generated map (tested).

## Generalization tests

Train on a fixed pool of (world seed, request) pairs; evaluate on
1. **new worlds**: unseen seeds, same goals;
2. **new goal combinations**: some goal pairs/triples held out of training;
3. **new mechanics** (later): recipe variants hidden in the world seed, held out of training;
4. **bigger worlds** (later): larger grids, so the policy must be size-agnostic.

## Speed (measured 2026-10-21, 1 CPU core)

- Batched environment: ~570k steps/s at B = 1,024 (random actions).
- Expert: ~300 us/step (Python BFS); all 150 random 1–3-goal requests solved, mean 64 steps, max 172.
  Expert runs for ideal feedback are cached per (seed, request) in the training pool, so they are computed once.
