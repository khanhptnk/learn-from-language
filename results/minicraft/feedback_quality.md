# MiniCraft teacher feedback quality

1000 episodes (250 each from expert, noisy experts at 0.7 / 0.4, random); success rate 0.61. Held-out halves for all classifier numbers.

| Profile | truthfulness errors | AUROC success | AUROC per requirement | success − failure score | failures scored ≥ median success | G* calib. AUROC (expert-only) | G* calib. AUROC (bank) | request-word share | words/msg | distinct |
|---|---|---|---|---|---|---|---|---|---|---|
| reporter | 0 | 0.963 | 0.960 | +0.611 | 0.01 | 0.600 | 0.654 | 0.18 | 27.0 | 0.93 |
| diagnoser | 0 | 0.945 | 0.920 | +0.566 | 0.01 | 1.000 | 0.839 | 0.26 | 43.7 | 0.97 |
| coach | 0 | 0.990 | 0.961 | +0.873 | 0.02 | 0.580 | 0.530 | 0.33 | 10.3 | 0.56 |
| grader | 0 | 0.997 | 0.986 | +0.901 | 0.00 | 0.614 | 0.667 | 0.47 | 12.4 | 0.81 |
| mixed | 0 | 0.989 | 0.962 | +0.797 | 0.01 | 0.994 | 0.872 | 0.23 | 43.6 | 0.99 |

Examples (noisy0.4 episodes):

- request goals ['get across the river', 'get ready to mine'], satisfied [False, True]
  - *reporter*: Done: get ready to mine. You picked up 1 stone. You made 1 iron. Not done: get across the river. You crafted 1 stone pickaxe. You crafted 3 planks. You made 2 sticks. You collected 1 ore. You made 1 wood pickaxe.
  - *diagnoser*: You picked up 1 ore. You made 2 sticks. Done: get ready to mine. You tried to make a stick without a plank. You crafted 1 wood pickaxe. You made 3 planks. You collected 1 stone. You were not standing by a workbench when crafting. Crafting failed 12 times because no furnace was next to you. You tried to make a wood pickaxe without a stick. You tried to make a stone pickaxe without a stick. You made 1 stone pickaxe.
  - *coach*: You still needed a plank. A plank is made from wood at a workbench.
  - *grader*: You did get ready to mine. 1 of 2 done. You did not get across the river.
  - *mixed*: Not quite there. You picked up 1 ore. Crafting failed 38 times because no workbench was next to you. You collected 5 wood. You picked up 1 stone. You tried to make a stick without a plank. You took 200 steps. You made 1 iron. You made 1 stone pickaxe. You were not standing by a furnace when crafting.
- request goals ['upgrade your tools', 'make some planks'], satisfied [True, True]
  - *reporter*: You crafted 6 planks. You made 3 sticks. You did upgrade your tools. That run lasted 170 steps. You made 1 stone pickaxe. You bumped into things 14 times. You crafted 1 wood pickaxe.
  - *diagnoser*: You crafted 1 stone pickaxe. You did upgrade your tools. Crafting failed 4 times because no furnace was next to you. You made 1 wood pickaxe. You tried to make a bridge without iron. You tried to make a stick without a plank. You collected 6 wood. You were not standing by a workbench when crafting. You tried to work the ore without a stone pickaxe.
  - *coach*: Done: upgrade your tools.
  - *grader*: Great job! Done: make some planks. Done: upgrade your tools.
  - *mixed*: Done: upgrade your tools. Crafting failed 4 times because no furnace was next to you. You used empty ground 7 times. 2 of 2 done. You crafted 1 stone pickaxe. You made 3 sticks. You picked up 6 wood. You collected 1 stone. You made 1 wood pickaxe. You made 6 planks. You did make some planks.

(14 s)
