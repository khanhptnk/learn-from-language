# Policy / reader architecture for pixel MiniCraft (design, 2026-10-23)

One network family serves every method: the deployed policy pi (request -> actions; GFT, GFT+PPO, PPO), and GFT's
reader pi^fb, a copy of pi prompted with request + feedback. Comparisons stay fair because all methods share it.

## What is wrong with the current network (`Encoder`, `POLICY_ARCH = dict(d=256, rec=8, trunk=3, text_attn=True)`)

1. **Text cannot decide where to look.** On pixels the agent crop is replaced by 4 attention maps computed from image
   features alone (`self.look(f)`), so "you still needed ore" cannot make the network attend to ore.
2. **Text enters once, late, through one vector.** The 8 recurrent conv passes (the path-finding) never see the text;
   the whole prompt is reduced to one d-vector added at the trunk input.
3. **No coordinates.** The attended features carry no position, so "where is X relative to me" must be encoded
   implicitly in channels.
4. **Tokenizer:** crc32 hashing into 4096 buckets; a few real words collide (e.g. "had" / "9"). Minor.

Evidence: pi follows requests weakly (firewood 0.45: it turns the wood into planks); on the reader benchmark
(`reader_bench.py`, pi's own episodes, held-out tasks) the reader picks the right feedback for a trajectory in 0.31 of
cases (chance 0.25), and does not fit its training episodes either. The earlier architecture search used symbolic
observations with an agent-centred crop, which does the "where to look" job for free, and only tested requests.

## Design principles

- Text must act **early and spatially**: it should mark which cells matter before planning, not only bias the output.
- Keep the **planning prior** that worked: a shared conv applied K times (value-iteration style) over the cell grid.
- Recover the **egocentric view** from pixels: find the agent, read its surroundings at known relative positions.
- Read the text **again at decision time**, queried by the current state (what does the message say about now?).
- Every text pathway starts as an exact no-op (zero-init last layers), so the text-free base is unchanged at init.

## Candidate A: grounded VIN (recommended first)

```
pixels 78x78 --patch conv 6x6/6--> cells (13x13xC) + coord channels
text tokens --exact vocab embed + 2-layer transformer--> word states W (L x dt), summary w
1. grounding:  each cell queries W (cell->text cross-attention) -> per-cell text-relevance features R (13x13xC)
               + FiLM(w, inventory) on cell features                      # "the ore cells matter"
2. planning:   u = stem([cells, R]);  f <- relu(conv(f) + u)  K times     # relevance propagates into a field
3. locate:     agent map a = softmax(1x1 conv(cells)), p = E[coords]      # agent is visibly drawn; text-free
4. read:       7x7 egocentric crop of f around p (grid_sample) + global pool + inventory
5. decide:     state query attends W again (current text_attn) -> trunk MLP -> policy, value
```

Cost: per state ~169 x L cell-text attention plus K small convs; a few times the current network.

## Candidate B: fusion transformer (general, VLM-like)

Tokens: 169 cell tokens (patch embed + 2D position) + inventory token + text tokens + [ACT]; 4-6 pre-LN layers,
d=128, 4 heads, full joint attention; positions also relative to the located agent (step 3 above); 2 conv passes
before tokenizing for local structure. Policy/value from [ACT]. ~10-50x the current compute (fine on GPU: budgets are
in episodes, not FLOPs). Less planning bias than A; more flexible text-vision binding.

## Text encoder options (orthogonal to A/B)

- **T1 (default):** exact-vocabulary tokenizer (no hashing), 2-layer transformer from scratch.
- **T2:** frozen pretrained sentence encoder (e.g. MiniLM-L6, 22M) with a trainable adapter; token states cached per
  unique text (prompts repeat), so near-zero cost. Gives negation / paraphrase / unseen-claim generalization (the
  Stage II G* problem) for free, and is the natural bridge to an LLM teacher. It is extra pretraining for all methods
  equally -- needs your call.

## Validation before any GFT / PPO run (pixels, offline, chosen on validation splits only)

1. **Request-following:** BC on expert demos, success per goal on new worlds / new combos. Must clearly beat the
   current network (firewood, deep goals).
2. **Reader benchmark** (`reader_bench.py` on the collected 16k pi episodes; every candidate starts from its own
   text-free base): retrieval and specific gain on held-out tasks, plus a held-out claim-combination split.
3. **Steering margin:** roll out the trained reader under G* vs the request alone on curriculum tasks -- the quantity
   GFT actually needs.
4. **Diagnosis if 2 stays weak:** the claims given as structured input instead of text. If that works, the language
   side is the bottleneck (-> T2); if not, the conditioning is.
5. **Speed:** forward + backward throughput at rollout / training batch sizes.

Then retrain the text-free base with the winner and rerun GFT, GFT+PPO and PPO at their tuned entropy targets.

## Training-recipe items to test on the benchmark (not architecture, same validation)

- Sentence-level feedback dropout (drop whole claims) instead of word dropout: teaches each claim's effect on its
  own, which is what conditioning on an unseen combination (G*) needs.
- Text-pathway learning rate and initialization (zero-init gates delay gradients to the text encoder).
