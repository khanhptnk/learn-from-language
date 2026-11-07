Caveats first: every proposal's generalization verifier predicted a null on well/sling, and the common cause is the instance, not the operators (identifiability, below). Numbers come from the verdicts and the two primitive-test logs; nothing new was trained.

RANKING (paradigm criterion first)

1. Explicit-EM GFT with a verifier teacher (the critic's combination; recommended). Paradigm: the one operator every lens passed (reweight own samples by a learned feedback likelihood, imitate, replay; Lemma 4.1 in sample form), with the recipe parts of R-GFT and MB-GFT removed (no content-word list, skeleton, window max, OOV rule, regex G*) and two additions nobody had: a verifier interface for negatives and a pre-registered identifiability condition. Its honest prediction for MiniCraft as configured is a null.

2. R-GFT / MB-GFT (same operator; MB-GFT's world model is dispensable by its own admission). Recipe parts: CONTENT_WORDS slot grammar, the crafted template as skeleton, 16-step windows, zero-variance zeroing that names exactly the held-out items, infer_ideal as G*. Fatal: output factorization leaves the input joint intact (7-8 consumption patterns; rope never first after "from"); the station slot has a perfect HUD shortcut (stone drops iff site) that agrees with the tile on the fence proxy and disagrees on well and sling; (site, rope+brick, CRAFT) is censored (0 of 74 steps).

3. SN-GFT. Fixed product/argmax operator; hindsight density (1.34 positives per step) is real and transplants. Recipe parts: chain-rule slots are the template frame, claim_steps times, the inventory vector for input factors, infer_ideal.ideal with a transitive-closure subgoal, proposed teacher wording edits. Fatal: P(rope | "at a site from") has zero support, and F inherits survivor bias at the fence state (0 of 39 labels mention the site). Survives only as per-word reachability (twin B), an explorer, not an improvement step.

4. HNH. The most generally stated loop (HER x HAL under Stage I/II), but RELABEL reads claim tuples, claim_steps and the goal check, not text, and complete narration is the event log in English. Fatal: pi_hi's station slot is unidentifiable: item determines station in every training sentence, so copy-from-hint and lookup predict identically and lookup wins.

5. Checklist GFT. GFT operator untouched, honest paradigm map, but the scheduler ("latch all words, then try each action once") is the recipe schema with labels removed, valid only for one-action holes over co-held preconditions; G* is infer_ideal; the 10.2% came from an inventory-switched harness. Its station-swap control has now finished (below) and is the one positive empirical fact in hand.

6. HSQ. Relabel plus Bellman max is fixed, but the relabel is a deterministic function of the agent's own state (all 14,452 attempt cells reproduce from the replay arrays with no teacher), so the words add nothing; coverage is bought by a per-press narrator (4.8 extra sentences per message); the priority rule makes fence and sling seen under supersets, so only the well is zero-shot and the gates certify lookup.

7. Retry Distillation. The operator is learned (AD-style), with an inventory-change keyframe detector and a praise parse inside the "fixed" step; the last edit has no covered point (input set determines station in all 8 seen recipes); the desired slot carries the narrated answer in training and is empty at test.

RECOMMENDED PARADIGM, GENERAL

Loop. (1) Sample N trajectories per request from pi_t, per-step mixed with a frozen text-free base (p_0 coverage, as the paper assumes). (2) The teacher narrates each in hindsight; on request it states what a success would be described as (G*, declared and costed per the paper's item 7) and answers true/false/unknown for any (episode, sentence) pair (verifier). (3) Fixed operator: w_n proportional to F_theta(f*|x_n), self-normalized within the request (top-k is best-of-N); weighted BC of pi under the request alone, with replay. (4) Refit F_theta by MLE on the round's (x, f) pairs and on verifier-labelled cross pairs (false sentences are negatives). (5) Optional word-level HER explorer V(s|w) from span ends, declared as exploration, never as improvement.

Learned. pi as now. F_theta(f|x): a conditional language model of the whole feedback given frames and actions; every word scored (no content list, skeleton or window machinery), request dropped from its input, observation channels with zero training variance dropped.

Fixed. Reweight-and-imitate, which sorts and averages numbers and so never generalizes; the verifier relabel (true of x is a positive, false a negative), which gives HER's transition-by-goal product without a designer-chosen frame.

Coverage. Hindsight density (every narrated sentence about every trajectory) plus the verifier cross product. Pre-registered condition: F_theta can score a never-seen f* only if each content word of f* carries positive conditional information given the others on the training distribution. Where that fails the paradigm predicts its own null.

Closest relative. RAFT/ReST/best-of-N with a learned reward model, the reward model being a generative likelihood of the teacher's sentences; GFT's Lemma 4.1 with F learned; HER through the verifier. The novelty claim is recognition instead of generation in Stage I, not a new operator.

MINICRAFT INSTANCE

Feedback: the realistic teacher unchanged, plus verify.check as the verifier (it already reconstructs all 77,600 sentences by replay). Sentences consumed: "You made a wall at a site from a brick and stone." / "You pressed craft at a kiln holding only rope and bricks, so nothing happened." / "Not done: build a well. A well is made from rope and a brick at a site." G* for the well: the teacher's own success narration via plan_facts (declared oracle access), not infer_ideal.

Pretraining changes: pretrain.py only. Reader retrained as a whole-feedback conditional LM (all words scored), request removed from the reader prompt, zero-variance inventory columns dropped, station prediction from pixels only. Corpus, teacher, explorer, drop filter untouched; 31 min per base.

Post-training: 512 episodes per round on the curriculum including "build a well" / "make a sling"; frozen text-free base as step-mixed explorer (never pretrain.Explorer with post-training requests); w = softmax(log F_theta(f*|x)/tau) within request; weighted BC + pretraining replay + continued reader MLE on (x_n, f_n). Comparators from the same base and budget: PPO success; PPO reward=parse; PPO or RWR on r = log F_theta(f*|x) with the same G* (the only arm that separates the operator from the critic). Ablations: permuted content words in f*; the wall's f*; verdict-only f*; inventory-blind F. Words count iff full beats both reward arms at equal teacher calls and the permuted and other-composite arms collapse.

CHEAPEST FALSIFYING TEST

Step 0 (CPU, minutes, nothing trained): from core.RECIPES and the fence-held-out corpus compute the conditional mutual information between the station word and the station given the held inputs, and the count of "rope" in first input position. Expected: zero and zero. If so, no learned F can place the well from this corpus, and the ladder measures how it fails, not whether.

Step 1 (CPU, under an hour, checkpoint base_v9_grounded_pixels_h9-11-13): the splice test. Take the 74 replay steps (29 episodes) where the explorer stands by a site holding rope+brick, append CRAFT (test data only), score the window under the well sentence with all content words masked, request dropped, zero-variance columns zeroed; controls: wall-craft windows, the same episodes unspliced (near-miss), rope+brick crafts at kiln/workbench. Kill numbers: AUC(well vs wall) < 0.6 or AUC(well vs near-miss) < 0.6. Report p(site | station slot), p(rope | input slot), the inventory-blind variant, and precision@1 on pools of 1 positive plus 63 base-policy candidates (the operator's real use).

Ladder: reader-only retrain under the general objective (31 min GPU, ~$0.40), rerun Step 1; if it still fails and you accept the minimal identifiability change (one pair of seen recipes sharing an input set at two stations; wording and drop filter unchanged), re-pretrain both bases (~$0.80), rerun; only then the fence-proxy loop with the real environment as candidate source (4 rounds x 256, N=64) and the main-base arms (~$5, each launch approved).

PRIMITIVE-TEST EVIDENCE (fence-held-out checkpoint, n=256, scripted inventory stage switch)

Stage reach under one primitive sentence each: pickaxe 46%, stone 38%, plank 23%. Fence made: 10.2% (craft-held last sentence), 9.4% (made-form), vs 1.4% one-shot, 0.4% nothing; seen wall programme 7.4-8.2%; raft 72.7%. Last-stage craft at the site: 43-47% of arrivals under the fence sentence, 49-55% under the seen wall sentence; the control that finished this morning: 0% under a raft (workbench) sentence, and 9.8% of episodes (equal to the fence sentence) under a wall sentence. Reading: the executor follows the station word at the last step (site vs workbench, 0 vs ~45% of arrivals) and ignores the input words (fence and wall sentences indistinguishable). Any paradigm routed through "execute what a sentence describes" has a base for the station slot and none for the input slots; the bottleneck is the early primitives and the sequencing, which the harness did with an inventory switch no paradigm may use.

RESIDUAL RISKS

- The general statement may still be MiniCraft in disguise: the verifier is cheap only because the world is fully observed and the teacher is a template; with a human it is G*-level supervision per query.
- Full observability: verify.py reconstructs every sentence from the trajectory, so descriptive feedback carries nothing the agent cannot compute; only prescriptive content is informative, and the brief's rule reads that as shaping in words. The constraints plus full observability may be jointly unsatisfiable; the untaken angle is hidden state only the teacher can name.
- Censoring: the drop filter removes every (site, rope+brick, CRAFT) transition and leaves 74 steps with zero crafts, so every imitation-trained component is trained against the needed action. No fixed operator removes this.
- With the current seen recipes the identifiability condition is zero by construction; the likely outcome is that no paradigm meets the criterion on this instance and the honest result is a measured null plus the condition, not a win.
- Recognition-easier-than-generation is unproven; the HUD shortcut can let a retrained reader pass the fence proxy and fail the well.
- G* via plan_facts is oracle access; report and cost it. One seed and 1.4% base rates: n=512 with paired controls is the minimum.