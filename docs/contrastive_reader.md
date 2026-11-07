# Contrastive Stage I: learning the reader from positives and negatives (2026-10-24)

## The problem

GFT's gradient is a policy gradient with weight w(x) = sum_f F(f|x) G*(f) / m(f): an episode counts only if its
feedback f could also have been said about an optimal run (G*(f) > 0). With the realistic teacher a failed run never
receives such a message, so w(x) = 0 for every failure; where pi never succeeds, the exact gradient is zero. Any
progress past that frontier must come from the reader generalizing to the ideal message f* it never saw as a
positive. Stage I trains the reader by MLE on (x, f(x)) pairs only: it never sees a mismatched pair, so it learns
nothing about which x does NOT go with a given f, nor anything about f*. A baseline cannot help (all failures get the
same weight).

## The objective: NCE for the posterior ratio

The reader's job is the ratio rho(x, f) = F(f|x) / m(f) (the posterior is r(x|f) = p(x) rho(x, f)). Density ratios
are what noise-contrastive estimation learns: with positives (x, f) ~ joint and negatives (x, f~) with f~ from a
proposal q, the softmax-NCE (ranking) loss

    L = - log [ e^{s(x,f)} / ( e^{s(x,f)} + sum_{f~} (m(f~)/q(f~)) e^{s(x,f~)} ) ]

has optimum s = log rho + const (Ma & Collins 2018; Oord et al. 2018). Same optimum as the Stage I MLE; a different
estimator, one that uses negatives. GFT's objective J and Stage II (sample f* ~ G*, roll out the reader on f*, distill
into pi) are unchanged; Prop. 9.2 bounds the Stage II gradient bias by the reader's distance from the true posterior,
so a better estimate of the posterior is the whole gain.

The reader's score is trajectory-level, built from the residual/FiLM reader:
s(x, f) = sum_t [ log pi^fb(a_t | s_t, f) - log pi(a_t | s_t) ]  (the total adjustment along x; with the FiLM
reader, log pi(a_t|s_t) is the same network prompted without feedback). Steps where the feedback does not matter
cancel between candidates, so the negative signal lands on the decision steps.

## Candidate set per episode (the hard negatives are given, not mined)

The teacher is a generative channel over atomic claims, and the ideal sampler is already assumed available (it is the
same supervision GFT's Stage II uses). For episode x with feedback f and task D:

1. **the ideal message f* of D** (always): the negative that matters, and the hardest one at the frontier (same
   request, most facts shared);
2. **one-fact-flipped feedback**: f with one failure claim turned into its success counterpart ("not done: get some
   metal" -> "done: get some metal", "still needed ore" dropped); true of a trajectory one fact better than x;
3. **feedback of other episodes of the same request** whose achieved-fact set differs (medium);
4. **random in-batch messages** (easy).

False negatives: a candidate whose claim set is a subset of what x actually achieved is not a negative (the teacher
could have said it about x); such candidates are excluded. Proposal weights: candidates 1-2 are sampled by a fixed
rule, so their importance weight m(f~)/q(f~) is unknown; we treat them as a hard-negative curriculum (fixed count,
temperature tau on the scores) rather than an unbiased estimate -- the standard practice (Robinson et al. 2021).

## What is standard and what is the bet

Standard: NCE / InfoNCE for conditional ratios; contrastive goal-conditioned RL (Eysenbach et al. 2020, 2022);
hard-negative mining with reweighting; discriminator-guided instruction following. The bet: that a reader trained
with negatives extrapolates to f* in a useful direction at the frontier. Only an experiment answers this:
(1) reader benchmark: held-out retrieval vs the MLE-trained FiLM reader; (2) steering: pi^fb under f* vs pi on
frontier goals.
