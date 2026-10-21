# Numerical verification of GFT_revised_2026-10-20.pdf

All checks use small enumerated instances; section and equation numbers refer to the revised paper. The same claims are asserted in `tests/test_theory_gft.py`.

## 1. Exact population results

| Result | Status | Evidence |
|---|---|---|
| Lemma 5.1: normalized multiplicative update | PASS | 300 random channels, max normalization error 7.8e-16 |
| Thm 5.2 (14): Φ_t − Φ_{t+1} ≥ KL(p_{t+1}‖p_t), V_t − V_{t+1} ≥ Φ_t | PASS | max violations 3.5e-16, 1.0e-15 |
| Thm 5.2 (15): Φ_T ≤ KL(p̄‖p0)/(T+1) | PASS | max violation 0.0e+00 |
| Thm 6.3 (20) with the best separating statistic (LP) | PASS | 200 audit instances, 0 violations; LP γ / (κ_min/m) median 2.00 (≥1 always) |
| Aliasing example has no separating statistic | PASS | LP optimum γ = 0 |
| Prop 6.4: monotone success under common optimal feedback | PASS | 0/200 truthful-audit instances (2–3 optimal executions) ever decrease |
| §6.3: success can fall without a common optimal law | PASS | p(C): 0.981 → 0.23974 (paper: 0.981 → 0.23974) |
| §10.1: conditioning directly on f_B | PASS | success 0.0567 (paper ≈ 0.0567) |
| Eq. 47: m_{r(·|f+)}(f+) = m(f+) + Var[F(f+|X)]/m(f+) | PASS | max error 2.2e-16 |
| Prop 3.1: invariance to relabeling feedback values | PASS | max difference 1.4e-16 |
| Prop C.1: damped operator, (75)–(76) | PASS | 60 instances × η ∈ {0.2, 0.5, 1}: 0 violations |
| Eq. 54: approximate calibration G̃ = F p̃ | PASS | 60 instances × ε ∈ {0.05, 0.2, 0.5}: 0 violations |

Exact intent-profile instances (m = 2 requirements, N = 3 outcomes, q = 0.02; from `experiments/l1/verify_intents_exact.py`):

| Teacher profile | LP γ | p_T(C) at T = 10 / 100 / 1000 | Thm 6.3 bound holds | success monotone |
|---|---|---|---|---|
| reporter | 0.500 | 0.998 / 1.000 / 1.000 | yes | yes |
| grader | 0.500 | 0.998 / 1.000 / 1.000 | yes | yes |
| explainer | 0.500 | 0.998 / 1.000 / 1.000 | yes | yes |
| coach | 0.500 | 0.998 / 1.000 / 1.000 | yes | yes |
| scorer | 1.000 | 1.000 / 1.000 / 1.000 | yes | yes |
| vague | 1.000 | 1.000 / 1.000 / 1.000 | yes | yes |
| mixed | 0.853 | 1.000 / 1.000 / 1.000 | yes | yes |
| desc0.1 | 0.500 | 0.998 / 1.000 / 1.000 | yes | yes |
| chatter only | -0.000 | 0.000 / 0.000 / 0.000 | yes (no separation) | yes |

## 2. FCP and SDPO (Sections 10–11)

| Result | Status | Evidence |
|---|---|---|
| Prop 10.1: exact-population online FCP = GFT operator | PASS | FCP implemented via its conditional table; max difference over 30 instances × 25 rounds 3.3e-16 |
| Eq. 51: one fixed desired message concentrates on argmax F(f+|x) | PASS | closed form exact (err 1.1e-16); suboptimal execution with higher F(f+|x) reaches 1.0000 |
| Prop 11.1: one step from u0 = 0.2 | PASS | SDPO u1 = 0.0801 (paper 0.0801), GFT p1 = 0.6258 (paper 0.6258); brute-force reverse-KL minimizer 0.0801 |
| Prop 11.1: SDPO → bad action for u0 < 1/2, GFT → good action | PASS | u0=0.05: SDPO→0, GFT→1.000; u0=0.2: SDPO→0, GFT→1.000; u0=0.45: SDPO→0, GFT→1.000; u0=0.55: SDPO→1, GFT→1.000; u0=0.8: SDPO→1, GFT→1.000 |

## 3. Gradient view (Section 9, Appendix F)

| Result | Status | Evidence |
|---|---|---|
| Prop 9.1: ∇J = (44) = (45) = ∇L_t(θ_t) | PASS | 50 instances vs central differences, max error 2.2e-10 |
| Prop 9.2: ‖g(θ,φ) − ∇J‖ ≤ 2Bδ | PASS | 50 perturbed posteriors: 0 violations |
| Appendix F: rejection sampling is exact, cost 1/m(f*) | PASS | TV(accepted, posterior) = 0.0063 (20k draws); mean trials 4.59 vs 1/m = 4.63 |

## 4. Finite-sample convergence

### Theorem 8.1 — finite-tabular Batch GFT, exact empirical fits on fresh batches

Instance: 7 executions, 9 messages, γ = 0.50, v = KL(p̄‖p0) = 3.00. Theorem constants b = 2.38e-13, β = 2.01e-15; its sufficient batch sizes at t = 0 are N_0 ≈ 2.7e+58, M_0 ≈ 7.9e+31 (δ = 0.05) — not runnable, as the paper notes. Practical schedules instead:

| Batch schedule (N_t = M_t) | seeds meeting (34) and (35) at every T ≤ 25 | seeds that lost all optimal mass | mean final p_T(C) |
|---|---|---|---|
| 20 (fixed) | 27/40 | 8/40 | 0.800 |
| 200 (fixed) | 40/40 | 0/40 | 1.000 |
| 50 (t+1)^2 | 37/40 | 0/40 | 1.000 |
| 500 (t+1)^2 | 40/40 | 0/40 | 1.000 |

### Theorem 8.2 — linear-softmax Batch GFT (the paper's two-action example)

| Batch schedule | θ_0 → θ_25 | projected-gradient residual at t = 0, 5, 25 | mean φ₁ (exact log 9 = 2.197) |
|---|---|---|---|
| 20 (fixed) | -2.00 → 2.00 | 0.377, 0.0229, 0.0509 | 1.718 |
| 200 (t+1)^2 | -2.00 → 5.02 | 0.377, 0.0107, 0.0003 | 2.173 |

### Theorem D.1 — Online GFT (Algorithm 3), β_t ∝ t^−0.6, α_t ∝ t^−0.9, single samples

Informative full-support channel; G* from an interior mixture, so J has an interior maximizer (A1 holds).

| iteration | posterior-tracking error E TV(r_φ, r_θ) | ‖∇J(θ_t)‖ |
|---|---|---|
| 0 | 0.3289 | 0.2331 |
| 20000 | 0.0119 | 0.0116 |
| 40000 | 0.0088 | 0.0094 |
| 60000 | 0.0074 | 0.0085 |
| 80000 | 0.0039 | 0.0079 |
| 100000 | 0.0046 | 0.0071 |
| 120000 | 0.0058 | 0.0070 |
| 140000 | 0.0049 | 0.0068 |
| 160000 | 0.0047 | 0.0072 |
| 180000 | 0.0051 | 0.0069 |
| 199999 | 0.0018 | 0.0064 |

§D.3 restricted actor p(success) = σ(θ²): at θ = 0, dJ/dθ = 0.0 with success 0.50 — a stationary point that is not optimal, so stationarity alone does not imply task success. **PASS**

## 5. Proof internals and assumptions

| Check | Status | Evidence |
|---|---|---|
| Thm 8.1 proof steps (56), (57), (59)–(62), batches fixed 200 | PASS | 349 sampled rounds; smallest slack per inequality: eq56 4.2e-02, step1 0.0e+00, eq57a 0.0e+00, eq57b 0.0e+00, eq59 4.2e-03, eq60 0.0e+00, eq61 0.0e+00, eq62a 0.0e+00, eq62b 0.0e+00 |
| Thm 8.1 proof steps (56), (57), (59)–(62), batches 500 (t+1)^2 | PASS | 500 sampled rounds; smallest slack per inequality: eq56 4.2e-02, step1 2.5e-06, eq57a 0.0e+00, eq57b 1.5e-05, eq59 3.9e-02, eq60 1.5e-05, eq61 1.5e-05, eq62a 1.6e-05, eq62b 1.5e-05 |
| Eq. 89 finite-time approximate-gradient bound | PASS | α=0.5, aux mix λ=0.0: 0.0312 ≤ 0.3141; α=0.5, aux mix λ=0.3: 0.0370 ≤ 0.5449; α=0.1, aux mix λ=0.1: 0.0252 ≤ 0.1184 (L = 0.292 from Hessians over visited iterates, B = √2; the bound is ~5–15× loose) |
| Thm 8.2 hypotheses and Eq. 70 on the paper's example, batches fixed 20 | PASS | B3: μ = 0.00247 > 0; B2: κ = 8.83; L_x = 6.00; η_t 75.2 → 54.8; Eq. 70 holds every round but only vacuously here — η_t is far from summable at these batch sizes (the theorem needs N_t ≍ (t+1)^8, M_t ≍ (t+1)^4) |
| Thm 8.2 hypotheses and Eq. 70 on the paper's example, batches 200 (t+1)^2 | PASS | B3: μ = 0.00247 > 0; B2: κ = 8.83; L_x = 6.00; η_t 27.8 → 6.3; Eq. 70 holds every round but only vacuously here — η_t is far from summable at these batch sizes (the theorem needs N_t ≍ (t+1)^8, M_t ≍ (t+1)^4) |
| Thm D.1 with A2 satisfied (projected probability-vector auxiliary on a floored simplex) | PASS | strong convexity μ ≥ min P(x,f) = 0.0042; tracking 0.336 → 0.0228, ‖∇J‖ 0.233 → 0.0238 over 200k single-sample steps |

