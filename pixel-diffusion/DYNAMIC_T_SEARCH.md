# Discrete dynamic-T schedule search

Phase 0 (validation) + Phase 1 (4-D search), 2026-08-23 .. 2026-08-25.
Machine-readable data: `results/dyn_search_results.json`.

A schedule is 5 control points at training fractions [0, .25, .5, .75, 1] with
linear interpolation (`piecewise` t_schedule in `training/training_loop.py`).
The first is pinned at T=0, so a schedule is the 4-vector
**x = [T2, T3, T4, T5]**, monotone non-decreasing in [0,1].

All runs: `batch=64 seed=0 s_max=4.0 total_kimg=2000 arch=ddpmpp --cond=0`,
objective MIND on 5000 generated vs a 20000-image holdout (lower better).

## The number that governs how to read everything below

**Noise sd = 0.00089**, from 8 same-config replicate pairs. A single run can
deviate by **2.7 sd** — observed directly: the two `cosine_pw5` draws differ by
0.00258. **n=1 rankings among the top ~6 schedules are not meaningful.**

Demonstrated concretely: at n=1 `warmup_pw10` (0.029223) appeared to beat
`warmup40` (0.029764) by 0.00054. At n=2 each they are 0.029622 vs 0.029537 —
the order flips and the gap is 0.1 sd.

## Findings

### 1. Start at T=0 — dominant, ~5 sd
Regression across all four coordinates gives T2 coefficient **+0.00776 (t=2.6)**,
worth +0.00465 across the observed range. Confirmed by a pair differing in T2
alone: `static_T050` [0.50,0.50,0.50,0.50] = 0.034904 vs `plateau_low`
[0.30,0.50,0.50,0.50] = 0.033323. Observed delta 0.001581 against a predicted
0.001552 — 2% agreement on a comparison not used to fit the model.

### 2. Ramp near the middle — both extremes are worse than no curriculum
| regime | example | MIND |
|---|---|---|
| ramp early & hard | `sobol03` [0.46,0.89,0.93,0.97] | 0.038191 |
| ramp early & hard | `early_steep` [0.60,0.70,0.80,0.90] | 0.038194 |
| ramp far too late | `sobol06` [0.01,0.03,0.12,0.83] | 0.035239 |
| **no curriculum** | `static_T050` | 0.034904 |
| warmup-like | `warmup40` [0.00,0.16,0.55,0.95] | 0.029537 |

Within the T2~0 family, ordering by T4: 0.12 -> 0.035239, 0.15 -> 0.031826,
0.40 -> 0.031649, 0.55 -> 0.029764, 0.63 -> 0.029223. Monotone over ~6 sd.

`sobol03` and `early_steep` were chosen independently (Sobol vs hand-designed)
and land 3e-6 apart, and `sobol03` has a near-ideal ceiling (0.97) which buys it
nothing.

**A linear model reports T3/T4 as insignificant (t=0.3, 0.9). That is wrong** —
the response is U-shaped, which has near-zero linear slope by construction. Use
the GP, not a regression, for these two coordinates.

### 3. Ceiling matters but saturates by T5 ~= 0.8-0.95
T5 series at roughly matched T2, eight independently chosen schedules:

| T5 | MIND | run |
|---|---|---|
| 0.50 | 0.034655 | `ceiling_050` |
| 0.50 | 0.033323 | `plateau_low` |
| 0.61 | 0.033469 | `sobol10` |
| 0.70 | 0.033075 | `plateau_mid` |
| 0.78 | 0.031528 | `sobol11` |
| 0.95 | 0.031990 | `cosine_pw10` |
| 0.95 | 0.032056 | `twophase` |
| 1.00 | 0.032233 | `ceiling_100` |

**Pearson r = -0.814 (t = -3.44, n=8).** But the gain is spent by ~0.78-0.95:
0.78 -> 0.95 -> 1.00 reads 0.03153 -> 0.03202 -> 0.03223, i.e. flat to slightly
worse. `ceiling_100` vs `twophase` is a near-controlled pair (T2/T3 identical,
T4 within 0.03) and differs by only 0.2 sd, in the wrong direction.

**T5 = 1.00 is not better than 0.95.** A BO fitted before `ceiling_100` landed
proposed T5 = 1.00 for all four of its suggestions; that region is a dead end.

`ceiling_050` [0.15,0.30,0.45,0.50] = 0.034655 is statistically
indistinguishable from static T=0.50 — a well-shaped curriculum with too low a
ceiling is worth roughly nothing.

### Summary shape
**Start at 0, ramp through the middle, finish around 0.8-0.95.** That is the
warmup family, consistent with the earlier hand-crafted sweep. The top schedules
form a plateau near MIND ~ 0.0295 that this noise floor cannot separate.

## Phase 0 — the discretisation is validated
| run | MIND | reference (n=3) | delta |
|---|---|---|---|
| `static_T050` | 0.034904 | 0.035229 | -0.3 sd |
| `warmup_cont` | 0.030354 | 0.029622 | +0.7 sd |
| `warmup_pw5` (n=2) | 0.030319 | 0.029622 | +0.7 sd |
| `warmup_pw10` (n=2) | 0.029622 | 0.029622 | 0.0 sd |
| `cosine_pw5` (n=2) | 0.031539 | 0.032169 | -0.7 sd |
| `cosine_pw10` | 0.031990 | 0.032169 | -0.2 sd |

**Knot placement, not knot count, is what matters.** `warmup_linear(frac=0.25)`
has its only kink at p=0.25, which *is* a 5-point control point, so the 5-point
"discretisation" of warmup is that schedule exactly (max |dsigma| = 2e-15,
pinned by `tests/test_piecewise_schedule.py`). The 10-point grid sits at k/9 and
misses the kink, making it the *less* faithful of the two.

So the warmup arms measure seed noise, not discretisation. **Cosine is the only
real test** (max |dT| = 0.033 at 5 knots vs 0.007 at 10) and it passes at
-0.7 sd. Five control points are sufficient for schedules this smooth.

## All results

| run | [T2,T3,T4,T5] | MIND | n | sd | source |
|---|---|---|---|---|---|
| `p1_a_warmup40_0to095` | [0.00, 0.16, 0.55, 0.95] | 0.029537 | 2 | 0.000321 | file |
| `p0_warmup_pw10` | [0.03, 0.32, 0.63, 0.95] | 0.029622 | 2 | 0.000564 | file |
| `p0_warmup_pw5` | [0.00, 0.32, 0.63, 0.95] | 0.030319 | 2 | 0.001257 | file |
| `p0_warmup_cont` | (continuous) | 0.030354 | 1 | — | file |
| `p1_q_sobol09` | [0.03, 0.34, 0.42, 1.00] | 0.030618 | 2 | 0.000048 | transcribed |
| `p1_a_warmup15_0to095` | [0.11, 0.39, 0.67, 0.95] | 0.030875 | 2 | 0.000261 | file |
| `p1_s_late_hard` | [0.00, 0.00, 0.40, 0.95] | 0.031007 | 2 | 0.000908 | transcribed |
| `p1_q_sobol04` | [0.27, 0.44, 0.52, 0.82] | 0.031222 | 1 | — | transcribed |
| `p1_q_sobol11` | [0.10, 0.23, 0.55, 0.78] | 0.031528 | 1 | — | transcribed |
| `p0_cosine_pw5` | [0.14, 0.47, 0.81, 0.95] | 0.031539 | 2 | 0.001824 | file |
| `p1_s_late_extreme` | [0.00, 0.05, 0.15, 1.00] | 0.031989 | 2 | 0.000229 | transcribed |
| `p0_cosine_pw10` | [0.14, 0.47, 0.81, 0.95] | 0.031990 | 1 | — | transcribed |
| `p1_q_sobol02` | [0.27, 0.28, 0.35, 0.90] | 0.032047 | 1 | — | file |
| `p1_a_twophase_050` | [0.25, 0.50, 0.72, 0.95] | 0.032056 | 1 | — | transcribed |
| `p1_q_sobol01` | [0.22, 0.23, 0.57, 0.74] | 0.032108 | 1 | — | transcribed |
| `p1_s_ceiling_100` | [0.25, 0.50, 0.75, 1.00] | 0.032233 | 1 | — | file |
| `p1_q_sobol05` | [0.13, 0.49, 0.83, 0.84] | 0.032429 | 1 | — | file |
| `p1_a_linear_0to095` | [0.24, 0.47, 0.71, 0.95] | 0.032766 | 1 | — | transcribed |
| `p1_s_plateau_mid` | [0.20, 0.60, 0.60, 0.70] | 0.033075 | 1 | — | transcribed |
| `p1_s_plateau_low` | [0.30, 0.50, 0.50, 0.50] | 0.033323 | 1 | — | file |
| `p1_q_sobol10` | [0.17, 0.40, 0.54, 0.61] | 0.033469 | 1 | — | transcribed |
| `p1_q_sobol07` | [0.45, 0.60, 0.64, 0.69] | 0.033685 | 1 | — | transcribed |
| `p1_s_early_mid` | [0.45, 0.60, 0.75, 0.95] | 0.033732 | 1 | — | transcribed |
| `p1_q_sobol00` | [0.10, 0.22, 0.59, 0.67] | 0.033861 | 1 | — | transcribed |
| `p1_s_ceiling_050` | [0.15, 0.30, 0.45, 0.50] | 0.034655 | 1 | — | file |
| `p0_static_T050` | [0.50, 0.50, 0.50, 0.50] | 0.034904 | 1 | — | file |
| `p1_q_sobol06` | [0.01, 0.03, 0.12, 0.83] | 0.035239 | 1 | — | transcribed |
| `p1_q_sobol03` | [0.46, 0.89, 0.93, 0.97] | 0.038191 | 1 | — | transcribed |
| `p1_s_early_steep` | [0.60, 0.70, 0.80, 0.90] | 0.038194 | 1 | — | transcribed |

## Tooling
| file | purpose |
|---|---|
| `dynamic_t_search.py` | generates Phase 0 + Phase 1 specs -> manifest |
| `run_dyn_job.sh` | one run: train 2000 kimg -> 5k gen -> MIND + FID |
| `run_dyn_queue.sh` | opportunistic GPU-slot scheduler (proline/lysine) |
| `submit_dyn_csail.sh` | one sbatch per run on CSAIL Slurm |
| `watchdog_dyn_csail.sh` | resubmits failed/lost CSAIL runs; runs as a tig-cpu job |
| `collect_dyn_results.py` | per-machine table; refuses argmin across unequal n |
| `bo_suggest_4d.py` | Phase 2: GP (Matern-5/2 ARD) + EI, numpy/scipy only |
| `tests/test_piecewise_schedule.py` | 14 checks on the piecewise schedule type |

Two BO design choices that are load-bearing, not cosmetic:
1. **GP noise clamped at 0.00089**, not free-fitted. Unclamped it goes to zero
   and the search chases lucky runs.
2. **EI anchored on best posterior mean**, not lowest observed value. With ~30
   points the running minimum sits ~2 sd below the best true value by luck; the
   `warmup_pw10` case above is exactly that failure caught in the wild.

## Status at time of writing
Phase 0 complete (6/6). Phase 1 29/30 — `p1_q_sobol08` still training.
Phase 2 **designed and validated but not run**: the GP identifies all four
coordinates at n=26 (no lengthscale pinned at a bound, first time in the study),
but its proposals were generated before `ceiling_100` landed and all push
T5 -> 1.00, which finding 3 shows is a dead end. **Regenerate proposals against
the completed data before running Phase 2.**
