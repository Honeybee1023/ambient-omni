# Principled dynamic-T: letting the model find its own schedule

Companion to `DYNAMIC_T_SEARCH.md`. That document records the *discrete* search —
~30 trained models, a 4-D schedule space, GP/EI over MIND. This one records the
attempt to get the same answer from a single run, by asking the model directly
where corrupted data stops being distinguishable from clean data.

**Status: headline result in, remaining runs in flight.** The method works
mechanically — non-invasive, curriculum-independent, warmup-shaped trajectory —
but the criterion it optimises is not the one that matters: `pr_predvar` lands at
MIND 0.034158, indistinguishable from a *static* schedule and 5.2 sd worse than
the best hand-found warmup. See section 7.

---

## 1. The idea

At noise level σ, noise a clean held-out image and a blurred held-out image and
denoise both. At small σ the model reconstructs the clean one better — blurred
inputs are off the manifold it has learned. As σ grows the two converge, because
the noise swamps the blur. The σ where they merge is the boundary, and

    T = Φ((ln σ + 1.2) / 1.2)

is that boundary in schedule units. As the model sharpens it should distinguish
them at higher and higher σ, so T should rise on its own — recovering the warmup
shape the discrete search found, without being told.

Divergence lives at **low** σ and vanishes at **high** σ, so the boundary is read
as the *top of the diverging region* scanning upward, not the first crossing. A
first-crossing rule is pinned for the rest of the run by one noisy grid point.

## 2. What was built

| file | role |
|---|---|
| `training/probe.py` | the probe: per-σ statistics, threshold rules, the closed-loop controller |
| `training/training_loop.py` | `probe` key on the schedule dict; `type: principled` obeys it |
| `dataset_creation/create_probe_holdout.py` | builds both probe arms from raw CelebA |
| `probe_checkpoint.py` | runs the probe offline against saved snapshots |
| `principled_t_search.py` | the run definitions |
| `analyze_probe.py` | scores every metric × rule against the known-good curve |
| `tests/test_probe.py` | 50 checks, including synthetic denoisers with a planted boundary |

Probing is **orthogonal to steering**: `probe` turns the measurement on, and
`type: principled` separately says to obey it. That is what lets a run follow a
schedule we already trust while recording what the probe *would* have chosen.

### Design decisions that carry weight

- **Held-out images, both arms.** The 500 clean training images have been seen
  ~4000 times by 2000 kimg; probing on them would measure memorisation. Both
  arms are built from the 20k holdout split, **from raw CelebA through the same
  `process_image` that made bucket b5** — blurring the already-saved holdout
  jpgs would put an extra JPEG cycle on the corrupt arm only, so part of the
  measured gap would be compression artifacts.
- **Common random numbers.** One noise tensor per probe, from a fixed seed,
  reused across every σ, both arms, and every probe of the run. The arms differ
  only by the blur, and the T trajectory moves because the model moved.
- **Paired design.** Both arms cover the same faces, so the comparison is
  within-face. Between-face variance is far larger than the effect. A disjoint
  unpaired estimate is computed and logged alongside it.
- **The probe never touches the training RNG.** Its own `torch.Generator`, under
  `no_grad`, with training mode restored. A probing run and a non-probing run at
  the same seed see the same batches. Pinned by a test.
- **Every rule, every probe.** Each probe stores the full per-σ statistics, so
  any threshold can be re-evaluated offline. `analyze_probe.py` **recomputes**
  decisions from those numbers rather than trusting what the run wrote — which
  is what made the calibration failure in §4 cost minutes instead of GPU-days.
- **Monotonicity is OFF by default.** We already know the answer is
  non-decreasing; enforcing it would make "the method discovered warmup"
  circular. Available as the `pr_skill_mono` ablation.

## 3. Two nuisances that break the obvious metrics

Both were found by synthetic denoisers in `tests/test_probe.py` before any GPU
time was spent, and both were then confirmed on real checkpoints.

### 3.1 Pixel energy contaminates every MSE-based metric

As σ grows, any denoiser's best answer shrinks toward zero, so its MSE tends to
the image's own mean square. Blurred images carry less energy, so **the arms
separate for a reason that has nothing to do with distinguishability.**

For a denoiser that provably cannot tell the arms apart (the exact Gaussian
posterior mean), the measured clean/corrupt MSE ratio still slides **0.996 →
0.148** across the grid. Worse, it drifts *monotonically*, so subtracting a
high-σ baseline does not remove it — it re-centres a slope and turns "no
divergence anywhere" into "divergence everywhere".

EDM's preconditioning gives the correction in closed form. With
`D(x;σ) = c_skip·x`, `c_skip = σ_data²/(σ_data²+σ²)`:

    blind_mse(x, σ) = (1-c_skip)²·mean(x²) + c_skip²·σ²

Dividing measured MSE by this gives a **skill** score: 1.0 means "no better than
uninformative". It needs no extra forward passes — `mean(x²)` comes from the
probe images. On the blind denoiser it collapses the 0.996→0.148 slide to
**0.9939 → 1.0051**, a ~60× reduction in nuisance.

### 3.2 Prediction variance is nuisance-free by construction

For a blind denoiser `D = c_skip·x_t`, the variance of the prediction across
noise realisations is `c_skip²σ²·I` for **both** arms — the images cancel
entirely. Measured: exactly `1.00000` at all 20 levels. This is the one metric
that needs no correction at all, and it is the one that behaved best on real
data (§4).

## 4. Calibration, and the failure it caught

Intermediate checkpoints are deleted by `run_dyn_job.sh` on completion. Eight of
them were copied out of a still-running job first (`$AMBIENT_BASE/probe_ckpts`),
which made it possible to calibrate against a real training trajectory **in
minutes instead of spending a 9-hour probe-only run on it**.

The first calibration pass returned a degenerate answer for *every* shipped
threshold: almost all rules said T = 0.000 at every checkpoint, and
`skill_ratio/baseline/1.02` said T = 1.000. Four closed-loop runs were queued
behind those thresholds — ~36 GPU-hours that would have trained at a constant T
and taught us nothing.

The cause: **the divergence test was one-sided.** Measured prediction-variance
ratio at 1001 kimg:

| T | 0.025 | 0.325 | 0.625 | 0.775 | 0.875 | 0.925 | 0.975 |
|---|---|---|---|---|---|---|---|
| `predvar_ratio` | 0.873 | 0.853 | 0.915 | 0.964 | 0.986 | 0.995 | 0.998 |

It converges on its null of 1.0 beautifully — **from below**, because the model's
answer wobbles *less* on blurred inputs than on clean ones. A `ratio > 1.05` test
never fires on that. Divergence is |ratio − null|, in either direction; the tests
now pin this against the measured curve.

The raw `loss_ratio` over the same checkpoint sits at 0.80–0.88 at *every* σ and
never approaches 1.0 — §3.1 in the wild.

### 4.1 What the calibrated sweep actually says

With the test made two-sided, `analyze_probe.py` re-scored every metric x rule x
threshold against all 8 checkpoints. Open-loop T trajectories (kimg 0 -> 1751):

```
pred_var    /fixed   /1.10   0.00 0.52 0.57 0.62 0.62 0.62 0.52 0.47
pred_var    /fixed   /1.05   0.00 0.72 0.72 0.72 0.77 0.77 0.72 0.67
pred_var    /fixed   /1.02   0.00 0.82 0.88 0.88 0.88 0.88 0.88 0.88
skill_ratio /baseline/1.05   0.00 0.67 0.72 0.72 0.72 0.72 0.72 0.67
loss_ratio  /fixed   /1.20   0.00 0.92 0.92 0.92 0.92 0.92 0.92 0.92
mse_gap     /adaptive/any    1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
```

Three things follow, and the second is the uncomfortable one.

**`pred_var` is the only well-behaved metric.** It is nuisance-free by
construction (§3.2), its curve converges cleanly on its null, and it is the only
one with a usable dynamic range: threshold 1.01 -> T=0.925, 1.10 -> T=0.625.

**The boundary does not ramp.** Apart from the jump off the untrained
checkpoint — where every ratio is exactly 1.000, so T=0 correctly — the
trajectories are flat or drift *down*. The per-sigma curves move by only ~0.013
(`pred_var`) to ~0.023 (`skill_ratio`) between 250 and 1751 kimg, against a
spread of ~0.14 across sigma, and not monotonically: at T=0.47 the ratio runs
0.899 -> 0.868 -> 0.911, a U rather than a ramp. **The model learns to separate
blurred from clean inside the first ~12% of training and its boundary then sits
still.** The premise "the boundary drifts as the model improves, recovering
warmup" is not supported open loop.

It is worth noting what the probe *does* get right: it settles at T ~ 0.88 under
a tight threshold, and the discrete search independently found the ceiling
saturates at T5 ~ 0.8-0.95. The probe reads the right endpoint immediately; it
is the trajectory to that endpoint it does not produce.

**`mse_gap` is unusable at any threshold.** Its paired t-statistic is 33-46 at
*every* sigma, so |t| > 16 still flags everything. Common random numbers make the
standard error tiny while the pixel-energy nuisance stays systematic — a t-test
is the wrong instrument when the contamination is a bias rather than noise. The
raw `loss_ratio` sits at 0.80-0.88 at every sigma and needs a threshold above
1.14 merely to stop being pinned.

### 4.2 Why the runs go ahead anyway

Open loop is not the experiment. In a closed loop T feeds back into which data
the model sees, which changes the next probe — a loop that cannot be simulated
from one fixed trajectory. And the calibration model followed `sobol08`'s
schedule (a late ramp), not the known-good warmup; `gt_warmup` probes a model
trained under the curriculum we actually care about.

Thresholds were chosen to cut where each curve is steepest, and deliberately
land the four closed-loop runs at different T levels (~0.5, ~0.7, ~0.9, ~0.9) so
the batch also sweeps the level axis.

### 4.2b Correction: the boundary DOES ramp — the calibration grid was too coarse

The first live probes of `gt_warmup` overturn part of §4.1. Recomputed with the
two-sided rules, under the known-good warmup curriculum and at 100-kimg spacing:

```
kimg                        0     100     200     300
pred_var/fixed/1.10      0.00    0.12    0.47    0.52
pred_var/fixed/1.05      0.00    0.38    0.67    0.72
```

That is a real ramp in the **raw** signal, over three probes — not the step §4.1
predicted. The two are not in conflict: sobol08's snapshots are 250 kimg apart,
so its first post-zero sample already sat at the top of the rise (it read 0.52 at
250 kimg; `gt_warmup` reads 0.52 at 300). The calibration was not wrong about the
plateau, it simply had no sample inside the climb. Checkpoint spacing, not
curriculum, is the explanation — the two agree wherever they overlap.

**What survives from §4.1**: the boundary saturates early, by ~300 kimg (15% of
training), and then sits still for the remaining 85%.

**What this changes**: the probe and the empirically best schedule *disagree
about shape*. The probe says the model can already tolerate T~0.5 by 300 kimg;
the best known schedule holds T=0 until 500 kimg and only then ramps, reaching
0.95 at the end. The probe front-loads what warmup back-loads. Whether that is
the probe being right and the discrete search's grid too coarse, or the probe
measuring the wrong thing, is exactly what the closed-loop runs decide.

It also means any future version should probe **densely over the first ~500
kimg** rather than uniformly: at 100-kimg spacing only three probes land inside
the part of the run where anything moves.

### 4.2c The probe reads the model, not the curriculum — and it disagrees with what works

`gt_warmup` (open loop, trained under warmup 0->0.95) and `pr_predvar` (closed
loop, driven by its own probe) give almost the same raw reading, though the two
models saw very different data:

```
kimg              0   100   200   300   400   500   600   700   800  ...  1500
gt_warmup      0.00  0.12  0.47  0.52  0.57  0.57  0.57  0.57  0.57       0.67
pr_predvar     0.00  0.17  0.47  0.52  0.57  0.57  0.62  0.62  0.62
warmup sched   0.00  0.00  0.00  0.00  0.00  0.00  0.06  0.13  0.19       0.63
```

mean |gt_warmup - pr_predvar| = **0.022** over the overlap, about one grid step.
Two things follow.

**The reading is a property of the model's competence, not of the schedule that
produced it.** That is the robustness the `gt_static50` control was added to
test, and it already holds across two curricula that differ enormously early on
(one holds T=0 for 500 kimg, the other is at T=0.37 by kimg 400).

**There is therefore no feedback amplification.** The closed loop neither runs
away nor reinforces itself; it tracks the same curve the open loop does. Any
ramp in the applied schedule is the probe's own reading plus the EMA, not a
loop effect.

**And the probe disagrees with the schedule that actually wins.** Higher T means
*less* corrupt data. The probe wants T~0.57 by kimg 400; the best known schedule
holds T=0 — maximally permissive — until kimg 500 and only tightens later. They
converge only near the end. So the criterion "use corrupted data only where it is
indistinguishable from clean" is **not** the same criterion as "use corrupted
data where it helps": early in training the model appears to need data *volume*
more than data *purity*, and 500 clean images is not enough on its own. The
probe rejects exactly the data the discrete search says to use.

That yields a falsifiable prediction, recorded before the run finished: because
`pr_predvar` restricts early where warmup does not, its MIND should land nearer
the static-T band (~0.0335-0.0350, cf. `static_T050` = 0.034904) than the warmup
plateau (~0.0295). **TODO: check against the measured value.**

### 4.3 The smoothing can manufacture the result — read T_raw, not T_smoothed

The controller applies an EMA over probes (`alpha=0.3`) because a raw per-probe T
is quantised to the grid and jumps. That smoother is also capable of inventing
the headline finding. Feed it the step function the calibration predicts —
untrained model reads 0, every subsequent probe reads 0.55 — and it emits:

```
T_raw       0.00 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 0.55 ...
T_smoothed  0.00 0.17 0.28 0.36 0.42 0.46 0.49 0.50 0.52 0.53 ...
```

reaching 95% of its final value at ~900 kimg. Plotted, that is a textbook warmup
ramp. It contains no discovery whatsoever.

So: **the claim "the model discovered a warmup schedule" can only rest on
`T_raw`.** Every probe logs both, and `analyze_probe.py` prints both, precisely
so this cannot be glossed over. If `T_raw` turns out to be a step and
`T_smoothed` a ramp, the honest framing is that *the probe supplies the ceiling
and the EMA supplies the ramp* — a real but much weaker contribution, and the
`pr_skill_nosmooth` ablation (`alpha=1.0`) becomes the run that matters rather
than a nice-to-have.

## 5. Runs

Dataset `celeba_dynamic_t_v2` (500 clean b0 + 26,014 blurred b5, σ_blur=0.5),
2000 kimg, batch 64, seed 0 — identical to the discrete search, so MIND is
directly comparable.

**Watch the dataset name.** lysine's `celeba_dynamic_t_v2` is the historical
182,598-image build; the matching 26,514-image one is
`celeba_dynamic_t_v2_b0b5`. proline's `celeba_dynamic_t_v2` *is* the 26,514 one.
The queues are launched with `DYN_DATASET` set per machine.

| run | probe | drives T? |
|---|---|---|
| `gt_warmup` | on | no — follows warmup 0→0.95 |
| `gt_static50` | on | no — follows static T=0.50 |
| `pr_skill` | `skill_ratio` | yes |
| `pr_predvar` | `pred_var` | yes |
| `pr_lossratio` | `loss_ratio` | yes |
| `pr_msegap` | `mse_gap` | yes |

`gt_static50` exists to answer the obvious reviewer question: if the probe reads
the same boundary under two different curricula, the signal is a property of the
model's competence rather than of the schedule that produced it.

Ablations (`pr_skill_mono`, `pr_skill_nosmooth`, `pr_skill_pct`,
`pr_skill_fast`) are defined but deliberately not queued until the thresholds
are known to be non-degenerate.

## 6. Cost

Measured, not estimated: 40.0 s per probe at 100 images on an idle H200, against
14.78 sec/kimg training. The spec'd 200 images would have cost **5.4%** of wall
clock — just over the 5% budget — so the probe runs at **160 images**, ≈4.3%.
Probe `batch_size` is 80 rather than 200 because at 200 the probe raised peak GPU
memory from ~30 GB to 43 GB, which would stop two jobs sharing an 80 GB A100.

Every run reports its own cost live as `probe_ovh` in the tick line, so this is
checked rather than assumed.

## 7. Results

### The probe is non-invasive — first hard evidence

`gt_warmup` follows warmup 0->0.95 with the probe attached but not steering, so
it is the same configuration as the discrete search's `p0_warmup_pw5`. Measured:

| | MIND | n |
|---|---|---|
| `gt_warmup` (probe attached) | 0.031173 | 1 |
| `p0_warmup_pw5` (no probe) | 0.030319 | 2 |

**+0.96 sd** apart on a run-to-run noise floor of 0.00089 — indistinguishable.
Probing 20 times mid-run, 12.8k forward passes each, costs nothing measurable in
final quality. That is the empirical counterpart to the RNG-isolation design
(the probe draws from its own `torch.Generator`, so a probing run sees the same
training batches as a non-probing one), and it is what makes the `gt_*` runs
usable as ground truth rather than as a perturbed system.

### The headline: indistinguishability is not the right objective

`pr_predvar` — the best-behaved metric, driving T closed-loop — finished at
**MIND = 0.034158**, against a run-to-run noise floor of 0.00089:

| | MIND | vs pr_predvar |
|---|---|---|
| `warmup40` (best discrete) | 0.029537 | **+5.19 sd worse** |
| `warmup` 0->0.95 | 0.030319 | +4.31 sd worse |
| `gt_warmup` (same schedule, probed) | 0.031173 | +3.35 sd worse |
| `ceiling_050` | 0.034655 | -0.56 sd |
| **`static_T050`** | 0.034904 | **-0.84 sd, indistinguishable** |

This value was **predicted in advance** — 0.0335-0.0350, committed in 729e005
before the run finished — from the shape disagreement in section 4.2c. It landed
at 0.034158.

The schedule it chose, against the one that works:

```
kimg          0    300    600    900   1200   1500   1801
T applied  0.00   0.28   0.49   0.58   0.61   0.62   0.59
warmup ref 0.00   0.00   0.06   0.25   0.44   0.63   0.82
```

Wrong at both ends. Far **too restrictive early** — T=0.49 at kimg 600 where
warmup is still at 0.06, so it throws away corrupted data the good schedule is
still using. And **not restrictive enough late** — 0.59 against 0.82, so it keeps
using corrupted data the good schedule has dropped.

So the conclusion is not "the probe is noisy" or "the threshold was wrong". The
probe works: it is non-invasive, its reading is curriculum-independent to within
one grid step, and it produces a monotone warmup-shaped trajectory in the raw
signal. **The criterion it optimises is simply not the criterion that matters.**

Ambient-o's premise — corrupted data is *safe* above the noise level where it
becomes indistinguishable from clean — is a statement about safety, and this
result says it is the wrong objective to *maximise*. With only 500 clean images,
early training benefits from corrupted data at noise levels where it is plainly
distinguishable: volume beats purity while the model still knows nothing. The
probe rejects exactly that data, and pays 5 sd for it.

That is a useful negative result rather than a failed experiment, and the
pre-registered prediction makes it a strong one. It also says where a principled
method should look instead: not at whether corrupted data is distinguishable, but
at whether including it still reduces held-out loss — a marginal-value question,
not a distinguishability one.

### Why it fails: only the early phase matters

`pr_skill` (skill_ratio/baseline/1.05) came in at **0.042679** — worse than any of
the 30 discrete runs. Not degenerate: it settled at T~0.78 with 16-18% of batches
corrupt, a real but very restrictive schedule. Placing it alongside the others
separates *how much* a schedule restricts from *when*:

| schedule | T over first 25% | T over last 25% | mean T | MIND |
|---|---|---|---|---|
| `warmup40` (best) | 0.079 | 0.950 | 0.534 | 0.029537 |
| warmup 0->0.95 | 0.000 | 0.792 | 0.356 | 0.030319 |
| `pr_predvar` | 0.220 | 0.596 | 0.490 | 0.034158 |
| `static_T050` | 0.500 | 0.500 | 0.500 | 0.034904 |
| `early_steep` | 0.299 | 0.850 | 0.637 | 0.038194 |
| `pr_skill` | 0.462 | 0.783 | 0.697 | 0.042679 |

```
r(T over first 25%, MIND) = +0.785      restricting early hurts, badly
r(T over last  25%, MIND) = -0.137      restricting late barely registers
r(mean T,           MIND) = +0.829      driven entirely by the early term
```

The best schedule settles it on its own: `warmup40` has a **higher** mean T than
`pr_predvar` (0.534 vs 0.490) and is 5.2 sd **better**, purely because its
restriction is back-loaded. So the rule is not "use more corrupted data", it is
**"use it early"** — the first quarter of training decides the outcome and the
last quarter is nearly free.

That is exactly the window the probe gets wrong. Its reading climbs from 0 to
~0.57 inside the first 400 kimg (section 4.2b), so it starts restricting
precisely when restriction is most expensive. The failure is not a bad threshold;
a probe faithfully tracking distinguishability *must* rise there, because that is
when the model actually learns to distinguish. The criterion and the objective
point in opposite directions during the only phase that matters.

Caveat: n=6 and the points are not independent, so treat +0.785 as direction and
mechanism rather than a calibrated effect size. The ordering and the `warmup40`
counterexample carry the argument, not the coefficient.

### The probe is a clock, not a controller

Three models trained under radically different data regimes — warmup (T=0 for the
first 500 kimg, rising to 0.95), static T=0.50, and the probe's own closed loop —
were probed with the identical instrument. Recomputed at `pred_var/fixed/1.10`:

```
kimg                          0    300    600    900   1200   1500
gt_warmup   (0 -> 0.95)    0.00   0.52   0.57   0.62   0.62   0.67
gt_static50 (0.50 flat)    0.00   0.57   0.62   0.62   0.62   0.62
pr_predvar  (closed loop)  0.00   0.52   0.62   0.62   0.62   0.57

pairwise mean |difference| = 0.019 - 0.028   (grid spacing is 0.05)
```

Every pair agrees to within **half a grid step**. That is a genuine robustness
property — the reading is a stable function of the model, not an artifact of the
curriculum — and it is also the deepest reason the method cannot work as posed.

**A control signal that does not respond to the control action cannot close a
useful loop.** The probe returns very nearly the same trajectory no matter what
data the model was fed, so it carries almost no information about whether the
*current* schedule is appropriate. It measures how far training has progressed,
not what data is safe given this model. It is a clock.

That reframes the negative result usefully. The failure is not a bad metric or a
bad threshold — `pred_var` is well-behaved, nuisance-free and reproducible. It is
that clean-vs-corrupt distinguishability is nearly independent of the training
schedule, so no threshold on it can steer that schedule. Anything genuinely
adaptive has to measure a quantity that *moves when the schedule moves* — e.g.
the marginal effect of including corrupted data on held-out loss, which by
construction depends on whether that data is currently being used.

### Control 2: probing is non-invasive on a second schedule

| | MIND | n |
|---|---|---|
| `gt_static50` (probe attached) | 0.035261 | 1 |
| `static_T050` (no probe) | 0.034904 | 1 |

**+0.40 sd.** With `gt_warmup` at +0.96 sd, non-invasiveness now holds on two
schedules as different as constant-T and full warmup.

### The EMA was doing the work: a matched pair

`pr_predvar_nosmooth` is `pr_predvar` with one character changed — `alpha` 0.3 ->
1.0, so T follows the raw probe reading with no smoothing. Same metric, same
threshold, same probe seed, same everything else.

| | early T (first 25%) | MIND | |
|---|---|---|---|
| `pr_predvar`, alpha=0.3 | 0.220 | 0.034158 | |
| `pr_predvar_nosmooth`, alpha=1.0 | 0.432 | 0.036033 | **+2.11 sd worse** |

Predicted in advance (commit 2d11674) on the strength of the early-phase result,
and confirmed. This is the controlled version of that finding: everything is held
fixed except how fast T is allowed to rise early, and letting it rise at the
probe's own pace costs 2.11 sd.

So the smoother was not a cosmetic detail — **its lag was the only thing keeping
`pr_predvar` out of the static-T band.** Section 4.3 worried that the EMA might
manufacture a warmup ramp and steal credit from the probe; the truth is worse for
the method than that. The EMA was contributing the one property that helps
(permissiveness early), and the probe was contributing the property that hurts.

With ten schedules now measured:

```
r(T over first 25%, MIND) = +0.762
r(T over last  25%, MIND) = +0.014      late T spans 0.50-0.95 and MIND does not care
```

### `pr_msegap`: the prediction held

Queued last and at the only non-degenerate threshold found, `mse_gap/baseline/16`
finished at **0.043754** — the worst of all 40 runs in both studies. Its applied T
sat at 0.82-0.87 from the very first probe (early T = 0.843, the most restrictive
schedule measured), which is precisely what the offline calibration said it would
do. The metric is unusable, and now measured rather than only predicted.

### Full ordering, all ten measured schedules

| schedule | early T | late T | MIND |
|---|---|---|---|
| `warmup40` (best discrete) | 0.079 | 0.950 | 0.029537 |
| warmup 0->0.95 | 0.000 | 0.792 | 0.030319 |
| `gt_warmup` (probed) | 0.000 | 0.792 | 0.031173 |
| `pr_predvar` | 0.220 | 0.596 | 0.034158 |
| `static_T050` | 0.500 | 0.500 | 0.034904 |
| `gt_static50` (probed) | 0.500 | 0.500 | 0.035261 |
| `pr_predvar_nosmooth` | 0.432 | 0.530 | 0.036033 |
| `early_steep` | 0.299 | 0.850 | 0.038194 |
| `pr_lossratio` | 0.480 | 0.876 | 0.042591 |
| `pr_skill` | 0.462 | 0.783 | 0.042679 |
| `pr_msegap` | 0.843 | 0.831 | 0.043754 |

Every probe-driven run sits below every hand-designed warmup, and the ordering is
early-T ordering.

### Replication

Every principled run is n=1, while the discrete baselines it is being compared
against are n=2. That is exactly the unequal-n comparison this project has been
burned by before (see `DYNAMIC_T_SEARCH.md`), so a seed-1 replicate of
`pr_predvar` — the headline number — is running. **TODO: fold in.**

### CORRECTION: the critical window is the SECOND quarter, not the first

`pr_predvar_hold25` holds T=0 through the first 25% of training and then obeys
the probe. Predicted (from the n=7 early-phase result) to reach the warmup
plateau, ~0.0295-0.0315. **It landed at 0.033964** — 0.22 sd from plain
`pr_predvar` (0.034158). Holding the entire first quarter bought nothing. The
prediction is refuted and the "only the first quarter matters" claim with it.

Scanning windows over all 12 measured schedules:

```
window of training      r(mean T in window, MIND)
[0.00, 0.10]                   +0.617
[0.00, 0.25]                   +0.812     <- the earlier claim
[0.00, 0.50]                   +0.909
[0.10, 0.40]                   +0.916
[0.25, 0.50]                   +0.939     <- strongest
[0.50, 0.75]                   +0.728
[0.75, 1.00]                   +0.141
```

The controlling variable is T over the **second quarter**. A two-predictor fit
confirms the late phase is inert:

```
MIND ~ 0.02456 + 0.01908 * T[.25,.50] + 0.00192 * T[.75,1]
       R^2 = 0.886        (T[.25,.50] alone: 0.882)
       +2.1 sd per +0.1 of T in the second quarter
       +0.2 sd per +0.1 of T in the last quarter
```

That explains `hold25` cleanly. It protected the first quarter, but the EMA then
ramped hard and put T at 0.396 across [.25,.50] — close to `pr_predvar`'s 0.534
and far above warmup's 0.158. It defended the wrong window.

And it sharpens the verdict on the method. To match warmup, the probe's output
would have to be suppressed through the **first half** of training — by which
point the schedule is warmup, and the probe is contributing nothing but a
ceiling that the last-quarter coefficient says barely matters.

**Caveat, stated rather than buried:** `warmup40` sits 4.1 sd *better* than this
model predicts. It is also the argmin of a 30-run search, so it is precisely the
point most likely to be an optimistic draw; n=2 helps but does not remove
selection bias. The model should be read as "T in the second quarter dominates",
not as a calibrated predictor, and `warmup40`'s residual is not evidence of extra
structure.

### Cost, confirmed on a finished run

1308 s of probing across 20 probes, against ~34,280 s of training: **3.8%**,
inside the 5% budget.

### Remaining runs

`pr_skill`, `pr_lossratio`, `gt_static50`, `pr_predvar_nosmooth`, `pr_msegap`.

A second prediction, recorded before the fact: `pr_predvar_nosmooth` (alpha=1.0)
follows T_raw directly, which is *more* restrictive early than the EMA-smoothed
version (0.52 vs 0.28 at kimg 300). If "too restrictive early" is really what
costs `pr_predvar` its 5 sd, the no-smoothing run should come in **worse than
0.034158**, and the EMA's lag will have been doing useful work the probe was not. — MIND per run, discovered T trajectories vs the reference, which
metric/rule tracked it, and whether any discovered schedule beats the discrete
search's best (MIND ≈ 0.0295).
