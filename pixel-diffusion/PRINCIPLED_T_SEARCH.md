# Principled dynamic-T: letting the model find its own schedule

Companion to `DYNAMIC_T_SEARCH.md`. That document records the *discrete* search —
~30 trained models, a 4-D schedule space, GP/EI over MIND. This one records the
attempt to get the same answer from a single run, by asking the model directly
where corrupted data stops being distinguishable from clean data.

**Status: runs in flight.** Design, tooling and calibration are complete and are
recorded below. Results sections are marked TODO until the runs land.

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

**TODO** — MIND per run, discovered T trajectories vs the reference, which
metric/rule tracked it, and whether any discovered schedule beats the discrete
search's best (MIND ≈ 0.0295).
