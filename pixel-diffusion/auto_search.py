"""Automatic-schedule search: controllers that set T online from a measurement.

Round 1 (2026-09-13). Each entry differs in KIND from the failed
distinguishability rules (which read a saturating skill and therefore produce a
concave, static-like curve):

  auto_soft_slope     bottleneck detector: withdraw blur from a noise level once
                      the output's fine-band energy there has stopped improving
                      under the blurred diet (learning limited by the targets,
                      not by time). Walks upward from the lowest eligible level;
                      max_step 0.1 per probe keeps the withdrawal smooth.
  auto_trigger_concave two-stage: T=0 until the memorisation gap on the model's
                      own clean training faces (vs held-out) exceeds 3% at low
                      noise, then a fixed concave withdrawal to 0.95 over the
                      remaining budget (the shape study's best).
  auto_trigger_linear same trigger, straight withdrawal (shape control).
  auto_starve         starvation only (control): per level, "starved" = the
                      model keeps improving on its own clean training faces
                      while held-out has stopped (memorisation onset). Blur is
                      used where starved and withdrawn elsewhere. Expected to
                      want blur more over time (wrong late).
  auto_beam_mind      look-ahead alone (run_beam.sh, SCORE=mind): branch 50
                      kimg under T / T+0.2 / T-0.2 every 250 kimg, keep the
                      arm with the best MIND on 2k samples.
  auto_beam_probe     combination (run_beam.sh, SCORE=probe): same branching,
                      arms judged by high-noise softness minus low-noise
                      memorisation gap from the probe.
  auto_soft_target    backward plan from a measured recovery time (300 kimg per
                      level, 4 levels recovering concurrently): withdraw a level
                      only when the remaining budget requires it. Reads the
                      clock and the measured constant, not the model -- the
                      "is budget-awareness alone enough" control.

Usage: python auto_search.py [--dry-run]   (adds/replaces entries in the manifest)
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")
import argparse, json, os, shutil
MANIFEST = f"{AMBIENT_BASE}/generated/dyn_search_manifest.json"
PHASE = "auto1"
TRAIN_DIR = f"{AMBIENT_BASE}/annotated_datasets/" + os.environ.get("DYN_DATASET", "celeba_dynamic_t_v2")

BASE = {"every_kimg": 100, "n_images": 160, "n_draws": 2, "n_levels": 20, "batch_size": 80,
        "probe_seed": 12345, "alpha": 1.0, "monotone": False, "t_init": 0.0,
        "train_dir": TRAIN_DIR, "n_train": 160}

def probe(**kw):
    c = dict(BASE); c.update(kw); return c


def log_probe(dataset):
    """A probe that only LOGS: no controller drives it, because the schedule it rides on is not
    'principled'. It records the same 20-level fine-detail and memorisation traces the controller
    runs produce, at ~3% of wall clock.

    Why baselines need it: without a baseline's trace we can only compare the rule against itself.
    That is exactly what stopped us checking the under-recovery forecast on the 250-clean/blur-1.0
    corner -- the rule's trace existed, its baselines' did not, so 'why did it win' could not be
    answered from logs. FUTURE BASELINE ENTRIES SHOULD TAKE THIS BY DEFAULT.

    Not applied retroactively: a manifest entry is read when the job launches, so editing one that
    is already queued or running would change what that run does.
    """
    return probe(every_kimg=200, train_dir=f"{AMBIENT_BASE}/annotated_datasets/{dataset}")

RUNS = [
    {"name": "auto_soft_slope", "note": "bottleneck: withdraw a level when its fine-band energy stops improving under blur",
     "schedule": {"type": "principled", "probe": probe(controller="soft_slope", ctl={"window": 4, "eps": 0.002}, max_step=0.1)}},
    {"name": "auto_trigger_concave", "note": "T=0 until mem gap at low noise > 5%, then concave withdrawal to 0.95",
     "schedule": {"type": "principled", "probe": probe(controller="trigger_ramp", ctl={"gap_thr": 0.03, "gap_t_max": 0.35, "shape": "concave", "force_at": 0.7})}},
    {"name": "auto_trigger_linear", "note": "same trigger, linear withdrawal (shape control)",
     "schedule": {"type": "principled", "probe": probe(controller="trigger_ramp", ctl={"gap_thr": 0.03, "gap_t_max": 0.35, "shape": "linear", "force_at": 0.7})}},
    {"name": "auto_starve", "note": "starvation only (control): blur where the model is memorising (train error falling, held-out flat), withdrawn elsewhere",
     "schedule": {"type": "principled", "probe": probe(controller="starve", ctl={"window": 4, "eps_train": -0.005, "eps_hold": 0.005}, max_step=0.15)}},
    {"name": "auto_soft_target", "note": "backward plan from a measured 300-kimg recovery time; clock + constant only",
     "schedule": {"type": "principled", "probe": probe(controller="soft_target", ctl={"tau_kimg": 300, "parallel": 4, "total_kimg": 2000})}},
]

# Round 2 (2026-09-14), designed from the look-ahead result: no controller may
# ask "is sample quality better now?" -- that question is confidently inverted
# at every affordable horizon (T=0.95 wins at 250 kimg with a 300-kimg horizon).
# Both entries below ask only *when withdrawal must start so recovery finishes*.
PHASE2 = "auto2"
ROUND2 = [
    {"name": "auto_backplan", "note": "self-calibrating backward plan; recovery time measured on this run, all levels recover concurrently",
     "schedule": {"type": "principled", "probe": probe(controller="backplan",
                  ctl={"tau_prior": 300, "parallel": 20, "total_kimg": 2000, "t_end": 0.95})}},
    {"name": "auto_backplan_slow", "note": "same, but assuming only 4 levels recover at a time: withdrawal starts much earlier and ramps",
     "schedule": {"type": "principled", "probe": probe(controller="backplan",
                  ctl={"tau_prior": 300, "parallel": 4, "total_kimg": 2000, "t_end": 0.95})}},
]


# Round 3 (2026-09-14), designed from measured recovery times, not from MIND.
# Paired same-seed probes (auto_soft_slope vs auto_trigger_concave) show fine
# detail at low noise comes back within one 100-kimg probe of withdrawing blur;
# the 9-checkpoint softness trajectories put it at ~250-500 kimg at sigma~0.85
# and later. Recovery time grows with noise level -- the opposite order to the
# one a threshold T can express (it always withdraws low noise first).
PHASE3 = "auto3"
# tau(t): kimg a level needs between withdrawal and the end. Upper ends of the
# measured ranges (recovery must finish); 600 at t=1 keeps the top from jumping.
TOPDOWN_TAU = [[0.0, 100], [0.2, 100], [0.3, 300], [0.5, 400], [0.7, 500], [1.0, 600]]
ROUND3 = [
    {"name": "auto_topdown", "note": "top-down withdrawal: each noise level loses blur its measured recovery time before the end (high noise first, low noise last)",
     "schedule": {"type": "topdown", "tau_points": TOPDOWN_TAU, "total_kimg": 2000, "probe": probe()}},
    {"name": "auto_test_step75", "note": "principle test, hand-set: T=0 then a jump to 0.95 at 75% (every level withdrawn 500 kimg before the end); probes every 50 kimg measure recovery at all 20 levels",
     "schedule": {"type": "piecewise", "control_points": [[0.0, 0.0], [0.75, 0.0], [0.75, 0.95], [1.0, 0.95]], "probe": probe(every_kimg=50)}},
]


# Round 4 (2026-09-15): measure the recovery time properly instead of reading back the
# probe spacing. auto_backplan "measured" 100/200/300 kimg, which is 1/2/3 probe intervals.
# Two changes: probe every 25 kimg once blur starts leaving levels (~+18% wall clock on an
# A100, ~a third of a GPU-day on top of a 10 h run), and fit an exponential to each level's
# softness since its own withdrawal rather than testing a 3-probe window for flatness.
# The controller withdraws TOP-DOWN, which is the order that makes the measurement useful:
# the slow high levels are withdrawn first, so their fitted recovery times are in hand
# before the fast low levels have to be planned. Priors are the recovery times measured on
# the look-ahead branches (tau90: ~100 kimg at t<=0.3, ~200 at 0.5, ~300 at 0.7-0.9),
# not tuned on MIND. NOT QUEUED: awaiting the user's decision on the remaining GPU time.
PHASE4 = "auto4"
ROUND4 = [
    {"name": "auto_topdown_live", "note": "top-down withdrawal, recovery time fitted per level on this run (dense probing from the first withdrawal)",
     "schedule": {"type": "principled", "probe": probe(
         controller="topdown_live", every_kimg=100, every_kimg_dense=25, dense_from_progress=0.65,
         ctl={"prior_points": [[0.1, 100], [0.3, 100], [0.5, 200], [0.7, 300], [0.9, 300]],
              "total_kimg": 2000, "tau_min": 50, "tau_max": 800, "min_points": 4})}},
]


# Round 5 (2026-09-15): robustness across four settings, one thing changed each time from
# 500 clean + 26,014 blurred at sigma_b 0.5. Datasets: dataset_creation/create_robustness_datasets.py
# (rb_c250, rb_c1000, rb_b03, rb_b10). The run's dataset is chosen by DYN_DATASET at launch;
# the name says which setting it belongs to. Per setting: our rule (backplan), the best
# hand-set schedule (warmup40), the best static we already measured at that blur level, and
# for the clean-count settings a clean-only floor. One seed each: at sd 0.00104 only gaps
# above ~0.003 mean anything, so the question is whether the rule TRACKS the hand-set
# schedule everywhere, not whether it beats it.
PHASE5 = "auto5"
WARMUP40 = {"type": "piecewise", "control_points": [[0.0, 0.0], [0.4, 0.0], [1.0, 0.95]]}
BACKPLAN_CTL = {"tau_prior": 300, "parallel": 20, "total_kimg": 2000, "t_end": 0.95}
# Clean-only floor: the blurred bucket parked at T=0.999 (eligible only above sigma 12.28,
# ~0.066% of draws) rather than a separate dataset, because run_dyn_job.sh always passes a
# schedule and the sentinel needs one. Same device the parked buckets already use.
CLEAN_ONLY = {"type": "static", "t_start": 0.999}
_SETTINGS = {
    "b03":   [("static0475", 0.475)],
    "b10":   [("static070", 0.70), ("static085", 0.85)],
    "c250":  [("static045", 0.45), ("cleanonly", None)],
    "c1000": [("static045", 0.45), ("cleanonly", None)],
}
ROUND5 = []
for _tag, _statics in _SETTINGS.items():
    _ds = f"{AMBIENT_BASE}/annotated_datasets/rb_{_tag}"
    ROUND5.append({"name": f"rb_{_tag}_backplan", "note": f"robustness {_tag}: backward planner (our rule)",
                   "schedule": {"type": "principled",
                                "probe": probe(controller="backplan", ctl=dict(BACKPLAN_CTL), train_dir=_ds)}})
    ROUND5.append({"name": f"rb_{_tag}_warmup40", "note": f"robustness {_tag}: hand-set baseline, T=0 to 40% then linear to 0.95",
                   "schedule": dict(WARMUP40)})
    for _sname, _t in _statics:
        ROUND5.append({"name": f"rb_{_tag}_{_sname}",
                       "note": f"robustness {_tag}: {'clean-only floor' if _t is None else f'best static T={_t}'}",
                       "schedule": dict(CLEAN_ONLY) if _t is None else {"type": "static", "t_start": _t}})


# Round 6 (2026-09-16): the clean-exposure controller, across all five settings.
# Control variable = clean exposure (the integral of the clean share of the batch), because it
# is the quantity that (a) predicts the end-of-run memorisation gap at rank 0.97 on our own
# table, and (b) actually moves when the schedule moves -- the property every earlier signal
# lacked. Two constants, both stated: target_epochs 1259 = the median of the base table's top
# eight (a calibrated scalar, from the 500-clean table only), and tau_rec 300 kimg = the
# recovery time measured on the look-ahead branches. The recovery floor wins any conflict.
# The dataset is chosen per run at launch (DYN_DATASET); the name says which setting.
PHASE6 = "auto6"
# target_epochs: the median MEASURED clean exposure of the eight best base-setting runs
# (1109 passes over the clean set; the band runs ~1000-1600, with 707 too little at 0.0349 and
# 1954 too much at 0.0346). Measured from the logged batch composition with the DataLoader's
# prefetch ticks excluded, i.e. in the same units the controller accumulates at run time.
# ONE calibrated scalar, taken from the base setting only and applied unchanged everywhere.
# f_hi: the clean share after full withdrawal, measured at 0.94 across those runs; the run
# replaces it with its own observation once it has withdrawn. tau_rec: 300 kimg, the recovery
# time measured on the look-ahead branches.
EXPOSURE_CTL = {"target_epochs": 1109, "tau_rec": 300, "total_kimg": 2000, "t_end": 0.95,
                "f_hi": 0.94, "shape": "jump"}
# The base dataset is called celeba_dynamic_t_v2_b0b5 on lysine and celeba_dynamic_t_v2 on
# proline (same 26,514-image b0+b5 content, different name). Resolve it per machine: a manifest
# entry naming a directory this cluster does not have would fail the probe at load time, after
# the job had taken a card.
BASE_DATASET = next((_n for _n in ("celeba_dynamic_t_v2_b0b5", "celeba_dynamic_t_v2")
                     if os.path.isdir(f"{AMBIENT_BASE}/annotated_datasets/{_n}")), "celeba_dynamic_t_v2")
_EXP_SETTINGS = {"base": BASE_DATASET, "c250": "rb_c250", "c1000": "rb_c1000",
                 "b03": "rb_b03", "b10": "rb_b10"}
ROUND6 = [
    {"name": f"exp_{_tag}", "note": f"clean-exposure controller, setting {_tag}",
     "setting": _tag,
     "schedule": {"type": "principled",
                  "probe": probe(controller="exposure", ctl=dict(EXPOSURE_CTL), every_kimg=200,
                                 train_dir=f"{AMBIENT_BASE}/annotated_datasets/{_ds}")}}
    for _tag, _ds in _EXP_SETTINGS.items()
]


# Round 7 (2026-09-17): the two tests that attack the rule's untested assumptions.
#
# (1) BUDGET. Every result so far is a 2000-kimg run, and the target is expressed in passes over
# the clean set, so it should transfer to another length -- that is an assumption, not a result.
# At 1000 kimg the same target forces a MUCH earlier drop (~42% of training) than the ~72% the
# 2000-kimg runs chose, so a "withdraw at about 70%" reading of our results predicts something
# different and the two are separated by one run. total_kimg is threaded into the controller's
# projection and its recovery floor; the job must also be launched with the shorter duration
# (run_budget_job.sh), since run_dyn_job.sh hardcodes 2000.
#
# (2) CORNER. Few clean images AND heavy blur, the one combination the robustness grid lacks.
# Here the exposure target wants a drop past the floor, so the floor binds and fixes blur-free
# time at 300 kimg -- a value measured at blur 0.5, where recovery runs about twice as fast as
# the blur-1.0 run showed. This is where I expect the rule to under-recover; it is registered as
# a predicted failure, and the prediction is in the log before it runs.
PHASE7 = "auto7"
_K1000 = dict(EXPOSURE_CTL); _K1000.update({"total_kimg": 1000})
ROUND7 = [
    # --- budget test, base setting, 1000 kimg
    {"name": "exp_base_k1000", "note": "budget test: clean-exposure controller at a 1000-kimg budget",
     "setting": "base_k1000", "total_kimg": 1000,
     "schedule": {"type": "principled",
                  "probe": probe(controller="exposure", ctl=dict(_K1000), every_kimg=100,
                                 train_dir=f"{AMBIENT_BASE}/annotated_datasets/{BASE_DATASET}")}},
    {"name": "hand_warmup40_k1000", "note": "budget test baseline: T=0 to 40% then linear to 0.95, at 1000 kimg",
     "setting": "base_k1000", "total_kimg": 1000, "schedule": dict(WARMUP40)},
    {"name": "hand_static045_k1000", "note": "budget test baseline: best static T=0.45, at 1000 kimg",
     "setting": "base_k1000", "total_kimg": 1000, "schedule": {"type": "static", "t_start": 0.45}},
    # optional fourth arm: the "withdraw at ~70%" reading of our 2000-kimg results, at this budget
    {"name": "hand_drop70_k1000", "note": "budget test contrast: jump to 0.95 at 70% of a 1000-kimg run",
     "setting": "base_k1000", "total_kimg": 1000,
     "schedule": {"type": "piecewise", "control_points": [[0.0, 0.0], [0.7, 0.0], [0.7, 0.95], [1.0, 0.95]]}},
    # --- failure corner, 250 clean at blur 1.0, 2000 kimg
    {"name": "exp_c250b10", "note": "failure corner: clean-exposure controller, 250 clean at blur 1.0",
     "setting": "c250b10",
     "schedule": {"type": "principled",
                  "probe": probe(controller="exposure", ctl=dict(EXPOSURE_CTL), every_kimg=200,
                                 train_dir=f"{AMBIENT_BASE}/annotated_datasets/rb_c250b10")}},
    {"name": "rb_c250b10_warmup40", "note": "failure corner baseline: warmup40", "setting": "c250b10",
     "schedule": dict(WARMUP40)},
    {"name": "rb_c250b10_static070", "note": "failure corner baseline: static T=0.70", "setting": "c250b10",
     "schedule": {"type": "static", "t_start": 0.70}},
    {"name": "rb_c250b10_static085", "note": "failure corner baseline: static T=0.85 (run whichever wins at 500 clean); carries a logging-only probe",
     "setting": "c250b10", "schedule": {"type": "static", "t_start": 0.85,
                                        "probe": log_probe("rb_c250b10")}},
]


# Round 8 (2026-09-18): the crossover test the analysis asks for.
# From the clean-only ladder (0.05467 at 250 clean, 0.04479 at 500, 0.03073 at 1000) against a
# schedule floor of ~0.0274-0.0287, blurred data stops being worth anything at roughly 1150-1250
# clean images. At 1250 the prediction is that clean-only MATCHES the schedules, i.e. our method
# has no advantage left -- a bound on the method's domain, run deliberately to find its edge.
# Dataset rb_c1250 (lysine only): the 500 b0 faces + 750 more made clean from bucket b1, which no
# run has trained on, + the same 26,014 blurred at 0.5.
PHASE8 = "auto8"
ROUND8 = [
    {"name": "exp_c1250", "note": "crossover test: clean-exposure controller at 1250 clean faces",
     "setting": "c1250",
     "schedule": {"type": "principled",
                  "probe": probe(controller="exposure", ctl=dict(EXPOSURE_CTL), every_kimg=200,
                                 train_dir=f"{AMBIENT_BASE}/annotated_datasets/rb_c1250")}},
    {"name": "rb_c1250_warmup40", "note": "crossover test baseline: warmup40", "setting": "c1250",
     "schedule": dict(WARMUP40)},
    {"name": "rb_c1250_static045", "note": "crossover test baseline: best static T=0.45", "setting": "c1250",
     "schedule": {"type": "static", "t_start": 0.45}},
    {"name": "rb_c1250_cleanonly", "note": "crossover test: clean-only floor at 1250 clean faces",
     "setting": "c1250", "schedule": dict(CLEAN_ONLY)},
]


# Round 9 (2026-09-18): two cheap tests against the user's open problems.
#
# ITEM 4 -- the ending threshold was inherited, never chosen. T_end = 0.95 is where the ~94%
# post-drop clean share comes from. Low endings are tested and bad; 0.95 against 1.0 was never run.
# With blur excluded entirely after the drop the post-drop clean share rises to ~0.98, so the
# controller re-solves its own withdrawal time (later, ~1463 kimg) rather than keeping 1439.
#
# ITEM 1 -- circularity. The target was calibrated on eight FINISHED runs; anyone who can afford
# eight finished runs does not need the method. These four arms calibrate it instead on SHORT runs:
# fixed drop fractions at a 1000-kimg budget spanning ~400 to ~1500 passes over the clean set, which
# brackets the calibrated 1109. If the optimum located from four half-length runs agrees with the
# one calibrated from eight full ones, the constant can be obtained cheaply and the circularity
# objection loses most of its force. Launch via run_budget_job.sh with 1000.
PHASE9 = "auto9"
_END100 = dict(EXPOSURE_CTL); _END100.update({"t_end": 1.0, "f_hi": 0.98})
ROUND9 = [
    {"name": "exp_base_end100", "note": "item 4: same rule, blur excluded entirely after the drop (T_end 1.0 rather than 0.95)",
     "setting": "base", "schedule": {"type": "principled",
                  "probe": probe(controller="exposure", ctl=dict(_END100), every_kimg=200,
                                 train_dir=f"{AMBIENT_BASE}/annotated_datasets/{BASE_DATASET}")}},
]
for _frac in (0.2, 0.4, 0.6, 0.8):
    ROUND9.append({"name": f"cal_k1000_drop{int(_frac*100):02d}",
                   "note": f"item 1: short-run calibration arm, jump to 0.95 at {int(_frac*100)}% of a 1000-kimg run",
                   "setting": "base_k1000", "total_kimg": 1000,
                   "schedule": {"type": "piecewise",
                                "control_points": [[0.0, 0.0], [_frac, 0.0], [_frac, 0.95], [1.0, 0.95]]}})


# Round 10 (2026-09-18): the corruption-axis boundary, and the missing case.
#
# The fine-detail LEVEL (not its slope) is an early, clean measurement of corruption severity: at
# 200 kimg it reads 0.435 / 0.289 / 0.142 of the truth's at blur 0.3 / 0.5 / 1.0, and it is nearly
# blind to clean count (spread 0.01 across 250-1250 clean). It does NOT predict the score across
# schedules within a setting -- the apparent correlation there is the restrictive-early confound
# (level@500k vs T@500k: rho 0.90).
#
# So it cannot trigger a withdrawal, but it does measure the axis our rule ignores. To use it in a
# two-sided rule we need MIND per unit of deficit, and that constant is NOT estimable from what we
# have: within a setting the deficit barely varies, and across settings MIND is not comparable.
# It becomes estimable only where the quality term dominates, which we have never measured.
# rb_b20 (blur 2.0) is that case: blur should be worth nothing, so a rule that keeps using it for
# 1437 kimg should LOSE to clean-only (0.0448, already measured and valid at any blur level).
PHASE10 = "auto10"
ROUND10 = [
    {"name": "exp_b20", "note": "corruption boundary: the rule at blur 2.0, where blur should be worthless",
     "setting": "b20",
     "schedule": {"type": "principled",
                  "probe": probe(controller="exposure", ctl=dict(EXPOSURE_CTL), every_kimg=200,
                                 train_dir=f"{AMBIENT_BASE}/annotated_datasets/rb_b20")}},
    {"name": "rb_b20_warmup40", "note": "corruption boundary baseline: warmup40 at blur 2.0",
     "setting": "b20", "schedule": dict(WARMUP40)},
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=MANIFEST); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for r in RUNS: r["phase"] = PHASE
    for r in ROUND2: r["phase"] = PHASE2
    for r in ROUND3: r["phase"] = PHASE3
    for r in ROUND4: r["phase"] = PHASE4
    for r in ROUND5: r["phase"] = PHASE5
    for r in ROUND6: r["phase"] = PHASE6
    for r in ROUND7: r["phase"] = PHASE7
    for r in ROUND8: r["phase"] = PHASE8
    for r in ROUND9: r["phase"] = PHASE9
    for r in ROUND10: r["phase"] = PHASE10
    RUNS.extend(ROUND2); RUNS.extend(ROUND3); RUNS.extend(ROUND4); RUNS.extend(ROUND5); RUNS.extend(ROUND6); RUNS.extend(ROUND7); RUNS.extend(ROUND8); RUNS.extend(ROUND9); RUNS.extend(ROUND10)
    m = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {"runs": []}
    names = {r["name"] for r in RUNS}
    m["runs"] = [e for e in m.get("runs", []) if e.get("name") not in names] + RUNS
    for r in RUNS:
        sch = r["schedule"]; pr = sch.get("probe") or {}
        kind = pr.get("controller", sch["type"])
        detail = pr.get("ctl") or sch.get("tau_points") or sch.get("control_points") or sch.get("t_start")
        print(f"  {r['name']:<24} {kind:<14} {detail}")
    if a.dry_run: return
    if os.path.exists(a.manifest): shutil.copy2(a.manifest, a.manifest + ".bak")
    json.dump(m, open(a.manifest, "w"), indent=2); print(f"wrote {a.manifest} ({len(m['runs'])} runs)")

if __name__ == "__main__":
    main()
