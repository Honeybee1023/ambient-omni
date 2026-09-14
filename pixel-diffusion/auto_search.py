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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=MANIFEST); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for r in RUNS: r["phase"] = PHASE
    for r in ROUND2: r["phase"] = PHASE2
    for r in ROUND3: r["phase"] = PHASE3
    RUNS.extend(ROUND2); RUNS.extend(ROUND3)
    m = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {"runs": []}
    names = {r["name"] for r in RUNS}
    m["runs"] = [e for e in m.get("runs", []) if e.get("name") not in names] + RUNS
    for r in RUNS: print(f"  {r['name']:<22} {r['schedule']['probe'].get('controller', r['schedule']['type']):<14} {r['schedule']['probe'].get('ctl', r['schedule'].get('tau_points', r['schedule'].get('control_points')))}")
    if a.dry_run: return
    if os.path.exists(a.manifest): shutil.copy2(a.manifest, a.manifest + ".bak")
    json.dump(m, open(a.manifest, "w"), indent=2); print(f"wrote {a.manifest} ({len(m['runs'])} runs)")

if __name__ == "__main__":
    main()
