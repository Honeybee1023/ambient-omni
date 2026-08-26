"""Define the runs for the principled (online-probe) T-schedule experiments.

The discrete search found the best T(p) curve by training ~30 models and ranking
them on MIND. This batch asks whether one model can find that curve by itself:
every 100 kimg it probes its own denoiser to see where corrupted data stops
being distinguishable from clean data, and sets T there. See training/probe.py.

Runs land in the same manifest that run_dyn_job.sh already reads, under
phase="principled", so the existing job/queue machinery runs them unchanged --
the only new thing is a `probe` key on the schedule dict.

Two groups:

  gt_*   probe attached to a schedule we already trust, LOGGING ONLY. The probe
         records what it would have chosen but does not steer training. These are
         the ground truth: what do the per-sigma curves look like when the
         schedule is known to be good? Two different schedules are probed
         (warmup and static) on purpose -- if the probe reads the same boundary
         under both, the signal is a property of the model's competence rather
         than of the curriculum that produced it, which is the first thing a
         reviewer will ask.

  pr_*   closed loop: the probe sets T. One run per metric.

Every probe logs what EVERY metric/rule combination would have chosen (see
probe.COUNTERFACTUALS), so the threshold comparison mostly comes out of runs we
were going to do anyway rather than out of extra GPU-hours.

Usage:
    python principled_t_search.py                 # write manifest entries
    python principled_t_search.py --dry-run       # print, change nothing
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse
import json
import os
import shutil

MANIFEST = f"{AMBIENT_BASE}/generated/dyn_search_manifest.json"
PHASE = "principled"

# The reference schedule. warmup_linear(t_start=0, t_end=0.95, warmup_frac=0.25)
# written as control points -- tests/test_piecewise_schedule.py pins that this
# reproduces the continuous form to 2e-15, so "5 knots" costs no fidelity here.
WARMUP_0_TO_095 = [[0.0, 0.0], [0.25, 0.0], [0.5, 0.31667],
                   [0.75, 0.63333], [1.0, 0.95]]

# Shared probe settings.
#   every_kimg 100 -> 20 probes over a 2000 kimg run, as specified.
#   160 images x 2 draws x 20 levels x 2 arms = 12.8k forward passes per probe.
#   Measured, not guessed: 40.0 s per probe at 100 images on an idle H200 against
#   14.78 sec/kimg training, so the spec's 200 images would have cost 5.4% of
#   wall clock -- just over the 5% budget. 160 lands at ~4.3% and costs only a
#   12% wider standard error on the paired gap, which the common-random-numbers
#   design has already made small. Each run also reports its own cost live as
#   `probe_ovh` in the tick line, so this is checked rather than assumed.
#   batch_size 80 rather than 200: at 200 the probe pushed peak GPU memory from
#   ~30 GB (training alone) to 43 GB, which would stop two jobs fitting on one
#   80 GB A100. At 80 the peak is back to 29.97 GB -- training's own -- and 160
#   images split into two even batches with no ragged tail.
#   alpha 0.3 smooths the grid-quantised raw T over probes.
#   monotone stays FALSE: the discrete search already told us the answer is
#   non-decreasing, so enforcing it would hand the method its result and make
#   "it discovered warmup" circular. Available as an ablation.
PROBE_BASE = {
    "every_kimg": 100,
    "n_images": 160,
    "n_draws": 2,
    "n_levels": 20,
    "batch_size": 80,
    "probe_seed": 12345,
    "alpha": 0.3,
    "monotone": False,
    "t_init": 0.0,
}


def probe(**overrides):
    cfg = dict(PROBE_BASE)
    cfg.update(overrides)
    return cfg


RUNS = [
    # -- ground truth: probe rides along, does not steer -------------------
    {
        "name": "gt_warmup",
        "note": "known-good warmup 0->0.95; probe logs only. The reference "
                "trajectory the discovered ones are compared against.",
        "schedule": {"type": "piecewise", "control_points": WARMUP_0_TO_095,
                     "probe": probe()},
    },
    {
        "name": "gt_static50",
        "note": "static T=0.50; probe logs only. Tests whether the probe's "
                "reading depends on the curriculum that produced the model.",
        "schedule": {"type": "static", "t_start": 0.50, "probe": probe()},
    },

    # -- closed loop: one run per metric -----------------------------------
    {
        "name": "pr_skill",
        "note": "energy-corrected MSE (blind_mse). The MSE metric whose null is "
                "genuinely 1.0 at every sigma.",
        "schedule": {"type": "principled",
                     "probe": probe(metric="skill_ratio", rule="baseline", threshold=1.02)},
    },
    {
        "name": "pr_predvar",
        "note": "prediction variance across noise draws; nuisance-free by "
                "construction (exactly 1.0 for a blind denoiser).",
        "schedule": {"type": "principled",
                     "probe": probe(metric="pred_var", rule="fixed", threshold=1.05)},
    },
    {
        "name": "pr_lossratio",
        "note": "raw per-sigma loss ratio, as specified. Uncorrected, so it "
                "carries the pixel-energy nuisance -- included as the "
                "comparison that shows why the correction matters.",
        "schedule": {"type": "principled",
                     "probe": probe(metric="loss_ratio", rule="fixed", threshold=1.05)},
    },
    {
        "name": "pr_msegap",
        "note": "raw MSE gap as a paired t-statistic. Also uncorrected.",
        "schedule": {"type": "principled",
                     "probe": probe(metric="mse_gap", rule="adaptive", threshold=2.0)},
    },
]

# Rule ablations. Deliberately NOT queued by default: which thresholds are even
# reachable is answered for free by the offline calibration
# (probe_checkpoint.py) and by the gt_* counterfactual logs. Queueing them blind
# risks spending 9 GPU-hours on a threshold the metric never crosses, which
# would train a model at a constant T and tell us nothing.
ABLATIONS = [
    {
        "name": "pr_skill_mono",
        "note": "as pr_skill but T forced non-decreasing.",
        "schedule": {"type": "principled",
                     "probe": probe(metric="skill_ratio", rule="baseline",
                                    threshold=1.02, monotone=True)},
    },
    {
        "name": "pr_skill_nosmooth",
        "note": "as pr_skill with no EMA over probes; isolates how much of the "
                "trajectory's shape is smoothing rather than signal.",
        "schedule": {"type": "principled",
                     "probe": probe(metric="skill_ratio", rule="baseline",
                                    threshold=1.02, alpha=1.0)},
    },
    {
        "name": "pr_skill_pct",
        "note": "as pr_skill with the percentile boundary scan instead of "
                "last-crossing; tolerates stray levels.",
        "schedule": {"type": "principled",
                     "probe": probe(metric="skill_ratio", rule="percentile",
                                    threshold=1.02, q=0.90)},
    },
    {
        "name": "pr_skill_fast",
        "note": "as pr_skill probing every 50 kimg; does a faster loop help or "
                "just add jitter?",
        "schedule": {"type": "principled",
                     "probe": probe(metric="skill_ratio", rule="baseline",
                                    threshold=1.02, every_kimg=50)},
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--include-ablations", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = list(RUNS) + (list(ABLATIONS) if args.include_ablations else [])
    for r in runs:
        r["phase"] = PHASE

    existing = {"runs": []}
    if os.path.exists(args.manifest):
        with open(args.manifest) as f:
            existing = json.load(f)

    names = {r["name"] for r in runs}
    # Replace our own entries, leave every other phase alone. The discrete
    # search's 30 runs live in this same file and must survive untouched.
    kept = [e for e in existing.get("runs", []) if e.get("name") not in names]
    merged = dict(existing)
    merged["runs"] = kept + runs

    print(f"manifest : {args.manifest}")
    print(f"existing : {len(existing.get('runs', []))} runs "
          f"({len(existing.get('runs', [])) - len(kept)} of them ours, replaced)")
    print(f"adding   : {len(runs)} runs under phase={PHASE!r}\n")
    for r in runs:
        s = r["schedule"]
        p = s["probe"]
        drives = "DRIVES T " if s["type"] == "principled" else "logs only"
        rule = f"{p.get('metric', '-')}/{p.get('rule', '-')}/{p.get('threshold', '-')}"
        print(f"  {r['name']:<18} {drives}  every {p['every_kimg']:>3} kimg  "
              f"{rule:<32} {r['note'].splitlines()[0][:44]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if os.path.exists(args.manifest):
        shutil.copy2(args.manifest, args.manifest + ".bak")
        print(f"\nbacked up -> {args.manifest}.bak")
    with open(args.manifest, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {args.manifest} ({len(merged['runs'])} runs total)")
    print(f"\nQueue them with:  bash run_dyn_queue.sh init {PHASE}")


if __name__ == "__main__":
    main()
