#!/usr/bin/env python3
"""Arms of the mixed-blur go/no-go study (dataset celeba_mix4_v1), registered into the
CSAIL job manifest that run_dyn_job.sh reads.

Groups: g03 / g05 / g10 / g20 = blur 0.3 / 0.5 / 1.0 / 2.0, ~6,500 faces each, + 500 clean.
Every arm is a per_group schedule (see training_loop.per_group_state).

Question: does any per-bucket schedule beat the best baseline by >= 3 noise units
(~0.003 MIND), holding on a second seed?  Yes -> build the automatic function.
No -> stop this direction.

Baselines that need no classifier are here. The Ambient-o arms (per-image thresholds,
and the three Ambient-o + fine-tune orderings) and Dataloops are added once the
classifier has annotated the dataset.

Usage (on CSAIL):  python mix4_manifest.py            # merge arms into the manifest
                   python mix4_manifest.py --list     # print them
Submit:            DYN_DATASET=celeba_mix4_v1 bash submit_dyn_csail.sh <names...>
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json, os, sys

MANIFEST = f"{AMBIENT_BASE}/generated/dyn_search_manifest.json"
G = ("g03", "g05", "g10", "g20")


def same(spec):
    return {"type": "per_group", "groups": {g: spec for g in G}}


def each(**specs):
    return {"type": "per_group", "groups": {g: specs[g] for g in G}}


def step(frac, end=0.95):
    """Blur at every noise level until `frac`, then threshold `end` (the ledger's best
    shape: hold T=0, then jump)."""
    return {"phases": [[0, 0.0], [frac, end]]}


ARMS = [
    # --- baselines ---
    {"name": "mix4_cleanonly", "schedule": same({"phases": [[0, "off"]]}),
     "note": "baseline 1: clean only, rerun in this batch (500 clean are the ledger's)"},
    {"name": "mix4_static", "note": "baseline 2: static per bucket at each blur's single-bucket optimum "
                                     "(0.3->0.50 v2b, 0.5->0.50 ledger n=3, 1.0/2.0->0.95 v2)",
     "schedule": each(g03={"phases": [[0, 0.50]]}, g05={"phases": [[0, 0.50]]},
                      g10={"phases": [[0, 0.95]]}, g20={"phases": [[0, 0.95]]})},
    *[{"name": f"mix4_finetune{int(f * 100)}", "schedule": same({"phases": [[0, 0.0], [f, "off"]]}),
       "note": f"baseline 4: all data at every noise level, then clean only from {f:.0%}"}
      for f in (0.60, 0.75, 0.90)],
    # --- per-bucket candidates (exploit-first) ---
    {"name": "mix4_c1_global72", "schedule": same(step(0.72)),
     "note": "anchor: one global step at 72% (the exposure rule's drop in the base setting)"},
    {"name": "mix4_c2_stagger", "schedule": each(g03=step(0.85), g05=step(0.75), g10=step(0.65), g20=step(0.55)),
     "note": "heavier blur withdrawn earlier (more bias to forget), 10% apart"},
    {"name": "mix4_c3_stagger_wide", "schedule": each(g03=step(0.85), g05=step(0.70), g10=step(0.55), g20=step(0.40)),
     "note": "same ordering, 15% apart"},
    {"name": "mix4_c4_stagger_mildkeep",
     "schedule": each(g03=step(0.85, 0.50), g05=step(0.75, 0.50), g10=step(0.65), g20=step(0.55)),
     "note": "c2, but the mild buckets end at their static optimum 0.50 instead of 0.95"},
    {"name": "mix4_c5_heavy_early",
     "schedule": each(g03=step(0.75), g05=step(0.75),
                      g10={"phases": [[0, 0.0], [0.40, "off"]]}, g20={"phases": [[0, 0.0], [0.40, "off"]]}),
     "note": "heavy buckets as early-only data (off at 40%), mild buckets step at 75%"},
]


# --- Ambient-o arms: dataset celeba_mix4_ambo (per-image classifier annotations, paper recipe).
# Submit with  DYN_DATASET=celeba_mix4_ambo TRAIN_EXTRA=--cls_ema_window=1  because the annotation
# used a 64-point sigma grid, where train.py's default window of 32 (meant for 2048 points) would
# smooth over half the grid. Switch fractions match the middle fine-tune arm (75%).
AMBO_ARMS = [
    {"name": "mix4_ambo", "schedule": same({"phases": [[0, "annot"]]}),
     "note": "baseline 3: Ambient-o, per-image classifier thresholds, fixed for the run"},
    {"name": "mix4_ambo_then_clean", "schedule": same({"phases": [[0, "annot"], [0.75, "off"]]}),
     "note": "baseline 5: Ambient-o thresholds, then clean only from 75%"},
    {"name": "mix4_all_then_ambo", "schedule": same({"phases": [[0, 0.0], [0.75, "annot"]]}),
     "note": "baseline 6: all data at every noise level, then Ambient-o thresholds from 75%"},
    {"name": "mix4_all_ambo_clean", "schedule": same({"phases": [[0, 0.0], [0.40, "annot"], [0.75, "off"]]}),
     "note": "baseline 7: all data, Ambient-o thresholds from 40%, clean only from 75%"},
]
ARMS = ARMS + [dict(a, dataset="celeba_mix4_ambo") for a in AMBO_ARMS]

# --- Dataloops (baseline 8): loop 1 of the official recipe. dataloops_restore.py restores every
# blurred image with the loop-0 model (mix4_ambo) from its sigma_tn to zero and labels it at
# sigma_tn/8; loop 1 trains a NEW model from scratch on that set with those labels fixed.
# Annotations are plain sigmas, so no --cls_ema_window is needed.  DYN_DATASET=celeba_mix4_loop1
ARMS = ARMS + [{"name": "mix4_dataloops1", "dataset": "celeba_mix4_loop1",
                "schedule": same({"phases": [[0, "annot"]]}),
                "note": "baseline 8: Ambient Dataloops loop 1 (restore sigma_tn -> 0, label sigma_tn/8, retrain)"}]


# --- Continuous per-level curves (proposed with the user 2026-09-29). Hold T=0, then ramp.
def cp(*pts):
    return {"control_points": [list(q) for q in pts]}
_MILD = {"g03": cp((0, 0), (0.60, 0), (1, 0.95)), "g05": cp((0, 0), (0.50, 0), (1, 0.95))}
_HEAVY_OFF = {"g10": cp((0, 0), (0.35, 0), (0.50, 1.0), (1, 1.0)), "g20": cp((0, 0), (0.30, 0), (0.45, 1.0), (1, 1.0))}
ARMS = ARMS + [
    {"name": "mix4_p1_ramp_stagger", "schedule": each(**_MILD, g10=cp((0, 0), (0.40, 0), (1, 0.95)), g20=cp((0, 0), (0.30, 0), (1, 0.95))),
     "note": "P1: hold 0 then linear to 0.95 at the end; ramp starts 0.3->60%, 0.5->50%, 1.0->40%, 2.0->30%"},
    {"name": "mix4_p3_smooth_c5", "schedule": each(**_MILD, **_HEAVY_OFF),
     "note": "P3: smooth c5 -- mild as P1, heavy ramp to off over 30-45% (2.0) / 35-50% (1.0)"},
    {"name": "mix4_p4_concave", "schedule": each(g03=cp((0, 0), (0.60, 0), (0.80, 0.75), (1, 0.95)),
                                                  g05=cp((0, 0), (0.50, 0), (0.75, 0.75), (1, 0.95)), **_HEAVY_OFF),
     "note": "P4: P3 with the mild levels on the concave shape (0.75 by 75%/80%, then 0.95)"},
]

# --- Last hand-picked family (user, 2026-09-30): every level at 0 early, linear to 0.95 ending at
# 75%, then flat; heavier blur starts its ramp earlier.
def _end75(s03, s05, s10, s20):
    r = lambda st: cp((0, 0), (st, 0), (0.75, 0.95), (1, 0.95))
    return each(g03=r(s03), g05=r(s05), g10=r(s10), g20=r(s20))
ARMS = ARMS + [
    {"name": "mix4_q1_end75", "schedule": _end75(0.70, 0.65, 0.55, 0.45),
     "note": "Q1: all reach 0.95 at 75%; ramps start 0.3->70%, 0.5->65%, 1.0->55%, 2.0->45%"},
    {"name": "mix4_q2_end75_steep", "schedule": _end75(0.73, 0.70, 0.62, 0.55),
     "note": "Q2: as Q1 but steeper; ramps start 0.3->73%, 0.5->70%, 1.0->62%, 2.0->55%"},
]

# --- Fair Dataloops (user, 2026-09-30): the official CIFAR recipe's FIXED sigma_min per corruption
# (~2-2.4x the blur strength: 0.6->1.2, 0.8->1.9, 1.0->2.4), extrapolated to our levels, instead of
# the classifier annotations that are blind to blur 0.3. Loop 0 doubles as Ambient-o fixed annotation.
ARMS = ARMS + [
    {"name": "mix4_dlfix_loop0", "dataset": "celeba_mix4_fixed", "schedule": same({"phases": [[0, "annot"]]}),
     "note": "Ambient-o fixed annotation = Dataloops loop 0: sigma_min 0.7/1.1/2.4/4.8 for blur 0.3/0.5/1.0/2.0"},
    {"name": "mix4_dlfix_loop1", "dataset": "celeba_mix4_fixed_loop1", "schedule": same({"phases": [[0, "annot"]]}),
     "note": "Dataloops loop 1 from the fixed-annotation loop 0 (restore sigma -> 0, relabel sigma/8, retrain)"},
]

# --- Half-length (1000 kimg) references for the policy search, which runs at 1000 kimg.
# Submit with RUN_KIMG=1000 (the ambo one also with its dataset and TRAIN_EXTRA=--cls_ema_window=1).
ARMS = ARMS + [
    {"name": "mix4h_c1", "schedule": same(step(0.72)), "note": "c1 at 1000 kimg (policy-search reference)"},
    {"name": "mix4h_c5", "schedule": each(g03=step(0.75), g05=step(0.75), g10={"phases": [[0, 0.0], [0.40, "off"]]},
                                          g20={"phases": [[0, 0.0], [0.40, "off"]]}), "note": "c5 at 1000 kimg (policy-search reference)"},
    {"name": "mix4h_all_ambo_clean", "dataset": "celeba_mix4_ambo",
     "schedule": same({"phases": [[0, 0.0], [0.40, "annot"], [0.75, "off"]]}),
     "note": "best baseline at 1000 kimg (policy-search reference)"},
]


def main():
    if "--list" in sys.argv:
        for a in ARMS:
            print(f"{a['name']:28s} {a['note']}")
        return
    m = json.load(open(MANIFEST))
    have = {r["name"]: r for r in m["runs"]}
    added = 0
    for a in ARMS:
        entry = dict({"dataset": "celeba_mix4_v1"}, **a, study="mix4")
        if a["name"] in have:
            if have[a["name"]].get("schedule") != a["schedule"]:
                sys.exit(f"{a['name']} already in the manifest with a DIFFERENT schedule; refusing")
            continue
        m["runs"].append(entry)
        added += 1
    tmp = MANIFEST + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, MANIFEST)
    print(f"added {added} arms ({len(ARMS) - added} already present); manifest now {len(m['runs'])} runs")


if __name__ == "__main__":
    main()
