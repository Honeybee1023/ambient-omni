"""
Collect v2b independence test results (shift tests + conditional sweep).
Idempotent — re-run after new jobs finish to pick up new results.
Output: $AMBIENT_BASE/generated/v2b_indep_results.json
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import json, os, glob, re

GEN_DIR = f"{AMBIENT_BASE}/generated"
OUTPUT = os.path.join(GEN_DIR, "v2b_indep_results.json")
ALL_RESULTS = os.path.join(GEN_DIR, "v2b_all_results.json")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def suffix_to_t(suffix):
    if len(suffix) == 3:
        return int(suffix) / 100.0
    elif len(suffix) == 4:
        return int(suffix) / 1000.0
    return None


def get_references():
    """Pull baseline + B1/B2 argmin from v2b_all_results.json."""
    ref = {}
    data = read_json(ALL_RESULTS)
    if data is None:
        print("WARNING: cannot read %s — references will be empty" % ALL_RESULTS)
        return ref

    if data.get("baseline"):
        ref["baseline"] = data["baseline"]

    for b in ["1", "2"]:
        pts = data.get("buckets", {}).get(b, {})
        if pts:
            best_t, best_entry = min(pts.items(),
                                     key=lambda x: x[1].get("mind", 999))
            ref["b%s_argmin" % b] = {"T": float(best_t), **best_entry}

    return ref


def collect_conditional():
    """Collect cond_b1_T* results (B2=0.55 fixed, sweep B1)."""
    points = {}
    pattern = os.path.join(GEN_DIR, "mind_celeba_v2b_cond_b1_T*_2000kimg.json")
    for mf in sorted(glob.glob(pattern)):
        basename = os.path.basename(mf)
        match = re.search(r'cond_b1_T(\d{3,4})_2000kimg', basename)
        if not match:
            continue
        t_suffix = match.group(1)
        t_val = suffix_to_t(t_suffix)
        if t_val is None:
            continue

        name = basename.replace("mind_", "").replace("_2000kimg.json", "")
        mind_data = read_json(mf)
        vl_path = os.path.join(GEN_DIR, "val_loss_%s_2000kimg.json" % name)
        vl_data = read_json(vl_path)

        entry = {}
        if mind_data and "mind" in mind_data:
            entry["mind"] = round(mind_data["mind"], 6)
        if vl_data and "weighted_loss_mean" in vl_data:
            entry["val_loss"] = round(vl_data["weighted_loss_mean"], 6)
        if entry:
            points[str(t_val)] = entry
            print("  cond B1=%.3f: MIND=%s val_loss=%s" % (
                t_val, entry.get("mind"), entry.get("val_loss")))

    return {
        "description": "B2 fixed at T=0.55, sweep B1 across T values",
        "fixed": {"B2": 0.55},
        "swept_bucket": "B1",
        "points": points
    }


def collect_shifts():
    """Collect shift_* results (combinatorial B1+B2 configurations)."""
    # Known configurations from create_v2b_indep_datasets.py
    configs = {
        "bothup": {"B1": 0.6,   "B2": 0.65},
        "bothdn": {"B1": 0.4,   "B2": 0.45},
        "apart":  {"B1": 0.4,   "B2": 0.65},
        "close":  {"B1": 0.525, "B2": 0.525},
    }
    tests = {}
    for label, config in configs.items():
        name = "celeba_v2b_shift_%s" % label
        mf = os.path.join(GEN_DIR, "mind_%s_2000kimg.json" % name)
        vl_path = os.path.join(GEN_DIR, "val_loss_%s_2000kimg.json" % name)

        if not os.path.exists(mf):
            print("  shift_%s: NOT YET AVAILABLE" % label)
            continue

        mind_data = read_json(mf)
        vl_data = read_json(vl_path)

        entry = {"config": config}
        if mind_data and "mind" in mind_data:
            entry["mind"] = round(mind_data["mind"], 6)
        if vl_data and "weighted_loss_mean" in vl_data:
            entry["val_loss"] = round(vl_data["weighted_loss_mean"], 6)
        tests[label] = entry
        print("  shift_%s (B1=%.3f, B2=%.3f): MIND=%s val_loss=%s" % (
            label, config["B1"], config["B2"],
            entry.get("mind"), entry.get("val_loss")))

    return tests


def main():
    print("=== Collecting v2b independence test results ===\n")

    print("Conditional sweep (B2=0.55 fixed, sweep B1):")
    cond = collect_conditional()

    print("\nShift tests:")
    shifts = collect_shifts()

    print("\nReferences from v2b_all_results.json:")
    refs = get_references()
    for k, v in refs.items():
        print("  %s: %s" % (k, v))

    results = {
        "conditional_sweep": cond,
        "shift_tests": shifts,
        "reference": refs,
        "metadata": {
            "description": "v2b independence tests: does optimal B1 T shift when B2 is active?",
            "test2_description": "4 shift configs testing B1+B2 combinations vs pairwise best",
            "test3_description": "B2=0.55 fixed, B1 swept 0.0-0.95 to find conditional argmin",
            "pairwise_best": "cond_b1_T050 = B1@0.5 + B2@0.55 (Test 1)",
            "metric": "MIND (lower=better)",
            "training": "2k kimg, seed=0",
        }
    }

    n_cond = len(cond["points"])
    n_shift = len(shifts)
    print("\n=== Summary: %d conditional points, %d/4 shift tests ===" % (n_cond, n_shift))

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to %s" % OUTPUT)


if __name__ == "__main__":
    main()
