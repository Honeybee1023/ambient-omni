"""
Collect all v2b MIND + val loss results into a single JSON.
Output: $AMBIENT_BASE/generated/v2b_all_results.json
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
OUTPUT = os.path.join(GEN_DIR, "v2b_all_results.json")

# T suffix -> T value mapping
def suffix_to_t(suffix):
    if len(suffix) == 3:
        return int(suffix) / 100.0
    elif len(suffix) == 4:
        return int(suffix) / 1000.0
    else:
        return None

def read_mind(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("mind")
    except:
        return None

def read_val_loss(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("weighted_loss_mean")
    except:
        return None

def main():
    results = {"buckets": {}, "baseline": {}, "baseline_s1": {}, "metadata": {}}

    for b in range(1, 8):
        results["buckets"][str(b)] = {}

    # Find all MIND files matching v2b or v2 naming
    mind_files = sorted(glob.glob(os.path.join(GEN_DIR, "mind_celeba_v2*_2000kimg.json")))
    print(f"Found {len(mind_files)} MIND files")

    for mf in mind_files:
        basename = os.path.basename(mf)
        name = basename.replace("mind_", "").replace("_2000kimg.json", "")

        mind_val = read_mind(mf)
        vl_path = os.path.join(GEN_DIR, f"val_loss_{name}_2000kimg.json")
        vl_val = read_val_loss(vl_path) if os.path.exists(vl_path) else None

        entry = {}
        if mind_val is not None:
            entry["mind"] = round(mind_val, 6)
        if vl_val is not None:
            entry["val_loss"] = round(vl_val, 6)

        if not entry:
            continue

        # Classify this result
        if "baseline_s1" in name:
            results["baseline_s1"] = entry
            print(f"  baseline_s1: MIND={mind_val}, val_loss={vl_val}")
        elif "baseline" in name and "_b" not in name:
            results["baseline"] = entry
            print(f"  baseline: MIND={mind_val}, val_loss={vl_val}")
        else:
            # Parse bucket and T: celeba_v2b_b3_T045 or celeba_v2_b3_T045
            match = re.search(r'_b(\d)_T(\d{3,4})', name)
            if match:
                bucket = match.group(1)
                t_suffix = match.group(2)
                t_val = suffix_to_t(t_suffix)
                if t_val is not None and bucket in results["buckets"]:
                    results["buckets"][bucket][str(t_val)] = entry
                    print(f"  B{bucket} T={t_val}: MIND={mind_val}, val_loss={vl_val}")

    # Summary
    total = 0
    print("\n=== Summary ===")
    for b in range(1, 8):
        n = len(results["buckets"][str(b)])
        total += n
        if n > 0:
            minds = [v["mind"] for v in results["buckets"][str(b)].values() if "mind" in v]
            best_t = min(results["buckets"][str(b)].items(),
                        key=lambda x: x[1].get("mind", 999))
            print(f"  B{b}: {n} points, best MIND={best_t[1].get('mind'):.4f} at T={best_t[0]}")
    print(f"  Total sweep points: {total}")
    print(f"  Baseline: {results.get('baseline', {})}")
    print(f"  Baseline s1: {results.get('baseline_s1', {})}")

    # Metadata
    results["metadata"] = {
        "metric": "MIND (lower=better)",
        "regime": "500 clean images, CelebA 64x64",
        "blur_sigmas": {
            "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4,
            "5": 0.5, "6": 0.6, "7": 0.7
        },
        "training": "2k kimg, seed=0",
        "eval": "5K generated images for MIND, holdout for val loss"
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT}")

if __name__ == "__main__":
    main()
