"""Collect all v2 coarse sweep MIND + val loss results into a single summary JSON."""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json

GEN_DIR = f"{AMBIENT_BASE}/generated"
MANIFEST_PATH = f"{AMBIENT_BASE}/annotated_datasets/celeba_v2_coarse_manifest.json"
OUT_PATH = os.path.join(GEN_DIR, "v2_coarse_results.json")

BLUR_SIGMAS = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.75, 5: 1.0, 6: 1.5, 7: 2.0}

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

results = {"baseline": {}, "sweeps": {}}

# Baseline
bl_mind = os.path.join(GEN_DIR, "mind_celeba_v2_baseline_2000kimg.json")
bl_vl = os.path.join(GEN_DIR, "val_loss_celeba_v2_baseline_2000kimg.json")
if os.path.exists(bl_mind):
    with open(bl_mind) as f:
        results["baseline"]["mind"] = json.load(f)["mind"]
if os.path.exists(bl_vl):
    with open(bl_vl) as f:
        results["baseline"]["val_loss"] = json.load(f)["weighted_loss_mean"]

# Sweep results
for b in range(1, 8):
    bkey = f"B{b}"
    results["sweeps"][bkey] = {"sigma": BLUR_SIGMAS[b], "points": []}

    for name, info in manifest["datasets"].items():
        if name == "celeba_v2_baseline":
            continue
        if info.get("active_buckets") != [b]:
            continue

        t_val = info["active_t"]
        entry = {"t": t_val}

        mind_path = os.path.join(GEN_DIR, f"mind_{name}_2000kimg.json")
        vl_path = os.path.join(GEN_DIR, f"val_loss_{name}_2000kimg.json")

        if os.path.exists(mind_path):
            with open(mind_path) as f:
                entry["mind"] = json.load(f)["mind"]
        if os.path.exists(vl_path):
            with open(vl_path) as f:
                entry["val_loss"] = json.load(f)["weighted_loss_mean"]

        results["sweeps"][bkey]["points"].append(entry)

    results["sweeps"][bkey]["points"].sort(key=lambda x: x["t"])

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("Baseline: MIND=%.6f, val_loss=%.6f" % (
    results["baseline"].get("mind", float('nan')),
    results["baseline"].get("val_loss", float('nan'))))
print()
for bkey in sorted(results["sweeps"]):
    info = results["sweeps"][bkey]
    print(f"{bkey} (sigma={info['sigma']}):")
    for p in info["points"]:
        mind = p.get("mind", float("nan"))
        vl = p.get("val_loss", float("nan"))
        print(f"  T={p['t']:.2f}: MIND={mind:.6f}, val_loss={vl:.6f}")
