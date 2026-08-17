#!/usr/bin/env python3
"""Collect conditional B2 sweep results (B1 fixed at T=0.5)."""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)


import os
import json
import glob
import re

GENERATED_DIR = f"{AMBIENT_BASE}/generated"

def decode_t_suffix(suffix):
    """T040 -> 0.4, T0525 -> 0.525"""
    digits = suffix[1:]  # strip 'T'
    if len(digits) == 3:
        return round(int(digits) / 100, 4)
    elif len(digits) == 4:
        return round(int(digits) / 1000, 4)
    raise ValueError(f"Unknown suffix: {suffix}")

def main():
    pattern = os.path.join(GENERATED_DIR, "mind_celeba_v2b_cond_b2_*_2000kimg.json")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} MIND files")

    results = {}
    for f in files:
        match = re.search(r'cond_b2_(T\d+)_', os.path.basename(f))
        if not match:
            print(f"  SKIP {f} (no T suffix found)")
            continue
        t_val = decode_t_suffix(match.group(1))
        with open(f) as fh:
            d = json.load(fh)
        results[str(t_val)] = {"mind": d["mind"]}
        print(f"  B2 T={t_val}: MIND={d['mind']:.6f}")

    out = {
        "description": "Conditional B2 sweep: B1 fixed at T=0.5, B3-B7 at T=0.999",
        "b1_fixed_t": 0.5,
        "b2_sweep": results
    }
    out_path = os.path.join(GENERATED_DIR, "v2b_cond_b2_sweep_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
