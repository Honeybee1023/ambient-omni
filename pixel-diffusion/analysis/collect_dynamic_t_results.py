#!/usr/bin/env python
"""Collect dynamic-T v2 results into one table.

    python analysis/collect_dynamic_t_results.py

Reads generated/mind_v2_*.json + fid_v2_*.json and prints every finished
experiment sorted by MIND, with the deltas that matter for the paper's claim:
does any dynamic schedule beat the best static T?

Interpretation guard: our measured seed-to-seed noise floor is ~0.001 MIND, so
differences below ~0.002 are NOT results at a single seed.
"""

import glob
import json
import os
import re

GENDIR = "/data/honjar/generated"
NOISE_FLOOR = 0.001

# Known reference points measured on this machine (static b5 sweep, seed 0).
REFERENCE = {
    "validate_T0475": ("celeba_v2b_b5_T0475 (static, pre-existing)", 0.035401),
}

STATIC_PREFIXES = ("clean_only", "static_", "validate_")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(GENDIR, "mind_v2_*.json"))):
        name = re.sub(r"^mind_v2_|\.json$", "", os.path.basename(p))
        d = load(p)
        if not d or "mind" not in d:
            continue
        tag, seed = (name.rsplit("_s", 1) + ["?"])[:2]
        fid = load(os.path.join(GENDIR, f"fid_v2_{name}.json")) or {}
        rows.append({
            "tag": tag,
            "seed": seed,
            "mind": d["mind"],
            "fid": fid.get("fid_score"),
            "static": tag.startswith(STATIC_PREFIXES),
        })

    if not rows:
        print("No finished experiments yet.")
        return

    rows.sort(key=lambda r: r["mind"])

    print(f"{'experiment':<26} {'seed':>4} {'MIND':>9} {'FID':>8}   kind")
    print("-" * 66)
    for r in rows:
        fid = f"{r['fid']:8.3f}" if isinstance(r["fid"], (int, float)) else "       -"
        print(f"{r['tag']:<26} {r['seed']:>4} {r['mind']:9.5f} {fid}   "
              f"{'static' if r['static'] else 'DYNAMIC'}")

    # Validation gate.
    print()
    for tag, (desc, expected) in REFERENCE.items():
        got = next((r["mind"] for r in rows if r["tag"] == tag), None)
        if got is None:
            print(f"[gate] {tag}: not finished yet")
            continue
        delta = abs(got - expected)
        ok = delta < 2 * NOISE_FLOOR
        print(f"[gate] {tag}: {got:.5f} vs {expected:.5f} ({desc})")
        print(f"       delta {delta:.5f} -> {'PASS' if ok else 'FAIL - pipeline suspect'}")

    statics = [r for r in rows if r["static"]]
    dynamics = [r for r in rows if not r["static"]]
    if statics and dynamics:
        best_s = min(statics, key=lambda r: r["mind"])
        best_d = min(dynamics, key=lambda r: r["mind"])
        gap = best_s["mind"] - best_d["mind"]
        print()
        print(f"best static : {best_s['tag']:<24} {best_s['mind']:.5f}")
        print(f"best dynamic: {best_d['tag']:<24} {best_d['mind']:.5f}")
        print(f"dynamic - static = {-gap:+.5f}")
        if gap > 2 * NOISE_FLOOR:
            print("  -> dynamic ahead by more than the noise floor; replicate with more seeds.")
        elif gap < -2 * NOISE_FLOOR:
            print("  -> static ahead by more than the noise floor.")
        else:
            print(f"  -> WITHIN NOISE (|gap| < {2*NOISE_FLOOR}). Not a result yet.")


if __name__ == "__main__":
    main()
