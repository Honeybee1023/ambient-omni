"""Manual schedule search inside the user's parameterisation (2026-09-13).

Parameters: b = where the T=0 stretch ends; T_end in [0.8, 1.0]; any number of
free control points between b and the end, anywhere in [0, 1]. Round 1 builds on
the current best, shape_b50_concave = [[0,0],[0.5,0],[0.75,0.75],[1,0.95]]
(MIND 0.0285, n=1): bend it higher, bend it earlier, end at exactly 1.0, and
move the blade end. Piecewise-linear control points, as in shape_search.py.

Usage: python manual_search.py [--dry-run]   # adds phase "manual" entries to the manifest
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")
import argparse, json, os, shutil
MANIFEST = f"{AMBIENT_BASE}/generated/dyn_search_manifest.json"
PHASE = "manual"

def pw(points):
    return {"type": "piecewise", "control_points": [[float(a), float(b)] for a, b in points]}

ROUND1 = [
    ("man_b50_bend85",        "winner bent higher: 0.85 at 75%",               pw([(0,0),(0.5,0),(0.75,0.85),(1,0.95)])),
    ("man_b50_bend95",        "winner bent to the ceiling: 0.95 at 75%, flat after", pw([(0,0),(0.5,0),(0.75,0.95),(1,0.95)])),
    ("man_b50_bendearly",     "winner bent earlier: 0.75 at 62.5%",            pw([(0,0),(0.5,0),(0.625,0.75),(1,0.95)])),
    ("man_b50_concave_end100","winner ending at exactly 1.0",                   pw([(0,0),(0.5,0),(0.75,0.75),(1,1.0)])),
    ("man_b40_concave",       "winner shape with the blade ending at 40%",     pw([(0,0),(0.4,0),(0.7,0.75),(1,0.95)])),
    ("man_b60_concave",       "winner shape with the blade ending at 60%",     pw([(0,0),(0.6,0),(0.8,0.75),(1,0.95)])),
]
ROUND2 = []   # filled after round 1
RUNS = [{"name": n, "note": note, "schedule": s} for n, note, s in ROUND1 + ROUND2]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=MANIFEST); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for r in RUNS: r["phase"] = PHASE
    existing = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {"runs": []}
    names = {r["name"] for r in RUNS}
    merged = dict(existing); merged["runs"] = [e for e in existing.get("runs", []) if e.get("name") not in names] + RUNS
    for r in RUNS: print(f"  {r['name']:<24} {r['schedule']['control_points']}")
    if a.dry_run: return
    if os.path.exists(a.manifest): shutil.copy2(a.manifest, a.manifest + ".bak")
    json.dump(merged, open(a.manifest, "w"), indent=2); print(f"wrote {a.manifest} ({len(merged['runs'])} runs)")

if __name__ == "__main__":
    main()
