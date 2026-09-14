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
ROUND2 = [   # coverage round (2026-09-13): fill the empty cells of (blade end, T_end, middle shape)
    ("man_b75_linear",     "very late blade: T=0 until 75%, straight to 0.95",            pw([(0,0),(0.75,0),(1,0.95)])),
    ("man_b90_linear",     "extreme late blade: T=0 until 90%, straight to 0.95",         pw([(0,0),(0.9,0),(1,0.95)])),
    ("man_b75_concave",    "late blade, concave: 0.75 at 87.5%",                          pw([(0,0),(0.75,0),(0.875,0.75),(1,0.95)])),
    ("man_b50_end80",      "blade to 50%, straight to T_end = 0.80",                      pw([(0,0),(0.5,0),(1,0.80)])),
    ("man_b50_end90",      "blade to 50%, straight to T_end = 0.90",                      pw([(0,0),(0.5,0),(1,0.90)])),
    ("man_b25_end100",     "blade to 25%, straight to T_end = 1.0 exactly",               pw([(0,0),(0.25,0),(1,1.0)])),
    ("man_b50_overshoot",  "blade to 50%, overshoot to 1.0 at 70%, settle at 0.90",       pw([(0,0),(0.5,0),(0.7,1.0),(1,0.90)])),
    ("man_b25_dip",        "blade to 25%, rise to 0.6, dip to 0.2 at 65%, rise to 0.95",  pw([(0,0),(0.25,0),(0.5,0.6),(0.65,0.2),(1,0.95)])),
    ("man_noblade_030",    "no blade: start at T=0.3, straight to 0.95",                  pw([(0,0.3),(1,0.95)])),
    ("man_b10_plateau50",  "short blade to 10%, plateau at 0.5 from 30% to 70%, then 0.95", pw([(0,0),(0.1,0),(0.3,0.5),(0.7,0.5),(1,0.95)])),
]
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
