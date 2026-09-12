"""Shape study: does the *shape* of the withdrawal matter, holding its start and end fixed?

Every good schedule so far is a "hockey stick": T = 0 for a while (the blade),
then a rise to ~0.95 by the end (the stick). This study pins the two knots the
blade and the stick share -- where T leaves 0 (blade end, b) and where it ends
(0.95 at the last kimg) -- and varies only what happens in between:

    linear   the straight line from (b, 0) to (1, 0.95)      [already run]
    convex   below that line: slow at first, steep at the end
    concave  above that line: steep at first, flat at the end
    step     the extreme concave case: jump to 0.95 at b
    wobble   non-monotone, up and down around the line (b = 0.5 only)

for b = 0.25 and b = 0.50. Hypothesis on record before running (user's): convex
beats linear beats concave beats step, at both b. Every automatic rule we have
produced a concave curve; if concave is genuinely worse this is the reason
those rules fail even when their start and end are right.

Usage:
    python shape_search.py            # add the entries to the manifest
    python shape_search.py --dry-run
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse, json, os, shutil

MANIFEST = f"{AMBIENT_BASE}/generated/dyn_search_manifest.json"
PHASE = "shape"
END = 0.95


def pw(points):
    return {"type": "piecewise", "control_points": [[float(a), float(b)] for a, b in points]}


def family(b):
    mid = b + (1 - b) / 2
    return [
        (f"shape_b{int(b*100)}_convex",  f"T=0 until {b:.0%}, then BELOW the straight line to 0.95 (0.2 at the midpoint)",
         pw([(0, 0), (b, 0), (mid, 0.2), (1, END)])),
        (f"shape_b{int(b*100)}_concave", f"T=0 until {b:.0%}, then ABOVE the straight line to 0.95 (0.75 at the midpoint)",
         pw([(0, 0), (b, 0), (mid, 0.75), (1, END)])),
        (f"shape_b{int(b*100)}_step",    f"T=0 until {b:.0%}, then jump straight to 0.95",
         pw([(0, 0), (b, 0), (b, END), (1, END)])),
    ]


RUNS = []
for b in (0.25, 0.50):
    for name, note, sched in family(b):
        RUNS.append({"name": name, "note": note, "schedule": sched})
RUNS.append({"name": "shape_b50_wobble",
             "note": "T=0 until 50%, then up-down-up around the straight line: 0.5, 0.2, 0.7, 0.95",
             "schedule": pw([(0, 0), (0.5, 0), (0.625, 0.5), (0.75, 0.2), (0.875, 0.7), (1, END)])})
# Reference rows already measured: linear b=0.25 = warmup25 (0.0296, n=3);
# linear b=0.50 = sched_hold50_ceil95 (0.0296, n=2); old-build step b=0.50 = 0.0346 (n=1).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for r in RUNS:
        r["phase"] = PHASE
    existing = json.load(open(a.manifest)) if os.path.exists(a.manifest) else {"runs": []}
    names = {r["name"] for r in RUNS}
    kept = [e for e in existing.get("runs", []) if e.get("name") not in names]
    merged = dict(existing); merged["runs"] = kept + RUNS
    for r in RUNS:
        print(f"  {r['name']:<20} {r['schedule']['control_points']}")
    if a.dry_run:
        return
    if os.path.exists(a.manifest):
        shutil.copy2(a.manifest, a.manifest + ".bak")
    json.dump(merged, open(a.manifest, "w"), indent=2)
    print(f"wrote {a.manifest} ({len(merged['runs'])} runs)")


if __name__ == "__main__":
    main()
