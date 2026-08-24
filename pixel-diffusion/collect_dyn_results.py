#!/usr/bin/env python3
"""Collect dynamic-T schedule-search results into one table + JSON.

Reads generated/mind_dyn_<run>_s<seed>.json (and fid_ alongside), joins them to
the 4-vectors in dyn_search_manifest.json, and writes dyn_all_results.json.

ONE RULE THIS TOOL ENFORCES
---------------------------
It will not hand you an argmin over points with unequal replication. Pooling
runs by name gives a curve with n=3 at a few settings and n=1 at the rest; the
n=1 points carry ~sqrt(n) times the standard error of the n=3 ones and take the
minimum by chance. That mistake was made twice on the conditional sweeps in this
project and reported as a real effect both times. So: `--argmin` reports the
best point only within each replication level, and says so.

Usage:  python collect_dyn_results.py [--json] [--phase 0|1]
"""

import argparse, glob, json, os, re
from collections import defaultdict

import numpy as np

AMBIENT_BASE = os.environ.get("AMBIENT_BASE") or next(
    (p for p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if os.path.isdir(p)), "/data/scratch/honjar")
GENERATED = f"{AMBIENT_BASE}/generated"
MANIFEST = f"{GENERATED}/dyn_search_manifest.json"

# Seed-averaged reference values for the hand-crafted continuous schedules,
# from mind_v2_*.json on lysine. Used only to say "this reproduces that";
# never mixed into the BO training set, which must come from one batch.
REFERENCE = {
    "p0_static_T050": ("static_T050", 0.035229, 3),
    "p0_warmup_cont": ("warmup25_0to095", 0.029622, 3),
    "p0_warmup_pw5": ("warmup25_0to095", 0.029622, 3),
    "p0_warmup_pw10": ("warmup25_0to095", 0.029622, 3),
    "p0_cosine_pw5": ("cosine_0to095", 0.032169, 3),
    "p0_cosine_pw10": ("cosine_0to095", 0.032169, 3),
    "p1_a_linear_0to095": ("linear_0to095", 0.031211, 3),
    "p1_a_twophase_050": ("twophase_0_050_095", 0.032425, 3),
    "p1_a_warmup15_0to095": ("warmup15_0to095", 0.029527, 1),
    "p1_a_warmup40_0to095": ("warmup40_0to095", 0.029433, 1),
}


def load_metric(path, key):
    try:
        with open(path) as f:
            return json.load(f).get(key)
    except Exception:
        return None


def load():
    """-> {run_name: {seed: {'mind':.., 'fid':..}}}"""
    out = defaultdict(dict)
    for path in sorted(glob.glob(f"{GENERATED}/mind_dyn_*.json")):
        m = re.match(r"mind_dyn_(.+)_s(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        run, seed = m.group(1), int(m.group(2))
        mind = load_metric(path, "mind")
        if mind is None:
            continue
        fid = load_metric(f"{GENERATED}/fid_dyn_{run}_s{seed}.json", "fid")
        out[run][seed] = {"mind": mind, "fid": fid}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None, help="0 or 1")
    ap.add_argument("--json", action="store_true", help="write dyn_all_results.json")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    specs = {e["name"]: e for e in manifest["runs"]}
    results = load()

    rows = []
    for name, spec in specs.items():
        if args.phase is not None and str(spec["phase"]) != args.phase:
            continue
        seeds = results.get(name, {})
        if not seeds:
            continue
        vals = [s["mind"] for s in seeds.values()]
        rows.append({
            "name": name, "phase": spec["phase"], "x": spec["x"],
            "n": len(vals), "mind_mean": float(np.mean(vals)),
            "mind_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "mind_all": {str(k): v["mind"] for k, v in sorted(seeds.items())},
            "fid_mean": (float(np.mean([s["fid"] for s in seeds.values() if s["fid"]]))
                         if any(s["fid"] for s in seeds.values()) else None),
            "note": spec.get("note", ""),
        })

    total = sum(1 for e in manifest["runs"]
                if args.phase is None or str(e["phase"]) == args.phase)
    print("=" * 92)
    print(f"DYNAMIC-T SCHEDULE SEARCH -- {len(rows)}/{total} runs with results"
          + (f" (phase {args.phase})" if args.phase else ""))
    print("=" * 92)
    if not rows:
        print("\nNothing yet. Progress:  bash run_dyn_queue.sh status")
        return

    rows.sort(key=lambda r: r["mind_mean"])
    print(f"\n{'run':<24}{'[T2,T3,T4,T5]':<30}{'n':>2}  {'MIND':>9}  {'sd':>8}  reference")
    print("-" * 92)
    for r in rows:
        xs = "[" + ", ".join(f"{v:.2f}" for v in r["x"]) + "]" if r["x"] else "(continuous)"
        sd = f"{r['mind_sd']:.6f}" if r["mind_sd"] is not None else "     -- "
        ref = ""
        if r["name"] in REFERENCE:
            label, val, n = REFERENCE[r["name"]]
            ref = f"{label} = {val:.6f} (n={n}), d = {r['mind_mean'] - val:+.6f}"
        print(f"{r['name']:<24}{xs:<30}{r['n']:>2}  {r['mind_mean']:.6f}  {sd:>8}  {ref}")

    # --- Replication-aware "best". Never one argmin across mixed n. ---
    print("\n" + "-" * 92)
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    print("Best point WITHIN each replication level (never argmin across mixed n --")
    print("an n=1 point has ~sqrt(3)x the standard error of an n=3 one and wins by chance):")
    for n in sorted(by_n, reverse=True):
        grp = sorted(by_n[n], key=lambda r: r["mind_mean"])
        print(f"  n={n} ({len(grp)} runs): {grp[0]['name']} -> {grp[0]['mind_mean']:.6f}")

    # Noise floor from whatever replicates exist in this batch.
    reps = [r for r in rows if r["n"] > 1]
    if reps:
        pooled = float(np.sqrt(np.mean([r["mind_sd"] ** 2 for r in reps])))
        print(f"\nPooled within-run SD over {len(reps)} replicated run(s): {pooled:.6f}")
        print("  Treat any gap smaller than this as indistinguishable.")
    else:
        print("\nNo replicated runs in this batch yet. The hand-crafted runs put the")
        print("  1-seed SD near 0.0012 (warmup 0->0.95 over 3 seeds), so treat gaps")
        print("  below ~0.0012 as noise until this batch has its own replicates.")

    if args.json:
        out = f"{GENERATED}/dyn_all_results.json"
        with open(out, "w") as f:
            json.dump({
                "fractions": manifest["fractions"],
                "t_first_pinned": manifest["t_first_pinned"],
                "dataset": manifest["dataset"],
                "reference_seed_averaged": manifest["known_seed_averaged_reference"],
                "caveat": ("mind_mean pools seeds per run. Do NOT argmin across runs "
                           "with different n -- compare within a replication level."),
                "runs": rows,
            }, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
