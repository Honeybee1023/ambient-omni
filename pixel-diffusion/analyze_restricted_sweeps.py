#!/usr/bin/env python3
"""Compare the restricted-bucket sweeps against the overlapping (conditional) ones.

The question is narrow: in the conditional sweeps a second bucket bought at most
~0.0015 MIND, which is the same size as the run-to-run spread, so it was not
possible to say whether the extra bucket was genuinely redundant. Restricting each
bucket to an exclusive noise band should, under the single-bottleneck hypothesis,
make the second bucket matter -- removing it now leaves a real hole.

Everything here is reported against *in-batch* baselines. The two degenerate ends
of each sweep reproduce a single-bucket configuration, so they are the baselines:

    T_swept == T_fixed  ->  swept bucket solo at T_fixed
    T_swept == 1.0      ->  fixed bucket solo at T_fixed

Usage:  python analyze_restricted_sweeps.py
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, glob

GENERATED = f"{AMBIENT_BASE}/generated"

# --- Reference: the overlapping conditional sweeps, from v2b_all_results.json ---
# Copied in rather than read, because that file lives on CSAIL and this runs on
# lysine. Keep in sync with $AMBIENT_BASE/generated/v2b_all_results.json.
COND = {
    "b3": {0.0: 0.035958, 0.2: 0.036571, 0.4: 0.034723, 0.45: 0.034215, 0.5: 0.031498,
           0.55: 0.031099, 0.6: 0.032279, 0.8: 0.032166, 1.0: 0.032623},
    "b4": {0.0: 0.036509, 0.2: 0.035866, 0.4: 0.033055, 0.45: 0.032723, 0.5: 0.032217,
           0.525: 0.032454, 0.55: 0.032471, 0.6: 0.032918, 0.7: 0.032021,
           0.8: 0.030852, 0.9: 0.031673},
    "b2": {0.0: 0.036033, 0.2: 0.034300, 0.3: 0.033382, 0.4: 0.033216, 0.45: 0.032800,
           0.5: 0.031355, 0.525: 0.031317, 0.55: 0.030982, 0.575: 0.033119,
           0.6: 0.031427, 0.65: 0.030644, 0.7: 0.030748, 0.8: 0.032183,
           0.9: 0.030818, 0.95: 0.032249, 0.99: 0.030772, 1.0: 0.030296},
}

# Solo-sweep values for the same single-bucket configurations the degenerate ends
# reproduce. Where a config was measured twice, both are listed -- the spread is
# the point.
SOLO = {
    ("b3", 0.55): [0.031825],                 # b3 solo @0.55
    ("b3", 1.0): [0.031144, 0.032623],        # b2 solo @0.55, measured twice
    ("b4", 0.55): [0.032316],                 # b4 solo @0.55
    ("b4", 1.0): [0.031144, 0.032623],        # same config as ("b3", 1.0)
    ("b2", 0.50): [0.031831],                 # b2 solo @0.50
    ("b2", 1.0): [0.030599, 0.030296],        # b1 solo @0.50, measured twice
}

SWEEPS = [
    ("b3", "b2 fixed @0.55, sweeping b3", 0.55),
    ("b4", "b2 fixed @0.55, sweeping b4", 0.55),
    ("b2", "b1 fixed @0.50, sweeping b2", 0.50),
]


def t_from_suffix(s):
    return int(s) / 100 if len(s) == 3 else int(s) / 1000


def load_restricted():
    out = {}
    for path in sorted(glob.glob(f"{GENERATED}/mind_celeba_v2b_restr_*_2000kimg.json")):
        base = os.path.basename(path)
        stem = base[len("mind_celeba_v2b_restr_"):-len("_2000kimg.json")]
        label, _, tsuf = stem.partition("_T")
        if not tsuf:
            continue
        with open(path) as f:
            out.setdefault(label, {})[t_from_suffix(tsuf)] = json.load(f)["mind"]
    return out


def fmt(v):
    return f"{v:.6f}" if v is not None else "    --   "


def main():
    restr = load_restricted()
    total = sum(len(v) for v in restr.values())
    print("=" * 78)
    print(f"RESTRICTED-BUCKET SWEEPS -- {total}/24 experiments complete")
    print("=" * 78)

    if total == 0:
        print("\nNo results yet. Check progress with:")
        print("    bash run_restr_sweep_queue.sh status")
        return

    # Noise estimate: restr_b3_T100 and restr_b4_T100 are the same configuration.
    rep = [restr.get("b3", {}).get(1.0), restr.get("b4", {}).get(1.0)]
    if all(r is not None for r in rep):
        print(f"\nIn-batch replicate (b3_T100 vs b4_T100 -- identical configs):")
        print(f"    {rep[0]:.6f}  vs  {rep[1]:.6f}   |diff| = {abs(rep[0]-rep[1]):.6f}")
        print(f"    Treat any gap below this as indistinguishable from noise.")
        noise = abs(rep[0] - rep[1])
    else:
        noise = None
        print("\nIn-batch replicate not yet available (needs restr_b3_T100 + restr_b4_T100).")

    for label, title, t_fixed in SWEEPS:
        data = restr.get(label, {})
        if not data:
            continue
        print("\n" + "-" * 78)
        print(f"{title}   [restricted vs overlapping]")
        print("-" * 78)
        print(f"{'T':>7}  {'restricted':>11}  {'overlapping':>12}  {'delta':>10}   note")

        base_fixed = data.get(1.0)        # fixed bucket solo
        base_swept = data.get(t_fixed)    # swept bucket solo

        for t in sorted(data):
            r = data[t]
            c = COND[label].get(t)
            delta = (r - c) if c is not None else None
            note = ""
            if t >= 1.0:
                note = "fixed bucket solo (baseline)"
            elif t <= t_fixed:
                note = "swept bucket solo"
            print(f"{t:>7.3f}  {fmt(r):>11}  {fmt(c):>12}  {fmt(delta):>10}   {note}")

        # Does the swept bucket earn its place under restriction?
        interior = {t: v for t, v in data.items() if t_fixed < t < 1.0}
        if interior and base_fixed is not None:
            best_t = min(interior, key=interior.get)
            best = interior[best_t]
            gain = base_fixed - best
            print(f"\n  best interior point : T={best_t:.3f}  MIND={best:.6f}")
            print(f"  fixed bucket solo   : MIND={base_fixed:.6f}")
            if base_swept is not None:
                print(f"  swept bucket solo   : MIND={base_swept:.6f}")
            verdict = ("INCONCLUSIVE (within replicate noise)" if noise and gain <= noise
                       else "the partition beats the single bucket" if gain > 0
                       else "no gain from splitting")
            print(f"  gain over fixed solo: {gain:+.6f}   -> {verdict}")

            cond_best_t = min(COND[label], key=COND[label].get)
            print(f"  overlapping sweep optimum was T={cond_best_t:.3f} "
                  f"(MIND={COND[label][cond_best_t]:.6f})")

    print("\n" + "=" * 78)
    print("Reminder: the old B2-only@0.55 was recorded as both 0.031144 and 0.032623")
    print("at seed 0. Compare within this batch, not against the older JSON.")
    print("=" * 78)


if __name__ == "__main__":
    main()
