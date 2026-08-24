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
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar"
)

import os, json, glob

GENERATED = f"{AMBIENT_BASE}/generated"

# --- Reference: overlapping conditional sweeps -- NOT SEED-AVERAGED, DO NOT
# --- TAKE AN ARGMIN OVER THESE.
#
# These were pooled by filename across lysine, CSAIL and proline, so the curve
# has n=3 at a few T values and n=1 at most others. An argmin across points of
# unequal replication is meaningless: the n=1 points carry ~sqrt(3)x the variance
# of the n=3 points and will take the minimum by chance. Earlier versions of this
# file reported optima at T=0.55 / 0.80 / 0.95 from exactly this mistake.
#
# Per the project owner, properly seed-averaged conditional sweeps put the
# optimum at T=1.00 for every "B2 fixed, other bucket swept" sweep -- the added
# bucket is redundant. That data has not been located in this repo or on the
# machines reachable from here; until it is, these values are usable only as
# per-point references at matching T, never for locating an optimum.
COND = {
    "b3": {0.0: (0.035958, 1), 0.2: (0.036571, 1), 0.4: (0.034723, 1),
           0.45: (0.033354, 2), 0.5: (0.031896, 2), 0.525: (0.032405, 1),
           0.55: (0.031818, 3), 0.575: (0.032742, 1), 0.6: (0.032897, 2),
           0.7: (0.032441, 1), 0.8: (0.032166, 1), 1.0: (0.032623, 1)},
    "b4": {0.0: (0.036509, 1), 0.2: (0.035866, 1), 0.4: (0.033055, 1),
           0.45: (0.032723, 1), 0.5: (0.032217, 1), 0.525: (0.033098, 2),
           0.55: (0.032471, 1), 0.6: (0.032918, 1), 0.7: (0.031669, 2),
           0.75: (0.032456, 1), 0.8: (0.031439, 3), 0.85: (0.032124, 2),
           0.9: (0.032245, 2)},
    "b2": {0.0: (0.036033, 1), 0.2: (0.034300, 1), 0.3: (0.033382, 1),
           0.4: (0.033216, 1), 0.45: (0.032800, 1), 0.5: (0.031355, 1),
           0.525: (0.031317, 1), 0.55: (0.030982, 1), 0.575: (0.033119, 1),
           0.6: (0.031427, 1), 0.65: (0.030644, 1), 0.7: (0.030748, 1),
           0.8: (0.032183, 1), 0.9: (0.030818, 1), 0.95: (0.032249, 1),
           0.99: (0.030772, 1), 1.0: (0.030296, 1)},
    # b5 against b2 fixed -- the conditional sweep exists, all points single-run.
    "b5g2": {0.0: (0.037695, 1), 0.2: (0.037055, 1), 0.4: (0.033269, 1),
             0.45: (0.033491, 1), 0.5: (0.033363, 1), 0.525: (0.033132, 1),
             0.6: (0.033879, 1), 0.7: (0.032832, 1), 0.8: (0.032900, 1),
             0.9: (0.032929, 1), 0.95: (0.031427, 1), 0.99: (0.031624, 1)},
    # b3 against b1 fixed has NO overlapping counterpart: the conditional sweeps
    # only ever paired b2|b1, and b3/b4/b5|b2. So the "useless with overlap ->
    # useful when restricted" comparison cannot be made for this one, and it is
    # readable only against its own in-batch single-bucket baselines.
    "b3g1": {},
}

# Spread between repeated runs of the same configuration, averaged over the T
# values that actually got reruns. This is the bar a difference has to clear.
# (Meeting prep quotes 0.00104 pooled across the dynamic-T arms -- same order.)
POOLED_SD = {"b3": 0.000857, "b4": 0.000662, "b2": 0.000857,
             "b5g2": 0.000857, "b3g1": 0.000857}

# Single-bucket configurations that the degenerate ends of each sweep reproduce.
# Every one of these is thinly measured -- note in particular that no T=1.0
# baseline in the conditional sweeps was ever replicated, so the published
# "the extra bucket is redundant" compares a 3-run mean against a single run.
# Re-measuring them in-batch is the point of the degenerate endpoints.
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
    ("b5g2", "b2 fixed @0.55, sweeping b5", 0.55),
    ("b3g1", "b1 fixed @0.50, sweeping b3", 0.50),
]

# At T=1.00 the swept bucket goes inactive, so every sweep sharing a fixed bucket
# collapses onto the same run. It is trained once and the others borrow it.
SHARED_BASELINE = {"b4": "b3", "b5g2": "b3", "b3g1": "b2"}


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
    # 23 original + 7 (b5 given b2) + 7 (b3 given b1); the T=1.00 point of each
    # later sweep is not retrained, it reuses the earlier sweep's identical run.
    print(f"RESTRICTED-BUCKET SWEEPS -- {total}/37 experiments complete")
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
        print(f"{'T':>7}  {'restricted':>11}  {'overlapping':>12} {'n':>3}  {'delta':>10}   note")

        # Fixed-bucket-solo baseline. restr_b3_T100 and restr_b4_T100 are the same
        # configuration by construction (b2 banded to [sigma(0.55), sigma(0.999)),
        # every other bucket inactive), so only b3_T100 is trained and the b4 sweep
        # borrows it rather than repeating the run.
        base_fixed = data.get(1.0)
        shared = False
        if base_fixed is None and label in SHARED_BASELINE:
            base_fixed = restr.get(SHARED_BASELINE[label], {}).get(1.0)
            shared = base_fixed is not None
        base_swept = data.get(t_fixed)    # swept bucket solo

        for t in sorted(data):
            r = data[t]
            ref = COND[label].get(t)
            c, n = ref if ref else (None, 0)
            delta = (r - c) if c is not None else None
            note = ""
            if t >= 1.0:
                note = "fixed bucket solo (baseline)"
            elif t <= t_fixed:
                note = "swept bucket solo"
            nstr = str(n) if n else "-"
            print(f"{t:>7.3f}  {fmt(r):>11}  {fmt(c):>12} {nstr:>3}  {fmt(delta):>10}   {note}")

        # Does the swept bucket earn its place under restriction?
        interior = {t: v for t, v in data.items() if t_fixed < t < 1.0}
        if interior and base_fixed is not None:
            best_t = min(interior, key=interior.get)
            best = interior[best_t]
            gain = base_fixed - best
            print(f"\n  best interior point : T={best_t:.3f}  MIND={best:.6f}")
            print(f"  fixed bucket solo   : MIND={base_fixed:.6f}"
                  + (f"   (shared with the {SHARED_BASELINE.get(label,'')} sweep -- same config)" if shared else ""))
            if base_swept is not None:
                print(f"  swept bucket solo   : MIND={base_swept:.6f}")
            # Bar to clear: the spread between repeated runs of one configuration
            # in the overlapping sweeps, or this batch's own replicate if larger.
            bar = max(POOLED_SD.get(label, 0.0), noise or 0.0)
            verdict = ("INCONCLUSIVE (within replicate noise)" if gain <= bar
                       else "the partition beats the single bucket" if gain > 0
                       else "no gain from splitting")
            print(f"  gain over fixed solo: {gain:+.6f}  (bar {bar:.6f})  -> {verdict}")

            if not COND[label]:
                print("  no overlapping counterpart exists for this pairing")
                continue
            # Deliberately not reported: see the header. The reference curve is
            # not seed-balanced, so its argmin is an artifact of which points
            # happen to have one run instead of three.
            print("  overlapping optimum: NOT COMPUTED -- reference curve is not "
                  "seed-balanced (see header)")
            # Only claim the optimum moved if it also beat the baseline by more
            # than the replicate spread -- an interior argmin that is within noise
            # of T=1.0 is not evidence of anything, which is the trap the
            # single-run conditional results fell into.
            # The "did the optimum move off T=1.0" claim needs a trustworthy
            # before-side, which we do not have. Report only the restricted side.
            if best_t < 1.0 and gain > bar:
                print(f"  ** restricted optimum is interior (T={best_t:.2f}) and clears "
                      f"the noise bar; before/after vs overlap NOT established **")

    print("\n" + "=" * 78)
    print("Reminder: the old B2-only@0.55 was recorded as both 0.031144 and 0.032623")
    print("at seed 0. Compare within this batch, not against the older JSON.")
    print("=" * 78)


if __name__ == "__main__":
    main()
