#!/usr/bin/env python3
"""Build the mixed-blur go/no-go dataset: 500 clean faces + 26,014 blurred faces split
evenly over four blur levels (0.3, 0.5, 1.0, 2.0).

Why these sources. Each blur level is taken from the round whose single-bucket static
sweep we reuse as a reference, so every bucket has a matching earlier result:
    g03  sigma_B 0.3   v2b bucket b3   (fine static sweep, best T 0.50)
    g05  sigma_B 0.5   v2b bucket b5   (the exact file set of every dynamic-T ledger run)
    g10  sigma_B 1.0   v2  bucket b5   (coarse sweep; static never beats clean-only)
    g20  sigma_B 2.0   v2  bucket b7   (same)
The 500 clean faces (b0) are byte-identical in v2 and v2b.

v2b b5 and v2 b5 hold the SAME 26,014 faces at two blur strengths, so g05 and g10 draw
disjoint halves of one shuffled id list. The builder refuses to write anything if any
face would appear twice.

Total corrupted count stays 26,014 (6504+6504+6503+6503), the same as every earlier
run, so the clean share of a batch (1.9%) and the clean-only reference are unchanged.

Every blurred image carries the sigma_min=999 sentinel: the per_group t_schedule sets
each group's threshold during training. Files are named <group>_<id>.jpg so the
schedule can find its group from the filename.

Run on CSAIL:  python dataset_creation/create_mixed_blur_dataset.py [--name celeba_mix4_v1]
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import argparse, json, os, random, sys

V2 = f"{AMBIENT_BASE}/celeba_processed_v2/shared_buckets_64"
V2B = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
SEED = 20260927
SENTINEL = 999.0
# group -> (source dir, source bucket prefix, blur sigma, count)
GROUPS = {
    "g03": (V2B, "b3", 0.3, 6504),
    "g05": (V2B, "b5", 0.5, 6504),
    "g10": (V2, "b5", 1.0, 6503),
    "g20": (V2, "b7", 2.0, 6503),
}


def ids_in(src, prefix):
    return sorted(f[len(prefix) + 1:] for f in os.listdir(src) if f.startswith(prefix + "_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="celeba_mix4_v1")
    ap.add_argument("--dry", action="store_true", help="select and verify, write nothing")
    args = ap.parse_args()
    out = f"{AMBIENT_BASE}/annotated_datasets/{args.name}"
    if os.path.exists(out):
        sys.exit(f"{out} already exists; refusing to overwrite")

    rng = random.Random(SEED)
    chosen = {}
    # g05 and g10 share faces: shuffle that id list once and split it.
    shared = ids_in(V2B, "b5")
    assert shared == ids_in(V2, "b5"), "v2 b5 and v2b b5 no longer hold the same ids"
    rng.shuffle(shared)
    chosen["g05"] = shared[:GROUPS["g05"][3]]
    chosen["g10"] = shared[GROUPS["g05"][3]:GROUPS["g05"][3] + GROUPS["g10"][3]]
    for g in ("g03", "g20"):
        src, pre, _, n = GROUPS[g]
        ids = ids_in(src, pre)
        rng.shuffle(ids)
        chosen[g] = ids[:n]

    # Face-level disjointness across groups (ids are CelebA source ids).
    seen = {}
    for g, ids in chosen.items():
        for i in ids:
            if i in seen:
                sys.exit(f"face {i} would appear in both {seen[i]} and {g}; refusing")
            seen[i] = g
    clean = ids_in(V2B, "b0")
    assert clean == ids_in(V2, "b0") and len(clean) == 500, "clean sets differ between v2 and v2b"
    overlap = set(clean) & set(seen)
    assert not overlap, f"{len(overlap)} clean faces also appear blurred"
    total = sum(len(v) for v in chosen.values())
    assert total == 26014, total
    print("selected:", {g: len(v) for g, v in chosen.items()}, "clean:", len(clean))
    if args.dry:
        print("dry run: nothing written")
        return

    os.makedirs(out)
    lines = []
    for i in clean:
        os.symlink(f"{V2B}/b0_{i}", f"{out}/b0_{i}")
        lines.append({"filename": f"b0_{i}", "sigma_min": 0.0, "sigma_max": 0.0})
    for g, ids in chosen.items():
        src, pre, _, _ = GROUPS[g]
        for i in ids:
            os.symlink(f"{src}/{pre}_{i}", f"{out}/{g}_{i}")
            lines.append({"filename": f"{g}_{i}", "sigma_min": SENTINEL, "sigma_max": 0.0})
    with open(f"{out}/annotations.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    meta = {"seed": SEED, "groups": {g: {"source": f"{GROUPS[g][0]}/{GROUPS[g][1]}",
                                          "blur_sigma": GROUPS[g][2], "n": len(chosen[g])}
                                      for g in GROUPS},
            "n_clean": len(clean), "n_total": len(lines)}
    json.dump(meta, open(f"{out}.meta.json", "w"), indent=1)   # beside, not inside: the loader scans the folder

    # Verify what was written, not what was intended.
    bad = [l["filename"] for l in lines if not os.path.isfile(f"{out}/{l['filename']}")]
    assert not bad, f"{len(bad)} broken symlinks, e.g. {bad[:3]}"
    print(f"wrote {out}: {len(lines)} annotations, all symlinks resolve")


if __name__ == "__main__":
    main()
