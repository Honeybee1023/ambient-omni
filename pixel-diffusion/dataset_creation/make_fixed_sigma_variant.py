#!/usr/bin/env python3
"""Copy a per-group dataset (e.g. celeba_mix4_v1) with every corrupted image given a FIXED
sigma_min per group, written into annotations.jsonl -- Ambient-o's "fixed annotation" mode and
the starting point of the official Dataloops CIFAR recipe (annotate_fixed_sigma: blur 0.6 ->
sigma 1.2, 0.8 -> 1.9, 1.0 -> 2.4, i.e. about 2-2.4x the blur strength).

Writing the values into the file (rather than setting them with a schedule) matters for
Dataloops: dataloops_restore.py reads each image's sigma_tn from the loop-0 run's
annotations_processed.jsonl, which records the values present at load time.

Usage: python dataset_creation/make_fixed_sigma_variant.py --src celeba_mix4_v1 \
           --name celeba_mix4_fixed --sigmas g03=0.7,g05=1.1,g10=2.4,g20=4.8
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sigmas", required=True, help="group=sigma,... e.g. g03=0.7,g05=1.1")
    a = ap.parse_args()
    sig = {k: float(v) for k, v in (kv.split("=") for kv in a.sigmas.split(","))}
    src = f"{AMBIENT_BASE}/annotated_datasets/{a.src}"
    out = f"{AMBIENT_BASE}/annotated_datasets/{a.name}"
    if os.path.exists(out):
        sys.exit(f"{out} exists; refusing to overwrite")
    os.makedirs(out)
    n = {g: 0 for g in sig}
    with open(f"{src}/annotations.jsonl") as fi, open(f"{out}/annotations.jsonl", "w") as fo:
        for line in fi:
            d = json.loads(line)
            g = d["filename"].split("_", 1)[0]
            if g in sig:
                d["sigma_min"] = sig[g]
                n[g] += 1
            elif d["sigma_min"] != 0.0:
                sys.exit(f"{d['filename']}: not a listed group and not clean ({d['sigma_min']})")
            os.symlink(os.path.realpath(f"{src}/{d['filename']}"), f"{out}/{d['filename']}")
            fo.write(json.dumps(d) + "\n")
    missing = [g for g, c in n.items() if c == 0]
    if missing:
        sys.exit(f"groups with no files: {missing}")
    print(f"wrote {out}: {n}")


if __name__ == "__main__":
    main()
