#!/usr/bin/env python3
"""Derive a fewer-clean copy of celeba70k_mix4: same 63,000 blurred faces (same four buckets), only the
first N of its clean faces kept. Default N = 700 (1% of the original 70k), giving 63,700 images.

Why this shape: the 10% set (7,000 clean) never memorizes within 20M images, so blurred data has nothing
to fix there. Cutting only the clean count isolates that one variable against the 10% run.

The clean subset is nested in the 10% one: it is the first N indices of the same fixed-seed permutation
create_celeba70k_mix4.py used (perm[:7000] were its clean faces), so no parquet re-read is needed.
Files are HARD LINKS to celeba70k_mix4 (identical bytes, no extra inodes on the scratch file quota).
FID/MIND reference and memorization search set stay the same as for the 10% set
($AMBIENT_BASE/celeba70k_processed/train_clean_64 and test_64), so the two settings are scored alike.

    python dataset_creation/create_celeba_mix4_subclean.py --n_clean 700 --name celeba_c700_mix4
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import argparse, json, os, sys

import numpy as np

SEED = 20261010          # create_celeba70k_mix4.SEED
N_OFFICIAL_TRAIN = 162770
N_SRC_CLEAN = 7000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="celeba70k_mix4")
    ap.add_argument("--n_clean", type=int, default=700)
    ap.add_argument("--name", default="celeba_c700_mix4")
    args = ap.parse_args()
    D = f"{AMBIENT_BASE}/annotated_datasets"
    src, out = f"{D}/{args.src}", f"{D}/{args.name}"
    src_meta = json.load(open(f"{src}.meta.json"))
    if os.path.exists(out) or os.path.exists(f"{out}.meta.json"):
        sys.exit(f"{out} already exists; refusing to overwrite")
    assert 0 < args.n_clean <= N_SRC_CLEAN

    perm = np.random.RandomState(SEED).permutation(N_OFFICIAL_TRAIN)
    src_clean = sorted(f"tr{int(i):06d}" for i in perm[:N_SRC_CLEAN])
    assert src_clean == sorted(src_meta["ids"]["b0"]), "permutation does not reproduce the source's clean set"
    keep = sorted(f"tr{int(i):06d}" for i in perm[:args.n_clean])

    src_lines = [json.loads(l) for l in open(f"{src}/annotations.jsonl")]
    keep_set = set(f"b0_{t}.png" for t in keep)
    lines = [l for l in src_lines if not l["filename"].startswith("b0_") or l["filename"] in keep_set]
    os.makedirs(out)
    for l in lines:
        os.link(f"{src}/{l['filename']}", f"{out}/{l['filename']}")
    with open(f"{out}/annotations.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

    groups = {g: v for g, v in src_meta["groups"].items()}
    meta = dict(src_meta)
    meta.update({
        "derived_from": args.src, "n_clean": len(keep), "clean_frac_of_source_total": len(keep) / src_meta["n_total"],
        "clean_rule": f"first {len(keep)} of the source permutation (nested in the source's clean set)",
        "n_total": len(lines), "groups": groups,
        "ids": {"b0": keep, **{g: src_meta["ids"][g] for g in groups}},
        "files": "hard links to the source dataset",
    })
    meta.pop("clean_frac", None)
    with open(f"{out}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    n_b0 = sum(l["filename"].startswith("b0_") for l in lines)
    print(f"wrote {out}: {len(lines)} images ({n_b0} clean, {len(lines) - n_b0} blurred); meta {out}.meta.json")


if __name__ == "__main__":
    main()
