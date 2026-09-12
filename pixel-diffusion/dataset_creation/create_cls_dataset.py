"""Build the training set for Ambient-o's clean-vs-corrupt classifier.

The experiment this serves: does the threshold Ambient-o's own classifier
assigns to our blur bucket match the MIND-optimal static threshold (0.50)? See
PRINCIPLED_T_SEARCH.md and the authors' notes. The classifier is trained with
`train.py --precond=edmcls`, which reads a `corruption_label` per image; we
supply it through --overwrite_cls_labels_path as a JSONL of
{"image_file": <filename>, "label": 0|1}. The key must match what the dataset
yields as `filename`, which for a flat image folder is the bare file name.

Two things this has to get right:

* Every image must be eligible at EVERY noise level. The classifier has to see
  both classes across the whole sigma range to learn where they stop being
  separable, so annotations.jsonl carries sigma_min = 0 for all files. (The
  dynamic-T dataset carries a 999 sentinel on the blurred files; that would
  restrict them and is exactly wrong here.)

* The classes must be balanced. The source has 500 clean and 26,014 blurred
  images. Trained on that as-is, the classifier learns "say blurred" and is
  right 98% of the time; its probability never drops toward 0.5, so every
  image annotates to the top of the sigma range and the experiment measures
  the imbalance instead of the corruption. Ambient-o trains its classifier at
  corruption_probability = 0.5. We match that by repeating the clean symlinks
  under suffixed names until the two classes are the same size. The 500 clean
  images are then seen ~52x as often as any single blurred one -- which is
  also what the diffusion model itself experiences.

Usage (on the machine with the data):
    python dataset_creation/create_cls_dataset.py --src celeba_dynamic_t_v2_b0b5
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse
import json
import os

CLEAN_PREFIX, CORRUPT_PREFIX = "b0_", "b5_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="celeba_dynamic_t_v2_b0b5",
                    help="existing dataset dir under annotated_datasets holding b0_*/b5_* files")
    ap.add_argument("--name", default="celeba_cls_b0b5")
    ap.add_argument("--no-balance", action="store_true",
                    help="do NOT repeat clean images to balance classes (not recommended)")
    args = ap.parse_args()

    src = os.path.join(AMBIENT_BASE, "annotated_datasets", args.src)
    out = os.path.join(AMBIENT_BASE, "annotated_datasets", args.name)
    if not os.path.isdir(src):
        raise SystemExit(f"source dataset not found: {src}")
    os.makedirs(out, exist_ok=True)

    files = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
    clean = [f for f in files if f.startswith(CLEAN_PREFIX)]
    corrupt = [f for f in files if f.startswith(CORRUPT_PREFIX)]
    if not clean or not corrupt:
        raise SystemExit(f"expected {CLEAN_PREFIX}* and {CORRUPT_PREFIX}* files in {src}, "
                         f"found {len(clean)} / {len(corrupt)}")

    reps = 1 if args.no_balance else max(1, round(len(corrupt) / len(clean)))
    annotations, labels = [], []

    def add(link_name, target, label):
        p = os.path.join(out, link_name)
        if not os.path.lexists(p):
            os.symlink(os.path.realpath(os.path.join(src, target)), p)
        annotations.append({"filename": link_name, "sigma_min": 0.0, "sigma_max": 0.0})
        labels.append({"image_file": link_name, "label": label})

    for f in corrupt:
        add(f, f, 1)
    for f in clean:
        add(f, f, 0)
        for r in range(1, reps):
            # b0_000123.jpg -> b0dup03_000123.jpg : unique name, same bytes
            add(f.replace(CLEAN_PREFIX, f"b0dup{r:02d}_", 1), f, 0)

    with open(os.path.join(out, "annotations.jsonl"), "w") as fh:
        for a in annotations:
            fh.write(json.dumps(a) + "\n")
    labels_path = os.path.join(out, "cls_labels.jsonl")
    with open(labels_path, "w") as fh:
        for l in labels:
            fh.write(json.dumps(l) + "\n")

    n0 = sum(1 for l in labels if l["label"] == 0)
    n1 = len(labels) - n0
    print(f"{out}")
    print(f"  clean   (label 0): {n0:>6}   ({len(clean)} distinct x {reps})")
    print(f"  blurred (label 1): {n1:>6}")
    print(f"  all sigma_min = 0 -> both classes eligible at every noise level")
    print(f"  labels: {labels_path}")
    if abs(n0 - n1) > 0.1 * max(n0, n1):
        print("  WARNING: classes are not balanced; the classifier will learn the prior")


if __name__ == "__main__":
    main()
