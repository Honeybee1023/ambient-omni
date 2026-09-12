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

* **Identity must carry no label information.** The first version paired 500
  clean faces against 26,014 *different* blurred faces. Then "is this one of
  the 500?" predicts the label perfectly, at every noise level, and a
  classifier that learns that shortcut annotates every blurred face as
  "never confused" -> threshold at the top of the range. Ambient-o never meets
  this problem because it corrupts a random half of the SAME images on the
  fly. We match that: the blurred class is the 500 clean faces blurred by us
  (sigma_blur = 0.5, the b5 recipe), so each face appears once clean and once
  blurred and identity is useless. 500 distinct faces is few -- but it is
  exactly what the diffusion model itself has, and if the annotator is
  data-starved in this regime, that is part of the finding, not a bug.

  (The blur is applied to the already-saved 64x64 jpg, whereas the real b5
  files were blurred before JPEG encoding. That is a slight mismatch for
  *training* the classifier; annotation is run on the real b5 files.)

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
    ap.add_argument("--name", default="celeba_cls_paired")
    args = ap.parse_args()

    src = os.path.join(AMBIENT_BASE, "annotated_datasets", args.src)
    out = os.path.join(AMBIENT_BASE, "annotated_datasets", args.name)
    if not os.path.isdir(src):
        raise SystemExit(f"source dataset not found: {src}")
    os.makedirs(out, exist_ok=True)

    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    BLUR_SIGMA = 0.5                                  # bucket b5

    files = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
    clean = [f for f in files if f.startswith(CLEAN_PREFIX)]
    if not clean:
        raise SystemExit(f"expected {CLEAN_PREFIX}* files in {src}")

    annotations, labels = [], []

    def add(name, label):
        annotations.append({"filename": name, "sigma_min": 0.0, "sigma_max": 0.0})
        labels.append({"image_file": name, "label": label})

    for f in clean:
        # clean copy: symlink to the training file
        p = os.path.join(out, f)
        if not os.path.lexists(p):
            os.symlink(os.path.realpath(os.path.join(src, f)), p)
        add(f, 0)
        # blurred copy of the SAME face: a real file, b5 recipe
        bname = f.replace(CLEAN_PREFIX, "b5self_", 1)
        bp = os.path.join(out, bname)
        if not os.path.exists(bp):
            arr = np.array(Image.open(os.path.join(src, f)).convert("RGB"), dtype=np.float32)
            arr = gaussian_filter(arr, sigma=(BLUR_SIGMA, BLUR_SIGMA, 0))
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(bp)
        add(bname, 1)

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
    print(f"  clean   (label 0): {n0:>6}   (the {len(clean)} training faces)")
    print(f"  blurred (label 1): {n1:>6}   (the SAME faces, blurred sigma={BLUR_SIGMA})")
    print(f"  all sigma_min = 0 -> both classes eligible at every noise level")
    print(f"  labels: {labels_path}")
    if abs(n0 - n1) > 0.1 * max(n0, n1):
        print("  WARNING: classes are not balanced; the classifier will learn the prior")


if __name__ == "__main__":
    main()
