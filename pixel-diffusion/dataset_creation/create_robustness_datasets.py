"""Four robustness datasets: one thing changed each time from the base setting.

Base setting: 500 clean + 26,014 blurred (sigma_b = 0.5), built by
create_dynamic_t_dataset_v2.py. Each dataset below changes exactly one thing and keeps
everything else identical, so a schedule's number can be compared across settings.

    rb_c250    250 clean   + the same 26,014 blurred at 0.5   (first 250 b0 files by name,
                                                               a strict subset of the base clean set)
    rb_c1000   1000 clean  + the same 26,014 blurred at 0.5   (the 500 b0 files plus 500 more,
                                                               made clean from raw; see below)
    rb_c1250   1250 clean  + the same 26,014 blurred at 0.5   (the crossover test: the analysis
                                                               puts the point where blurred data
                                                               stops being worth anything at
                                                               ~1150-1250 clean images)
    rb_b03     500 clean   + the SAME 26,014 source images blurred at 0.3
    rb_b10     500 clean   + the SAME 26,014 source images blurred at 1.0
    rb_c250b10 250 clean   + the SAME 26,014 source images blurred at 1.0  (the corner where
               scarce clean data and heavy blur meet: the exposure target wants a late drop and
               the recovery floor caps it at 300 kimg, a figure measured at blur 0.5 where
               recovery is about twice as fast. Predicted failure case, built to be measured.)

The blurred sets for 0.3 and 1.0 are regenerated from raw for the b5 index list rather than
borrowed from buckets b3/b7, which hold different subsets of images: reusing them would change
the image set and the blur strength at once.

The 500 extra clean faces come from bucket b1 (sigma_b = 0.1), which no run of ours has ever
trained on, processed at sigma 0 and renamed b0_<rawindex>.jpg so the loader treats them as
clean. They are checked against the 20,000-file holdout list, which must stay disjoint: MIND is
scored against that holdout with a fixed cache in every setting.

Processing matches the v2b pipeline exactly: PIL open, convert RGB, resize to 64x64 with
LANCZOS, scipy.ndimage.gaussian_filter(sigma=(s, s, 0)) when s > 0, clip, uint8.

Annotations mirror create_dynamic_t_dataset_v2.py: clean sigma_min 0.0, the blurred bucket
sigma_min 999.0 (the schedule sentinel), nothing parked (these are b0+b5-only sets).

Usage:  python dataset_creation/create_robustness_datasets.py [--only rb_b03]
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or next(
    (_p for _p in ("/data-local/honjar", "/var/local/honjar", "/data/scratch/honjar")
     if _os.path.isdir(_p)), "/data/scratch/honjar")

import argparse, json, os, shutil, sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SRC = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
RAW = f"{AMBIENT_BASE}/celeba_raw/img_align_celeba"
SPLIT = f"{AMBIENT_BASE}/celeba_processed_v2b/celeba_split_v2b.json"
OUT_ROOT = f"{AMBIENT_BASE}/annotated_datasets"
IMG_ROOT = f"{AMBIENT_BASE}/celeba_robustness_64"      # generated images live here
SENTINEL = 999.0
RES = 64


def raw_ids(prefix):
    """Raw indices of one bucket, in sorted filename order."""
    return [f[len(prefix) + 1:-4] for f in sorted(os.listdir(SRC))
            if f.startswith(prefix + "_") and f.endswith(".jpg")]


def holdout_ids():
    with open(SPLIT) as f:
        d = json.load(f)
    out = set()
    for h in d["holdout_files"]:
        out.add(os.path.splitext(os.path.basename(str(h)))[0].lstrip("0") or "0")
    return out


def process(raw_id, sigma, dst):
    src = os.path.join(RAW, f"{raw_id}.jpg")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    a = np.array(Image.open(src).convert("RGB").resize((RES, RES), Image.LANCZOS), dtype=np.float32)
    if sigma > 0:
        a = gaussian_filter(a, sigma=(sigma, sigma, 0))
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(dst, quality=95)


def make_images(ids, sigma, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for i, rid in enumerate(ids):
        dst = os.path.join(out_dir, f"{prefix}_{rid}.jpg")
        if not os.path.exists(dst):
            process(rid, sigma, dst); made += 1
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(ids)}", flush=True)
    print(f"  {out_dir}: {len(ids)} images ({made} new, sigma={sigma})")
    return out_dir


def build(name, clean, blurred):
    """clean / blurred: lists of (abs_path, filename-in-dataset)."""
    out = os.path.join(OUT_ROOT, name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    ann = []
    for src, fname in clean:
        os.symlink(src, os.path.join(out, fname))
        ann.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})
    for src, fname in blurred:
        os.symlink(src, os.path.join(out, fname))
        ann.append({"filename": fname, "sigma_min": SENTINEL, "sigma_max": 0.0})
    with open(os.path.join(out, "annotations.jsonl"), "w") as f:
        for a in ann:
            f.write(json.dumps(a) + "\n")
    print(f"{out}: {len(clean)} clean + {len(blurred)} blurred = {len(ann)} annotations")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); a = ap.parse_args()
    b0, b1, b5 = raw_ids("b0"), raw_ids("b1"), raw_ids("b5")
    print(f"buckets: b0 {len(b0)}, b1 {len(b1)}, b5 {len(b5)}")
    hold = holdout_ids()
    base_clean = [(os.path.join(SRC, f"b0_{r}.jpg"), f"b0_{r}.jpg") for r in b0]
    base_b5 = [(os.path.join(SRC, f"b5_{r}.jpg"), f"b5_{r}.jpg") for r in b5]

    want = lambda n: a.only is None or a.only == n

    if want("rb_c250"):
        build("rb_c250", base_clean[:250], base_b5)

    def extra_clean_faces(n_extra):
        """n_extra clean faces beyond the 500, taken in order from bucket b1 (sigma_b 0.1), which
        no run has ever trained on, processed at sigma 0. Refuses on any overlap with the MIND
        holdout. rb_c1000 uses the first 500 of this same series and rb_c1250 the first 750, so
        the larger set is a superset of the smaller one."""
        extra = b1[:n_extra]
        clash = [r for r in extra if (r.lstrip("0") or "0") in hold]
        print(f"  extra clean from b1: {len(extra)} ids, first {extra[0]}, last {extra[-1]}, "
              f"in holdout: {len(clash)}")
        if clash:
            sys.exit(f"REFUSING: {len(clash)} of the extra clean faces are in the MIND holdout")
        d = make_images(extra, 0.0, os.path.join(IMG_ROOT, "extra_clean"), "b0")
        with open(os.path.join(IMG_ROOT, f"extra_clean_ids_{n_extra}.json"), "w") as f:
            json.dump({"source_bucket": "b1", "sigma": 0.0, "n": len(extra), "raw_ids": extra}, f)
        return [(os.path.join(d, f"b0_{r}.jpg"), f"b0_{r}.jpg") for r in extra]

    if want("rb_c1000"):
        build("rb_c1000", base_clean + extra_clean_faces(500), base_b5)

    if want("rb_c1250"):
        build("rb_c1250", base_clean + extra_clean_faces(750), base_b5)

    for name, sigma, n_clean in (("rb_b03", 0.3, 500), ("rb_b10", 1.0, 500), ("rb_c250b10", 1.0, 250)):
        if not want(name):
            continue
        d = make_images(b5, sigma, os.path.join(IMG_ROOT, f"blur{str(sigma).replace('.','')}"), "b5")
        blurred = [(os.path.join(d, f"b5_{r}.jpg"), f"b5_{r}.jpg") for r in b5]
        build(name, base_clean[:n_clean], blurred)


if __name__ == "__main__":
    main()
