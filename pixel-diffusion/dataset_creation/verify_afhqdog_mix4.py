#!/usr/bin/env python3
"""Check afhqdog_mix4_v1 / afhqdog_cls_mix4 / the clean reference sets ON DISK, and draw an example grid.

Checks (any failure exits non-zero):
  * counts per group (b0 474, g03 1067, g05/g10/g20 1066), reference 4,739, val 500, cls 1,896 + 1,896
  * every file is a real PNG, RGB, 64x64; annotations match the files; sigma_min 0 for b0, 999 otherwise
  * groups disjoint and together exactly the 4,739 train dogs; val shares no id AND no identical image
  * every blurred file is byte-for-byte the recipe applied to its clean reference copy (all 4,265)
  * cls set: labels balanced, clean copies resolve to b0 files, blurred copies equal the recipe
  * mean grey level of blurred minus clean for the same dog: rounded (what we wrote) vs truncated
    (the old CelebA conversion), to show the bias the fix removes

The grid (4 dogs x [clean, 0.3, 0.5, 1.0, 2.0], from the paired cls set so each row is ONE dog,
upscaled 4x nearest) goes to $AMBIENT_BASE/generated/afhqdog_examples.png.
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import hashlib, json, os, sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_afhqdog_mix4 import LEVELS, PROC, blur  # noqa: E402

D = f"{AMBIENT_BASE}/annotated_datasets"
MIX, CLS = f"{D}/afhqdog_mix4_v1", f"{D}/afhqdog_cls_mix4"
REF, VAL = f"{PROC}/train_clean_64", f"{PROC}/val_64"
EXPECT = {"b0": 474, "g03": 1067, "g05": 1066, "g10": 1066, "g20": 1066}
SIG = dict(LEVELS)
fails = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        fails.append(msg)


def load(p):
    im = Image.open(p)
    return im.format, im.mode, im.size, np.array(im.convert("RGB"))


def main():
    print("== afhqdog_mix4_v1")
    files = sorted(f for f in os.listdir(MIX) if f != "annotations.jsonl")
    by = {}
    for f in files:
        by.setdefault(f.split("_", 1)[0], []).append(f.split("_", 1)[1][:-4])
    check({g: len(v) for g, v in by.items()} == EXPECT, f"group counts {({g: len(v) for g, v in by.items()})}")
    ref_ids = sorted(f[:-4] for f in os.listdir(REF))
    val_ids = sorted(f[:-4] for f in os.listdir(VAL))
    check(len(ref_ids) == 4739 and len(val_ids) == 500, f"reference {len(ref_ids)}, val {len(val_ids)}")
    allids = [i for v in by.values() for i in v]
    check(len(allids) == len(set(allids)), "groups disjoint (no dog twice)")
    check(sorted(allids) == ref_ids, "groups cover exactly the 4,739 train dogs")
    check(not set(val_ids) & set(ref_ids), "val ids disjoint from train ids")

    ann = [json.loads(l) for l in open(f"{MIX}/annotations.jsonl")]
    check(sorted(a["filename"] for a in ann) == files, f"annotations ({len(ann)}) match files ({len(files)})")
    check(all((a["sigma_min"] == 0.0) == a["filename"].startswith("b0_") and a["sigma_min"] in (0.0, 999.0)
              and a["sigma_max"] == 0.0 for a in ann), "sigma_min 0 for b0, 999 sentinel for blurred")
    meta = json.load(open(f"{MIX}.meta.json"))
    check(all(sorted(meta["ids"][g]) == sorted(by[g]) for g in EXPECT), "meta.json id lists match disk")

    bad_fmt, mism, diffs = [], [], {g: [] for g in SIG}
    trunc = {g: [] for g in SIG}
    ref_hash = {}
    for i in ref_ids:
        fmt, mode, size, a = load(f"{REF}/{i}.png")
        if (fmt, mode, size) != ("PNG", "RGB", (64, 64)):
            bad_fmt.append(f"ref/{i}")
        ref_hash[hashlib.md5(a.tobytes()).hexdigest()] = i
    for f in files:
        g, i = f.split("_", 1)[0], f.split("_", 1)[1][:-4]
        fmt, mode, size, a = load(f"{MIX}/{f}")
        if (fmt, mode, size) != ("PNG", "RGB", (64, 64)):
            bad_fmt.append(f)
        clean_im = Image.open(f"{REF}/{i}.png").convert("RGB")
        c = np.array(clean_im)
        if g == "b0":
            if not np.array_equal(a, c):
                mism.append(f)
            continue
        if not np.array_equal(a, np.array(blur(clean_im, SIG[g]))):
            mism.append(f)
        diffs[g].append(a.astype(np.float64).mean() - c.astype(np.float64).mean())
        t = np.clip(gaussian_filter(c.astype(np.float32), sigma=(SIG[g], SIG[g], 0)), 0, 255).astype(np.uint8)
        trunc[g].append(t.astype(np.float64).mean() - c.astype(np.float64).mean())
    val_dup = []
    for i in val_ids:
        fmt, mode, size, a = load(f"{VAL}/{i}.png")
        if (fmt, mode, size) != ("PNG", "RGB", (64, 64)):
            bad_fmt.append(f"val/{i}")
        h = hashlib.md5(a.tobytes()).hexdigest()
        if h in ref_hash:
            val_dup.append((i, ref_hash[h]))
    check(not bad_fmt, f"all PNG / RGB / 64x64 ({len(bad_fmt)} bad, e.g. {bad_fmt[:3]})")
    check(not mism, f"every file equals the recipe applied to its clean reference ({len(mism)} mismatches, e.g. {mism[:3]})")
    check(not val_dup, f"no val image is pixel-identical to a train image ({len(val_dup)} found, e.g. {val_dup[:3]})")
    print("  mean grey level, blurred - clean of the same dog (all dogs per group):")
    for g in SIG:
        d, t = np.array(diffs[g]), np.array(trunc[g])
        print(f"    {g} (sigma {SIG[g]}): rounded mean {d.mean():+.4f} max|.| {np.abs(d).max():.4f}   "
              f"| truncated (old CelebA) mean {t.mean():+.4f}")
    d03 = np.abs(np.array(diffs["g03"]))
    check(np.abs(np.array(diffs["g03"])).mean() < 0.05, f"sigma 0.3: |rounded diff| mean {d03.mean():.4f} < 0.05")
    print("    first 5 g03 dogs:", ", ".join(f"{x:+.4f}" for x in diffs["g03"][:5]))

    print("== afhqdog_cls_mix4")
    lab = [json.loads(l) for l in open(f"{CLS}/cls_labels.jsonl")]
    cann = [json.loads(l) for l in open(f"{CLS}/annotations.jsonl")]
    n0 = sum(l["label"] == 0 for l in lab)
    check(n0 == 1896 and len(lab) - n0 == 1896, f"labels {n0} clean / {len(lab) - n0} blurred")
    cfiles = sorted(f for f in os.listdir(CLS) if f.endswith(".png"))
    check(sorted(l["image_file"] for l in lab) == cfiles == sorted(a["filename"] for a in cann),
          "labels, annotations and files agree")
    check(all(a["sigma_min"] == 0.0 for a in cann), "all sigma_min 0 (both classes at every noise level)")
    dogs = sorted(f[3:-4] for f in cfiles if f.startswith("b0_"))
    check(dogs == sorted(by["b0"]), "cls dogs are exactly the 474 clean dogs")
    cm = []
    for f in cfiles:
        pre, i = f.split("_", 1)[0], f.split("_", 1)[1][:-4]
        a = np.array(Image.open(f"{CLS}/{f}").convert("RGB"))
        c = Image.open(f"{REF}/{i}.png").convert("RGB")
        if pre.startswith("b0"):
            ok = np.array_equal(a, np.array(c)) and os.path.realpath(f"{CLS}/{f}") == os.path.realpath(f"{MIX}/b0_{i}.png")
        else:
            ok = np.array_equal(a, np.array(blur(c, int(pre[2:5]) / 100)))
        if not ok:
            cm.append(f)
    check(not cm, f"cls clean copies resolve to b0 files, blurred copies equal the recipe ({len(cm)} bad)")

    # Grid: 4 dogs x (clean, 0.3, 0.5, 1.0, 2.0), one dog per row.
    rows = []
    for i in dogs[:4]:
        tiles = [Image.open(f"{CLS}/b0_{i}.png")] + [Image.open(f"{CLS}/bl{round(s * 100):03d}self_{i}.png")
                                                     for _, s in LEVELS]
        rows.append(np.concatenate([np.array(t.convert("RGB")) for t in tiles], axis=1))
    grid = Image.fromarray(np.concatenate(rows, axis=0)).resize((5 * 256, 4 * 256), Image.NEAREST)
    gp = f"{AMBIENT_BASE}/generated/afhqdog_examples.png"
    grid.save(gp)
    print(f"grid (rows: {dogs[:4]}; columns: clean, 0.3, 0.5, 1.0, 2.0) -> {gp}")

    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
