"""
Update celeba_2k_analysis.json with all 161 dense 2-domain points.
Fixed T suffix parsing: 3-digit → /100, 4-digit → /1000.
"""
import json
import os
import glob

METRICS_DIR = "/data/scratch/honjar/generated"
OUTPUT = os.path.join(METRICS_DIR, "celeba_2k_analysis.json")

with open(OUTPUT) as f:
    existing = json.load(f)

def parse_t_suffix(suffix):
    """Parse T suffix to float. 3-digit → /100, 4-digit → /1000."""
    try:
        n = int(suffix)
        if len(suffix) == 3:
            return n / 100.0
        elif len(suffix) == 4:
            return n / 1000.0
        elif len(suffix) == 2:
            return n / 100.0  # unlikely but handle
        else:
            return n / (10 ** len(suffix))
    except ValueError:
        return None

new_data_2k = {}
for b in range(1, 8):
    points = {}
    pattern = os.path.join(METRICS_DIR, f"metrics_celeba_2d_b{b}_T*_2000kimg.json")
    for fpath in sorted(glob.glob(pattern)):
        basename = os.path.basename(fpath)
        after_b = basename.split(f"_b{b}_T")[1]
        t_suffix = after_b.replace("_2000kimg.json", "")
        
        t_val = parse_t_suffix(t_suffix)
        if t_val is None:
            print(f"  WARNING: can't parse '{t_suffix}' in {basename}")
            continue
        
        try:
            with open(fpath) as f:
                d = json.load(f)
            fid = d.get("fid_score") or d.get("results", {}).get("fid50k_full")
            if fid is not None:
                points[t_val] = round(fid, 2)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    sorted_ts = sorted(points.keys())
    new_data_2k[str(b)] = {
        "T": [t for t in sorted_ts],
        "FID": [points[t] for t in sorted_ts]
    }
    print(f"  B{b}: {len(sorted_ts)} points, T range [{sorted_ts[0]:.3f}, {sorted_ts[-1]:.3f}]")

existing['data_2k'] = new_data_2k

with open(OUTPUT, 'w') as f:
    json.dump(existing, f, indent=2)
print(f"\nSaved. Now verify no more suspicious dips:")
for b in [5, 6, 7]:
    ts = new_data_2k[str(b)]["T"]
    fids = new_data_2k[str(b)]["FID"]
    for t, fid in zip(ts, fids):
        if t < 0.15 and fid < 30:
            print(f"  STILL SUSPICIOUS: B{b} T={t:.4f} FID={fid:.2f}")
print("(no output above = all clean)")
