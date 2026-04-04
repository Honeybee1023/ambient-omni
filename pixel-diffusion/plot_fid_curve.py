import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import json
import os

# Seed A: seeds 0-999
checkpoints_A = [
    (1001, "/data/scratch/honjar/generated/baseline_wolves_only_1000kimg/fid_out.json"),
    (2001, "/data/scratch/honjar/generated/baseline_wolves_only_2000kimg/fid_out.json"),
    (3002, "/data/scratch/honjar/generated/baseline_wolves_only_3000kimg/fid_out.json"),
    (4003, "/data/scratch/honjar/generated/baseline_wolves_only_4000kimg/fid_out.json"),
]

# Seed B: seeds 1000-1999
checkpoints_B = [
    (1001, "/data/scratch/honjar/generated/baseline_wolves_only_1000kimg_seedB/fid_out.json"),
    (2001, "/data/scratch/honjar/generated/baseline_wolves_only_2000kimg_seedB/fid_out.json"),
    (3002, "/data/scratch/honjar/generated/baseline_wolves_only_3000kimg_seedB/fid_out.json"),
    (4003, "/data/scratch/honjar/generated/baseline_wolves_only_4000kimg_seedB/fid_out.json"),
]

def load_fids(checkpoints, label):
    kimgs, fids = [], []
    for kimg, path in checkpoints:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            kimgs.append(kimg)
            fids.append(data["fid_score"])
            print(f"  {label} - {kimg} kimg: FID = {data['fid_score']:.2f}")
        else:
            print(f"  {label} - {kimg} kimg: MISSING ({path})")
    return kimgs, fids

print("Loading FID scores...")
kimgs_A, fids_A = load_fids(checkpoints_A, "Seed A")
kimgs_B, fids_B = load_fids(checkpoints_B, "Seed B")

plt.figure(figsize=(8, 5))
plt.plot(kimgs_A, fids_A, 'o-', linewidth=2, markersize=8, label='Seeds 0-999', color='#2196F3')
plt.plot(kimgs_B, fids_B, 's--', linewidth=2, markersize=8, label='Seeds 1000-1999', color='#FF9800')

plt.xlabel('Training Duration (kimg)', fontsize=12)
plt.ylabel('FID Score (lower = better)', fontsize=12)
plt.title('Baseline Wolves-Only: FID vs Training Duration', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# annotate each point
for k, f in zip(kimgs_A, fids_A):
    plt.annotate(f'{f:.1f}', (k, f), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=9, color='#2196F3')
for k, f in zip(kimgs_B, fids_B):
    plt.annotate(f'{f:.1f}', (k, f), textcoords="offset points",
                 xytext=(0, -18), ha='center', fontsize=9, color='#FF9800')

plt.tight_layout()
out_path = "/data/scratch/honjar/generated/fid_curve_baseline_wolves_both_seeds.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")
