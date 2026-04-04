import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import json
import os

# (kimg, path to fid_out.json)
checkpoints = [
    (1001, "/data/scratch/honjar/generated/baseline_wolves_only_1000kimg/fid_out.json"),
    (2001, "/data/scratch/honjar/generated/baseline_wolves_only_2000kimg/fid_out.json"),
    (3002, "/data/scratch/honjar/generated/baseline_wolves_only_3000kimg/fid_out.json"),
    (4003, "/data/scratch/honjar/generated/baseline_wolves_only_4000kimg/fid_out.json"),
]

kimgs = []
fids = []

for kimg, path in checkpoints:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        kimgs.append(kimg)
        fids.append(data["fid_score"])
        print(f"  {kimg} kimg: FID = {data['fid_score']:.2f}")
    else:
        print(f"  {kimg} kimg: MISSING ({path})")

plt.figure(figsize=(8, 5))
plt.plot(kimgs, fids, 'o-', linewidth=2, markersize=8)
plt.xlabel('Training Duration (kimg)', fontsize=12)
plt.ylabel('FID Score (lower = better)', fontsize=12)
plt.title('Baseline Wolves-Only: FID vs Training Duration', fontsize=14)
plt.grid(True, alpha=0.3)

# annotate each point
for k, f in zip(kimgs, fids):
    plt.annotate(f'{f:.1f}', (k, f), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=10)

plt.tight_layout()
out_path = "/data/scratch/honjar/generated/fid_curve_baseline_wolves.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")
