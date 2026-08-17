#!/usr/bin/env python3
"""Plot FID vs kimg for all 12 training runs."""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GEN_DIR = f"{AMBIENT_BASE}/generated"
data = {}

for dirname in os.listdir(GEN_DIR):
    fid_path = os.path.join(GEN_DIR, dirname, "fid_out.json")
    if not os.path.isfile(fid_path):
        continue
    if "seedB" in dirname:
        continue
    match = re.match(r'^(.+?)_(\d+)kimg$', dirname)
    if not match:
        continue
    dataset_name = match.group(1)
    kimg = int(match.group(2))
    with open(fid_path) as f:
        result = json.load(f)
    fid = result["fid_score"]
    if dataset_name not in data:
        data[dataset_name] = []
    data[dataset_name].append((kimg, fid))

for name in data:
    data[name].sort(key=lambda x: x[0])

# Plot 1: FID curves (two panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
colors_t = plt.cm.tab10(np.linspace(0, 1, 10))

for name in sorted(data.keys()):
    kimgs = [x[0] for x in data[name]]
    fids = [x[1] for x in data[name]]
    if name == "baseline_wolves_only":
        ax1.plot(kimgs, fids, 'k-o', linewidth=2.5, markersize=5, label=name, zorder=10)
        ax2.plot(kimgs, fids, 'k-o', linewidth=2.5, markersize=5, label=name, zorder=10)
    elif name == "baseline_naive_all":
        ax1.plot(kimgs, fids, 'r-s', linewidth=2.5, markersize=5, label=name, zorder=9)
        ax2.plot(kimgs, fids, 'r-s', linewidth=2.5, markersize=5, label=name, zorder=9)
    else:
        idx = int(re.search(r'random_t_vector_(\d+)', name).group(1))
        ax1.plot(kimgs, fids, '-', color=colors_t[idx], linewidth=1.2, alpha=0.7, markersize=3, marker='o', label=f't_vec_{idx:03d}')
        ax2.plot(kimgs, fids, '-', color=colors_t[idx], linewidth=1.2, alpha=0.7, markersize=3, marker='o', label=f't_vec_{idx:03d}')

ax1.set_xlabel('Training kimg', fontsize=12)
ax1.set_ylabel('FID (lower = better)', fontsize=12)
ax1.set_title('FID vs Training Duration - All 12 Runs (Full scale)', fontsize=13)
ax1.legend(fontsize=8, ncol=2, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)

ax2.set_xlabel('Training kimg', fontsize=12)
ax2.set_ylabel('FID (lower = better)', fontsize=12)
ax2.set_title('FID vs Training Duration - Mixed-Data Runs (Zoomed)', fontsize=13)
ax2.set_ylim(148, 180)
ax2.legend(fontsize=8, ncol=2, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)

plt.tight_layout()
plt.savefig(f"{AMBIENT_BASE}/generated/fid_all_runs.png", dpi=150, bbox_inches='tight')
print("Saved: fid_all_runs.png")

# Plot 2: Bar chart of best FID per run
fig2, ax3 = plt.subplots(figsize=(14, 6))
best_fids = []
for name in sorted(data.keys()):
    fids_list = [x[1] for x in data[name]]
    best_fid = min(fids_list)
    best_kimg = [x[0] for x in data[name] if x[1] == best_fid][0]
    best_fids.append((name, best_fid, best_kimg))
best_fids.sort(key=lambda x: x[1])

names = [x[0].replace('random_t_vector_', 'tvec_').replace('_seed', '\nseed') for x in best_fids]
fids_vals = [x[1] for x in best_fids]
kimg_labels = [f'{x[2]}k' for x in best_fids]
bar_colors = ['#2ecc71' if 'wolves_only' in x[0] else '#e74c3c' if 'naive_all' in x[0] else '#3498db' for x in best_fids]

bars = ax3.bar(range(len(names)), fids_vals, color=bar_colors, edgecolor='white')
for i, (bar, fid, kl) in enumerate(zip(bars, fids_vals, kimg_labels)):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{fid:.1f}\n@{kl}', ha='center', va='bottom', fontsize=7)
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels(names, fontsize=7)
ax3.set_ylabel('Best FID (lower = better)', fontsize=12)
ax3.set_title('Best FID Across All 12 Runs (Green=wolves only, Red=naive all, Blue=T-vector)', fontsize=13)
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{AMBIENT_BASE}/generated/fid_best_per_run.png", dpi=150, bbox_inches='tight')
print("Saved: fid_best_per_run.png")

# Print summary
print("\n=== SUMMARY: Best FID per run ===")
print(f"{'Run':<55} {'Best FID':>10} {'@ kimg':>10}")
print("-" * 77)
for name, fid, kimg in best_fids:
    print(f"{name:<55} {fid:>10.2f} {kimg:>10}")
