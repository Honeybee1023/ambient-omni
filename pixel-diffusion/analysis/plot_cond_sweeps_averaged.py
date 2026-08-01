#!/usr/bin/env python3
"""Generate noise-averaged conditional sweep plots with error bars.

For points with multiple runs, shows mean ± std.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GENERATED = "/var/local/honjar/generated"
REPO_ROOT = "/var/local/honjar/ambient-omni/pixel-diffusion"
PLOT_DIR = os.path.join(REPO_ROOT, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

B2_SOLO_055 = 0.031144
BASELINE_MIND = 0.04432

# B3 solo sweep (from lysine)
B3_SOLO = {
    0.0: 0.03688, 0.2: 0.03556, 0.4: 0.03391, 0.45: 0.03328,
    0.5: 0.03265, 0.55: 0.03180, 0.6: 0.03329, 0.8: 0.03898,
    0.9: 0.04336, 0.95: 0.04486
}

# B4 solo sweep (from lysine)
B4_SOLO = {
    0.0: 0.03665, 0.2: 0.03577, 0.4: 0.03483, 0.5: 0.03451,
    0.55: 0.03528, 0.6: 0.03618, 0.8: 0.04153, 0.9: 0.04501, 0.95: 0.04615
}

# B5 solo sweep (from lysine)
B5_SOLO = {
    0.0: 0.03699, 0.2: 0.03652, 0.4: 0.03621, 0.5: 0.03656,
    0.55: 0.03706, 0.6: 0.03825, 0.8: 0.04249, 0.9: 0.04506, 0.95: 0.04627
}


def collect_all_runs(sweep_prefix, bucket_label):
    """Collect all runs (original + reruns) for a conditional sweep.
    Returns dict: {t_val: [mind1, mind2, ...]}
    """
    all_runs = {}
    pattern = f"{GENERATED}/mind_celeba_v2b_cond_{sweep_prefix}_T*_*2000kimg.json"
    for fpath in sorted(glob.glob(pattern)):
        basename = os.path.basename(fpath)
        # Parse T value and run number
        parts = basename.replace(f"mind_celeba_v2b_cond_{sweep_prefix}_T", "").replace("_2000kimg.json", "")
        # Could be "055", "0525", "055_r2", "080_r3" etc
        if "_r" in parts:
            t_str, _ = parts.rsplit("_r", 1)
        else:
            t_str = parts
        
        if len(t_str) == 3:
            t_val = int(t_str) / 100
        elif len(t_str) == 4:
            t_val = int(t_str) / 1000
        elif len(t_str) == 2:
            t_val = int(t_str) / 10
        else:
            continue
        
        with open(fpath) as f:
            mind_val = json.load(f)["mind"]
        
        if t_val not in all_runs:
            all_runs[t_val] = []
        all_runs[t_val].append(mind_val)
    
    return all_runs


def plot_sweep_with_errorbars(solo_data, cond_runs, title, xlabel_bucket, filename,
                              extra_cond_pts=None):
    """Plot sweep with error bars for points that have multiple runs."""
    # Solo curve
    solo_ts = sorted(solo_data.keys())
    solo_minds = [solo_data[t] for t in solo_ts]
    
    # Conditional: merge local + any extra (lysine) points
    all_cond = dict(cond_runs)  # {t: [list of values]}
    if extra_cond_pts:
        for t, v in extra_cond_pts.items():
            if t not in all_cond:
                all_cond[t] = []
            if isinstance(v, list):
                all_cond[t].extend(v)
            else:
                all_cond[t].append(v)
    
    # Add T=1.0 (bucket off = B2 solo)
    all_cond[1.0] = [B2_SOLO_055]
    
    cond_ts = sorted(all_cond.keys())
    cond_means = [np.mean(all_cond[t]) for t in cond_ts]
    cond_stds = [np.std(all_cond[t]) if len(all_cond[t]) > 1 else 0 for t in cond_ts]
    cond_n = [len(all_cond[t]) for t in cond_ts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Solo
    ax.plot(solo_ts, solo_minds, 'b-o', label=f'{xlabel_bucket} solo (B2 inactive)', 
            markersize=5, alpha=0.7)
    
    # Conditional with error bars
    ax.errorbar(cond_ts, cond_means, yerr=cond_stds, fmt='r-s', 
                label=f'{xlabel_bucket} with B2=0.55 (mean ± std)', 
                markersize=6, capsize=3, capthick=1.5)
    
    # Mark points with multiple runs in green
    multi_ts = [t for t in cond_ts if len(all_cond[t]) > 1]
    multi_means = [np.mean(all_cond[t]) for t in multi_ts]
    ax.plot(multi_ts, multi_means, 'gs', markersize=10, alpha=0.3, 
            label=f'Averaged ({len(multi_ts)} pts with 2-3 runs)')
    
    # Reference lines
    ax.axhline(B2_SOLO_055, color='gray', linestyle='--', alpha=0.7, 
               label=f'B2 solo ({B2_SOLO_055:.5f})')
    ax.axhline(BASELINE_MIND, color='lightgray', linestyle=':', alpha=0.5)
    
    # Circle the minimum
    min_idx = np.argmin(cond_means)
    ax.plot(cond_ts[min_idx], cond_means[min_idx], 'o',
            markersize=14, markeredgecolor='black', markeredgewidth=2, markerfacecolor='none')
    
    ax.set_xlabel(f'T (noise threshold for {xlabel_bucket})')
    ax.set_ylabel('MIND (lower is better)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outpath = os.path.join(PLOT_DIR, filename)
    plt.savefig(outpath, dpi=150)
    plt.close()
    
    print(f"  Saved: {outpath}")
    print(f"  Min: T={cond_ts[min_idx]:.3f}, MIND={cond_means[min_idx]:.6f} ± {cond_stds[min_idx]:.6f} (n={cond_n[min_idx]})")
    print(f"  vs B2 solo: {cond_means[min_idx]-B2_SOLO_055:+.6f}")
    
    return cond_ts, cond_means, cond_stds


def main():
    print("=" * 60)
    print("Noise-averaged conditional sweep plots")
    print("=" * 60)
    
    # --- B3 ---
    print("\n--- B3 sweep (B2=0.55 fixed) ---")
    b3_runs = collect_all_runs("b3", "B3")
    # Add lysine original points (these were the first runs)
    b3_lysine_only = {
        0.0: 0.03566, 0.2: 0.03437, 0.4: 0.03217, 0.45: 0.03196,
        0.5: 0.03162, 0.55: 0.03110, 0.6: 0.03174,
        0.8: 0.03356, 0.9: 0.03423, 0.95: 0.03469
    }
    for t in sorted(b3_runs.keys()):
        print(f"    T={t:.3f}: {b3_runs[t]}  mean={np.mean(b3_runs[t]):.6f}")
    
    b3_ts, b3_means, b3_stds = plot_sweep_with_errorbars(
        solo_data=B3_SOLO,
        cond_runs=b3_runs,
        title="B3 sweep: solo vs conditional on B2=0.55 (noise-averaged)",
        xlabel_bucket="B3",
        filename="cond_b3_sweep_averaged.png",
        extra_cond_pts={t: v for t, v in b3_lysine_only.items()}
    )
    
    # --- B4 ---
    print("\n--- B4 sweep (B2=0.55 fixed) ---")
    b4_runs = collect_all_runs("b4", "B4")
    # Add lysine original points
    b4_lysine_only = {
        0.7: 0.03202, 0.8: 0.03085, 0.9: 0.03167
    }
    for t in sorted(b4_runs.keys()):
        print(f"    T={t:.3f}: {b4_runs[t]}  mean={np.mean(b4_runs[t]):.6f}")
    
    b4_ts, b4_means, b4_stds = plot_sweep_with_errorbars(
        solo_data=B4_SOLO,
        cond_runs=b4_runs,
        title="B4 sweep: solo vs conditional on B2=0.55 (noise-averaged)",
        xlabel_bucket="B4",
        filename="cond_b4_sweep_averaged.png",
        extra_cond_pts={t: v for t, v in b4_lysine_only.items()}
    )
    
    # --- B5 (no reruns, just single runs) ---
    print("\n--- B5 sweep (B2=0.55 fixed) ---")
    b5_runs = collect_all_runs("b5", "B5")
    for t in sorted(b5_runs.keys()):
        print(f"    T={t:.3f}: {b5_runs[t]}  mean={np.mean(b5_runs[t]):.6f}")
    
    plot_sweep_with_errorbars(
        solo_data=B5_SOLO,
        cond_runs=b5_runs,
        title="B5 sweep: solo vs conditional on B2=0.55",
        xlabel_bucket="B5",
        filename="cond_b5_sweep_averaged.png"
    )
    
    # --- Combined comparison with error bars ---
    print("\n--- Combined comparison ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sweeps = [
        ("B3|B2=0.55", b3_runs, b3_lysine_only, 'tab:blue', 's'),
        ("B4|B2=0.55", b4_runs, b4_lysine_only, 'tab:red', 'o'),
        ("B5|B2=0.55", b5_runs, {}, 'tab:green', '^'),
    ]
    
    for label, runs, extra, color, marker in sweeps:
        all_pts = dict(runs)
        for t, v in extra.items():
            if t not in all_pts:
                all_pts[t] = []
            all_pts[t].append(v)
        all_pts[1.0] = [B2_SOLO_055]
        
        ts = sorted(all_pts.keys())
        means = [np.mean(all_pts[t]) for t in ts]
        stds = [np.std(all_pts[t]) if len(all_pts[t]) > 1 else 0 for t in ts]
        
        ax.errorbar(ts, means, yerr=stds, fmt=f'-{marker}', color=color, 
                    label=label, markersize=6, capsize=2)
        min_idx = np.argmin(means)
        ax.plot(ts[min_idx], means[min_idx], 'o', markersize=12, 
                markeredgecolor='black', markeredgewidth=2, markerfacecolor='none')
    
    ax.axhline(B2_SOLO_055, color='gray', linestyle='--', alpha=0.7, 
               label=f'B2 solo ({B2_SOLO_055:.5f})')
    ax.set_xlabel('T (noise threshold for added bucket)')
    ax.set_ylabel('MIND (lower is better)')
    ax.set_title('Conditional sweeps: B3/B4/B5 with B2=0.55 (noise-averaged)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outpath = os.path.join(PLOT_DIR, "cond_sweeps_combined_averaged.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")
    
    # --- Summary table ---
    print("\n" + "=" * 60)
    print("FINAL SUMMARY (noise-averaged)")
    print("=" * 60)
    print(f"B2 solo: {B2_SOLO_055:.6f}")
    print()
    
    for name, runs, extra in [("B3", b3_runs, b3_lysine_only), ("B4", b4_runs, b4_lysine_only), ("B5", b5_runs, {})]:
        all_pts = dict(runs)
        for t, v in extra.items():
            if t not in all_pts:
                all_pts[t] = []
            all_pts[t].append(v)
        all_pts[1.0] = [B2_SOLO_055]
        
        ts = sorted(all_pts.keys())
        means = {t: np.mean(all_pts[t]) for t in ts}
        min_t = min(means, key=means.get)
        n_runs = len(all_pts[min_t])
        std = np.std(all_pts[min_t]) if n_runs > 1 else float('nan')
        print(f"  {name}: min at T={min_t:.3f}, MIND={means[min_t]:.6f} ± {std:.6f} (n={n_runs}), vs B2: {means[min_t]-B2_SOLO_055:+.6f}")


if __name__ == "__main__":
    main()
