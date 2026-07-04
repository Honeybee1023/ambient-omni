#!/usr/bin/env python3
"""Generate conditional sweep plots from all available results.

Reads MIND JSON files from /var/local/honjar/generated/ and produces PNG plots
in plots/ directory. Includes hardcoded reference points from lysine for
sweeps that were partially run there.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GENERATED = "/var/local/honjar/generated"
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Reference values (from lysine/CSAIL v2b_all_results.json) ---
# B2 solo at T=0.55 (used as B4/B5 "off" baseline)
B2_SOLO_055 = 0.031144  # B2 solo optimum

# Baseline (clean only)
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

# B3 conditional sweep original points (from lysine, B2=0.55 fixed)
B3_COND_LYSINE = {
    0.0: 0.03566, 0.2: 0.03437, 0.4: 0.03217, 0.45: 0.03196,
    0.5: 0.03162, 0.55: 0.03110, 0.6: 0.03174, 0.8: 0.03356,
    0.9: 0.03423, 0.95: 0.03469
}

# B4 conditional from lysine (these were on lysine)
B4_COND_LYSINE = {
    0.7: 0.03202, 0.8: 0.03085, 0.9: 0.03167
}


def load_mind_results(pattern):
    """Load MIND values from JSON files matching a glob pattern."""
    results = {}
    for fpath in sorted(glob.glob(pattern)):
        basename = os.path.basename(fpath)
        # Extract T value from filename like mind_celeba_v2b_cond_b4_T045_2000kimg.json
        # or mind_celeba_v2b_cond_b4_T0525_2000kimg.json
        parts = basename.split("_T")
        if len(parts) < 2:
            continue
        t_str = parts[-1].split("_2000kimg")[0]
        # Skip reruns for now (handle separately)
        if t_str.endswith("_r2"):
            continue
        if len(t_str) == 3:  # e.g., "045" -> 0.45
            t_val = int(t_str) / 100
        elif len(t_str) == 4:  # e.g., "0525" -> 0.525
            t_val = int(t_str) / 1000
        else:
            continue
        with open(fpath) as f:
            data = json.load(f)
        results[t_val] = data["mind"]
    return results


def load_rerun_results(pattern_prefix):
    """Load rerun results (files ending in _r2_2000kimg.json)."""
    results = {}
    for fpath in sorted(glob.glob(f"{pattern_prefix}*_r2_2000kimg.json")):
        basename = os.path.basename(fpath)
        parts = basename.split("_T")
        if len(parts) < 2:
            continue
        t_str = parts[-1].split("_r2_2000kimg")[0]
        if len(t_str) == 3:
            t_val = int(t_str) / 100
        elif len(t_str) == 4:
            t_val = int(t_str) / 1000
        else:
            continue
        with open(fpath) as f:
            data = json.load(f)
        results[t_val] = data["mind"]
    return results


def plot_sweep(solo_data, cond_data, rerun_data, title, xlabel_bucket, 
               off_value, off_label, filename, cond_label="with B2 fixed at T=0.55"):
    """Generate a single sweep plot."""
    solo_ts = sorted(solo_data.keys())
    solo_minds = [solo_data[t] for t in solo_ts]
    
    # Add T=1.0 point to conditional (bucket off = B2 solo)
    cond_with_off = dict(cond_data)
    cond_with_off[1.0] = off_value
    cond_ts = sorted(cond_with_off.keys())
    cond_minds = [cond_with_off[t] for t in cond_ts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(solo_ts, solo_minds, 'b-o', label=f'{xlabel_bucket} solo (B2 inactive)', markersize=5)
    ax.plot(cond_ts, cond_minds, 'r-s', label=f'{xlabel_bucket} {cond_label}', markersize=6)
    ax.axhline(BASELINE_MIND, color='gray', linestyle='--', alpha=0.7, label=f'Baseline ({BASELINE_MIND:.5f})')
    
    # Plot reruns as green triangles
    if rerun_data:
        rerun_ts = sorted(rerun_data.keys())
        rerun_minds = [rerun_data[t] for t in rerun_ts]
        ax.plot(rerun_ts, rerun_minds, 'g^', label='Reruns', markersize=10, zorder=5)
    
    # Black circles around minima
    solo_min_idx = np.argmin(solo_minds)
    cond_min_idx = np.argmin(cond_minds)
    ax.plot(solo_ts[solo_min_idx], solo_minds[solo_min_idx], 'o',
            markersize=14, markeredgecolor='black', markeredgewidth=2, markerfacecolor='none')
    ax.plot(cond_ts[cond_min_idx], cond_minds[cond_min_idx], 'o',
            markersize=14, markeredgecolor='black', markeredgewidth=2, markerfacecolor='none')
    
    ax.set_xlabel(f'T (noise threshold for {xlabel_bucket})')
    ax.set_ylabel('MIND (lower is better)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outpath = os.path.join(PLOT_DIR, filename)
    plt.savefig(outpath, dpi=150)
    plt.close()
    
    print(f"  Saved: {outpath}")
    print(f"  Solo min:  T={solo_ts[solo_min_idx]:.3f}, MIND={solo_minds[solo_min_idx]:.6f}")
    print(f"  Cond min:  T={cond_ts[cond_min_idx]:.3f}, MIND={cond_minds[cond_min_idx]:.6f}")
    print(f"  {xlabel_bucket} off (T=1.0): MIND={off_value:.6f}")
    return outpath


def main():
    print("=" * 60)
    print("Generating conditional sweep plots")
    print("=" * 60)
    
    # --- B3 conditional sweep (B2=0.55 fixed, sweep B3) ---
    print("\n--- B3 sweep (B2=0.55 fixed) ---")
    b3_cond_proline = load_mind_results(f"{GENERATED}/mind_celeba_v2b_cond_b3_*_2000kimg.json")
    b3_reruns = load_rerun_results(f"{GENERATED}/mind_celeba_v2b_cond_b3")
    # Merge lysine + proline results (proline overrides if same T)
    b3_cond_all = dict(B3_COND_LYSINE)
    b3_cond_all.update(b3_cond_proline)
    print(f"  Points: {len(b3_cond_all)} ({len(B3_COND_LYSINE)} lysine + {len(b3_cond_proline)} proline)")
    print(f"  Reruns: {b3_reruns}")
    
    plot_sweep(
        solo_data=B3_SOLO,
        cond_data=b3_cond_all,
        rerun_data=b3_reruns,
        title="B3 sweep: solo vs conditional on B2=0.55",
        xlabel_bucket="B3",
        off_value=B2_SOLO_055,
        off_label="B2 solo (B3 inactive)",
        filename="cond_b3_sweep.png"
    )
    
    # --- B4 conditional sweep (B2=0.55 fixed, sweep B4) ---
    print("\n--- B4 sweep (B2=0.55 fixed) ---")
    b4_cond_proline = load_mind_results(f"{GENERATED}/mind_celeba_v2b_cond_b4_*_2000kimg.json")
    b4_reruns = load_rerun_results(f"{GENERATED}/mind_celeba_v2b_cond_b4")
    # Merge
    b4_cond_all = dict(b4_cond_proline)
    b4_cond_all.update(B4_COND_LYSINE)  # lysine has T=0.7, 0.8, 0.9
    # Replace anomalous T=0.525 with rerun if available
    if 0.525 in b4_reruns:
        b4_cond_all[0.525] = b4_reruns[0.525]
        print(f"  Replaced B4 T=0.525 with rerun: {b4_reruns[0.525]:.6f}")
    print(f"  Points: {len(b4_cond_all)}")
    print(f"  Reruns: {b4_reruns}")
    
    plot_sweep(
        solo_data=B4_SOLO,
        cond_data=b4_cond_all,
        rerun_data=b4_reruns,
        title="B4 sweep: solo vs conditional on B2=0.55",
        xlabel_bucket="B4",
        off_value=B2_SOLO_055,
        off_label="B2 solo (B4 inactive)",
        filename="cond_b4_sweep.png"
    )
    
    # --- B5 conditional sweep (B2=0.55 fixed, sweep B5) ---
    print("\n--- B5 sweep (B2=0.55 fixed) ---")
    b5_cond_proline = load_mind_results(f"{GENERATED}/mind_celeba_v2b_cond_b5_*_2000kimg.json")
    b5_reruns = load_rerun_results(f"{GENERATED}/mind_celeba_v2b_cond_b5")
    print(f"  Points: {len(b5_cond_proline)}")
    
    plot_sweep(
        solo_data=B5_SOLO,
        cond_data=b5_cond_proline,
        rerun_data=b5_reruns,
        title="B5 sweep: solo vs conditional on B2=0.55",
        xlabel_bucket="B5",
        off_value=B2_SOLO_055,
        off_label="B2 solo (B5 inactive)",
        filename="cond_b5_sweep.png"
    )
    
    # --- Combined comparison plot ---
    print("\n--- Combined: optimal T shift across buckets ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For each conditional sweep, plot the curve
    sweeps = [
        ("B3|B2=0.55", b3_cond_all, 'tab:blue', 's'),
        ("B4|B2=0.55", b4_cond_all, 'tab:red', 'o'),
        ("B5|B2=0.55", b5_cond_proline, 'tab:green', '^'),
    ]
    
    for label, data, color, marker in sweeps:
        # Add T=1.0
        d = dict(data)
        d[1.0] = B2_SOLO_055
        ts = sorted(d.keys())
        minds = [d[t] for t in ts]
        ax.plot(ts, minds, f'-{marker}', color=color, label=label, markersize=6)
        # Mark minimum
        min_idx = np.argmin(minds)
        ax.plot(ts[min_idx], minds[min_idx], 'o',
                markersize=12, markeredgecolor='black', markeredgewidth=2, markerfacecolor='none')
    
    ax.axhline(B2_SOLO_055, color='gray', linestyle='--', alpha=0.7, label=f'B2 solo ({B2_SOLO_055:.5f})')
    ax.axhline(BASELINE_MIND, color='lightgray', linestyle=':', alpha=0.7, label=f'Baseline ({BASELINE_MIND:.5f})')
    ax.set_xlabel('T (noise threshold for added bucket)')
    ax.set_ylabel('MIND (lower is better)')
    ax.set_title('Conditional sweeps comparison: B3/B4/B5 with B2=0.55 fixed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outpath = os.path.join(PLOT_DIR, "cond_sweeps_combined.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")
    
    # --- Print summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY: Conditional sweep minima (B2=0.55 fixed)")
    print("=" * 60)
    print(f"{'Sweep':<20} {'Best T':<10} {'MIND':<12} {'vs B2 solo':<12} {'Helps?'}")
    print("-" * 60)
    
    for name, data in [("B3|B2", b3_cond_all), ("B4|B2", b4_cond_all), ("B5|B2", b5_cond_proline)]:
        ts = sorted(data.keys())
        minds = [data[t] for t in ts]
        min_idx = np.argmin(minds)
        best_t = ts[min_idx]
        best_mind = minds[min_idx]
        diff = best_mind - B2_SOLO_055
        helps = "YES ✓" if diff < -0.0005 else "marginal" if diff < 0 else "NO"
        print(f"{name:<20} {best_t:<10.3f} {best_mind:<12.6f} {diff:+.6f}    {helps}")
    
    print(f"\nB2 solo reference: {B2_SOLO_055:.6f}")
    print(f"Baseline (clean):  {BASELINE_MIND:.6f}")
    
    # Print rerun comparison
    if b3_reruns or b4_reruns:
        print("\n--- Rerun comparison ---")
        if 0.55 in b3_reruns:
            print(f"  B3 T=0.55: original={B3_COND_LYSINE.get(0.55, 'N/A'):.6f}, rerun={b3_reruns[0.55]:.6f}")
        if 0.8 in b4_reruns:
            print(f"  B4 T=0.80: original={B4_COND_LYSINE.get(0.8, 'N/A'):.6f}, rerun={b4_reruns[0.8]:.6f}")


if __name__ == "__main__":
    main()
