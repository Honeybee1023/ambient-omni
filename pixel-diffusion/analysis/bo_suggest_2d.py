"""
2D Bayesian Optimization: (T_B1, T_B2) search minimizing MIND.
B3-B7 fixed at T=0.999 (inactive). Datasets use v2b multi-bucket format.

Seed data from 1D sweeps (edges), conditional sweeps, shift tests,
and any previous 2D BO rounds.

Usage:
  python bo_suggest_2d.py --round 1 --batch-size 10 --beta 1.0
  python bo_suggest_2d.py --round 1 --dry-run
"""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os, sys, json, glob, re, argparse
import numpy as np
from scipy.stats import norm as scipy_norm

# === Constants ===
P_MEAN = -1.2
P_STD = 1.2
PROCESSED_DIR = f"{AMBIENT_BASE}/celeba_processed_v2b/shared_buckets_64"
ANNOTATED_DIR = f"{AMBIENT_BASE}/annotated_datasets"
METRICS_DIR = f"{AMBIENT_BASE}/generated"
ALL_BUCKETS = [1, 2, 3, 4, 5, 6, 7]
INACTIVE_T = 0.999

# Known conditional sweep configurations
COND_B1_FIXED_B2 = 0.55   # cond_b1 sweep fixes B2 at this value
COND_B2_FIXED_B1 = 0.5    # cond_b2 sweep fixes B1 at this value

# Shift test configurations: label -> (T_B1, T_B2)
SHIFT_CONFIGS = {
    "bothup": (0.6,   0.65),
    "bothdn": (0.4,   0.45),
    "apart":  (0.4,   0.65),
    "close":  (0.525, 0.525),
}


def t_to_sigma_min(t):
    """Convert noise threshold T to sigma_min for annotations."""
    if t <= 0.001:
        return 0.0
    if t >= 0.999:
        return float(np.exp(P_STD * scipy_norm.ppf(0.999) + P_MEAN))
    return float(np.exp(P_STD * scipy_norm.ppf(t) + P_MEAN))


def parse_t_suffix(suffix):
    """Parse T suffix to float. 3 chars=hundredths, 4 chars=thousandths."""
    n = int(suffix)
    if len(suffix) == 3:
        return n / 100.0
    elif len(suffix) == 4:
        return n / 1000.0
    raise ValueError(f"Unknown suffix: {suffix}")


def read_mind(fpath):
    """Read MIND value from JSON file."""
    with open(fpath) as f:
        return json.load(f)["mind"]


# === Data Collection ===
def collect_all_seed_data():
    """Collect all existing MIND data mapped to (T1, T2) space, deduplicated."""
    raw = []

    # 1. B1 1D sweep -> (T1, 0.999)
    for fpath in sorted(glob.glob(os.path.join(METRICS_DIR, "mind_celeba_v2*_b1_T*_2000kimg.json"))):
        fname = os.path.basename(fpath)
        m = re.match(r'mind_celeba_v2b?_b1_T(\w+)_2000kimg\.json', fname)
        if not m:
            continue
        try:
            t1 = parse_t_suffix(m.group(1))
        except ValueError:
            continue
        raw.append((t1, INACTIVE_T, read_mind(fpath), f"b1_T{m.group(1)}"))

    # 2. B2 1D sweep -> (0.999, T2)
    for fpath in sorted(glob.glob(os.path.join(METRICS_DIR, "mind_celeba_v2*_b2_T*_2000kimg.json"))):
        fname = os.path.basename(fpath)
        m = re.match(r'mind_celeba_v2b?_b2_T(\w+)_2000kimg\.json', fname)
        if not m:
            continue
        try:
            t2 = parse_t_suffix(m.group(1))
        except ValueError:
            continue
        raw.append((INACTIVE_T, t2, read_mind(fpath), f"b2_T{m.group(1)}"))

    # 3. Baseline -> (0.999, 0.999)
    for pattern in ["mind_celeba_v2b_baseline_2000kimg.json",
                    "mind_celeba_v2_baseline_2000kimg.json"]:
        fpath = os.path.join(METRICS_DIR, pattern)
        if os.path.exists(fpath):
            raw.append((INACTIVE_T, INACTIVE_T, read_mind(fpath), "baseline"))
            break

    # 4. Conditional B1 sweep (B2=0.55 fixed) -> (T1, 0.55)
    for fpath in sorted(glob.glob(os.path.join(METRICS_DIR, "mind_celeba_v2b_cond_b1_T*_2000kimg.json"))):
        fname = os.path.basename(fpath)
        m = re.match(r'mind_celeba_v2b_cond_b1_T(\w+)_2000kimg\.json', fname)
        if not m:
            continue
        try:
            t1 = parse_t_suffix(m.group(1))
        except ValueError:
            continue
        raw.append((t1, COND_B1_FIXED_B2, read_mind(fpath), f"cond_b1_T{m.group(1)}"))

    # 5. Conditional B2 sweep (B1=0.5 fixed) -> (0.5, T2)
    for fpath in sorted(glob.glob(os.path.join(METRICS_DIR, "mind_celeba_v2b_cond_b2_T*_2000kimg.json"))):
        fname = os.path.basename(fpath)
        m = re.match(r'mind_celeba_v2b_cond_b2_T(\w+)_2000kimg\.json', fname)
        if not m:
            continue
        try:
            t2 = parse_t_suffix(m.group(1))
        except ValueError:
            continue
        raw.append((COND_B2_FIXED_B1, t2, read_mind(fpath), f"cond_b2_T{m.group(1)}"))

    # 6. Shift tests -> known (T1, T2) pairs
    for label, (t1, t2) in SHIFT_CONFIGS.items():
        fpath = os.path.join(METRICS_DIR, f"mind_celeba_v2b_shift_{label}_2000kimg.json")
        if os.path.exists(fpath):
            raw.append((t1, t2, read_mind(fpath), f"shift_{label}"))

    # 7. Previous 2D BO rounds
    for tvec_path in sorted(glob.glob(os.path.join(METRICS_DIR, "tvec_celeba_v2b_bo2d_*.json"))):
        base = os.path.basename(tvec_path).replace("tvec_", "").replace(".json", "")
        mind_path = os.path.join(METRICS_DIR, f"mind_{base}_2000kimg.json")
        if os.path.exists(mind_path):
            with open(tvec_path) as f:
                tvec = json.load(f)
            raw.append((tvec[0], tvec[1], read_mind(mind_path), base))

    # Deduplicate by (t1, t2) rounded to 4 digits — keep last occurrence
    seen = {}
    for t1, t2, mind, source in raw:
        key = (round(t1, 4), round(t2, 4))
        seen[key] = {"t1": key[0], "t2": key[1], "mind": mind, "source": source}

    points = sorted(seen.values(), key=lambda x: x["mind"])
    return points


# === GP Model ===
def build_and_fit_gp(X, Y_log, n_iter=200):
    """Fit Matern 5/2 GP with ARD on 2D log(MIND) data."""
    import torch
    import gpytorch

    class GP2D(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
            )

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    train_x = torch.tensor(X, dtype=torch.float64)
    train_y = torch.tensor(Y_log, dtype=torch.float64)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP2D(train_x, train_y, likelihood)
    model = model.double()
    likelihood = likelihood.double()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
        if (i + 1) % 50 == 0:
            print(f"  GP iter {i+1}/{n_iter}, loss={loss.item():.4f} (best={best_loss:.4f})")

    model.eval()
    likelihood.eval()

    noise = likelihood.noise.item()
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
    print(f"  Lengthscales: T1={ls[0]:.4f}, T2={ls[1]:.4f}")
    print(f"  Output scale: {model.covar_module.outputscale.item():.4f}")
    print(f"  Noise (log space): {noise:.6f}")

    return model, likelihood


def suggest_batch(model, likelihood, X_train, batch_size=10, beta=1.0, grid_res=200):
    """UCB batch selection on dense 2D grid."""
    import torch
    import gpytorch

    model.eval()
    likelihood.eval()

    # Dense grid over [0, 0.999]^2
    t1_vals = np.linspace(0.0, 0.999, grid_res)
    t2_vals = np.linspace(0.0, 0.999, grid_res)
    T1, T2 = np.meshgrid(t1_vals, t2_vals)
    candidates = np.column_stack([T1.ravel(), T2.ravel()])

    cand_t = torch.tensor(candidates, dtype=torch.float64)

    # Evaluate GP in chunks
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        chunk_size = 10000
        mus, sigmas = [], []
        for start in range(0, len(cand_t), chunk_size):
            chunk = cand_t[start:start + chunk_size]
            pred = likelihood(model(chunk))
            mus.append(pred.mean.numpy())
            sigmas.append(pred.variance.sqrt().numpy())
        mu_log = np.concatenate(mus)
        sigma_log = np.concatenate(sigmas)

    # LCB in log(MIND) space (lower = better)
    lcb = mu_log - beta * sigma_log

    # Greedy batch selection with diversity
    selected = []
    selected_info = []
    mask = np.ones(len(candidates), dtype=bool)

    for _ in range(batch_size):
        lcb_rem = lcb[mask]
        cand_rem = candidates[mask]
        mu_rem = mu_log[mask]
        sigma_rem = sigma_log[mask]

        if len(lcb_rem) == 0:
            break

        best_idx = np.argmin(lcb_rem)
        pt = cand_rem[best_idx].copy()
        selected.append(pt.tolist())
        selected_info.append({
            'predicted_mind': float(np.exp(mu_rem[best_idx])),
            'mind_lcb': float(np.exp(lcb_rem[best_idx])),
            'sigma_log': float(sigma_rem[best_idx]),
        })

        # Exclude nearby for diversity
        dist = np.linalg.norm(candidates - pt, axis=1)
        mask &= (dist > 0.05)

    return selected, selected_info


# === Dataset Creation ===
def create_bo_dataset(name, t1, t2):
    """Create multi-bucket dataset: B1=t1, B2=t2, B3-B7=0.999."""
    import shutil

    ds_dir = os.path.join(ANNOTATED_DIR, name)
    ann_path = os.path.join(ds_dir, "annotations.jsonl")

    if os.path.exists(ann_path):
        print(f"  SKIP (exists): {name}")
        return

    if os.path.exists(ds_dir):
        shutil.rmtree(ds_dir)
    os.makedirs(ds_dir)

    bucket_t = {1: t1, 2: t2}  # only B1, B2 active; rest -> INACTIVE_T
    annotations = []

    # Clean images (bucket 0)
    for src in sorted(glob.glob(os.path.join(PROCESSED_DIR, "b0_*.jpg"))):
        fname = os.path.basename(src)
        os.symlink(src, os.path.join(ds_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    # All blur buckets (B1-B7)
    for b in ALL_BUCKETS:
        t_val = bucket_t.get(b, INACTIVE_T)
        smin = t_to_sigma_min(t_val)
        for src in sorted(glob.glob(os.path.join(PROCESSED_DIR, f"b{b}_*.jpg"))):
            fname = os.path.basename(src)
            os.symlink(src, os.path.join(ds_dir, fname))
            annotations.append({"filename": fname, "sigma_min": smin, "sigma_max": 0.0})

    # Sort by filename for reproducibility
    annotations.sort(key=lambda x: x["filename"])

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    # Save T-vector sidecar as [t1, t2]
    tvec_path = os.path.join(METRICS_DIR, f"tvec_{name}.json")
    with open(tvec_path, "w") as f:
        json.dump([round(t1, 4), round(t2, 4)], f)

    smin1 = t_to_sigma_min(t1)
    smin2 = t_to_sigma_min(t2)
    print(f"  CREATED: {name} ({len(annotations)} images)")
    print(f"    B1: T={t1:.4f}  sigma_min={smin1:.4f}")
    print(f"    B2: T={t2:.4f}  sigma_min={smin2:.4f}")
    print(f"    B3-B7: T=0.999 (inactive)")


# === Main ===
def main():
    parser = argparse.ArgumentParser(description="2D BO for (T_B1, T_B2)")
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--beta', type=float, default=1.0,
                        help='UCB exploration weight (lower=more exploitative)')
    parser.add_argument('--gp-iters', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print(f"2D Bayesian Optimization — Round {args.round}")
    print(f"Search space: T_B1 x T_B2 in [0, 0.999]")
    print(f"Batch: {args.batch_size}, Beta: {args.beta}")
    print("=" * 60)

    # --- Collect seed data ---
    print("\n--- Collecting seed data ---")
    points = collect_all_seed_data()
    print(f"  Total unique (T1, T2) points: {len(points)}")

    edge_b1 = [p for p in points if abs(p['t2'] - INACTIVE_T) < 0.002
               and abs(p['t1'] - INACTIVE_T) > 0.002]
    edge_b2 = [p for p in points if abs(p['t1'] - INACTIVE_T) < 0.002
               and abs(p['t2'] - INACTIVE_T) > 0.002]
    interior = [p for p in points if abs(p['t1'] - INACTIVE_T) > 0.002
                and abs(p['t2'] - INACTIVE_T) > 0.002]
    baseline = [p for p in points if abs(p['t1'] - INACTIVE_T) < 0.002
                and abs(p['t2'] - INACTIVE_T) < 0.002]
    print(f"    B1 edge (T2~0.999): {len(edge_b1)}")
    print(f"    B2 edge (T1~0.999): {len(edge_b2)}")
    print(f"    Interior 2D:        {len(interior)}")
    print(f"    Baseline:           {len(baseline)}")

    X = np.array([[p['t1'], p['t2']] for p in points])
    Y = np.array([p['mind'] for p in points])
    Y_log = np.log(Y)

    print(f"  MIND range: {Y.min():.6f} – {Y.max():.6f}")
    best_i = np.argmin(Y)
    print(f"  Best: MIND={Y[best_i]:.6f} at ({X[best_i,0]:.4f}, {X[best_i,1]:.4f})"
          f"  [{points[best_i]['source']}]")

    if args.dry_run:
        print(f"\n--- All {len(points)} seed points (sorted by MIND) ---")
        for p in points:
            print(f"  MIND={p['mind']:.6f}  ({p['t1']:.4f}, {p['t2']:.4f})  {p['source']}")
        print("\n=== Dry run complete ===")
        return

    # --- Fit GP ---
    print("\n--- Fitting GP (Matern 5/2, ARD) in log(MIND) ---")
    model, likelihood = build_and_fit_gp(X, Y_log, n_iter=args.gp_iters)

    # --- Suggest batch ---
    print(f"\n--- Suggesting {args.batch_size} points (beta={args.beta}) ---")
    suggested, info = suggest_batch(
        model, likelihood, X,
        batch_size=args.batch_size, beta=args.beta
    )

    # --- Create datasets ---
    print(f"\n{'=' * 60}")
    print(f"SUGGESTED POINTS (Round {args.round})")
    print(f"{'=' * 60}")

    dataset_names = []
    for i, (pt, si) in enumerate(zip(suggested, info)):
        t1, t2 = round(pt[0], 4), round(pt[1], 4)
        name = f"celeba_v2b_bo2d_r{args.round}_p{i:02d}"
        dataset_names.append(name)

        print(f"\n  Point {i:2d}: {name}")
        print(f"    T_B1={t1:.4f}, T_B2={t2:.4f}")
        print(f"    Predicted MIND: {si['predicted_mind']:.6f}  "
              f"(LCB={si['mind_lcb']:.6f}, sigma_log={si['sigma_log']:.4f})")

        create_bo_dataset(name, t1, t2)

    # --- Save manifest ---
    manifest = {
        'round': args.round,
        'beta': args.beta,
        'n_seed_points': len(points),
        'datasets': dataset_names,
        'suggestions': [
            {'name': nm, 't1': round(pt[0], 4), 't2': round(pt[1], 4),
             'predicted_mind': si['predicted_mind'], 'mind_lcb': si['mind_lcb']}
            for nm, pt, si in zip(dataset_names, suggested, info)
        ]
    }
    mpath = os.path.join(METRICS_DIR, f"bo2d_round{args.round}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved: {mpath}")
    print(f"\n=== Next: bash launch_bo_2d.sh {args.round} ===")


if __name__ == "__main__":
    main()
