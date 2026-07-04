"""
Bayesian Optimization for 7D T-space exploration.
Reads all 2k kimg metrics, fits additive+interaction GP in log(FID) space,
suggests batch of points via UCB.

Usage:
  python bo_suggest.py --round 1 --batch-size 15 --beta 1.0
  python bo_suggest.py --round 1 --dry-run
"""
import os, sys, json, glob, re, argparse, shutil
import numpy as np
from scipy.stats import norm as scipy_norm

# === Constants ===
P_MEAN = -1.2
P_STD = 1.2
CELEBA_ROOT = "/data/scratch/honjar/celeba_processed"
SHARED_DIR = os.path.join(CELEBA_ROOT, "shared_buckets_64")
ANNOTATED_DIR = "/data/scratch/honjar/annotated_datasets"
METRICS_DIR = "/data/scratch/honjar/generated"
BUCKETS = [1, 2, 3, 4, 5, 6, 7]
BLUR_SIGMAS = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0, 7: 8.0}

# Per-bucket minimum T for candidate generation (post-transition from 2k sigmoid fits).
# Keeps candidates in the region where data could plausibly help, not the plateau.
T_FLOOR = np.array([0.00, 0.70, 0.84, 0.90, 0.93, 0.95, 0.96])


# === T <-> sigma_min (VERIFIED EXACT) ===
def t_to_sigma_min(t_value):
    if t_value <= 0.001:
        return 0.0
    if t_value >= 0.999:
        return float(np.exp(P_STD * scipy_norm.ppf(0.999) + P_MEAN))
    return float(np.exp(P_STD * scipy_norm.ppf(t_value) + P_MEAN))


def parse_t_suffix(suffix):
    """Parse T suffix string to float.
    3 chars = hundredths (T025 -> 0.25), 4 chars = thousandths (T0125 -> 0.125)."""
    n = int(suffix)
    if len(suffix) == 3:
        return n / 100.0
    elif len(suffix) == 4:
        return n / 1000.0
    else:
        raise ValueError(f"Unknown suffix format: {suffix}")


# === Data Collection ===
def collect_2domain_data():
    """Read all 2-domain 2k kimg metrics -> (7D T-vector, FID) pairs."""
    data_points = []
    pattern = os.path.join(METRICS_DIR, "metrics_celeba_2d_b*_T*_2000kimg.json")
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        m = re.match(r'metrics_celeba_2d_b(\d+)_T(\w+)_2000kimg\.json', fname)
        if not m:
            continue
        bucket = int(m.group(1))
        t_suffix = m.group(2)
        try:
            t_value = parse_t_suffix(t_suffix)
        except ValueError:
            print(f"  WARNING: skipping unparseable suffix: {t_suffix}")
            continue

        with open(fpath) as f:
            metrics = json.load(f)
        fid = metrics['fid_score']

        t_vec = [1.0] * 7
        t_vec[bucket - 1] = t_value

        data_points.append({
            'source': f'2d_b{bucket}_T{t_suffix}',
            't_vec': t_vec,
            'fid': fid
        })
    return data_points


def collect_independence_data():
    """Read independence test metrics with known T-vectors."""
    indep_configs = {
        'celeba_indep2k_all_argmin':  [0.95, 0.95, 1.0, 0.97, 0.99, 0.99, 1.0],
        'celeba_indep2k_argmin_low':  [0.95, 0.95, 1.0, 1.0,  1.0,  1.0,  1.0],
        'celeba_indep2k_argmin_high': [1.0,  1.0,  1.0, 0.97, 0.99, 0.99, 1.0],
        'celeba_indep_baseline':      [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    }

    data_points = []
    for name, t_vec in indep_configs.items():
        for suffix in ['2000kimg', '1000kimg']:
            fpath = os.path.join(METRICS_DIR, f"metrics_{name}_{suffix}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    metrics = json.load(f)
                data_points.append({
                    'source': f'{name}_{suffix}',
                    't_vec': t_vec,
                    'fid': metrics['fid_score']
                })
                break
    return data_points


def collect_bo_data():
    """Read any previous BO round metrics."""
    data_points = []
    pattern = os.path.join(METRICS_DIR, "tvec_celeba_bo_*.json")
    for tvec_path in sorted(glob.glob(pattern)):
        base = os.path.basename(tvec_path).replace('tvec_', '').replace('.json', '')
        metrics_path = os.path.join(METRICS_DIR, f"metrics_{base}_2000kimg.json")
        if os.path.exists(metrics_path):
            with open(tvec_path) as f:
                t_vec = json.load(f)
            with open(metrics_path) as f:
                metrics = json.load(f)
            data_points.append({
                'source': base,
                't_vec': t_vec,
                'fid': metrics['fid_score']
            })
    return data_points


# === GP Model ===
def build_and_fit_gp(X, Y_log, n_iter=200):
    """Fit additive + interaction GP on log(FID)."""
    import torch
    import gpytorch

    class AdditiveInteractionGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            self.additive_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.AdditiveStructureKernel(
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                    num_dims=7
                )
            )

            self.interaction_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=7)
            )

            self.covar_module = self.additive_kernel + self.interaction_kernel

        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)

    train_x = torch.tensor(X, dtype=torch.float64)
    train_y = torch.tensor(Y_log, dtype=torch.float64)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = AdditiveInteractionGP(train_x, train_y, likelihood)
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
            print(f"  GP iter {i+1}/{n_iter}, loss={loss.item():.3f} (best={best_loss:.3f})")

    model.eval()
    likelihood.eval()
    add_scale = model.additive_kernel.outputscale.item()
    int_scale = model.interaction_kernel.outputscale.item()
    total = add_scale + int_scale
    noise = likelihood.noise.item()
    print(f"\n  Kernel weights: additive={add_scale:.3f} ({100*add_scale/total:.0f}%), "
          f"interaction={int_scale:.3f} ({100*int_scale/total:.0f}%)")
    print(f"  Observation noise (log space): {noise:.4f} (sqrt={noise**0.5:.3f})")

    return model, likelihood, train_x


def suggest_batch(model, likelihood, train_x, batch_size=15, beta=1.0,
                  n_candidates=50000, seed=42):
    """UCB batch selection in log(FID) space with per-bucket T constraints."""
    import torch
    import gpytorch

    rng = np.random.RandomState(seed)
    model.eval()
    likelihood.eval()

    # --- Generate candidates with per-bucket T floors ---
    # T_FLOOR prevents exploring the plateau region where data clearly hurts
    t_lo = T_FLOOR.copy()
    t_hi = np.ones(7)

    parts = []

    # 60% near argmin: [max(floor, 0.92), 1.0] per bucket
    n1 = int(n_candidates * 0.6)
    lo_near = np.maximum(t_lo, 0.92)
    parts.append(rng.uniform(lo_near, t_hi, size=(n1, 7)))

    # 30% moderate: [max(floor, 0.85), 1.0] per bucket
    n2 = int(n_candidates * 0.3)
    lo_mod = np.maximum(t_lo, 0.85)
    parts.append(rng.uniform(lo_mod, t_hi, size=(n2, 7)))

    # 10% broader: [floor, 1.0] per bucket
    n3 = n_candidates - n1 - n2
    parts.append(rng.uniform(t_lo, t_hi, size=(n3, 7)))

    candidates = np.concatenate(parts, axis=0)
    cand_t = torch.tensor(candidates, dtype=torch.float64)

    # --- Evaluate LCB in log(FID) space ---
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        chunk_size = 5000
        mus_log, sigmas_log = [], []
        for start in range(0, len(cand_t), chunk_size):
            chunk = cand_t[start:start + chunk_size]
            pred = likelihood(model(chunk))
            mus_log.append(pred.mean.numpy())
            sigmas_log.append(pred.variance.sqrt().numpy())
        mu_log = np.concatenate(mus_log)
        sigma_log = np.concatenate(sigmas_log)

    # LCB in log space (lower log(FID) = lower FID = better)
    lcb_log = mu_log - beta * sigma_log

    # --- Greedy batch selection with diversity ---
    selected = []
    selected_info = []
    remaining_mask = np.ones(len(candidates), dtype=bool)

    for _ in range(batch_size):
        lcb_rem = lcb_log[remaining_mask]
        cand_rem = candidates[remaining_mask]
        mu_rem = mu_log[remaining_mask]
        sigma_rem = sigma_log[remaining_mask]

        if len(lcb_rem) == 0:
            break

        best_local = np.argmin(lcb_rem)
        point = cand_rem[best_local].copy()
        selected.append(point.tolist())
        selected_info.append({
            'predicted_fid': float(np.exp(mu_rem[best_local])),
            'fid_lcb': float(np.exp(lcb_rem[best_local])),
            'sigma_log': float(sigma_rem[best_local]),
        })

        dist = np.linalg.norm(candidates - point, axis=1)
        remaining_mask &= (dist > 0.03)

    return selected, selected_info


# === Dataset Creation ===
def get_bucket_files(bucket_num):
    prefix = f"b{bucket_num}_"
    return sorted([f for f in os.listdir(SHARED_DIR)
                   if f.startswith(prefix) and f.endswith('.jpg')
                   and not f.startswith('._')])


def create_bo_dataset(name, t_vec, target_files, all_bucket_files):
    """Create annotated dataset for a BO-suggested 7D T-vector."""
    dataset_dir = os.path.join(ANNOTATED_DIR, name)
    ann_path = os.path.join(dataset_dir, "annotations.jsonl")

    if os.path.exists(ann_path):
        print(f"  SKIP (exists): {name}")
        return

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    annotations = []

    for fname in target_files:
        src = os.path.join(SHARED_DIR, fname)
        os.symlink(src, os.path.join(dataset_dir, fname))
        annotations.append({"filename": fname, "sigma_min": 0.0, "sigma_max": 0.0})

    for i, b in enumerate(BUCKETS):
        t_val = t_vec[i]
        sigma_min = t_to_sigma_min(t_val)
        for fname in all_bucket_files[b]:
            src = os.path.join(SHARED_DIR, fname)
            os.symlink(src, os.path.join(dataset_dir, fname))
            annotations.append({"filename": fname, "sigma_min": sigma_min, "sigma_max": 0.0})

    with open(ann_path, "w") as f:
        for ann in annotations:
            f.write(json.dumps(ann) + "\n")

    t_vec_path = os.path.join(METRICS_DIR, f"tvec_{name}.json")
    with open(t_vec_path, "w") as f:
        json.dump(t_vec, f)

    print(f"  CREATED: {name} ({len(annotations)} images)")
    for i, b in enumerate(BUCKETS):
        t_val = t_vec[i]
        sigma_min = t_to_sigma_min(t_val)
        status = "ACTIVE" if t_val < 0.999 else "excluded"
        print(f"    B{b} (blur={BLUR_SIGMAS[b]:.1f}): T={t_val:.4f}  "
              f"sigma_min={sigma_min:.4f}  [{status}]")


# === Main ===
def main():
    parser = argparse.ArgumentParser(description="BO for 7D T-space")
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=15)
    parser.add_argument('--beta', type=float, default=1.0,
                        help='UCB exploration weight (lower=more exploitative)')
    parser.add_argument('--gp-iters', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Bayesian Optimization — Round {args.round}")
    print(f"Batch size: {args.batch_size}, Beta: {args.beta}")
    print("=" * 60)

    # --- 1. Collect data ---
    print("\n--- Collecting data ---")
    data_2d = collect_2domain_data()
    print(f"  2-domain points: {len(data_2d)}")
    data_indep = collect_independence_data()
    print(f"  Independence points: {len(data_indep)}")
    data_bo = collect_bo_data()
    print(f"  Previous BO points: {len(data_bo)}")

    all_data = data_2d + data_indep + data_bo
    print(f"  TOTAL: {len(all_data)} observations")

    X = np.array([d['t_vec'] for d in all_data])
    Y_raw = np.array([d['fid'] for d in all_data])

    # LOG TRANSFORM: GP works in log(FID) space to prevent negative predictions
    Y_log = np.log(Y_raw)

    print(f"  FID range: {Y_raw.min():.2f} – {Y_raw.max():.2f}")
    print(f"  log(FID) range: {Y_log.min():.3f} – {Y_log.max():.3f}")

    t1_mask = np.all(X == 1.0, axis=1)
    if t1_mask.sum() > 0:
        print(f"  FID at [1,...,1]: {Y_raw[t1_mask].mean():.2f} "
              f"± {Y_raw[t1_mask].std():.2f} (n={t1_mask.sum()})")

    if args.dry_run:
        print("\n--- Dry run: all data points ---")
        for d in sorted(all_data, key=lambda x: x['fid']):
            t_str = "[" + ", ".join(f"{t:.3f}" for t in d['t_vec']) + "]"
            print(f"  FID={d['fid']:6.2f}  {t_str}  {d['source']}")
        print("\n=== Dry run complete. ===")
        return

    # --- 2. Fit GP in log space ---
    print("\n--- Fitting GP (additive + interaction) in log(FID) space ---")
    model, likelihood, train_x = build_and_fit_gp(X, Y_log, n_iter=args.gp_iters)

    # --- 3. Suggest batch ---
    print(f"\n--- Suggesting {args.batch_size} points (beta={args.beta}) ---")
    suggested, info = suggest_batch(
        model, likelihood, train_x,
        batch_size=args.batch_size, beta=args.beta,
        seed=args.round * 1000
    )

    # --- 4. Print and create datasets ---
    print(f"\n{'='*60}")
    print(f"SUGGESTED POINTS (Round {args.round})")
    print(f"{'='*60}")

    target_files = get_bucket_files(0)
    all_bucket_files = {b: get_bucket_files(b) for b in BUCKETS}

    dataset_names = []
    for i, (t_vec, si) in enumerate(zip(suggested, info)):
        t_vec_rounded = [round(t, 4) for t in t_vec]
        name = f"celeba_bo_r{args.round}_p{i:02d}"
        dataset_names.append(name)

        print(f"\n  Point {i:2d}: {name}")
        t_parts = []
        for j, t in enumerate(t_vec_rounded):
            label = f"B{j+1}={t:.3f}"
            if t >= 0.999:
                label += "(off)"
            t_parts.append(label)
        print(f"    T = {', '.join(t_parts)}")
        print(f"    Predicted FID: {si['predicted_fid']:.2f}  "
              f"(LCB={si['fid_lcb']:.2f}, σ_log={si['sigma_log']:.3f})")

        create_bo_dataset(name, t_vec_rounded, target_files, all_bucket_files)

    # --- 5. Save manifest ---
    manifest = {
        'round': args.round,
        'beta': args.beta,
        'n_observations_used': len(all_data),
        'datasets': dataset_names,
        'suggestions': [
            {'name': name, 't_vec': [round(t, 4) for t in tvec],
             'predicted_fid': si['predicted_fid'],
             'fid_lcb': si['fid_lcb']}
            for name, tvec, si in zip(dataset_names, suggested, info)
        ]
    }
    manifest_path = os.path.join(METRICS_DIR, f"bo_round{args.round}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")
    print(f"\n=== Run: bash launch_bo_batch.sh {args.round} ===")


if __name__ == "__main__":
    main()
