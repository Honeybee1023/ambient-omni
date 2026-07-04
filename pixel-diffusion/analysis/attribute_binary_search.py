"""
Attribution analysis for binary search Round 1.

For each image, compares average model metrics when that image was at T=0
(included) vs T=100 (excluded). Primary decision metric: Vendi Score.
PickScore and Aesthetic tracked as secondary signals.

Outputs:
- Per-image attribution scores and winner decisions
- Category-level summaries (wolves, dogs, cats)
- Sanity check results
"""

import os
import json
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=str,
                        default="/data/scratch/honjar/annotated_datasets/bsearch_r1_assignments.npz",
                        help="Assignment matrix from dataset generation")
    parser.add_argument("--metrics_dir", type=str,
                        default="/data/scratch/honjar/generated",
                        help="Directory containing metric JSON files")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--num_models", type=int, default=20)
    parser.add_argument("--output", type=str,
                        default="/data/scratch/honjar/generated/bsearch_r1_attribution.json",
                        help="Output JSON with attribution results")
    args = parser.parse_args()

    # Load assignment matrix
    data = np.load(args.assignments, allow_pickle=True)
    assignments = data["assignments"]  # (num_models, num_images)
    filenames = list(data["filenames"])
    wolves = list(data["wolves"])
    dogs = list(data["dogs"])
    cats = list(data["cats"])

    num_models, num_images = assignments.shape
    print(f"Assignment matrix: {num_models} models x {num_images} images")
    print(f"  Wolves: {len(wolves)}, Dogs: {len(dogs)}, Cats: {len(cats)}")

    # Load metrics for all models
    model_pickscore = []
    model_aesthetic = []
    model_vendi = []

    for i in range(args.num_models):
        name = f"bsearch_r1_model_{i:03d}"
        json_path = os.path.join(args.metrics_dir, f"metrics_{name}_1000kimg.json")

        if not os.path.exists(json_path):
            print(f"WARNING: Missing metrics for {name}")
            model_pickscore.append(np.nan)
            model_aesthetic.append(np.nan)
            model_vendi.append(np.nan)
            continue

        with open(json_path) as f:
            metrics = json.load(f)

        model_pickscore.append(metrics["pickscore"]["mean"])
        model_aesthetic.append(metrics["aesthetic"]["mean"])
        model_vendi.append(metrics["vendi"]["score"])

    model_pickscore = np.array(model_pickscore)
    model_aesthetic = np.array(model_aesthetic)
    model_vendi = np.array(model_vendi)

    print(f"\nModel-level metric ranges:")
    print(f"  PickScore: {model_pickscore.min():.3f} - {model_pickscore.max():.3f} "
          f"(spread: {model_pickscore.max() - model_pickscore.min():.3f})")
    print(f"  Aesthetic: {model_aesthetic.min():.3f} - {model_aesthetic.max():.3f} "
          f"(spread: {model_aesthetic.max() - model_aesthetic.min():.3f})")
    print(f"  Vendi:     {model_vendi.min():.3f} - {model_vendi.max():.3f} "
          f"(spread: {model_vendi.max() - model_vendi.min():.3f})")

    # Per-image attribution
    # For each image: avg metric when T=0 vs avg metric when T=100
    # assignments: 0 = included (T=0), 1 = excluded (T=100)
    results = []

    for img_idx in range(num_images):
        img_assignments = assignments[:, img_idx]  # (num_models,)
        included_mask = img_assignments == 0  # T=0
        excluded_mask = img_assignments == 1  # T=100

        n_included = int(included_mask.sum())
        n_excluded = int(excluded_mask.sum())

        # Average metrics when this image was included vs excluded
        vendi_when_included = float(model_vendi[included_mask].mean())
        vendi_when_excluded = float(model_vendi[excluded_mask].mean())
        vendi_delta = vendi_when_included - vendi_when_excluded

        pick_when_included = float(model_pickscore[included_mask].mean())
        pick_when_excluded = float(model_pickscore[excluded_mask].mean())
        pick_delta = pick_when_included - pick_when_excluded

        aes_when_included = float(model_aesthetic[included_mask].mean())
        aes_when_excluded = float(model_aesthetic[excluded_mask].mean())
        aes_delta = aes_when_included - aes_when_excluded

        # Decision: based on Vendi (primary)
        # vendi_delta > 0 means including this image helps diversity -> winner is T=0
        # vendi_delta < 0 means excluding this image helps diversity -> winner is T=100
        winner = 0 if vendi_delta > 0 else 100

        fname = filenames[img_idx]
        category = "wolf" if fname.startswith("wolf_") else "dog" if fname.startswith("dog_") else "cat"

        results.append({
            "filename": fname,
            "category": category,
            "n_included": n_included,
            "n_excluded": n_excluded,
            "vendi_when_included": round(vendi_when_included, 4),
            "vendi_when_excluded": round(vendi_when_excluded, 4),
            "vendi_delta": round(vendi_delta, 4),
            "pick_when_included": round(pick_when_included, 4),
            "pick_when_excluded": round(pick_when_excluded, 4),
            "pick_delta": round(pick_delta, 4),
            "aes_when_included": round(aes_when_included, 4),
            "aes_when_excluded": round(aes_when_excluded, 4),
            "aes_delta": round(aes_delta, 4),
            "winner_T": winner,
        })

    # Category-level summaries
    wolf_results = [r for r in results if r["category"] == "wolf"]
    dog_results = [r for r in results if r["category"] == "dog"]
    cat_results = [r for r in results if r["category"] == "cat"]

    def summarize(group, name):
        winners_0 = sum(1 for r in group if r["winner_T"] == 0)
        winners_100 = sum(1 for r in group if r["winner_T"] == 100)
        avg_vendi_delta = np.mean([r["vendi_delta"] for r in group])
        avg_pick_delta = np.mean([r["pick_delta"] for r in group])
        avg_aes_delta = np.mean([r["aes_delta"] for r in group])
        print(f"\n  {name}: {len(group)} images")
        print(f"    Winner T=0 (include): {winners_0} ({100*winners_0/len(group):.1f}%)")
        print(f"    Winner T=100 (exclude): {winners_100} ({100*winners_100/len(group):.1f}%)")
        print(f"    Avg Vendi delta (incl - excl): {avg_vendi_delta:+.4f}")
        print(f"    Avg PickScore delta: {avg_pick_delta:+.4f}")
        print(f"    Avg Aesthetic delta: {avg_aes_delta:+.4f}")
        return {
            "count": len(group),
            "winners_T0": winners_0,
            "winners_T100": winners_100,
            "pct_T0": round(100 * winners_0 / len(group), 1),
            "avg_vendi_delta": round(float(avg_vendi_delta), 4),
            "avg_pick_delta": round(float(avg_pick_delta), 4),
            "avg_aes_delta": round(float(avg_aes_delta), 4),
        }

    print("\n=== CATEGORY SUMMARIES ===")
    wolf_summary = summarize(wolf_results, "Wolves")
    dog_summary = summarize(dog_results, "Dogs")
    cat_summary = summarize(cat_results, "Cats")

    # Sanity checks
    print("\n=== SANITY CHECKS ===")

    # Check 1: Wolves should mostly prefer T=0 (inclusion)
    wolf_pct_included = wolf_summary["pct_T0"]
    wolf_check = wolf_pct_included > 50
    print(f"  Wolves prefer inclusion (T=0): {wolf_pct_included:.1f}% "
          f"{'PASS' if wolf_check else 'FAIL'}")

    # Check 2: Dogs should have higher inclusion rate than cats
    dog_pct = dog_summary["pct_T0"]
    cat_pct = cat_summary["pct_T0"]
    dog_cat_check = dog_pct > cat_pct
    print(f"  Dogs prefer inclusion more than cats: {dog_pct:.1f}% vs {cat_pct:.1f}% "
          f"{'PASS' if dog_cat_check else 'FAIL'}")

    # Check 3: Wolves avg Vendi delta should be most positive
    wolf_vd = wolf_summary["avg_vendi_delta"]
    dog_vd = dog_summary["avg_vendi_delta"]
    cat_vd = cat_summary["avg_vendi_delta"]
    print(f"  Avg Vendi delta — Wolves: {wolf_vd:+.4f}, Dogs: {dog_vd:+.4f}, Cats: {cat_vd:+.4f}")

    # Save full results
    output = {
        "round": args.round,
        "num_models": args.num_models,
        "num_images": num_images,
        "decision_metric": "vendi",
        "model_metrics": {
            "pickscore": model_pickscore.tolist(),
            "aesthetic": model_aesthetic.tolist(),
            "vendi": model_vendi.tolist(),
        },
        "category_summaries": {
            "wolves": wolf_summary,
            "dogs": dog_summary,
            "cats": cat_summary,
        },
        "sanity_checks": {
            "wolves_prefer_inclusion": wolf_check,
            "wolves_pct_T0": wolf_pct_included,
            "dogs_more_included_than_cats": dog_cat_check,
            "dog_pct_T0": dog_pct,
            "cat_pct_T0": cat_pct,
        },
        "per_image_results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {args.output}")
    print(f"  {num_images} per-image attributions")
    print(f"  Use 'winner_T' field for Round 2 interval assignment")


if __name__ == "__main__":
    main()
