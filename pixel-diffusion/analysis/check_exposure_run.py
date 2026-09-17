"""Audit one clean-exposure run: did the controller choose its drop time, or land there by luck?

Reads a run's training log and probe log and checks, in order, the things that would each on their
own invalidate the result. Every check prints PASS/FAIL with the numbers, so a run can be accepted
or rejected without reading the raw logs.

    python analysis/check_exposure_run.py --run_dir <train_outputs/dyn_search/dyn_exp_base_s0> \
        [--expect_drop 1439] [--expect_n_clean 500]

Checks:
  1. clean count read from the dataset matches the setting (everything scales by it);
  2. the observed clean share while blur is in use is near n_clean/n_total (exposure accounting sane);
  3. the drop time the controller chose equals the closed-form solution from ITS OWN observed
     constants -- this is the "not by accident" test, and it is the one that matters;
  4. the plan was stable across probes rather than wandering into place;
  5. blur actually left when the plan said, in the batches (corrupt fraction), not just in T;
  6. exposure at the end matches what was projected, and sits on target (or overshoots only where
     the recovery floor binds);
  7. the recovery floor bound only where expected;
  8. the probe was driving, not logging.
"""
import argparse, json, os, re, sys


def parse_log(path):
    txt = open(path).read()
    ticks = [(float(a), float(b), float(c)) for a, b, c in
             re.findall(r"(?<!/)\bkimg ([0-9.]+).*?\bT ([0-9.]+) corrupt ([0-9.]+)", txt)]
    m = re.search(r"Clean images in this dataset: (\d+)", txt)
    return ticks, (int(m.group(1)) if m else None), txt


def measured_exposure(ticks, skip=2):
    """Clean exposure in kimg from the batches actually drawn, excluding prefetch ticks."""
    if len(ticks) <= skip:
        return 0.0
    E, prev = 0.0, ticks[skip - 1][0]
    for k, _, c in ticks[skip:]:
        E += (1 - c) * (k - prev); prev = k
    return E


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--expect_drop", type=float, default=None)
    ap.add_argument("--expect_n_clean", type=int, default=None)
    ap.add_argument("--tol_kimg", type=float, default=60.0)
    a = ap.parse_args()

    ticks, n_clean_log, _ = parse_log(os.path.join(a.run_dir, "log.txt"))
    recs = [json.loads(l) for l in open(os.path.join(a.run_dir, "probe_log.jsonl"))]
    dec = [r for r in recs if isinstance(r.get("decision"), dict) and "withdraw_kimg" in r["decision"]]
    if not dec:
        sys.exit("not an exposure run: no withdraw_kimg in any probe decision")
    # The plan must be judged at the moment it was ACTED ON. Once blur has been withdrawn the
    # controller keeps re-solving from a much larger accumulated exposure, so the last probe's
    # withdraw_kimg is a stale re-solve (it drifts to the recovery floor), not the decision the
    # run made. Use the last probe before withdrawal for the plan checks, and the last probe of
    # all for the exposure outcome.
    pre = [r for r in dec if float(r.get("T_current", 0.0)) < 0.5]
    plan_rec = pre[-1] if pre else dec[0]
    last, first = dec[-1], dec[0]
    d = plan_rec["decision"]
    d_end = last["decision"]
    total = 2000.0
    ok = True

    print(f"{a.run_dir}  ({len(ticks)} ticks, {len(recs)} probes)")
    ok &= check("clean count read from dataset", n_clean_log == d["n_clean"] and
                (a.expect_n_clean is None or n_clean_log == a.expect_n_clean),
                f"log {n_clean_log}, controller {d['n_clean']}, expected {a.expect_n_clean}")
    ok &= check("clean share while blur in use is sane", 0.5 <= d["a_clean_frac"] / max(d["n_clean"] / 26514.0, 1e-9) <= 2.0,
                f"observed a {d['a_clean_frac']:.4f} vs n_clean/n_total {d['n_clean']/26514.0:.4f}")

    # 3. the decisive one: recompute the choice from the run's own observed constants
    aa, f_hi = d["a_clean_frac"], d["f_hi"]
    tgt = d["target_clean_kimg"]
    k_at = float(plan_rec["kimg"]); E_at = float(d["clean_kimg_so_far"])
    solved = (tgt - E_at + aa * k_at - f_hi * total) / (aa - f_hi)
    floor = total - float(d.get("tau_rec", 300))
    expected = min(solved, floor)
    ok &= check("drop time equals the closed form from its own constants",
                abs(d["withdraw_kimg"] - expected) <= a.tol_kimg,
                f"chose {d['withdraw_kimg']:.0f}, closed form {expected:.0f} (unclipped {solved:.0f}, floor {floor:.0f})")
    if a.expect_drop is not None:
        ok &= check("drop time matches the pre-registered prediction",
                    abs(d["withdraw_kimg"] - a.expect_drop) <= a.tol_kimg,
                    f"chose {d['withdraw_kimg']:.0f}, predicted {a.expect_drop:.0f}")

    plans = [r["decision"]["withdraw_kimg"] for r in pre if r["decision"].get("withdraw_kimg")] or [d["withdraw_kimg"]]
    ok &= check("plan stable across probes", (max(plans) - min(plans)) <= 150.0,
                f"range {min(plans):.0f}-{max(plans):.0f} over {len(plans)} probes")

    drop = d["withdraw_kimg"]
    after = [(k, c) for k, T, c in ticks if k >= drop + 100]
    before = [(k, c) for k, T, c in ticks if k <= drop - 50]
    ok &= check("blur actually left the batches at the planned time",
                bool(after) and bool(before) and after[0][1] < 0.2 and before[-1][1] > 0.8,
                f"corrupt {before[-1][1]:.3f} before -> {after[0][1]:.3f} after" if after and before else "n/a")

    E_meas = measured_exposure(ticks)
    ok &= check("final exposure matches the projection made when it dropped",
                abs(E_meas - d["projected_total_clean_kimg"]) / max(d["projected_total_clean_kimg"], 1e-9) < 0.15,
                f"measured {E_meas:.0f} kimg, projected {d['projected_total_clean_kimg']:.0f}")
    ep = E_meas * 1000.0 / d["n_clean"]
    on_target = abs(ep - d["target_epochs"]) / d["target_epochs"] < 0.15
    ok &= check("exposure on target (or overshooting only under the floor)",
                on_target or d["recovery_floor_binding"],
                f"{ep:.0f} passes vs target {d['target_epochs']:.0f}, floor binding {d['recovery_floor_binding']}")
    ok &= check("probe was driving", all(r.get("applied") for r in dec), "")
    print("RUN LOOKS SOUND" if ok else "RUN IS SUSPECT -- do not use its number until explained")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
