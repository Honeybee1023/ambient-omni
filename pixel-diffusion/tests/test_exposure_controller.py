"""The clean-exposure controller: does it land the projected exposure on its target?

The controller withdraws blur when projected clean exposure (the integral of the clean share of
the batch) reaches a target scaled by the clean-set size. Checks:
  1. the chosen withdrawal time makes the projection equal the target, at several clean counts;
  2. the recovery floor wins when the two constraints conflict (few clean images);
  3. exposure already spent moves the withdrawal later (the loop is closed, not open);
  4. withdrawal never reverses once planned.

Run:  python tests/test_exposure_controller.py
"""
import os, sys, ast
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "probe.py")
_cls = [n for n in ast.parse(open(_SRC).read()).body if isinstance(n, ast.ClassDef) and n.name == "ProbeController"][0]
_m = next(m for m in _cls.body if isinstance(m, ast.FunctionDef) and m.name == "_exposure")
_ns = {"np": np}
exec(compile(ast.Module(body=[_m], type_ignores=[]), _SRC, "exec"), _ns)
_exposure = _ns["_exposure"]

TOTAL, TARGET, TAU, A, F_HI = 2000.0, 1259.0, 300.0, 0.019, 0.962


class Ctl:
    def __init__(self, n_clean, E_now=0.0, T=0.0, planned=None):
        self.ctl = {"target_epochs": TARGET, "tau_rec": TAU, "total_kimg": TOTAL,
                    "t_end": 0.95, "f_hi": F_HI, "shape": "jump"}
        self.n_clean = n_clean; self.clean_frac_at_zero = A; self.observed_clean_frac = None
        self.observed_clean_kimg = E_now; self.current_T = T; self.withdraw_kimg = planned


def step(c, kimg):
    """The controller returns (T, diagnostics); the tests read the diagnostics."""
    return _exposure(c, {"t_grid": []}, kimg / TOTAL, kimg)[1]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else "")); return ok


def main():
    ok = True
    for n in (500, 1000):
        d = step(Ctl(n), 0.0)
        ok &= check(f"projection hits target, n_clean={n}",
                    abs(d["projected_total_epochs"] - TARGET) < 1.0,
                    f"proj {d['projected_total_epochs']:.0f} vs {TARGET:.0f}, drop at {d['withdraw_kimg']:.0f}")
    # Beyond ~1900 clean images the target cannot be reached inside the budget: even withdrawing
    # at once, the clean set is too large to be cycled that many times. The controller must then
    # withdraw as early as it can and log that it is under target -- and this is the regime where
    # a pure exposure rule has nothing to say about blur's own value (see the log).
    d = step(Ctl(4000), 0.0)
    ok &= check("unreachable target -> withdraw immediately, under target",
                d["withdraw_kimg"] <= 1e-6 and d["projected_total_epochs"] < TARGET,
                f"drop at {d['withdraw_kimg']:.0f}, proj {d['projected_total_epochs']:.0f}")
    # The other unreachable direction: with very few clean images the recovery floor forces a
    # late withdrawal and the run overshoots the target however it is scheduled.
    d = step(Ctl(200), 0.0)
    ok &= check("few clean images -> floor binds and target is overshot",
                d["recovery_floor_binding"] and d["projected_total_epochs"] > TARGET,
                f"drop at {d['withdraw_kimg']:.0f}, proj {d['projected_total_epochs']:.0f}")
    d = step(Ctl(250), 0.0)
    ok &= check("recovery floor binds when clean images are few",
                d["recovery_floor_binding"] and abs(d["withdraw_kimg"] - (TOTAL - TAU)) < 1e-6,
                f"drop at {d['withdraw_kimg']:.0f}, unclipped {d['withdraw_kimg_unclipped']:.0f}")
    ok &= check("floor does not bind at n_clean=500", not step(Ctl(500), 0.0)["recovery_floor_binding"])
    base = step(Ctl(500), 600.0)["withdraw_kimg"]
    hot = step(Ctl(500, E_now=40.0), 600.0)["withdraw_kimg"]
    ok &= check("exposure spent moves the drop later", hot > base + 1.0, f"{base:.0f} -> {hot:.0f}")
    planned = step(Ctl(500), 0.0)["withdraw_kimg"]
    later = step(Ctl(500, E_now=0.0, planned=planned), 100.0)["withdraw_kimg"]
    ok &= check("plan does not drift earlier without cause", later <= planned + 1e-6,
                f"{planned:.0f} -> {later:.0f}")
    c = Ctl(500, T=0.95, planned=1300.0)
    t, d = _exposure(c, {"t_grid": []}, 1500 / TOTAL, 1500.0)
    ok &= check("withdrawal never reverses", t >= 0.95 - 1e-9, f"T={t}")
    # after the drop the plan is history and must stop moving (it used to wander, and once
    # produced -1412644 when a lag tick made the two clean shares equal)
    ok &= check("plan frozen after withdrawal", d.get("plan_frozen") and abs(d["withdraw_kimg"] - 1300.0) < 1e-6,
                f"reported {d['withdraw_kimg']:.0f}, frozen={d.get('plan_frozen')}")
    c = Ctl(500, T=0.95, planned=1300.0); c.observed_clean_frac = 0.046   # the lag-tick reading
    _, d = _exposure(c, {"t_grid": []}, 1500 / TOTAL, 1500.0)
    ok &= check("a lag-tick clean share is not believed as f_hi", d["f_hi"] > 0.5, f"f_hi={d['f_hi']:.3f}")
    c = Ctl(500); c.clean_frac_at_zero = 0.93                            # near-singular solve
    _, d = _exposure(c, {"t_grid": []}, 0.0, 0.0)
    ok &= check("near-singular solve falls back to the floor, not millions",
                abs(d["withdraw_kimg"] - (TOTAL - TAU)) < 1e-6, f"plan {d['withdraw_kimg']:.0f}")
    print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
