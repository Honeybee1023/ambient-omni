"""fit_recovery_tau: does the exponential fit recover a known recovery time?

The flatness test it replaces can only return multiples of the probe spacing, so a run
probing every 100 kimg "measures" 100/200/300 kimg whatever the truth is. This checks the
fit instead returns the right tau90 from noisy samples, and refuses when it cannot.

Run:  python tests/test_recovery_fit.py
"""
import os, sys, ast
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "probe.py")
_fn = next(n for n in ast.parse(open(_SRC).read()).body if isinstance(n, ast.FunctionDef) and n.name == "fit_recovery_tau")
_ns = {"np": np}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), _SRC, "exec"), _ns)
fit = _ns["fit_recovery_tau"]


def series(tau, cadence, span, s0=0.35, s_inf=0.53, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(0, span + 1, cadence, dtype=float)
    y = s_inf - (s_inf - s0) * np.exp(-t / tau)
    return t, y + rng.normal(0, noise, size=t.shape)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else "")); return ok


def main():
    ok = True
    for tau in (40.0, 90.0, 150.0):
        want = 2.302585 * tau
        t, y = series(tau, 25, 400)
        got = fit(t, y)[0]
        ok &= check(f"clean series, tau90={want:.0f}", got is not None and abs(got - want) / want < 0.12, f"got {got}")
        t, y = series(tau, 25, 400, noise=0.004, seed=1)   # probe noise ~0.004 on softness
        got = fit(t, y)[0]
        ok &= check(f"noisy series, tau90={want:.0f}", got is not None and abs(got - want) / want < 0.30, f"got {got}")
    t, y = series(90.0, 100, 300)
    got100 = fit(t, y)[0]
    ok &= check("coarse 100-kimg cadence still fits (4 points)", got100 is not None, f"got {got100}")
    ok &= check("too few points refused", fit(*series(90.0, 100, 100))[0] is None)
    flat = (np.arange(0, 401, 25.0), np.full(17, 0.42))
    ok &= check("flat series refused", fit(*flat)[0] is None)
    t = np.arange(0, 401, 25.0)
    ok &= check("falling series refused", fit(t, 0.5 - 0.0001 * t)[0] is None)
    print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
