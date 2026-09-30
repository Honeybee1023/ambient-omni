#!/usr/bin/env python3
"""Checks on the learned per-image x per-time policy (t_schedule type 'policy').

Run:  python tests/test_policy_schedule.py
"""
import os, sys, ast
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "training", "training_loop.py")
_tree = ast.parse(open(_SRC).read())
_want = {"compute_scheduled_sigma_min", "policy_features", "policy_T", "T_to_sigma_array"}
_nodes = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in _want]
_nodes += [n for n in _tree.body if isinstance(n, ast.Assign)
           and any(getattr(t, "id", "") == "POLICY_FEATURES" for t in n.targets)]
_ns = {"np": np}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), _SRC, "exec"), _ns)
feats, pT, T2s, sched, NAMES = (_ns["policy_features"], _ns["policy_T"], _ns["T_to_sigma_array"],
                                _ns["compute_scheduled_sigma_min"], _ns["POLICY_FEATURES"])
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


a = np.array([0.03, 0.73, 0.95, 0.99]); h = np.array([0.1, -0.4, -1.0, -2.0])
phi = feats(0.6, 0.8, 1.3, a, h)
check("feature matrix shape", phi.shape == (4, len(NAMES)), str(phi.shape))
check("columns in declared order", np.allclose(phi[:, NAMES.index('p*h')], 0.6 * h)
      and np.allclose(phi[:, NAMES.index('proj*a')], 1.3 * a) and np.allclose(phi[:, 0], 1))
th = np.zeros(len(NAMES)); th[0] = -8.6; th[1] = 12.0          # c1-like: step near p = 0.72
T = pT(th, feats(np.array(0.72), 0, 0, a, h))
check("c1-like init crosses 0.5 near p=0.72", np.allclose(T, 1 / (1 + np.exp(-(12 * 0.72 - 8.6)))))
T_lo = pT(th, feats(0.0, 0, 0, a, h)); T_hi = pT(th, feats(1.0, 0, 0, a, h))
check("low early, high late", T_lo.max() < 0.01 and T_hi.min() > 0.95, f"{T_lo.max():.4f} {T_hi.min():.4f}")
check("output within [0,1] for huge logits", np.all((pT(th * 1e6, phi) >= 0) & (pT(th * 1e6, phi) <= 1)))
try:
    pT(np.zeros(3), phi); check("refuses wrong-length theta", False)
except ValueError:
    check("refuses wrong-length theta", True)
for t in (0.0005, 0.2, 0.5, 0.9, 0.9995):
    want = sched({"type": "static", "t_start": t}, 0.3)
    got = float(T2s(np.array([t]))[0])
    check(f"T->sigma matches schedule mapping at T={t}", abs(got - want) < 1e-9 * max(1, want), f"{got} vs {want}")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}"); sys.exit(1)
print("all policy checks passed")
