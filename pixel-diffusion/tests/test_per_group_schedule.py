#!/usr/bin/env python3
"""Checks on the `per_group` t_schedule type (one threshold curve per blur bucket).

What must hold for the go/no-go runs to mean what they say:
  - control_points reproduce the global 'piecewise' schedule exactly;
  - phases are a step function that switches exactly at the listed fractions;
  - 'annot' and 'off' come back as their own states, never as a number;
  - malformed specs are refused before training rather than mid-run.

Run:  python tests/test_per_group_schedule.py
"""
import os, sys, ast
import numpy as np

# Same trick as test_piecewise_schedule.py: lift the pure functions out of the real
# source, since training_loop.py imports torch/wandb at module scope.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "training", "training_loop.py")
_tree = ast.parse(open(_SRC).read())
_want = {"compute_scheduled_sigma_min", "per_group_state", "group_of", "sigma_min_to_t"}
_nodes = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in _want]
_nodes += [n for n in _tree.body if isinstance(n, ast.Assign)
           and any(getattr(t, "id", "") == "PER_GROUP_OFF_SIGMA" for t in n.targets)]
_ns = {"np": np}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), _SRC, "exec"), _ns)
sched, state, group_of, s2t = (_ns["compute_scheduled_sigma_min"], _ns["per_group_state"],
                               _ns["group_of"], _ns["sigma_min_to_t"])

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


print("group_of")
check("clean prefix", group_of("b0_000100.jpg") == "b0")
check("bucket prefix", group_of("g10_004512.png") == "g10")

print("control_points == global piecewise")
cp = [[0, 0], [0.4, 0], [0.7, 0.6], [1, 0.95]]
grid = np.linspace(0, 1, 401)
same = all(state({"control_points": cp}, p) == ("T", sched({"type": "piecewise", "control_points": cp}, p))
           for p in grid)
check("identical on a 401-point grid", same)

print("phases step function")
ft = {"phases": [[0, 0.0], [0.75, "off"]]}          # fine-tune: all data, then clean only
check("before switch = T 0 (sigma 0)", state(ft, 0.7499) == ("T", 0.0))
check("at switch = off", state(ft, 0.75) == ("off", None))
check("after switch = off", state(ft, 1.0) == ("off", None))
v7 = {"phases": [[0, 0.0], [0.4, "annot"], [0.8, "off"]]}   # all -> Ambient-o -> clean only
check("phase 1", state(v7, 0.1) == ("T", 0.0))
check("phase 2 is annot", state(v7, 0.5) == ("annot", None))
check("phase 3 is off", state(v7, 0.9) == ("off", None))
st = {"phases": [[0, 0.5]]}                          # static per bucket
s = state(st, 0.3)
check("static 0.5 round-trips", s[0] == "T" and abs(s2t(s[1]) - 0.5) < 1e-6, f"{s}")

print("refusals")
for bad, why in [({"phases": []}, "empty phases"),
                 ({"phases": [[0.2, 0.0]]}, "phases not starting at 0"),
                 ({"phases": [[0, 0], [0.8, "off"], [0.5, "annot"]]}, "unsorted phases"),
                 ({"control_points": cp, "phases": [[0, 0]]}, "both kinds"),
                 ({}, "neither kind")]:
    try:
        state(bad, 0.5)
        check(f"refuses {why}", False, "no error raised")
    except ValueError:
        check(f"refuses {why}", True)

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all per_group checks passed")
