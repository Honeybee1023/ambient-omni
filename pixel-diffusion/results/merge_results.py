#!/usr/bin/env python3
"""Merge every dynamic-T result into one dyn_all_results.json the BO can read.

Two sources, and the distinction is recorded per-point:
  * "file"        -- rescued JSON, authoritative
  * "transcribed" -- value read from a run log while CSAIL was reachable and
                     copied here by hand. CSAIL has been unreachable for hours
                     and may not come back; without these the dataset is missing
                     12 of 27 runs. They should be replaced by the real files if
                     access returns -- rescue_results.sh does that automatically,
                     and this script prefers a file whenever one exists.
"""
import json, os, re, sys, glob
from collections import defaultdict
import statistics as st

SP = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(f"{SP}/dyn_search_manifest.json"))
SPECS = {e["name"]: e for e in MAN["runs"]}

# Transcribed from run output while CSAIL was up. seed 0 unless noted.
TRANSCRIBED = {
    "p0_cosine_pw10":       0.031990,
    "p1_a_linear_0to095":   0.032766,
    "p1_a_twophase_050":    0.032056,
    "p1_s_early_mid":       0.033732,
    "p1_s_early_steep":     0.038194,
    "p1_s_plateau_mid":     0.033074698414473615,
    "p1_q_sobol00":         0.033861,
    "p1_q_sobol01":         0.032107764298748065,
    "p1_q_sobol03":         0.03819071625646843,
    "p1_q_sobol04":         0.031221610796163778,
    "p1_q_sobol06":         0.03523885465189775,
    "p1_q_sobol07":         0.03368463973903148,
    "p1_q_sobol09":         0.03058392506575951,
    "p1_q_sobol10":         0.0334693138271771,
    "p1_q_sobol11":         0.031528179963325374,
    # Recovered on CSAIL from checkpoints after being cancelled mid-flight.
    # lysine independently reran both at the same seed, giving two same-seed
    # cross-machine replicate pairs.
    "p1_s_late_extreme":    0.03182631822037608,
    "p1_s_late_hard":       0.031649321522006627,
}

# Key by (seed, machine), not seed alone: late_extreme and late_hard were each
# run twice at SEED 0 on different machines (the cancellation incident), so a
# seed-only key silently drops one of every such pair -- and those are two of
# the five pairs the noise estimate rests on.
vals = defaultdict(dict)   # run -> {(seed, machine): (mind, source)}
for path in glob.glob(f"{SP}/rescued_results/*/mind_dyn_*.json"):
    m = re.match(r"mind_dyn_(.+)_s(\d+)\.json$", os.path.basename(path))
    if not m:
        continue
    machine = os.path.basename(os.path.dirname(path))
    try:
        vals[m.group(1)][(int(m.group(2)), machine)] = (json.load(open(path))["mind"], "file")
    except Exception:
        pass
for run, v in TRANSCRIBED.items():
    if not any(k[0] == 0 and k[1] == "csail-slurm" for k in vals[run]):
        vals[run][(0, "csail-slurm")] = (v, "transcribed")

rows = []
for run, seeds in sorted(vals.items()):
    if run not in SPECS:
        print(f"  WARN: {run} not in manifest, skipped"); continue
    ms = [v for v, _ in seeds.values()]
    rows.append({
        "name": run, "phase": SPECS[run]["phase"], "x": SPECS[run]["x"],
        "n": len(ms), "mind_mean": st.mean(ms),
        "mind_sd": st.stdev(ms) if len(ms) > 1 else None,
        "mind_by_run": {f"s{k[0]}@{k[1]}": v for k, (v, _) in sorted(seeds.items())},
        "sources": sorted({s for _, s in seeds.values()}),
    })

out = f"{SP}/bocheck/generated/dyn_all_results.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"fractions": MAN["fractions"], "t_first_pinned": MAN["t_first_pinned"],
           "dataset": "celeba_dynamic_t_v2 (26,514 b0+b5 variant)",
           "noise_sd_from_replicate_pairs": 0.00072,
           "runs": rows}, open(out, "w"), indent=2)

nfile = sum(1 for r in rows if r["sources"] == ["file"])
ntrans = len(rows) - nfile
reps = [r for r in rows if r["n"] > 1]
print(f"merged {len(rows)}/30 unique runs  ({nfile} from file, {ntrans} include a transcribed value)")
print(f"replicated runs: {len(reps)}  -> " + ", ".join(f"{r['name'].replace('p1_','').replace('p0_','')}(n={r['n']})" for r in reps))
if reps:
    d2 = [ (max(r['mind_by_run'].values())-min(r['mind_by_run'].values()))**2 for r in reps ]
    print(f"noise sd from {len(reps)} pairs: {(sum(d2)/len(d2)/2)**0.5:.6f}")
print(f"wrote {out}")
