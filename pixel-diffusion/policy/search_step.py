#!/usr/bin/env python3
"""One idempotent step of the CMA-ES search over the per-image x per-time threshold policy.

Run repeatedly (e.g. every 10 minutes by policy/drive_search.sh). Each call:
  1. if no generation is in flight: asks CMA-ES for a population, registers each candidate in
     the job manifest as a `policy` t_schedule, and submits it with submit_dyn_csail.sh;
  2. otherwise collects finished MIND scores, resubmits a candidate once if its job vanished
     without a score, and gives a failed candidate a penalty after that;
  3. when the whole generation is scored, updates CMA-ES and prints GEN_DONE (or SEARCH_DONE).
All state lives in one JSON file, so a crash or a restart loses nothing.

Usage (CSAIL):  python policy/search_step.py [--init]
"""
# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)
import argparse, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmaes import CMAES

B = AMBIENT_BASE
REPO = f"{B}/ambient-omni/pixel-diffusion"
STATE = f"{B}/generated/policy_search/state.json"
MANIFEST = f"{B}/generated/dyn_search_manifest.json"
CONFIG = {
    "dataset": "celeba_mix4_v1",
    "features_path": f"{B}/annotated_datasets/celeba_mix4_v1.policy_features.json",
    "run_kimg": 1000,
    "every_kimg": 50,
    "popsize": 10,
    "generations": 8,
    "sigma0": 1.5,
    # c1-like start: T crosses 0.5 near 72% of training for every image; all image and
    # exposure weights zero, so the search starts from our best global schedule.
    # order: bias, p, passes, proj, a, h, p*a, p*h, proj*a, proj*h
    "mean0": [-8.6, 12.0, 0, 0, 0, 0, 0, 0, 0, 0],
    "prefix": "mix4pol",
}


def save(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(STATE + ".tmp", STATE)


def register(entries):
    m = json.load(open(MANIFEST))
    have = {r["name"] for r in m["runs"]}
    for e in entries:
        if e["name"] not in have:
            m["runs"].append(e)
    with open(MANIFEST + ".tmp", "w") as f:
        json.dump(m, f, indent=1)
    os.replace(MANIFEST + ".tmp", MANIFEST)


def submit(names, cfg):
    env = dict(os.environ, DYN_DATASET=cfg["dataset"], RUN_KIMG=str(cfg["run_kimg"]), SEED="0")
    r = subprocess.run(["bash", f"{REPO}/submit_dyn_csail.sh", *names], env=env, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())


def queued(name):
    r = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "honjar"), "-n", f"dyn_{name}", "-o", "%i"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def mind_of(name):
    p = f"{B}/generated/mind_dyn_{name}_s0.json"
    return json.load(open(p))["mind"] if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true", help="create the state file (refuses if it exists)")
    a = ap.parse_args()
    if a.init:
        if os.path.exists(STATE):
            sys.exit(f"{STATE} exists; refusing to re-initialise")
        es = CMAES(CONFIG["mean0"], CONFIG["sigma0"], popsize=CONFIG["popsize"], seed=20260930)
        save({"config": CONFIG, "es": es.to_json(), "gens": [], "done": False})
        print("initialised"); return
    state = json.load(open(STATE))
    cfg = state["config"]
    if state["done"]:
        print("SEARCH_DONE (already)"); return
    es = CMAES.from_json(state["es"])
    gens = state["gens"]
    if not gens or gens[-1]["told"]:
        X = es.ask()
        g = len(gens)
        cands = [{"name": f"{cfg['prefix']}_g{g}_k{k}", "theta": [round(float(v), 5) for v in x],
                  "f": None, "retries": 0, "failed": False} for k, x in enumerate(X)]
        register([{"name": c["name"], "study": "mix4_policy", "dataset": cfg["dataset"],
                   "schedule": {"type": "policy", "theta": c["theta"], "features_path": cfg["features_path"],
                                "every_kimg": cfg["every_kimg"]},
                   "note": f"policy search gen {g}"} for c in cands])
        gens.append({"gen": g, "cands": cands, "told": False})
        state["es"] = es.to_json()
        save(state)
        submit([c["name"] for c in cands], cfg)
        print(f"SUBMITTED generation {g}: {len(cands)} candidates"); return
    cur = gens[-1]
    for c in cur["cands"]:
        if c["f"] is not None or c["failed"]:
            continue
        v = mind_of(c["name"])
        if v is not None:
            c["f"] = float(v)
        elif not queued(c["name"]):
            if c["retries"] < 1:
                c["retries"] += 1
                print(f"resubmitting {c['name']} (no score, not queued)")
                submit([c["name"]], cfg)
            else:
                c["failed"] = True
                print(f"FAILED {c['name']} twice; it will get a penalty")
    save(state)
    pending = [c["name"] for c in cur["cands"] if c["f"] is None and not c["failed"]]
    if pending:
        done = sum(c["f"] is not None for c in cur["cands"])
        print(f"generation {cur['gen']}: {done}/{len(cur['cands'])} scored, waiting on {len(pending)}"); return
    ok = [c["f"] for c in cur["cands"] if c["f"] is not None]
    pen = (max(ok) + 0.005) if ok else 0.08
    X = [c["theta"] for c in cur["cands"]]
    f = [c["f"] if c["f"] is not None else pen for c in cur["cands"]]
    es.tell(X, f)
    cur["told"] = True
    cur["best"] = min(ok) if ok else None
    cur["median"] = float(np.median(ok)) if ok else None
    state["es"] = es.to_json()
    best_all = min((c["f"], c["name"]) for g in gens for c in g["cands"] if c["f"] is not None)
    state["done"] = len(gens) >= cfg["generations"]
    save(state)
    print(f"{'SEARCH_DONE' if state['done'] else 'GEN_DONE'} generation {cur['gen']}: best {cur['best']:.5f} "
          f"median {cur['median']:.5f}; best so far {best_all[0]:.5f} ({best_all[1]}); sigma {es.sigma:.3f}")


if __name__ == "__main__":
    main()
