#!/usr/bin/env python3
"""Run CelebA mix4 schedules, unchanged, on the AFHQ-dog mix4 dataset (no search on AFHQ).

Each AFHQ run copies a CelebA schedule byte for byte (same per_group phases, same group keys, same
train.py / run_dyn_job.sh / batch 64 / 2000 kimg / seed 0); only the dataset and the MIND/FID
reference set change. Label-free schedules (k1..k4) run on afhqdog_mix4_km, whose k-means groups are
AFHQ's own (built from Ambient-o's classifier on AFHQ); Ambient-o runs on afhqdog_mix4_ambo.

Usage (on Engaging, via srun/sbatch, never on the login node):
    python launch_afhq.py list
    python launch_afhq.py submit afhq_c5 afhq_c1 ...      # register + sbatch on Sloan
"""
import argparse, json, os, shutil, sys, time

import bo_mix4 as bo

BASE = bo.BASE
LOGDIR = os.path.join(BASE, 'train_logs', 'afhqdog')
REF = os.path.join(BASE, 'afhqdog_processed', 'train_clean_64')
REF_CACHE = os.path.join(BASE, 'generated', 'mind_ref_cache_afhqdog.npz')


def ph(*pairs):
    return {'phases': [list(p) for p in pairs]}


G = ('g03', 'g05', 'g10', 'g20')
K = ('k1', 'k2', 'k3', 'k4')


def from_manifest(name):
    """A CelebA schedule exactly as it ran (Engaging manifest)."""
    m = json.load(open(bo.MANIFEST))
    r = [e for e in m['runs'] if e['name'] == name]
    if not r:
        sys.exit(f'{name} not in manifest')
    return r[0]['schedule']


# CelebA source -> (AFHQ dataset, schedule). CSAIL-era hand runs are written out from the CSAIL
# manifest (same JSON as the ledger shows); search runs are read from the Engaging manifest.
RUNS = {
    'afhq_cleanonly': ('afhqdog_mix4_v1', lambda: {'type': 'per_group', 'groups': {g: ph((0, 'off')) for g in G}}),
    'afhq_c5': ('afhqdog_mix4_v1', lambda: {'type': 'per_group', 'groups': {
        'g03': ph((0, 0.0), (0.75, 0.95)), 'g05': ph((0, 0.0), (0.75, 0.95)),
        'g10': ph((0, 0.0), (0.4, 'off')), 'g20': ph((0, 0.0), (0.4, 'off'))}}),
    'afhq_c1': ('afhqdog_mix4_v1', lambda: from_manifest('mix4_c1_global72_sloan')),
    'afhq_true038': ('afhqdog_mix4_v1', lambda: from_manifest('mix4bo_true_038')),
    'afhq_true032': ('afhqdog_mix4_v1', lambda: from_manifest('mix4bo_true_032')),
    # need the AFHQ classifier: Ambient-o as published (per-image thresholds, fixed all run), like CelebA mix4_ambo
    'afhq_ambo': ('afhqdog_mix4_ambo', lambda: {'type': 'per_group', 'groups': {g: ph((0, 'annot')) for g in G}}),
    # need afhqdog_mix4_km (after the AFHQ classifier + k-means):
    'afhq_km038': ('afhqdog_mix4_km', lambda: from_manifest('mix4bo_km_038')),
    'afhq_km041': ('afhqdog_mix4_km', lambda: from_manifest('mix4bo_km_041')),
}
SOURCE = {'afhq_c5': 'CSAIL mix4_c5_heavy_early', 'afhq_c1': 'mix4_c1_global72', 'afhq_cleanonly': 'CSAIL mix4_cleanonly',
          'afhq_true038': 'mix4bo_true_038', 'afhq_true032': 'mix4bo_true_032',
          'afhq_ambo': 'CSAIL mix4_ambo', 'afhq_km038': 'mix4bo_km_038', 'afhq_km041': 'mix4bo_km_041'}


# extra train.py flags, as for the CelebA runs: the 64-point annotation grid needs an EMA window of 1
EXTRA = {'afhq_ambo': '--cls_ema_window=1'}


def submit(run):
    dataset, sched_fn = RUNS[run]
    if not os.path.isdir(os.path.join(BASE, 'annotated_datasets', dataset)):
        sys.exit(f'{run}: dataset {dataset} not built yet')
    sched = sched_fn()
    bo.register(run, sched, f'AFHQ dog, schedule copied unchanged from CelebA {SOURCE[run]}')
    os.makedirs(LOGDIR, exist_ok=True)
    stamp = int(time.time())
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{stamp}.sh')
    shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={dataset} DYN_REF={REF} DYN_REF_CACHE={REF_CACHE} '
            f'KEEP_LAST_DUMPS=2' + (f' TRAIN_EXTRA={EXTRA[run]}' if run in EXTRA else '') + f'; nvidia-smi --query-gpu=name --format=csv,noheader; bash {frozen} {run} slurm 0 0')
    w = bo.SLOAN
    cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
           '-p', w['part'], f'--gres={w["gres"]}', '--cpus-per-task=8', '--mem=64G', '-t', w['time'],
           '--requeue', '--wrap', wrap]
    r = bo.sh(cmd)
    if r.returncode != 0:
        sys.exit(f'{run}: sbatch failed: {r.stderr}')
    print(run, r.stdout.strip(), dataset, json.dumps(sched))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'submit'])
    ap.add_argument('runs', nargs='*')
    a = ap.parse_args()
    if a.cmd == 'list':
        for k, (d, f) in RUNS.items():
            print(k, d, SOURCE[k])
    else:
        for run in a.runs:
            submit(run)
