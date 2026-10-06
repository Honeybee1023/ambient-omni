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
    # Oracle: every training dog, unblurred, no schedule -- the best a model can do at this length (upper bound)
    'afhq_oracle': ('afhqdog_oracle', lambda: None),
    # --- baselines, copied from the CelebA mix4 table (mix4_manifest.py) ---
    **{f'afhq_finetune{int(f * 100)}': ('afhqdog_mix4_v1', (lambda f=f: {'type': 'per_group', 'groups': {
        g: ph((0, 0.0), (f, 'off')) for g in G}})) for f in (0.60, 0.75, 0.90)},
    # static Ambient-o per level: CelebA's values (each level's best from its own CelebA sweep) ...
    'afhq_static_celeba': ('afhqdog_mix4_v1', lambda: {'type': 'per_group', 'groups': {
        'g03': ph((0, 0.50)), 'g05': ph((0, 0.50)), 'g10': ph((0, 0.95)), 'g20': ph((0, 0.95))}}),
    # ... and AFHQ's own: the median per-image Ambient-o threshold of each level (AFHQ classifier, no sweep)
    'afhq_static_cls': ('afhqdog_mix4_v1', lambda: {'type': 'per_group', 'groups': {
        'g03': ph((0, 0.027)), 'g05': ph((0, 0.733)), 'g10': ph((0, 0.955)), 'g20': ph((0, 0.985))}}),
    'afhq_ambo_then_clean': ('afhqdog_mix4_ambo', lambda: {'type': 'per_group', 'groups': {
        g: ph((0, 'annot'), (0.75, 'off')) for g in G}}),
    'afhq_all_then_ambo': ('afhqdog_mix4_ambo', lambda: {'type': 'per_group', 'groups': {
        g: ph((0, 0.0), (0.75, 'annot')) for g in G}}),
    'afhq_all_ambo_clean': ('afhqdog_mix4_ambo', lambda: {'type': 'per_group', 'groups': {
        g: ph((0, 0.0), (0.40, 'annot'), (0.75, 'off')) for g in G}}),
}
# Any CelebA search run copies the same way: afhq_true030 <- mix4bo_true_030 (true blur levels),
# afhq_km049 <- mix4bo_km_049 (k-means groups, AFHQ's own afhqdog_mix4_km).
for _n in [f'true_{i:03d}' for i in range(80)] + [f'km_{i:03d}' for i in range(80)]:
    _k = 'afhq_' + _n.replace('_', '')
    if _k not in RUNS:
        RUNS[_k] = ('afhqdog_mix4_km' if _n.startswith('km') else 'afhqdog_mix4_v1',
                    (lambda n=_n: from_manifest('mix4bo_' + n)))
# Proper-warm-up variants: <run>_wu = the same run with the LR held at EDM/Ambient-o's AFHQ value (2e-4) after a
# 100-kimg warm-up (5% of the run), instead of the default ramp to 1e-3 over 10,000 kimg (which ends at 2e-4).
WU_FLAGS = '--lr=2e-4 --lr_rampup_kimg=100'
for _k in list(RUNS):
    RUNS[_k + '_wu'] = RUNS[_k]
SOURCE = {'afhq_c5': 'CSAIL mix4_c5_heavy_early', 'afhq_c1': 'mix4_c1_global72', 'afhq_cleanonly': 'CSAIL mix4_cleanonly',
          'afhq_true038': 'mix4bo_true_038', 'afhq_true032': 'mix4bo_true_032',
          'afhq_ambo': 'CSAIL mix4_ambo', 'afhq_oracle': 'oracle: all 4,739 training dogs unblurred (no CelebA source)', 'afhq_km038': 'mix4bo_km_038', 'afhq_km041': 'mix4bo_km_041',
          'afhq_finetune60': 'CSAIL mix4_finetune60', 'afhq_finetune75': 'CSAIL mix4_finetune75',
          'afhq_finetune90': 'CSAIL mix4_finetune90', 'afhq_static_celeba': 'CSAIL mix4_static (same thresholds)',
          'afhq_static_cls': 'static Ambient-o at AFHQ classifier per-level medians (no CelebA sweep)',
          'afhq_ambo_then_clean': 'CSAIL mix4_ambo_then_clean', 'afhq_all_then_ambo': 'CSAIL mix4_all_then_ambo',
          'afhq_all_ambo_clean': 'CSAIL mix4_all_ambo_clean'}
for _k in [k for k in RUNS if k.endswith('_wu')]:
    SOURCE.setdefault(_k, SOURCE.get(_k[:-3], 'mix4bo_' + _k[5:-3].replace('true', 'true_').replace('km', 'km_')) + ' + proper warm-up')
for _k in RUNS:
    SOURCE.setdefault(_k, 'mix4bo_' + _k[5:].replace('true', 'true_').replace('km', 'km_'))


# extra train.py flags, as for the CelebA runs: the 64-point annotation grid needs an EMA window of 1
EXTRA = {k: '--cls_ema_window=1' for k in ('afhq_ambo', 'afhq_ambo_then_clean', 'afhq_all_then_ambo', 'afhq_all_ambo_clean')}


def submit(run):
    dataset, sched_fn = RUNS[run]
    if not os.path.isdir(os.path.join(BASE, 'annotated_datasets', dataset)):
        sys.exit(f'{run}: dataset {dataset} not built yet')
    sched = sched_fn()
    extra = ' '.join(x for x in (EXTRA.get(run.removesuffix('_wu'), ''), WU_FLAGS if run.endswith('_wu') else '') if x)
    bo.register(run, sched, f'AFHQ dog, schedule copied unchanged from CelebA {SOURCE[run]}')
    os.makedirs(LOGDIR, exist_ok=True)
    stamp = int(time.time())
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{stamp}.sh')
    shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={dataset} DYN_REF={REF} DYN_REF_CACHE={REF_CACHE} '
            f'KEEP_LAST_DUMPS=2' + (f' TRAIN_EXTRA="{extra}"' if extra else '') + f'; nvidia-smi --query-gpu=name --format=csv,noheader; bash {frozen} {run} slurm 0 0')
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
