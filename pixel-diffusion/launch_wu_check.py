#!/usr/bin/env python3
"""Regime check (2026-10-06): do the CelebA mix4 gains survive a proper LR schedule?

Every earlier 2000-kimg run used train.py's default LR ramp (to 1e-3 over 10,000 kimg), so the LR was still rising and
ended at 2e-4. Here the same schedules run with the LR held at 2e-4 after a 100-kimg warm-up (EDM / Ambient-o's
AFHQ-FFHQ value), same data, length, batch and seed. Names: celeba_wu_<arm>.

Usage (Engaging, via srun):  python launch_wu_check.py submit cleanonly static finetune75 km056
"""
import json, os, shutil, sys, time

import bo_mix4 as bo

BASE = bo.BASE
LOGDIR = os.path.join(BASE, 'train_logs', 'wu_check')
FLAGS = '--lr=2e-4 --lr_rampup_kimg=100'
G = ('g03', 'g05', 'g10', 'g20')


def ph(*p):
    return {'phases': [list(x) for x in p]}


def man(name):
    return {e['name']: e for e in json.load(open(bo.MANIFEST))['runs']}[name]['schedule']


ARMS = {   # name -> (dataset, schedule builder); schedules exactly as in mix4_manifest.py / the BO state
    'cleanonly': ('celeba_mix4_v1', lambda: {'type': 'per_group', 'groups': {g: ph((0, 'off')) for g in G}}),
    'static': ('celeba_mix4_v1', lambda: {'type': 'per_group', 'groups': {
        'g03': ph((0, 0.50)), 'g05': ph((0, 0.50)), 'g10': ph((0, 0.95)), 'g20': ph((0, 0.95))}}),
    'finetune75': ('celeba_mix4_v1', lambda: {'type': 'per_group', 'groups': {g: ph((0, 0.0), (0.75, 'off')) for g in G}}),
    'km056': ('celeba_mix4_km', lambda: man('mix4bo_km_056')),
}


def submit(arm):
    dataset, fn = ARMS[arm]
    run = f'celeba_wu_{arm}'
    bo.register(run, fn(), f'regime check: {arm} on {dataset} with {FLAGS}')
    os.makedirs(LOGDIR, exist_ok=True)
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{int(time.time())}.sh')
    shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={dataset} KEEP_LAST_DUMPS=2 TRAIN_EXTRA="{FLAGS}"; '
            f'nvidia-smi --query-gpu=name --format=csv,noheader; bash {frozen} {run} slurm 0 0')
    cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
           '-p', 'ou_sloan_gpu,sched_mit_sloan_gpu_r8,mit_preemptable', '--gres=gpu:1', '--cpus-per-task=6',
           '--mem=14G', '-t', '24:00:00', '--requeue', '--wrap', wrap]
    r = bo.sh(cmd)
    print(run, r.stdout.strip() or r.stderr.strip())


if __name__ == '__main__':
    for a in sys.argv[2:]:
        submit(a)
