#!/usr/bin/env python3
"""Short-run transfer test, step 1: do 500-kimg runs rank schedules the way the 2000-kimg runs did?

Each arm reruns existing CelebA mix4 k-means schedules (same per_group JSON, jump times as the same FRACTIONS of
training) at RUN_KIMG=500 with the LR warm-up scaled by the same factor (10000 -> 2500 kimg), so the LR follows the
same curve over the run as in every 2000-kimg run (it ends at 20% of peak in both). The arms differ only in data:
  full     celeba_mix4_km             all images: each clean face is seen 4x fewer times than in the 2000-kimg run
  quarter  celeba_mix4_km_q4          every group (clean included) subsampled to 1/4 with a fixed seed: each image is
                                      seen as often, at every fraction of training, as in the 2000-kimg run
MIND/FID use the same reference set as the 2000-kimg runs. SHORT_PART overrides the partition (a ~1.5-h run fits
the public mit_normal_gpu 6-h limit; Sloan allows ~24 submitted jobs per user).

Usage (Engaging, via srun/sbatch):
    python launch_short.py build                      # make celeba_mix4_km_q4 (symlinks)
    python launch_short.py build --src afhqdog_mix4_km --out afhqdog_mix4_km_q4
    python launch_short.py submit full km_056 ...     # short_full_km056 ...
    python launch_short.py submit quarter km_056 ...  # short_q4_km056 ...
    python launch_short.py copy SRC_RUN NEW_RUN DATASET --extra "--lr=2e-4 --lr_rampup_kimg=25" [--ref DIR --ref_cache NPZ]
                                                      # 500-kimg copy of any registered run (e.g. a baseline)
"""
import argparse, json, os, random, shutil, sys, time

import bo_mix4 as bo

BASE = bo.BASE
D = os.path.join(BASE, 'annotated_datasets')
SRC, Q4 = 'celeba_mix4_km', 'celeba_mix4_km_q4'
SEED = 20261005
SHORT_KIMG, RAMPUP = 500, 2500
ARMS = {'full': (SRC, 'short_full_'), 'quarter': (Q4, 'short_q4_')}
LOGDIR = os.path.join(BASE, 'train_logs', 'short')


def build(src=SRC, q4=Q4):
    out = os.path.join(D, q4)
    if os.path.exists(out):
        sys.exit(f'{out} exists')
    rows = [json.loads(l) for l in open(os.path.join(D, src, 'annotations.jsonl'))]
    groups = {}
    for r in rows:
        groups.setdefault(r['filename'].split('_')[0], []).append(r)
    rng = random.Random(SEED)
    keep = []
    for g in sorted(groups):
        rs = sorted(groups[g], key=lambda r: r['filename'])
        rng.shuffle(rs)
        keep += rs[:len(rs) // 4]
    os.makedirs(out)
    with open(os.path.join(out, 'annotations.jsonl'), 'w') as f:
        for r in keep:
            os.symlink(os.path.join(D, src, r['filename']), os.path.join(out, r['filename']))
            f.write(json.dumps(r) + '\n')
    sizes = {g: sum(r['filename'].startswith(g + '_') for r in keep) for g in groups}
    json.dump({'source': src, 'fraction': 0.25, 'seed': SEED, 'sizes': sizes}, open(out + '.meta.json', 'w'), indent=1)
    print('wrote', out, sizes)


def submit(arm, src_runs):
    dataset, prefix = ARMS[arm]
    man = {e['name']: e for e in json.load(open(bo.MANIFEST))['runs']}
    os.makedirs(LOGDIR, exist_ok=True)
    for s in src_runs:
        src = 'mix4bo_' + s
        run = prefix + s.replace('_', '')
        bo.register(run, man[src]['schedule'], f'short-run test ({arm}): {src} at {SHORT_KIMG} kimg, LR warm-up {RAMPUP} kimg')
        frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{int(time.time())}.sh')
        shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
        wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={dataset} RUN_KIMG={SHORT_KIMG} KEEP_LAST_DUMPS=2 '
                f'TRAIN_EXTRA=--lr_rampup_kimg={RAMPUP}; nvidia-smi --query-gpu=name --format=csv,noheader; '
                f'bash {frozen} {run} slurm 0 0')
        cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
               '-p', os.environ.get('SHORT_PART', bo.SLOAN['part']), '--gres=gpu:1', '--cpus-per-task=6', '--mem=14G', '-t', '06:00:00',
               '--requeue', '--wrap', wrap]
        r = bo.sh(cmd)
        print(run, r.stdout.strip() or r.stderr.strip())


def copy(src_run, run, dataset, extra, ref='', ref_cache='', mem='14G'):
    """500-kimg copy of any registered run's schedule (e.g. a baseline) on any dataset, for the short-vs-full plot.
    Every env var is set explicitly: a submit from inside a job would otherwise inherit that job's settings."""
    man = {e['name']: e for e in json.load(open(bo.MANIFEST))['runs']}
    bo.register(run, man[src_run]['schedule'], f'500-kimg copy of {src_run} on {dataset} ({extra})')
    os.makedirs(LOGDIR, exist_ok=True)
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{int(time.time())}.sh')
    shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={dataset} RUN_KIMG={SHORT_KIMG} KEEP_LAST_DUMPS=2 '
            f'DYN_REF={ref} DYN_REF_CACHE={ref_cache} TRAIN_EXTRA="{extra}"; '
            f'nvidia-smi --query-gpu=name --format=csv,noheader; bash {frozen} {run} slurm 0 0')
    cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
           '-p', os.environ.get('SHORT_PART', 'ou_sloan_gpu,sched_mit_sloan_gpu_r8,mit_preemptable'), '--gres=gpu:1',
           '--cpus-per-task=6', f'--mem={mem}', '-t', '03:00:00', '--requeue', '--wrap', wrap]
    r = bo.sh(cmd)
    print(run, r.stdout.strip() or r.stderr.strip())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'submit', 'copy'])
    ap.add_argument('arm', nargs='?')
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--extra', default='', help='copy: TRAIN_EXTRA for the short run')
    ap.add_argument('--ref', default='', help='copy: MIND reference dir (empty = CelebA default)')
    ap.add_argument('--ref_cache', default='')
    ap.add_argument('--mem', default='14G')
    ap.add_argument('--src', default=SRC, help='build: source dataset (e.g. afhqdog_mix4_km)')
    ap.add_argument('--out', default=Q4, help='build: name of the 1/4 copy (e.g. afhqdog_mix4_km_q4)')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.src, a.out)
    elif a.cmd == 'copy':          # copy SRC_RUN NEW_RUN DATASET
        copy(a.arm, a.runs[0], a.runs[1], a.extra, a.ref, a.ref_cache, a.mem)
    else:
        submit(a.arm, a.runs)
