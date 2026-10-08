#!/usr/bin/env python3
"""Scale short-search picks up to full 2000-kimg runs (2026-10-07).

For each checkpoint n (CHECKS) whose first n search points have all finished, take the top 3 of those n by short-run
MIND -- what the search would have recommended after n runs -- and run each at full scale: full k-means dataset,
2000 kimg, the same jump fractions and levels, and the search's own LR regime. A schedule picked at several
checkpoints runs once. Names: ss2m_<search>_<NNN> (NNN = the short run's index).

Usage (Engaging, via sbatch/srun):  python launch_ss2m.py q4a q4b     (SS2M_DRY=1: print, submit nothing)
Also called at the end of short_search.step for the searches in AUTO.
"""
import os, shutil, sys, time

import bo_mix4 as bo
import short_search as ss

BASE = bo.BASE
LOGDIR = os.path.join(BASE, 'train_logs', 'ss2m')
CHECKS = (8, 12, 16, 20, 24, 32, 40)
TOP = 3
AUTO = ('q4a', 'q4b', 'afq4')
FULL_DATASET = 'celeba_mix4_km'
LR_FLAGS = {'old': '', 'wu': '--lr=2e-4 --lr_rampup_kimg=100'}     # 2000-kimg versions of short_search.LR_FLAGS
PART = 'ou_sloan_gpu,sched_mit_sloan_gpu_r8'


def path():
    return os.path.join(ss.STATE_DIR, 'ss2m.json')


def fixed_from_end(x, short_kimg=500, full_kimg=2000):
    """Last jump (latest group) keeps the same number of images before the end as in the short run; other jumps and
    all levels unchanged. For 500 -> 2000 kimg: f -> 1 - (1 - f) / 4."""
    x = list(x)
    last = max(range(4), key=lambda i: x[i])
    x[last] = round(1 - (1 - x[last]) * short_kimg / full_kimg, 4)
    return x


def submit(st, q, suffix='', x=None, nice=0, note='same fractions'):
    run = 'ss2m_' + q['name'][len('ss_'):] + suffix
    x = q['x'] if x is None else x
    bo.register(run, ss.schedule(x), f"2000-kimg copy of {q['name']} ({st['search']}, LR {st['lr']}, {note})")
    os.makedirs(LOGDIR, exist_ok=True)
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{int(time.time())}.sh')
    shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
    # RUN_KIMG explicit: when called from a short job's step, the env would otherwise carry RUN_KIMG=500
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={st.get("full_dataset") or FULL_DATASET} RUN_KIMG=2000 '
            f'KEEP_LAST_DUMPS=2 ' + ss.ref_env(st) +
            f'TRAIN_EXTRA="{LR_FLAGS[st["lr"]]}"; nvidia-smi --query-gpu=name --format=csv,noheader; '
            f'bash {frozen} {run} slurm 0 0')
    cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
           '-p', PART, f'--nice={nice}', '--gres=gpu:1', '--cpus-per-task=6', f'--mem={st.get("mem", "14G")}', '-t', '24:00:00', '--requeue', '--wrap', wrap]
    return run, ss.sbatch(cmd)


def update(search):
    import fcntl
    with open(path() + '.lock', 'w') as lk:          # q4a and q4b jobs can finish at the same moment
        fcntl.flock(lk, fcntl.LOCK_EX)
        _update(search)


def _update(search):
    st = bo.load(ss.path(search), None)
    pts = st['points']
    for q in pts:                                   # scores straight from the result files (state may lag)
        if q.get('mind') is None:
            q['mind'], q['fid'] = ss.mind_fid(q['name'])
    log = bo.load(path(), {})
    mine = log.setdefault(search, {'checks': {}, 'runs': {}})
    byname = {q['name']: q for q in pts}
    for n in CHECKS:
        if str(n) in mine['checks']:                 # already decided; still retry any of its failed submissions
            top = [byname[k] for k in mine['checks'][str(n)]]
        elif len(pts) < n or any(q.get('mind') is None for q in pts[:n]):
            continue
        else:
            top = sorted(pts[:n], key=lambda q: q['mind'])[:TOP]
            mine['checks'][str(n)] = [q['name'] for q in top]
        for q in top:
            if mine['runs'].get(q['name'], {}).get('job', 'ERR').startswith('ERR'):   # new, or a failed sbatch
                                                                                   # (QOS submit limit): retry
                if os.environ.get('SS2M_DRY'):
                    print('would submit', q['name'], round(q['mind'], 5), 'for check', n); continue
                suffix = mine['runs'].get(q['name'], {}).get('suffix', '')     # set when a run had to be redone
                run, job = submit(st, q, suffix)
                mine['runs'][q['name']] = {'run': run, 'job': job, 'suffix': suffix,
                                           'submitted': time.strftime('%F %T')}
                print('submitted', run, job, 'for check', n)
    if not os.environ.get('SS2M_DRY'):
        bo.save(path(), log)


def fixed_end(search, checks=(8, 12, 16)):
    """Low-priority (nice) fixed-from-end versions of each checkpoint's top-1 pick: they start only on GPUs nobody
    else wants. Names ss2m_<search>_NNN_fe."""
    st = bo.load(ss.path(search), None)
    byname = {q['name']: q for q in st['points']}
    log = bo.load(path(), {})
    mine = log.setdefault(search + '_fe', {'runs': {}})
    for n in checks:
        q = byname[log[search]['checks'][str(n)][0]]
        if not mine['runs'].get(q['name'], {}).get('job', 'ERR').startswith('ERR'):
            continue
        x = fixed_from_end(q['x'])
        run, job = submit(st, q, '_fe', x=x, nice=10000, note='fixed-from-end last jump')
        mine['runs'][q['name']] = {'run': run, 'job': job, 'x': x, 'check': n, 'submitted': time.strftime('%F %T')}
        print('submitted', run, job, 'x', x)
    bo.save(path(), log)


if __name__ == '__main__':
    if sys.argv[1] == 'fixed_end':          # python launch_ss2m.py fixed_end fda
        import fcntl
        with open(path() + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            fixed_end(sys.argv[2])
    else:
        for s in sys.argv[1:]:
            update(s)
