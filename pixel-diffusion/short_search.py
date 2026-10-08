#!/usr/bin/env python3
"""Schedule search on cheap short runs (the candidate automatic method).

A short run copies a full run at reduced scale: every k-means group (clean included) subsampled to 1/SCALE and
RUN_KIMG = full_kimg / SCALE, with the LR warm-up scaled by the same factor, so passes per image and the LR curve
match the full run at every fraction of training and jump times carry over as plain fractions (transfer test,
2026-10-05/06: 1/4 data at 500 kimg ranked 8 CelebA schedules with Spearman 0.90 against their 2000-kimg MIND).

Search space (user, 2026-10-05): one jump per group, monotone and therefore non-crossing -- the worse a group, the
earlier AND at least as high it jumps. Groups are listed mildest -> worst (k1..k4). Levels from LEVELS. Objective:
log MIND (FID recorded alongside). First N_INIT=8 space-filling runs at once, then batches of 4: a GP (Matern 5/2,
bo_mix4's fit) picks each batch by expected improvement with "kriging believer" for points already chosen. A new
batch is submitted only when the previous one has fully finished.

Usage (Engaging, via srun/sbatch):
    python short_search.py init  --search q4a --dataset celeba_mix4_km_q4 --lr old|wu
    python short_search.py step  --search q4a        # collect results; submit the next batch if one is due
    python short_search.py noise --search q4a --source mix4bo_km_056   # one repeat (seed 1) to measure noise
    python short_search.py status --search q4a
A search stops submitting new batches when its state has "paused": true.
"""
import argparse, json, os, shutil, sys, time

import numpy as np
from scipy.stats import norm, qmc

import bo_mix4 as bo

BASE, GEN = bo.BASE, bo.GEN
STATE_DIR = os.path.join(GEN, 'short_search')
LOGDIR = os.path.join(BASE, 'train_logs', 'short_search')
GROUPS = ('k1', 'k2', 'k3', 'k4')               # mildest -> worst
LEVELS = (0.85, 0.90, 0.95, 1.0)
WHEN_RANGE = (0.10, 0.95)
BATCH, N_INIT = 4, 8
LR_FLAGS = {'old': '--lr_rampup_kimg=2500',                    # the default ramp (10000 kimg) scaled by 1/4
            'wu': '--lr=2e-4 --lr_rampup_kimg=25'}            # proper warm-up (100 kimg at 2000) scaled by 1/4
PART = os.environ.get('SHORT_PART', 'ou_sloan_gpu,sched_mit_sloan_gpu_r8')   # untyped gres on mit_normal_gpu = L40S


def path(search):
    return os.path.join(STATE_DIR, f'{search}.json')


def sample(rng, n):
    """n random monotone schedules: x = [w1..w4, l1..l4] (k1..k4); w4 <= w3 <= w2 <= w1, l4 >= l3 >= l2 >= l1."""
    w = np.sort(WHEN_RANGE[0] + (WHEN_RANGE[1] - WHEN_RANGE[0]) * rng.random((n, 4)), axis=1)[:, ::-1]
    lv = np.sort(rng.choice(LEVELS, size=(n, 4)), axis=1)
    return np.hstack([w, lv])


def space_filling(seed, n):
    """Sobol over the 4 jump times (sorted into monotone order) and stratified monotone level choices."""
    s = qmc.Sobol(d=8, scramble=True, seed=seed).random(n)
    w = np.sort(WHEN_RANGE[0] + (WHEN_RANGE[1] - WHEN_RANGE[0]) * s[:, :4], axis=1)[:, ::-1]
    lv = np.sort(np.array(LEVELS)[np.minimum((s[:, 4:] * len(LEVELS)).astype(int), len(LEVELS) - 1)], axis=1)
    return np.hstack([w, lv])


def feats(X):
    X = np.asarray(X, float)
    return np.hstack([X[:, :4], (X[:, 4:] - LEVELS[0]) / (LEVELS[-1] - LEVELS[0])])


def schedule(x):
    return {'type': 'per_group', 'groups': {g: {'phases': [[0, 0.0], [round(float(x[i]), 3), float(x[4 + i])]]}
                                            for i, g in enumerate(GROUPS)}}


def propose_batch(st, rng):
    done = [p for p in st['points'] if p.get('mind') is not None]
    if not st['points']:                       # one joint Sobol design of N_INIT spread-out runs, all at once
        return space_filling(st['seed'], N_INIT)
    X = feats([p['x'] for p in done]); y = np.log([p['mind'] for p in done])
    mu_y, sd_y = y.mean(), (y.std() or 1.0)
    ys = (y - mu_y) / sd_y
    floor = st['noise_log_sd'] / sd_y
    theta = bo.fit_gp(X, ys, floor, seed=len(done))
    top = np.asarray([p['x'] for p in done], float)[np.argsort(y)[:5]]
    loc = np.repeat(top, 600, 0)
    loc[:, :4] = np.clip(loc[:, :4] + rng.normal(0, 0.04, (len(loc), 4)), *WHEN_RANGE)
    loc[:, :4] = np.sort(loc[:, :4], axis=1)[:, ::-1]
    flip = rng.random((len(loc), 4)) < 0.2
    loc[:, 4:] = np.sort(np.where(flip, rng.choice(LEVELS, size=(len(loc), 4)), loc[:, 4:]), axis=1)
    C = np.vstack([sample(rng, 16384), loc])
    Xk, yk, out = X.copy(), ys.copy(), []
    best = ys.min()
    for _ in range(BATCH):
        mu, sd = bo.posterior(theta, Xk, yk, feats(C), floor)
        z = (best - mu) / sd
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        i = int(np.argmax(ei))
        out.append(C[i])
        Xk = np.vstack([Xk, feats(C[i:i + 1])]); yk = np.append(yk, mu[i])     # kriging believer
        C = np.delete(C, i, 0)
    return np.array(out)


def ref_env(st):
    """MIND/FID reference (AFHQ scores against its own clean training dogs). Always set, empty = run_dyn_job.sh's
    CelebA default, so a job submitted from inside another search's job never inherits that job's reference."""
    return ''.join(f'{k}={st.get(v) or ""} ' for k, v in (('DYN_REF', 'ref'), ('DYN_REF_CACHE', 'ref_cache')))


def submit(st, run, sched, seed=0):
    bo.register(run, sched, f"short search {st['search']} ({st['dataset']}, {st['run_kimg']} kimg, LR {st['lr']})")
    os.makedirs(LOGDIR, exist_ok=True)
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{int(time.time())}.sh')
    shutil.copy(os.path.join(bo.REPO, 'run_dyn_job.sh'), frozen)
    py = os.path.join(BASE, 'miniconda3', 'envs', 'ambient', 'bin', 'python')
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={st["dataset"]} RUN_KIMG={st["run_kimg"]} KEEP_LAST_DUMPS=2 '
            + ref_env(st) +
            f'TRAIN_EXTRA="{LR_FLAGS[st["lr"]]}"; nvidia-smi --query-gpu=name --format=csv,noheader; '
            f'bash {frozen} {run} slurm {seed} 0; cd {bo.REPO} && {py} short_search.py step --search {st["search"]}')
    cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
           '-p', PART, '--gres=gpu:1', '--cpus-per-task=6', f'--mem={st.get("mem", "14G")}', '-t', '03:00:00', '--requeue',
           '--wrap', wrap]
    r = bo.sh(cmd)
    return r.stdout.strip() or ('ERR ' + r.stderr.strip())


def mind_fid(run, seed=0):
    out = []
    for k, f in (('mind', f'mind_dyn_{run}_s{seed}.json'), ('fid_score', f'fid_dyn_{run}_s{seed}.json')):
        p = os.path.join(GEN, f)
        out.append(json.load(open(p))[k] if os.path.exists(p) else None)
    return out


def step(search):
    p = path(search)
    with open(p + '.lock', 'w') as lk:
        import fcntl
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = bo.load(p, None)
        if st is None:
            sys.exit(f'no search {search}; run init')
        for q in st['points'] + st.get('noise', []):
            if q.get('mind') is None:
                q['mind'], q['fid'] = mind_fid(q['name'], q.get('seed', 0))
        for q in st['points'] + st.get('noise', []):     # a failed sbatch (e.g. the QOS submit limit) is retried
            if q.get('mind') is None and str(q.get('job', '')).startswith('ERR'):
                sched = (schedule(q['x']) if q.get('x') else
                         {e['name']: e for e in json.load(open(bo.MANIFEST))['runs']}[q['source']]['schedule'])
                q['job'] = submit(st, q['name'], sched, seed=q.get('seed', 0))
                print('resubmitted', q['name'], q['job'])
        pending = [q for q in st['points'] if q.get('mind') is None]
        if (not pending and len(st['points']) < st['budget'] and not st.get('paused')
                and not os.path.exists(os.path.join(STATE_DIR, 'PAUSE'))):
            rng = np.random.default_rng(st['seed'] + 7 * len(st['points']))
            for x in propose_batch(st, rng):
                i = len(st['points'])
                run = f"ss_{search}_{i:03d}"
                job = submit(st, run, schedule(x))
                st['points'].append({'name': run, 'x': [float(v) for v in x], 'job': job, 'mind': None, 'fid': None,
                                     'submitted': time.strftime('%F %T')})
                print('submitted', run, job, [round(float(v), 3) for v in x])
        done = [q for q in st['points'] if q.get('mind') is not None]
        for n in (8, 16, 24, 40):          # the pick the search would make after n runs (budget curve)
            if len(done) >= n and str(n) not in st['picks']:
                first = sorted(st['points'][:n], key=lambda q: q['mind'])[0]
                st['picks'][str(n)] = first['name']
        bo.save(p, st)
    import launch_ss2m                                   # scale finished checkpoints' top 3 up to 2000 kimg
    if search in launch_ss2m.AUTO:
        launch_ss2m.update(search)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['init', 'step', 'noise', 'status'])
    ap.add_argument('--search', required=True)
    ap.add_argument('--dataset', default='celeba_mix4_km_q4')
    ap.add_argument('--lr', choices=list(LR_FLAGS), default='old')
    ap.add_argument('--run_kimg', type=int, default=500)
    ap.add_argument('--budget', type=int, default=40)
    ap.add_argument('--noise_log_sd', type=float, default=0.04, help='GP noise floor on log MIND (0.04 ~ 2 MIND units at 0.05)')
    ap.add_argument('--source', help='noise: manifest name of the schedule to repeat')
    ap.add_argument('--seed', type=int, default=1, help='noise: seed of the repeat (1 -> ss_<s>_noise, 0 -> _noise0)')
    ap.add_argument('--full_dataset', default='celeba_mix4_km', help='dataset of the 2000-kimg copies (launch_ss2m)')
    ap.add_argument('--ref', help='MIND/FID reference image dir (default: run_dyn_job.sh CelebA holdout)')
    ap.add_argument('--ref_cache', help='MIND reference feature cache matching --ref')
    ap.add_argument('--mem', default='14G')
    a = ap.parse_args()
    os.makedirs(STATE_DIR, exist_ok=True)
    if a.cmd == 'init':
        if os.path.exists(path(a.search)):
            sys.exit('exists')
        bo.save(path(a.search), {'search': a.search, 'dataset': a.dataset, 'lr': a.lr, 'run_kimg': a.run_kimg,
                                 'budget': a.budget, 'noise_log_sd': a.noise_log_sd, 'seed': 1234, 'points': [],
                                 'noise': [], 'picks': {}, 'created': time.strftime('%F %T'),
                                 'full_dataset': a.full_dataset, 'ref': a.ref, 'ref_cache': a.ref_cache, 'mem': a.mem})
        print('created', path(a.search))
    elif a.cmd == 'step':
        step(a.search)
    elif a.cmd == 'noise':
        st = bo.load(path(a.search), None)
        man = {e['name']: e for e in json.load(open(bo.MANIFEST))['runs']}
        run = f"ss_{a.search}_noise" + ('' if a.seed == 1 else str(a.seed))
        job = submit(st, run, man[a.source]['schedule'], seed=a.seed)
        st['noise'].append({'name': run, 'source': a.source, 'seed': a.seed, 'job': job, 'mind': None, 'fid': None})
        bo.save(path(a.search), st); print('noise run', run, job)
    else:
        st = bo.load(path(a.search), None)
        for q in st['points'] + st['noise']:
            print(q['name'], q.get('mind'), q.get('fid'), [round(v, 3) for v in q.get('x', [])])
        print('picks', st['picks'])


if __name__ == '__main__':
    main()
