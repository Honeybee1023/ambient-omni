#!/usr/bin/env python3
"""Bayesian optimisation of per-group "when to jump / where to jump to" on the mixed-blur set.

Every corrupted group g follows the same two-phase shape the mentors endorsed on 2026-10-03:
T = 0 (blurred images usable at every noise level) until `when_g`, then a single jump to
`where_g` for the rest of training. Clean images always stay at T = 0. With four groups the
design point is 8 numbers, all in [0, 1]:

    x = [when_1, u_1, when_2, u_2, when_3, u_3, when_4, u_4],   where_g = 0.5 + 0.5 * u_g

Two searches share this code and differ only in how images are grouped:
  true  groups are the four blur levels (g03 g05 g10 g20) -- uses the real labels, a ceiling
  km    groups are 1-D k-means clusters of Ambient-o's classifier thresholds (k1..k4),
        i.e. what an automatic method could actually see (dataset celeba_mix4_km)

The search drives itself on Slurm. `step` is idempotent and runs under a file lock:
  1. collect finished runs (MIND json present), resubmit a vanished run once, then mark failed
  2. top the search up to its concurrency with new proposals, submitting one sbatch each
Every training job runs `step` for its own search when it ends, so a finished run immediately
frees a slot for the next proposal; nothing waits for a whole batch. Proposals: the first
N_INIT points are scrambled Sobol (uniform coverage first); after that, expected improvement
under a GP, with still-running points included at their predicted value ("kriging believer")
so concurrent proposals spread out instead of piling onto one spot.

    python bo_mix4.py step --search true      # on a compute node (sbatch/srun), never a login node
    python bo_mix4.py status                  # read-only summary, safe anywhere
"""
import argparse, fcntl, glob, json, os, shutil, subprocess, sys, time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm, qmc

BASE = os.environ.get('AMBIENT_BASE') or sys.exit('AMBIENT_BASE must be set')
REPO = os.path.join(BASE, 'ambient-omni', 'pixel-diffusion')
GEN = os.path.join(BASE, 'generated')
STATE_DIR = os.path.join(GEN, 'bo_mix4')
LOGDIR = os.path.join(BASE, 'train_logs', 'bo_mix4')
MANIFEST = os.path.join(GEN, 'dyn_search_manifest.json')

SEARCHES = {
    'true': dict(groups=['g03', 'g05', 'g10', 'g20'], dataset='celeba_mix4_v1'),
    'km':   dict(groups=['k1', 'k2', 'k3', 'k4'], dataset='celeba_mix4_km'),
}
N_INIT = 8            # Sobol points before the GP takes over
BUDGET = 40           # total runs per search (initial + BO)
CONCURRENCY = 12      # runs in flight per search
NOISE_SD = 0.00089    # measured MIND replicate sd; the GP noise is clamped at or above it
MAX_ATTEMPTS = 2      # a run that vanishes without MIND is resubmitted once

# Good-citizen caps on OUR jobs (running + pending, both searches together).
# mit_preemptable's QOS allows 4 GPUs per user (sacctmgr, 2026-10-03), so more than a couple
# queued behind those 4 just sit idle; the Sloan partitions have no per-user GPU cap (ou_sloan_gpu
# allows 24 submitted jobs) and Giannis said to use them freely.
SLOAN_CAP = 12   # 2026-10-03 evening: we held 18 of 36 Sloan GPUs with ~15 jobs of the group waiting
PREEMPT_CAP = 6
SLOAN = dict(part='ou_sloan_gpu,sched_mit_sloan_gpu_r8', gres='gpu:1', time='24:00:00')
PREEMPT = dict(part='mit_preemptable', gres='gpu:l40s:1', time='2-00:00:00')


# ----------------------------------------------------------------------------- design space --

def decode(x, groups):
    """Unit-cube point -> per_group schedule. Values are rounded so names/specs stay readable."""
    spec = {}
    for i, g in enumerate(groups):
        when = round(float(x[2 * i]), 3)
        where = round(0.5 + 0.5 * float(x[2 * i + 1]), 3)
        if when <= 0.005:
            phases = [[0, where]]
        elif when >= 0.995:
            phases = [[0, 0.0]]
        else:
            phases = [[0, 0.0], [when, where]]
        spec[g] = {'phases': phases}
    return {'type': 'per_group', 'groups': spec}


# --------------------------------------------------------------------------------------- GP --

def matern52(A, B, ls, sf2):
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) / ls) ** 2, 0).sum(-1) + 1e-12)
    s = np.sqrt(5.0) * d
    return sf2 * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def _unpack(theta, D, floor):
    return np.exp(theta[:D]), np.exp(2 * theta[D]), np.exp(2 * theta[D + 1]) + floor ** 2


def nll(theta, X, y, floor):
    ls, sf2, sn2 = _unpack(theta, X.shape[1], floor)
    K = matern52(X, X, ls, sf2) + sn2 * np.eye(len(X))
    try:
        c = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    a = cho_solve(c, y)
    return float(0.5 * y @ a + np.log(np.diag(c[0])).sum())


def fit_gp(X, y, floor, restarts=16, seed=0):
    D = X.shape[1]
    rng = np.random.default_rng(seed)
    best, best_v = None, np.inf
    # Lengthscales capped at 2 on the unit cube: longer makes the surrogate near-linear and
    # puts the EI maximum on a corner by pure extrapolation (seen in the 4-D search).
    bounds = [(np.log(0.05), np.log(2.0))] * D + [(np.log(1e-2), np.log(3.0)), (np.log(1e-4), np.log(3.0))]
    for _ in range(restarts):
        x0 = np.concatenate([np.log(rng.uniform(0.15, 1.5, D)), [np.log(rng.uniform(0.5, 1.5))],
                             [np.log(rng.uniform(0.05, 0.5))]])
        try:
            r = minimize(nll, x0, args=(X, y, floor), method='L-BFGS-B', bounds=bounds)
        except Exception:
            continue
        if r.fun < best_v:
            best, best_v = r.x, r.fun
    if best is None:
        raise RuntimeError('GP fit failed on every restart')
    return best


def posterior(theta, X, y, Xs, floor):
    ls, sf2, sn2 = _unpack(theta, X.shape[1], floor)
    c = cho_factor(matern52(X, X, ls, sf2) + sn2 * np.eye(len(X)), lower=True)
    Ks = matern52(X, Xs, ls, sf2)
    mu = Ks.T @ cho_solve(c, y)
    var = np.maximum(sf2 - np.einsum('ij,ij->j', Ks, cho_solve(c, Ks)), 1e-12)
    return mu, np.sqrt(var)


def propose(done_X, done_y, pending_X, n_seen, sobol_seed, seed):
    """Next point. Space-filling Sobol until N_INIT runs have FINISHED (with 12 in flight the
    first proposals come before any result exists), else EI with kriging believer. The Sobol
    sequence is fixed per search, so point k is always row k however the calls are split."""
    D = 8
    if len(done_X) < N_INIT:
        return qmc.Sobol(d=D, scramble=True, seed=sobol_seed).random(BUDGET)[n_seen]
    X = np.asarray(done_X, float)
    y = np.asarray(done_y, float)
    mu_y, sd_y = y.mean(), y.std() if y.std() > 0 else 1.0
    ys = (y - mu_y) / sd_y
    floor = NOISE_SD / sd_y
    theta = fit_gp(X, ys, floor, seed=seed)
    # Running points enter at their predicted mean: the GP then treats them as known and
    # EI moves elsewhere, which is what keeps 12 concurrent proposals from being 12 copies.
    if len(pending_X):
        P = np.asarray(pending_X, float)
        mp, _ = posterior(theta, X, ys, P, floor)
        X = np.vstack([X, P]); ys = np.concatenate([ys, mp])
    rng = np.random.default_rng(seed)
    cand = [qmc.Sobol(d=D, scramble=True, seed=seed).random(8192)]
    top = X[np.argsort(ys)[:5]]
    cand.append(np.clip(np.repeat(top, 400, 0) + rng.normal(0, 0.07, (len(top) * 400, D)), 0, 1))
    C = np.vstack(cand)
    mu, sd = posterior(theta, X, ys, C, floor)
    best = ys[:len(done_y)].min()
    z = (best - mu) / sd
    ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
    return C[int(np.argmax(ei))]


# ------------------------------------------------------------------------------------ slurm --

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def our_jobs():
    """{job_name: (jobid, state, partition)} for our running/pending jobs."""
    r = sh(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%j|%i|%T|%P'])
    out = {}
    for line in r.stdout.splitlines():
        p = line.split('|')
        if len(p) == 4:
            out[p[0]] = (p[1], p[2], p[3])
    return out


def pick_partition(jobs):
    n_sloan = sum(1 for _, _, part in jobs.values() if 'sloan' in part)
    n_pre = sum(1 for _, _, part in jobs.values() if part == 'mit_preemptable')
    if n_sloan < SLOAN_CAP:
        return SLOAN
    if n_pre < PREEMPT_CAP:
        return PREEMPT
    return None


def submit(run, search, dataset, where):
    os.makedirs(LOGDIR, exist_ok=True)
    stamp = int(time.time())
    # Frozen copies: bash reads a script while executing it, so a git pull mid-run would
    # otherwise swap the job script underneath a running job.
    frozen = os.path.join(LOGDIR, f'run_dyn_job.{run}.{stamp}.sh')
    shutil.copy(os.path.join(REPO, 'run_dyn_job.sh'), frozen)
    me = os.path.join(LOGDIR, f'bo_mix4.{run}.{stamp}.py')
    shutil.copy(os.path.abspath(__file__), me)
    py = os.path.join(BASE, 'miniconda3', 'envs', 'ambient', 'bin', 'python')
    wrap = (f'export AMBIENT_BASE={BASE} DYN_DATASET={dataset} KEEP_LAST_DUMPS=2; '
            f'nvidia-smi --query-gpu=name --format=csv,noheader; '
            f'bash {frozen} {run} slurm 0 0; '
            f'cd {REPO} && {py} {os.path.join(REPO, "bo_mix4.py")} step --search {search}')
    cmd = ['sbatch', '--parsable', '-D', BASE, '-J', f'dyn_{run}', '-o', os.path.join(LOGDIR, f'{run}-%j.out'),
           '-p', where['part'], f'--gres={where["gres"]}', '--cpus-per-task=8', '--mem=64G',
           '-t', where['time'], '--requeue', '--wrap', wrap]
    r = sh(cmd)
    if r.returncode != 0:
        raise RuntimeError(f'sbatch failed for {run}: {r.stderr.strip()}')
    return r.stdout.strip().split(';')[0]


# ------------------------------------------------------------------------------------ state --

def load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save(path, obj):
    tmp = path + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1)
    os.replace(tmp, path)


def register(run, schedule, note):
    m = load(MANIFEST, {'runs': []})
    m['runs'] = [e for e in m['runs'] if e['name'] != run]
    m['runs'].append({'name': run, 'schedule': schedule, 'note': note})
    save(MANIFEST, m)


def mind_of(run):
    p = os.path.join(GEN, f'mind_dyn_{run}_s0.json')
    if not os.path.exists(p):
        return None, None
    mind = json.load(open(p))['mind']
    f = os.path.join(GEN, f'fid_dyn_{run}_s0.json')
    fid = None
    if os.path.exists(f):
        d = json.load(open(f)); fid = d.get('fid_score', d.get('fid'))
    return mind, fid


def step(search, dry=False):
    cfg = SEARCHES[search]
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f'{search}.json')
    with open(os.path.join(STATE_DIR, '.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load(path, {'search': search, 'points': []})
        jobs = our_jobs()
        # 1. collect
        for p in st['points']:
            if p['status'] != 'running':
                continue
            mind, fid = mind_of(p['name'])
            if mind is not None:
                p.update(status='done', mind=mind, fid=fid, finished=time.strftime('%F %T'))
            elif f'dyn_{p["name"]}' not in jobs:
                if p['attempts'] < MAX_ATTEMPTS and not dry:
                    where = pick_partition(jobs) or SLOAN
                    p['jobid'] = submit(p['name'], search, cfg['dataset'], where)
                    p['attempts'] += 1
                    p.setdefault('history', []).append(f'resubmitted {time.strftime("%F %T")} -> {where["part"]}')
                    jobs[f'dyn_{p["name"]}'] = (p['jobid'], 'PENDING', where['part'])
                else:
                    p['status'] = 'failed'
        # 2. top up
        while True:
            running = [p for p in st['points'] if p['status'] == 'running']
            if len(running) >= CONCURRENCY or len(st['points']) >= BUDGET:
                break
            where = pick_partition(jobs)
            if where is None:
                break
            done = [p for p in st['points'] if p['status'] == 'done']
            x = propose([p['x'] for p in done], [p['mind'] for p in done], [p['x'] for p in running],
                        len(st['points']), sobol_seed=11 if search == 'true' else 23,
                        seed=1000 + len(st['points']))
            idx = len(st['points'])
            run = f'mix4bo_{search}_{idx:03d}'
            sched = decode(x, cfg['groups'])
            if dry:
                print('would submit', run, where['part'], json.dumps(sched)); break
            register(run, sched, f'bo_mix4 {search} point {idx} ({"sobol" if len(done) < N_INIT else "EI"})')
            jobid = submit(run, search, cfg['dataset'], where)
            st['points'].append(dict(name=run, x=[float(v) for v in x], schedule=sched, status='running',
                                     jobid=jobid, partition=where['part'], attempts=1,
                                     submitted=time.strftime('%F %T')))
            jobs[f'dyn_{run}'] = (jobid, 'PENDING', where['part'])
            save(path, st)
        if not dry:
            save(path, st)


def status():
    for search in SEARCHES:
        st = load(os.path.join(STATE_DIR, f'{search}.json'), {'points': []})
        pts = st['points']
        done = sorted([p for p in pts if p['status'] == 'done'], key=lambda p: p['mind'])
        print(f'== {search}: {len(pts)} submitted, {len(done)} done, '
              f'{sum(p["status"] == "running" for p in pts)} running, {sum(p["status"] == "failed" for p in pts)} failed')
        for p in done[:8]:
            g = p['schedule']['groups']
            desc = '  '.join(f'{k}:' + ('/'.join(f'{a:g}@{b:g}' for a, b in v['phases'][1:]) or f'{v["phases"][0][1]:g}')
                             for k, v in g.items())
            print(f'   {p["mind"]:.5f}  fid {p["fid"]}  {p["name"]}  {desc}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['step', 'status'])
    ap.add_argument('--search', choices=list(SEARCHES))
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'status':
        status()
    else:
        searches = [a.search] if a.search else list(SEARCHES)
        for s in searches:
            step(s, dry=a.dry)
