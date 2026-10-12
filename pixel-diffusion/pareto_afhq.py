#!/usr/bin/env python3
"""Deployment Pareto study on AFHQ-dog (2026-10-12): cost of the short-run schedule search vs quality at T.

T = 10240 kimg (Ambient-o's AFHQ curve is converged there; its 10240 snapshot is the baseline). For each short length
L in LS, a scaled-down copy of the T run: every k-means group (clean included) subsampled by f = L / T (nested: the
1/20 subset sits inside the 1/10 one, ...), LR warm-up scaled by f, otherwise the training-curve settings
(run_curve.sh: EDM 64-px recipe, batch 256, lr 2e-4, dropout 0.25). Schedules: ONE scrambled Sobol sequence (seed
SOBOL_SEED, never used before) over short_search's space (one monotone jump per group, levels .85-1.0, jump times
.10-.95), the SAME sequence at every L, so N = 8, 12, 16, 20 are prefixes of it.

Pick = top-1 of the first N runs by MIND against the HELD-OUT clean dogs: the clean images that L's subset does not
contain -- what a practitioner can score with, since the short run never trained on them. MIND against all 4739
clean training dogs (the benchmark reference) is recorded too, to check the two rank runs alike. Each pick is trained
to T on the full k-means data with the same jump fractions and levels; picks that coincide across (L, N) run once.

Usage (Engaging; CPU via srun/sbatch, never the login node):
    python pareto_afhq.py build              # subsets + held-out reference folders (hard links)
    python pareto_afhq.py topup              # submit queued short runs up to CAP pending; also scales finished picks
    python pareto_afhq.py status             # table + picks
Every short job calls topup when it ends, so the study drives itself.
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or "/orcd/scratch/orcd/006/honjar"

import argparse, fcntl, json, os, subprocess, sys, time

import numpy as np

import short_search as ss

T_KIMG = 10240
LS = (512, 1024, 1536, 2048)
N_RUNS = {512: 20, 1024: 20, 1536: 12, 2048: 8}      # trimmed grid (N*L <= 20M); extend by raising these
CHECKS = (8, 12, 16, 20)
SOBOL_SEED = 20261012
SUB_SEED = 20261012
CAP = 10                                              # our pending pareto jobs at a time (Sloan QOS: 24 queued/user)
SRC = 'afhqdog_mix4_km15'                             # cls15 classifier -> Ambient-o T -> 1-D k-means, k1 = mildest
GROUPS = ('b0', 'k1', 'k2', 'k3', 'k4')
FULL_REF = f'{AMBIENT_BASE}/afhqdog_processed/train_clean_64'
DATASETS = f'{AMBIENT_BASE}/annotated_datasets'
STATE = f'{AMBIENT_BASE}/generated/pareto/afhq.json'
SCHED_DIR = f'{AMBIENT_BASE}/generated/pareto/schedules'
LOG = f'{AMBIENT_BASE}/train_logs/pareto'
REPO = f'{AMBIENT_BASE}/ambient-omni/pixel-diffusion'
X = '--exclude=node[4004,4007-4008,5003-5005,5101,5103-5104,5106,5202-5204]'
G2 = (f'{X} -p ou_sloan_gpu,sched_mit_sloan_gpu_r8,mit_normal_gpu,mit_preemptable --gres=gpu:2 -c 16 --mem=128G '
      f'--requeue --signal=B:USR1@600').split()
G1 = f'{X} -p sched_mit_sloan_gpu_r8,ou_sloan_gpu,mit_normal_gpu --gres=gpu:1 -c 8 --mem=96G -t 06:00:00'.split()


def short_name(L, i): return f'par_afhq_L{L}_s{i:02d}'
def full_name(i): return f'par_afhq_T_s{i:02d}'
def subset(L): return f'afhq_par_L{L}'
def heldout(L): return f'{AMBIENT_BASE}/afhqdog_processed/par_heldout_L{L}'


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else None


def save(st):
    tmp = STATE + '.tmp'
    json.dump(st, open(tmp, 'w'), indent=1)
    os.replace(tmp, STATE)


def build():
    src = f'{DATASETS}/{SRC}'
    lines = [json.loads(l) for l in open(f'{src}/annotations.jsonl')]
    by = {g: sorted(l['filename'] for l in lines if l['filename'].startswith(g + '_')) for g in GROUPS}
    assert sum(map(len, by.values())) == len(lines), 'unexpected file prefix'
    rng = np.random.RandomState(SUB_SEED)
    order = {g: [by[g][j] for j in rng.permutation(len(by[g]))] for g in GROUPS}    # nested: prefixes of one order
    line_of = {l['filename']: l for l in lines}
    summary = {}
    for L in LS:
        f = L / T_KIMG
        keep = {g: order[g][:max(1, int(round(f * len(order[g]))))] for g in GROUPS}
        out, ref = f'{DATASETS}/{subset(L)}', heldout(L)
        if os.path.exists(out) or os.path.exists(ref):
            sys.exit(f'{out} or {ref} exists; refusing to overwrite')
        os.makedirs(out); os.makedirs(ref)
        with open(f'{out}/annotations.jsonl', 'w') as fh:
            for g in GROUPS:
                for fn in keep[g]:
                    os.link(f'{src}/{fn}', f'{out}/{fn}')
                    fh.write(json.dumps(line_of[fn]) + '\n')
        seen = set(keep['b0'])
        for fn in by['b0']:
            if fn not in seen:                          # clean dogs this subset never trains on
                os.link(f'{src}/{fn}', f'{ref}/{fn[len("b0_"):]}')
        summary[L] = {g: len(keep[g]) for g in GROUPS}
        summary[L]['heldout_clean'] = len(by['b0']) - len(seen)
        print(L, summary[L])
    json.dump({'source': SRC, 'sub_seed': SUB_SEED, 'T_kimg': T_KIMG, 'subsets': summary},
              open(f'{DATASETS}/afhq_par_subsets.meta.json', 'w'), indent=1)


def init():
    if load():
        sys.exit('state exists')
    os.makedirs(SCHED_DIR, exist_ok=True)
    n = max(N_RUNS.values())
    xs = ss.space_filling(SOBOL_SEED, 32)[:n]          # one sequence; prefixes are the N-run designs
    st = {'sobol_seed': SOBOL_SEED, 'T_kimg': T_KIMG, 'n_runs': {str(k): v for k, v in N_RUNS.items()},
          'x': [[float(v) for v in x] for x in xs], 'short': {}, 'full': {}, 'picks': {},
          'created': time.strftime('%F %T')}
    for i, x in enumerate(xs):
        json.dump(ss.schedule(x), open(f'{SCHED_DIR}/s{i:02d}.json', 'w'))
    save(st)
    print('created', STATE, len(xs), 'schedules')


def sbatch(args):
    r = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else 'ERR ' + r.stderr.strip()


def env(extra):
    # every variable set explicitly: a job submitted from inside another job must not inherit its settings
    base = {'AMBIENT_BASE': AMBIENT_BASE, 'CURVE_DROPOUT': '0.25', 'CURVE_EXTRA': '', 'CURVE_REF': FULL_REF,
            'CURVE_EVAL_RESUBMIT': '0', 'CURVE_EVAL_NOWAIT': '1', 'CURVE_REF2': '', 'CURVE_MIND_REF2': '0'}
    base.update(extra)
    return 'export ' + ' '.join(f'{k}="{v}"' for k, v in base.items()) + '; '


def submit_short(st, L, i):
    name = short_name(L, i)
    f = L / T_KIMG
    e = env({'CURVE_DATA': subset(L), 'CURVE_SCHED': f'file:{SCHED_DIR}/s{i:02d}.json', 'CURVE_MIMG': f'{L / 1000:g}',
             'CURVE_LR_RAMP': f'{100 * f:g}', 'CURVE_REF2': heldout(L), 'CURVE_MIND_REF2': '1',
             'CURVE_FINAL_KIMG': str(L), 'CURVE_EVAL_KIMGS': str(L)})
    py = f'{AMBIENT_BASE}/miniconda3/envs/ambient/bin/python'
    wrap = (e + f'bash {REPO}/run_curve.sh {name} && CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES%%,*}} '
            f'bash {REPO}/run_curve_eval.sh {name}; cd {REPO} && {py} pareto_afhq.py topup')
    return sbatch(G2 + ['-t', '06:00:00', '-J', name, '-o', f'{LOG}/{name}-%j.out', '--wrap', wrap])


def submit_full(st, i):
    name = full_name(i)
    e = env({'CURVE_DATA': SRC, 'CURVE_SCHED': f'file:{SCHED_DIR}/s{i:02d}.json', 'CURVE_MIMG': f'{T_KIMG / 1000:g}',
             'CURVE_LR_RAMP': '100'})
    t = sbatch(G2 + ['-t', '06:00:00', '-J', name, '-o', f'{LOG}/{name}-%j.out',
                     '--wrap', e + f'exec bash {REPO}/run_curve.sh {name}'])
    ev = sbatch(G1 + [f'--dependency=afterok:{t}', '-J', f'ceval_{name}', '-o', f'{LOG}/eval_{name}-%j.out', '--wrap',
                      env({'CURVE_FINAL_KIMG': str(T_KIMG), 'CURVE_EVAL_KIMGS': str(T_KIMG)})
                      + f'exec bash {REPO}/run_curve_eval.sh {name}'])
    return t, ev


def result(name, kimg):
    p = f'{AMBIENT_BASE}/generated/curve/{name}/k{kimg:06d}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def pending_jobs():
    r = subprocess.run(['squeue', '-u', os.environ.get('USER', 'honjar'), '-h', '-t', 'PD', '-o', '%j'],
                       capture_output=True, text=True)
    return sum(1 for j in r.stdout.split() if j.startswith('par_afhq_'))


def topup():
    os.makedirs(LOG, exist_ok=True)
    with open(STATE + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load()
        for k, q in st['short'].items():
            if q.get('res') is None:
                q['res'] = result(k, q['L'])
        order = sorted(((i, L) for L in LS for i in range(int(st['n_runs'][str(L)]))), key=lambda t: (t[0] // 4, t[1], t[0]))
        free = CAP - pending_jobs()
        for i, L in order:
            name = short_name(L, i)
            q = st['short'].get(name)
            if q and not str(q.get('job', 'ERR')).startswith('ERR'):
                continue
            if free <= 0:
                break
            job = submit_short(st, L, i)
            st['short'][name] = {'L': L, 'i': i, 'job': job, 'res': None, 'submitted': time.strftime('%F %T')}
            print('submitted', name, job)
            free -= 1
        # picks: top-1 by held-out MIND among the first N runs at each L, once all N have results
        for L in LS:
            for n in CHECKS:
                key = f'L{L}_N{n}'
                if key in st['picks'] or n > int(st['n_runs'][str(L)]):
                    continue
                rs = [st['short'].get(short_name(L, i), {}).get('res') for i in range(n)]
                if any(r is None for r in rs):
                    continue
                i_best = int(np.argmin([r['mind_ref2'] for r in rs]))
                i_full = int(np.argmin([r['mind'] for r in rs]))
                st['picks'][key] = {'pick': i_best, 'pick_fullref': i_full, 'decided': time.strftime('%F %T')}
                print('pick', key, i_best, '(full-ref pick', i_full, ')')
        for p in st['picks'].values():                  # scale each distinct pick to T
            i = p['pick']
            q = st['full'].get(str(i))
            if q is None or str(q.get('job', 'ERR')).startswith('ERR'):
                t, ev = submit_full(st, i)
                st['full'][str(i)] = {'name': full_name(i), 'job': t, 'eval_job': ev, 'submitted': time.strftime('%F %T')}
                print('scaled to T', full_name(i), t, ev)
        save(st)


def status():
    st = load()
    for L in LS:
        print(f'--- L = {L} kimg (data x{L / T_KIMG:g})')
        for i in range(int(st['n_runs'][str(L)])):
            q = st['short'].get(short_name(L, i), {})
            r = q.get('res') or result(short_name(L, i), L)
            s = (f"MIND held-out {r['mind_ref2']:.4f}  full {r['mind']:.4f}  FID {r['fid_train']:.2f}" if r
                 else q.get('job', 'not submitted'))
            print(f'  s{i:02d} {[round(v, 2) for v in st["x"][i]]}  {s}')
    print('picks', json.dumps(st['picks']))
    for i, q in st['full'].items():
        r = result(q['name'], T_KIMG)
        print(q['name'], q['job'], f"MIND {r['mind']:.4f} FID {r['fid_train']:.2f}" if r else '')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'init', 'topup', 'status'])
    a = ap.parse_args()
    {'build': build, 'init': init, 'topup': topup, 'status': status}[a.cmd]()
