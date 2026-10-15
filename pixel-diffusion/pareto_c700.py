#!/usr/bin/env python3
"""Deployment Pareto study on CelebA 1% (2026-10-13): the AFHQ study (pareto_afhq.py) repeated on celeba_c700_mix4
(700 clean + 63,000 blurred faces at blur 0.3/0.5/1.0/2.0).

T = 15360 kimg (Ambient-o's CelebA-1% curve stops improving there; chosen by the user). Short lengths L = 512 / 1024 /
2048 / 4096 kimg, each a scaled-down copy of the T run: every k-means group (clean included) subsampled by f = L / T
(nested subsets), LR warm-up scaled by f, otherwise run_curve.sh's EDM 64-px recipe with CelebA's dropout 0.05.
Schedules: ONE scrambled Sobol sequence (SOBOL_SEED, new: not the AFHQ one) over short_search's space, the same 20
schedules at every L, so N = 8, 12, 16, 20 are prefixes. Pick = top-1 of the first N by MIND against the clean faces
that L's subset does not contain (held-out); FID is recorded for every run too. Each distinct pick is trained to T on
the full k-means data with its last jump kept the same distance from the end as in the short run it was picked at
(fixed-from-end; proportional copies only by hand, submit_prop) and scored at every 512-kimg snapshot (FID/MIND vs the 70k clean train faces and vs the test
split). Ambient-o and clean-only also run as short runs on each subset's exact images (diagnostics, as on AFHQ).

Usage (Engaging; CPU via srun/sbatch, never the login node):
    python pareto_c700.py build | build_diag | init | submit_diag | topup | status
Every short job calls topup when it ends, so the study drives itself.
"""
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or "/orcd/scratch/orcd/006/honjar"

import argparse, fcntl, json, os, shutil, subprocess, sys, time

import numpy as np

import short_search as ss

T_KIMG = 15360
LS = (512, 1024, 2048, 4096)
N_RUNS = {512: 20, 1024: 20, 2048: 20, 4096: 20}
CHECKS = (8, 12, 16, 20)
SOBOL_SEED = 20261013                                 # new for CelebA 1%; the AFHQ study used 20261012
SUB_SEED = 20261013
CAP = 10                                              # our pending pareto jobs at a time
SRC = 'celeba_c700_mix4_km15'                         # cls15 classifier -> Ambient-o T -> 1-D k-means, k1 = mildest
AMBO_SRC = 'celeba_c700_mix4_ambo15'                  # same images named by blur level, with per-image annotations
GROUPS = ('b0', 'k1', 'k2', 'k3', 'k4')
CELEBA = f'{AMBIENT_BASE}/celeba70k_processed'
FULL_REF = f'{CELEBA}/train_clean_64'                 # 70k clean train faces (benchmark reference)
TEST_REF = f'{CELEBA}/test_64'                        # official test split (used to choose T)
HOLDOUT = f'{CELEBA}/test_500_64'                     # 500 test faces for the held-out loss curve
DATASETS = f'{AMBIENT_BASE}/annotated_datasets'
STATE = f'{AMBIENT_BASE}/generated/pareto/c700.json'
DIAG = f'{AMBIENT_BASE}/generated/pareto/c700_diag.json'
SCHED_DIR = f'{AMBIENT_BASE}/generated/pareto/c700_schedules'
LOG = f'{AMBIENT_BASE}/train_logs/pareto'
REPO = f'{AMBIENT_BASE}/ambient-omni/pixel-diffusion'
PY = f'{AMBIENT_BASE}/miniconda3/envs/ambient/bin/python'
X = '--exclude=node[4004,4007-4008,5003-5005,5101,5103-5104,5106,5202-5204]'
# Order among our own jobs only (user, 2026-10-14: "low priority" means relative to our other jobs; always try to get
# our jobs onto free GPUs). Offsets are tiny next to the ~250k priorities, so they barely move us against other users.
# Full runs to T: nice 0; short runs: NICE_SHORT; proportional copies: NICE_PROP. (scontrol top is disabled here.)
NICE_SHORT, NICE_PROP = 10, 20
G2 = (f'{X} -p ou_sloan_gpu,sched_mit_sloan_gpu_r8,mit_normal_gpu,mit_preemptable --gres=gpu:2 -c 16 --mem=128G '
      f'--requeue --signal=B:USR1@600').split()
G1 = f'{X} -p sched_mit_sloan_gpu_r8,ou_sloan_gpu,mit_normal_gpu --gres=gpu:1 -c 8 --mem=96G -t 06:00:00'.split()


def short_name(L, i): return f'par_c700_L{L}_s{i:02d}'
def full_name(i): return f'par_c700_T_s{i:02d}'
def subset(L): return f'c700_par_L{L}'
def diag_subset(L): return f'c700_par_L{L}_ambo'
def heldout(L): return f'{CELEBA}/c700_par_heldout_L{L}'


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
            if fn not in seen:                          # clean faces this subset never trains on
                os.link(f'{src}/{fn}', f'{ref}/{fn[len("b0_"):]}')
        summary[L] = {g: len(keep[g]) for g in GROUPS}
        summary[L]['heldout_clean'] = len(by['b0']) - len(seen)
        print(L, summary[L])
    json.dump({'source': SRC, 'sub_seed': SUB_SEED, 'T_kimg': T_KIMG, 'subsets': summary},
              open(f'{DATASETS}/c700_par_subsets.meta.json', 'w'), indent=1)
    if not os.path.exists(HOLDOUT):                     # fixed 500 test faces for the held-out loss curve
        os.makedirs(HOLDOUT)
        for fn in sorted(os.listdir(TEST_REF))[:500]:
            os.link(f'{TEST_REF}/{fn}', f'{HOLDOUT}/{fn}')


def build_diag():
    """Each L subset re-expressed in Ambient-o's own dataset (k<n>_g<lvl>_x -> g<lvl>_x, per-image annotations)."""
    src = f'{DATASETS}/{AMBO_SRC}'
    amb = {json.loads(l)['filename']: l for l in open(f'{src}/annotations.jsonl')}
    for L in LS:
        out = f'{DATASETS}/{diag_subset(L)}'
        if os.path.exists(out):
            sys.exit(f'{out} exists; refusing to overwrite')
        names = [json.loads(l)['filename'] for l in open(f'{DATASETS}/{subset(L)}/annotations.jsonl')]
        names = [n if n.startswith('b0_') else n.split('_', 1)[1] for n in names]
        missing = [n for n in names if n not in amb]
        assert not missing, f'{len(missing)} images not in {AMBO_SRC}, e.g. {missing[:3]}'
        os.makedirs(out)
        with open(f'{out}/annotations.jsonl', 'w') as fh:
            for n in names:
                os.link(f'{src}/{n}', f'{out}/{n}')
                fh.write(amb[n] if amb[n].endswith('\n') else amb[n] + '\n')
        shutil.copy(f'{src}/sigmas.txt', out)
        print(L, len(names), 'images ->', out)


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
    """sbatch; if ou_sloan_gpu's per-user submit limit (QOS, 24 queued) is hit, retry without that partition."""
    r = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    if r.returncode != 0 and 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
        a = list(args)
        i = a.index('-p')
        a[i + 1] = ','.join(p for p in a[i + 1].split(',') if p != 'ou_sloan_gpu')
        r = subprocess.run(['sbatch', '--parsable'] + a, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else 'ERR ' + r.stderr.strip()


def env(extra):
    # every variable set explicitly: a job submitted from inside another job must not inherit its settings
    base = {'AMBIENT_BASE': AMBIENT_BASE, 'CURVE_DROPOUT': '0.05', 'CURVE_EXTRA': '', 'CURVE_REF': FULL_REF,
            'CURVE_EVAL_RESUBMIT': '0', 'CURVE_EVAL_NOWAIT': '1', 'CURVE_REF2': TEST_REF, 'CURVE_MIND_REF2': '0'}
    base.update(extra)
    return 'export ' + ' '.join(f'{k}="{v}"' for k, v in base.items()) + '; '


def short_job(name, L, e, after_eval=''):
    """Train a short run, score its last snapshot, then run `after_eval`. Written as a script and exec'd, with the
    time-limit signal forwarded to run_curve.sh, so a run that hits the 6-h limit requeues and resumes (a plain
    --wrap chain would leave sh as the batch shell and the run would just die at the limit)."""
    os.makedirs(f'{LOG}/scripts', exist_ok=True)
    path = f'{LOG}/scripts/{name}.sh'
    evpath = f'{LOG}/scripts/eval_{name}.sh'
    snap = f'{AMBIENT_BASE}/train_outputs/curve/{name}/network-snapshot-{L:06d}.pkl'
    # scoring alone, for when training ends too late in the 6-h job to score there (slow GPU: L2048_s06 on an A100
    # finished training at 4.5 h and timed out while scoring, 2026-10-15)
    with open(evpath, 'w') as fh:
        fh.write('#!/bin/bash\n' + e + '\n' + f'bash {REPO}/run_curve_eval.sh {name}\n' + after_eval + '\n')
    hand_off = ' '.join(['sbatch'] + G1 + ['-J', f'eval_{name}', '-o', f'{LOG}/eval_{name}-%j.out', '--wrap',
                                           f'"exec bash {evpath}"'])
    with open(path, 'w') as fh:
        fh.write('#!/bin/bash\n' + e + '\n'
                 f'bash {REPO}/run_curve.sh {name} & P=$!\n'
                 "trap 'kill -USR1 $P' USR1\n"
                 'wait $P; rc=$?; while kill -0 $P 2>/dev/null; do wait $P; rc=$?; done\n'
                 '[ $rc -eq 0 ] || exit $rc          # training failed: fail the job so the watcher sees it\n'
                 f'[ -f {snap} ] || exit 0          # requeued (or failed): nothing to score yet\n'
                 f'[ $SECONDS -gt 14400 ] && {{ {hand_off}; exit 0; }}   # under 2 h left: score in its own job\n'
                 f'CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES%%,*}} bash {REPO}/run_curve_eval.sh {name}\n'
                 + after_eval + '\n')
    return sbatch(G2 + ['-t', '06:00:00', f'--nice={NICE_SHORT}', '-J', name, '-o', f'{LOG}/{name}-%j.out',
                        '--wrap', f'exec bash {path}'])


def submit_short(st, L, i):
    name = short_name(L, i)
    f = L / T_KIMG
    e = env({'CURVE_DATA': subset(L), 'CURVE_SCHED': f'file:{SCHED_DIR}/s{i:02d}.json', 'CURVE_MIMG': f'{L / 1000:g}',
             'CURVE_LR_RAMP': f'{100 * f:g}', 'CURVE_REF2': heldout(L), 'CURVE_MIND_REF2': '1',
             'CURVE_FINAL_KIMG': str(L), 'CURVE_EVAL_KIMGS': str(L)})
    return short_job(name, L, e, f'cd {REPO} && {PY} pareto_c700.py topup')


def submit_diag():
    """Ambient-o (annot) and clean-only (off) as short runs at every L, scored like the search's short runs."""
    log = json.load(open(DIAG)) if os.path.exists(DIAG) else {}
    for L in LS:
        f = L / T_KIMG
        for tag, sched, extra in (('ambo', 'annot', '--cls_ema_window=1'), ('clean', 'off', '')):
            name = f'par_c700_L{L}_{tag}'
            if not str(log.get(name, {}).get('job', 'ERR')).startswith('ERR'):
                continue
            e = env({'CURVE_DATA': diag_subset(L), 'CURVE_SCHED': sched, 'CURVE_EXTRA': extra,
                     'CURVE_MIMG': f'{L / 1000:g}', 'CURVE_LR_RAMP': f'{100 * f:g}', 'CURVE_REF2': heldout(L),
                     'CURVE_MIND_REF2': '1', 'CURVE_FINAL_KIMG': str(L), 'CURVE_EVAL_KIMGS': str(L)})
            log[name] = {'job': short_job(name, L, e), 'L': L, 'sched': sched, 'submitted': time.strftime('%F %T')}
            print(name, log[name]['job'])
    json.dump(log, open(DIAG, 'w'), indent=1)


def submit_scoring(name, after):
    # scoring every 512-kimg snapshot (30 at T) at 10k samples: ~8 per 6-h job, so split into 4 interleaved jobs
    ks = list(range(512, T_KIMG + 1, 512))
    return ','.join(sbatch(G1 + [f'--dependency=afterok:{after}', '-J', f'ceval_{name}', '-o', f'{LOG}/eval_{name}-%j.out',
                                 '--wrap', env({'CURVE_FINAL_KIMG': str(T_KIMG), 'CURVE_EVAL_KIMGS': ' '.join(map(str, ks[j::4]))})
                                 + f'exec bash {REPO}/run_curve_eval.sh {name}']) for j in range(4))


def fe_sched(i, L):
    """Schedule i scaled to T with its LAST jump kept the same distance from the end as in the L-kimg short run
    (AFHQ test A, 2026-10-14: s02 scaled this way beat Ambient-o at T, proportional scaling did not). The other
    jumps keep their fraction of training. Returns the schedule file."""
    path = f'{SCHED_DIR}/s{i:02d}_feL{L}.json'
    if not os.path.exists(path):
        s = json.load(open(f'{SCHED_DIR}/s{i:02d}.json'))
        g = max((g for g, v in s['groups'].items() if v['phases'][1][0] < 1), key=lambda g: s['groups'][g]['phases'][1][0])
        w, lv = s['groups'][g]['phases'][1]
        s['groups'][g]['phases'][1] = [round(1 - (1 - w) * L / T_KIMG, 5), lv]
        json.dump(s, open(path, 'w'))
    return path


def submit_full(st, i, L=None, nice=0):
    """Pick i trained to T: last jump fixed-from-end of its pick length L (the default rule), or proportional (L None,
    lowest of our jobs)."""
    name = full_name(i) + (f'_feL{L}' if L else '')
    sched = fe_sched(i, L) if L else f'{SCHED_DIR}/s{i:02d}.json'
    e = env({'CURVE_DATA': SRC, 'CURVE_SCHED': f'file:{sched}', 'CURVE_MIMG': f'{T_KIMG / 1000:g}',
             'CURVE_LR_RAMP': '100'})
    t = sbatch(G2 + ['-t', '06:00:00'] + ([f'--nice={nice}'] if nice else []) + ['-J', name, '-o', f'{LOG}/{name}-%j.out',
                     '--wrap', e + f'exec bash {REPO}/run_curve.sh {name}'])
    ck = ' '.join(f'--ckpt {name}:{AMBIENT_BASE}/train_outputs/curve/{name}/network-snapshot-{k:06d}.pkl'
                  for k in range(1024, T_KIMG + 1, 1024))
    sbatch(G1 + [f'--dependency=afterok:{t}', '-J', f'losscurve_{name}', '-o', f'{LOG}/losscurve_{name}-%j.out',
                 '--wrap', f'export AMBIENT_BASE={AMBIENT_BASE}; cd {REPO} && exec {PY} analysis/eval_holdout_loss_curve.py '
                 f'--holdout_dir {HOLDOUT} --out {AMBIENT_BASE}/generated/pareto/c700_losscurve_{name}.json {ck}'])
    return t, submit_scoring(name, t)


def result(name, kimg):
    p = f'{AMBIENT_BASE}/generated/curve/{name}/k{kimg:06d}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def pending_jobs():
    r = subprocess.run(['squeue', '-u', os.environ.get('USER', 'honjar'), '-h', '-t', 'PD', '-o', '%j'],
                       capture_output=True, text=True)
    return sum(1 for j in r.stdout.split() if j.startswith('par_c700_'))


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
        # picks: top-1 by held-out MIND among the first N runs at each L (FID pick recorded alongside)
        for L in LS:
            for n in CHECKS:
                key = f'L{L}_N{n}'
                if key in st['picks'] or n > int(st['n_runs'][str(L)]):
                    continue
                rs = [st['short'].get(short_name(L, i), {}).get('res') for i in range(n)]
                if any(r is None for r in rs):
                    continue
                st['picks'][key] = {'pick': int(np.argmin([r['mind_ref2'] for r in rs])),
                                    'pick_fullref': int(np.argmin([r['mind'] for r in rs])),
                                    'pick_fid': int(np.argmin([r['fid_ref2'] for r in rs])),
                                    'decided': time.strftime('%F %T')}
                print('pick', key, st['picks'][key])
        for key, p in st['picks'].items():             # each distinct (pick, L) to T, last jump fixed-from-end
            i, L = p['pick'], int(key.split('_')[0][1:])
            k = f'{i}_feL{L}'
            q = st['full'].get(k)
            if q is None or str(q.get('job', 'ERR')).startswith('ERR'):
                t, ev = submit_full(st, i, L)
                st['full'][k] = {'name': f'{full_name(i)}_feL{L}', 'job': t, 'eval_job': ev,
                                 'submitted': time.strftime('%F %T')}
                print('scaled to T (last jump fixed-from-end)', k, t, ev)
        save(st)


def submit_prop():
    """Proportional versions of the picks (user, 2026-10-14): lowest priority among OUR jobs, submitted by hand."""
    with open(STATE + '.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = load()
        for i in sorted({p['pick'] for p in st['picks'].values()}):
            if str(i) in st['full']:
                continue
            t, ev = submit_full(st, i, None, nice=NICE_PROP)
            st['full'][str(i)] = {'name': full_name(i), 'job': t, 'eval_job': ev, 'submitted': time.strftime('%F %T')}
            print('proportional to T', full_name(i), t, ev)
        save(st)


def status():
    st = load()
    for L in LS:
        print(f'--- L = {L} kimg (data x{L / T_KIMG:.4f})')
        for i in range(int(st['n_runs'][str(L)])):
            q = st['short'].get(short_name(L, i), {})
            r = q.get('res') or result(short_name(L, i), L)
            s = (f"MIND held-out {r['mind_ref2']:.4f}  FID held-out {r['fid_ref2']:.2f}" if r else q.get('job', 'not submitted'))
            print(f'  s{i:02d} {[round(v, 2) for v in st["x"][i]]}  {s}')
    print('picks', json.dumps(st['picks']))
    for i, q in st['full'].items():
        r = result(q['name'], T_KIMG)
        print(q['name'], q['job'], f"MIND {r['mind']:.4f} FID test {r['fid_ref2']:.2f}" if r else '')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'build_diag', 'init', 'submit_diag', 'topup', 'status', 'submit_prop'])
    a = ap.parse_args()
    {'build': build, 'build_diag': build_diag, 'init': init, 'submit_diag': submit_diag, 'topup': topup,
     'status': status, 'submit_prop': submit_prop}[a.cmd]()
