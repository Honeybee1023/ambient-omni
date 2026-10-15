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

import argparse, fcntl, json, os, shutil, subprocess, sys, time

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


AMBO_SRC = 'afhqdog_mix4_ambo15'          # same images as SRC, named by blur level, with Ambient-o's per-image annotations
DIAG = f'{AMBIENT_BASE}/generated/pareto/afhq_diag.json'
def diag_subset(L): return f'afhq_par_L{L}_ambo'


def build_diag():
    """Each L subset re-expressed in Ambient-o's own dataset: the same images (k<n>_g<lvl>_x -> g<lvl>_x), with their
    per-image classifier annotations, so CURVE_SCHED=annot runs Ambient-o and =off runs clean-only on exactly the
    images the search's short runs saw."""
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


def submit_diag():
    """Ambient-o (annot) and clean-only (off) as short runs at every L, scored like the search's short runs."""
    log = json.load(open(DIAG)) if os.path.exists(DIAG) else {}
    py = f'{AMBIENT_BASE}/miniconda3/envs/ambient/bin/python'
    for L in LS:
        f = L / T_KIMG
        for tag, sched, extra in (('ambo', 'annot', '--cls_ema_window=1'), ('clean', 'off', '')):
            name = f'par_afhq_L{L}_{tag}'
            if not str(log.get(name, {}).get('job', 'ERR')).startswith('ERR'):
                continue
            e = env({'CURVE_DATA': diag_subset(L), 'CURVE_SCHED': sched, 'CURVE_EXTRA': extra,
                     'CURVE_MIMG': f'{L / 1000:g}', 'CURVE_LR_RAMP': f'{100 * f:g}', 'CURVE_REF2': heldout(L),
                     'CURVE_MIND_REF2': '1', 'CURVE_FINAL_KIMG': str(L), 'CURVE_EVAL_KIMGS': str(L)})
            wrap = (e + f'bash {REPO}/run_curve.sh {name} && CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES%%,*}} '
                    f'bash {REPO}/run_curve_eval.sh {name}')
            job = sbatch(G2 + ['-t', '06:00:00', '-J', name, '-o', f'{LOG}/{name}-%j.out', '--wrap', wrap])
            log[name] = {'job': job, 'L': L, 'sched': sched, 'submitted': time.strftime('%F %T')}
            print(name, job)
    json.dump(log, open(DIAG, 'w'), indent=1)


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
    if any(x.startswith('--dependency') for x in args):     # a job only waiting must not hold one of the 24 slots;
        args = list(args); i = args.index('-p')                 # sloan_slots() adds ou_sloan_gpu back once it is ready
        args[i + 1] = ','.join(p for p in args[i + 1].split(',') if p != 'ou_sloan_gpu')
    r = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
    if r.returncode != 0 and 'QOSMaxSubmitJobPerUserLimit' in r.stderr:
        a = list(args)
        i = a.index('-p')
        a[i + 1] = ','.join(p for p in a[i + 1].split(',') if p != 'ou_sloan_gpu')
        r = subprocess.run(['sbatch', '--parsable'] + a, capture_output=True, text=True)
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
    # nice: short runs yield to the long training curves and T runs when GPUs free up (they all share one QOS cap)
    return sbatch(G2 + ['-t', '06:00:00', '--nice=1000', '-J', name, '-o', f'{LOG}/{name}-%j.out', '--wrap', wrap])


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
            if q is not None and not str(q.get('job', 'ERR')).startswith('ERR') and str(q.get('eval_job', 'ERR')).startswith('ERR'):
                q['eval_job'] = sbatch(G1 + [f'--dependency=afterok:{q["job"]}', '-J', f'ceval_{q["name"]}', '-o',
                                             f'{LOG}/eval_{q["name"]}-%j.out', '--wrap',
                                             env({'CURVE_FINAL_KIMG': str(T_KIMG), 'CURVE_EVAL_KIMGS': str(T_KIMG)})
                                             + f'exec bash {REPO}/run_curve_eval.sh {q["name"]}'])
                print('eval resubmitted', q['name'], q['eval_job'])
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


FT_SWITCH = 0.75     # same switch point as the Ambient-o-then-clean baseline and CelebA's finetune75
HOLDOUT = f'{AMBIENT_BASE}/afhqdog_processed/val_64'     # 500 clean dogs never trained on (held-out loss curve)


def submit_baseline(name, data, sched):
    """A baseline trained to T, scored at EVERY saved snapshot (512 kimg, 10k samples) so its best snapshot <= T can be
    reported, then its held-out denoising loss by noise level at every snapshot (own output file, merged by the page)."""
    e = env({'CURVE_DATA': data, 'CURVE_SCHED': sched, 'CURVE_MIMG': f'{T_KIMG / 1000:g}', 'CURVE_LR_RAMP': '100'})
    t = sbatch(G2 + ['-t', '06:00:00', '-J', name, '-o', f'{LOG}/{name}-%j.out',
                     '--wrap', e + f'exec bash {REPO}/run_curve.sh {name}'])
    print(name, 'train', t)
    followups(name, t)


def submit_scoring(name, after):
    # two jobs (alternate snapshots): one job scoring all 20 at 10k samples would not fit the 6-h limit
    ks = list(range(512, T_KIMG + 1, 512))
    return ','.join(sbatch(G1 + [f'--dependency=afterok:{after}', '-J', f'ceval_{name}', '-o', f'{LOG}/eval_{name}-%j.out',
                                 '--wrap', env({'CURVE_FINAL_KIMG': str(T_KIMG), 'CURVE_EVAL_KIMGS': ' '.join(map(str, part))})
                                 + f'exec bash {REPO}/run_curve_eval.sh {name}']) for part in (ks[1::2], ks[0::2]))


def submit_finetune():
    """Fine-tune baseline at T: every blurred image used as clean (T 0) until FT_SWITCH, then clean images only."""
    path = f'{SCHED_DIR}/finetune75.json'
    if not os.path.exists(path):
        json.dump({'type': 'per_group', 'groups': {g: {'phases': [[0, 0.0], [FT_SWITCH, 'off']]} for g in GROUPS if g != 'b0'}},
                  open(path, 'w'))
    submit_baseline('par_afhq_T_finetune75', SRC, f'file:{path}')


def followups(name, after):
    """Scoring every 512 kimg + held-out loss curve for a baseline whose training job is already queued."""
    py = f'{AMBIENT_BASE}/miniconda3/envs/ambient/bin/python'
    ck = ' '.join(f'--ckpt {name}:{AMBIENT_BASE}/train_outputs/curve/{name}/network-snapshot-{k:06d}.pkl'
                  for k in range(512, T_KIMG + 1, 512))
    lc = sbatch(G1 + [f'--dependency=afterok:{after}', '-J', f'losscurve_{name}', '-o', f'{LOG}/losscurve_{name}-%j.out',
                      '--wrap', f'export AMBIENT_BASE={AMBIENT_BASE}; cd {REPO} && exec {py} analysis/eval_holdout_loss_curve.py '
                      f'--holdout_dir {HOLDOUT} --out {AMBIENT_BASE}/generated/pareto/afhq_losscurve_{name}.json {ck}'])
    print(name, 'eval', submit_scoring(name, after), 'losscurve', lc)


def submit_s02_lastjump():
    """Two one-change tests of s02's LAST jump (k1, the mildest group: T 0 -> 0.90 at 92.3%) at T:
    A (when): the jump kept the same distance from the end as in the L=1024 short run (s02's pick length),
              i.e. 1 - (1 - 0.923) * 1024 / T;   B (where): same time, level 0.90 -> 0.80."""
    base = json.load(open(f'{SCHED_DIR}/s02.json'))
    w, lv = base['groups']['k1']['phases'][1]
    for tag, nw, nlv in (('lastfe', round(1 - (1 - w) * 1024 / T_KIMG, 5), lv), ('last080', w, 0.8)):
        path = f'{SCHED_DIR}/s02_{tag}.json'
        if not os.path.exists(path):
            s = json.loads(json.dumps(base))
            s['groups']['k1']['phases'][1] = [nw, nlv]
            json.dump(s, open(path, 'w'))
        submit_baseline(f'par_afhq_T_s02_{tag}', SRC, f'file:{path}')


def submit_lastfe(i, L):
    """Pick i at T with the rule that won for s02 (test A): only the latest jump moves, to the same distance from the
    end as in the L-kimg short run it was picked at; the other jumps stay proportional. s06 (picked at L=512) is the
    check that the rule is not specific to s02 (user, 2026-10-15)."""
    s = json.load(open(f'{SCHED_DIR}/s{i:02d}.json'))
    g = max((g for g in s['groups'] if s['groups'][g]['phases'][1][0] < 1), key=lambda g: s['groups'][g]['phases'][1][0])
    w, lv = s['groups'][g]['phases'][1]
    s['groups'][g]['phases'][1] = [round(1 - (1 - w) * L / T_KIMG, 5), lv]
    path = f'{SCHED_DIR}/s{i:02d}_lastfe.json'
    json.dump(s, open(path, 'w'))
    submit_baseline(f'par_afhq_T_s{i:02d}_lastfe', SRC, f'file:{path}')


def submit_oracle():
    """Idealised upper-bound baseline: all 4,739 training dogs clean (no blur at all), no schedule."""
    submit_baseline('par_afhq_T_oracle', 'afhqdog_oracle', 'none')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name'); ap.add_argument('--after')
    ap.add_argument('cmd', choices=['followups', 'build', 'init', 'topup', 'status', 'build_diag', 'submit_diag', 'submit_finetune', 'submit_oracle', 'submit_s02_lastjump', 'submit_s06_lastfe'])
    a = ap.parse_args()
    if a.cmd == 'followups':
        followups(a.name, a.after); raise SystemExit
    {'build': build, 'init': init, 'topup': topup, 'status': status, 'build_diag': build_diag,
     'submit_diag': submit_diag, 'submit_finetune': submit_finetune, 'submit_oracle': submit_oracle, 'submit_s02_lastjump': submit_s02_lastjump,
     'submit_s06_lastfe': lambda: submit_lastfe(6, 512)}[a.cmd]()
