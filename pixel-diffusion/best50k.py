"""50k-sample scores at each run's best snapshots (user, 2026-10-19: final rows need 50k FID and MIND; this is the next
use for any idle GPU after our own training and 10k scoring).

For every listed run whose 10k curve is complete up to T, finds the snapshot with the best 10k FID and the one with the
best 10k MIND (at most 2 snapshots) and submits one single-GPU job per snapshot that has no 50k result yet
(run_curve_eval.sh with CURVE_EVAL_N=50000 -> k<kimg>_n50k.json). Jobs go to any partition we may use, below our
training and 10k scoring (nice 600) but above the gap-filler seeds (nice 1000), may requeue after preemption, and
already-submitted snapshots are skipped, so it is safe to run again whenever more seeds finish.

Usage: python best50k.py c700|afhq [--dry] <run> [<run> ...]
"""
import json, os, re, subprocess, sys

DATA = sys.argv[1]
P = __import__('pareto_c700' if DATA == 'c700' else 'pareto_afhq')
FK = 'fid_ref2' if DATA == 'c700' else 'fid_train'     # the FID key each tab reports
DRY = '--dry' in sys.argv
CURVE = f'{P.AMBIENT_BASE}/generated/curve'
PARTS = 'ou_sloan_gpu,sched_mit_sloan_gpu_r8,mit_normal_gpu,mit_preemptable'
queued = subprocess.run(['squeue', '-u', os.environ.get('USER', 'honjar'), '-h', '-o', '%j'],
                        capture_output=True, text=True).stdout.split()

for name in [a for a in sys.argv[2:] if not a.startswith('--')]:
    rows = []
    for f in os.listdir(f'{CURVE}/{name}') if os.path.isdir(f'{CURVE}/{name}') else []:
        if re.fullmatch(r'k\d{6}\.json', f):
            r = json.load(open(f'{CURVE}/{name}/{f}'))
            if 0 < r['kimg'] <= P.T_KIMG and r.get(FK) is not None and r.get('mind') is not None:
                rows.append(r)
    if len(rows) < P.T_KIMG // 512:
        print(f'{name}: {len(rows)} of {P.T_KIMG // 512} snapshots scored at 10k, skipped'); continue
    ks = sorted({min(rows, key=lambda r: r[FK])['kimg'], min(rows, key=lambda r: r['mind'])['kimg']})
    for k in ks:
        jn = f'c50k_{name}_k{k}'
        if os.path.exists(f'{CURVE}/{name}/k{k:06d}_n50k.json') or jn in queued:
            print(f'{name} k{k}: done or queued'); continue
        args = [P.X, '-p', PARTS, '--gres=gpu:1', '-c', '8', '--mem=96G', '-t', '06:00:00', '--time-min=04:30:00',  # CelebA 50k on an A100 needs >4 h (2026-10-22)
                '--nice=600', '--requeue', '-J', jn, '-o', f'{P.LOG}/{jn}-%j.out', '--wrap',
                P.env({'CURVE_FINAL_KIMG': str(P.T_KIMG), 'CURVE_EVAL_KIMGS': str(k), 'CURVE_EVAL_N': '50000',
                       'CURVE_EVAL_DEFER': '0'}) + f'exec bash {P.REPO}/run_curve_eval.sh {name}']
        out = 'dry' if DRY else subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
        if not DRY and out.returncode and 'QOSMaxSubmitJob' in out.stderr:   # ou_sloan_gpu takes 24 queued jobs per user
            args[args.index('-p') + 1] = PARTS.replace('ou_sloan_gpu,', '')
            out = subprocess.run(['sbatch', '--parsable'] + args, capture_output=True, text=True)
        print(name, f'k{k}', out if DRY else (out.stdout.strip() or out.stderr.strip()))
