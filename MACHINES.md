# Where the data lives

Code is synced through GitHub (see `SYNC.md`). **Data is not.** Datasets,
checkpoints and generated samples stay on the machine that produced them, so
this file records what is where. Regenerate the listings under `inventory/`
with `pixel-diffusion/scripts/inventory.sh`.

Snapshot taken 2026-08-16.

## RULE: never kill a job you did not start

**Every machine here is shared.** All three run under accounts used by more than
one person and more than one project: `pbhat21` has held all four lysine A100s
and later a single card; a `cov_capture.py` benchmark from
`/var/local/honjar/diffusion/` took 52.9GB on each usable proline card; a
`p3b_breadth` job shares the CSAIL account. Some of these run under **our own
username** and are still not ours.

So, without exception:

- **Never `kill`, `pkill`, `scancel` or otherwise stop a process you did not
  start yourself in this session.** Not if it is idle-looking, not if it is
  using a GPU you want, not if it is under our username.
- Wanting the GPU is not a reason. Wait for it, use another, or ask.
- If a machine looks full, it *is* full. Report that; do not make room.

Two habits that follow from this:

1. **`pkill -f <pattern>` is banned** for anything but a pattern you can prove
   matches only your own processes -- and even then prefer explicit PIDs
   collected from `ps`. `-f` matches the *whole command line*, so it has
   repeatedly matched the invoking shell itself, and it will just as happily
   match a colleague's job that mentions the same dataset name.
2. Before stopping anything, print `ps -o user=,pid=,cmd=` for every PID you are
   about to touch and confirm each one is yours.

Killing your *own* runs is fine, but check what it costs first: on 2026-08-24 a
`scancel` issued without checking job state destroyed two CSAIL runs that had
already finished training and generated all 5000 images. They were minutes from
done. Guard reassignment with "skip if already RUNNING".

## Machines

| | Lysine | CSAIL |
|---|---|---|
| SSH | `utcs-lysine` (`gdaras@lysine.cs.utexas.edu`) | `csail-slurm` (via `csail-login`) |
| `$AMBIENT_BASE` | `/data-local/honjar` | `/data/scratch/honjar` |
| Repo (running) | `$AMBIENT_BASE/ambient-omni` | `$AMBIENT_BASE/ambient-omni` |
| Repo (git) | same | `~/ambient-omni` |
| Python | `$AMBIENT_BASE/miniconda3/envs/ambient/bin/python` | same layout |
| Scheduler | none — plain `tmux` per GPU | Slurm |

Note the account mismatch: lysine's checkout lives under the shared `gdaras`
account, not `honjar`. That is why commits from there were misattributed.

## Data volumes

| | `annotated_datasets` | `train_outputs` | `generated` |
|---|---|---|---|
| **Lysine** | 106 entries, 68 GB | 77 entries, 354 GB | 330 entries, 4.6 GB |
| **CSAIL** | 873 entries | 1382 entries | 2236 entries |

CSAIL sizes are not recorded: `du` over that tree does not finish in a
reasonable time on CSAIL's NFS. Entry counts are exact.

**The two machines hold different eras of the project.** CSAIL has roughly 8×
the datasets and 18× the training outputs — the whole historical body of work
(AFHQ wolves, per-category, 2-domain, binary search, Bayesian optimisation,
CelebA v2/v2b sweeps). Lysine has far fewer but *newer* runs: the v2b
conditional sweeps and the dynamic-T work.

So neither machine is a superset. Before rerunning anything, check the relevant
`inventory/` listing — the run you want may already exist on the other cluster.

## Inventory listings

Exact directory listings, one filename per line:

- `inventory/lysine_annotated_datasets.txt`, `inventory/csail_annotated_datasets.txt`
- `inventory/lysine_train_outputs.txt`, `inventory/csail_train_outputs.txt`
- `inventory/lysine_generated.txt`, `inventory/csail_generated.txt`

To find which machine has a given run:

```bash
grep -l celeba_v2b_b3_T050 inventory/*_annotated_datasets.txt
```

## Not backed up anywhere

- **Everything above.** Neither cluster's data directory is replicated or
  backed up. A disk loss on either machine is unrecoverable.
- **`$AMBIENT_BASE/ambient-omni-private-notes/`** on lysine — `context.md`,
  `HANDOFF.md`, `lit_review/`, reached through symlinks from the repo and
  deliberately kept out of git. Roughly 240 KB of research notes existing in
  exactly one place.

Both are worth a periodic `rsync` somewhere durable.

## Known gotcha

Lysine's filesystem was renamed `/data` → `/data-local`; `/data` no longer
exists there at all. Any script with a literal `/data/honjar/...` path predates
the rename and is broken. Use `$AMBIENT_BASE` — see `SYNC.md`.
