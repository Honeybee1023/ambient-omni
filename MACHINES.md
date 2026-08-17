# Where the data lives

Code is synced through GitHub (see `SYNC.md`). **Data is not.** Datasets,
checkpoints and generated samples stay on the machine that produced them, so
this file records what is where. Regenerate the listings under `inventory/`
with `pixel-diffusion/scripts/inventory.sh`.

Snapshot taken 2026-08-16.

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
