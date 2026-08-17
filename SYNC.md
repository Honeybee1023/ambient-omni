# Keeping the checkouts in sync

There are **four** working copies of this repo and one remote. The remote is the
only source of truth; the checkouts are disposable views of it.

| Where | Path | Role |
|---|---|---|
| **Remote** | `github.com/Honeybee1023/ambient-omni` | **Source of truth.** Everything flows through here. |
| Lysine | `/data-local/honjar/ambient-omni` | UT Austin GPU box. Most recent experiments. |
| CSAIL scratch | `/data/scratch/honjar/ambient-omni` | MIT run directory — jobs execute here. |
| CSAIL home | `~/ambient-omni` | MIT git-side checkout (scratch has flaky git over NFS). |
| Mac | `~/Downloads/ambient-diffusion-omni` | Local editing and review. No GPUs. |

Working branch is `scaling-law-threshold-search`. `main` and `tpu-port` are
upstream history from `giannisdaras/ambient-omni` — leave them alone.

## The rule

**Never edit a checkout without pulling first, and never end a session without
pushing.** Four copies drift apart within days otherwise; that drift is exactly
what this document exists to stop. Concretely, every time you sit down at any
machine:

```bash
cd <that machine's checkout>
./sync.sh pull      # fast-forward from the remote, refuses if you have local edits
#   ... work ...
./sync.sh push      # commit + push, so the other three can pick it up
```

Then, on each *other* machine you care about, `./sync.sh pull`. There is no
push-to-all: GitHub is the hub, and each checkout pulls from it independently.

`./sync.sh status` prints where all reachable checkouts sit relative to the
remote, which is the fastest way to spot a machine that has fallen behind.

## Why commits kept showing the wrong author

Lysine's repo lives under the shared `gdaras` account, so it inherited that
account's git identity and 15 commits landed as
`papachristoumarios <kasraarabi@gmail.com>`. Those have been rewritten, and each
checkout now sets its identity locally. If you clone somewhere new, set it
before the first commit:

```bash
git config user.name  "Honeybee1023"
git config user.email "honjar@mit.edu"
```

`./sync.sh push` checks this and refuses to commit under the wrong identity,
because fixing authorship after the fact means a history rewrite and a force
push.

## Machine-specific paths

The tree is laid out identically everywhere, under one base directory:

```
$AMBIENT_BASE/
    ambient-omni/         <- this repo
    annotated_datasets/
    train_outputs/
    train_logs/
    generated/
    miniconda3/
```

Only the base differs: `/data-local/honjar` on lysine, `/data/scratch/honjar` on
CSAIL. So scripts refer to `$AMBIENT_BASE` rather than a literal path, and the
value is detected automatically. Before running anything interactively:

```bash
source env.sh          # exports AMBIENT_BASE, AMBIENT_PY, AMBIENT_GENERATED, ...
$AMBIENT_PY train.py   # instead of a hardcoded conda path
```

Scripts also self-resolve, so they work when launched directly without sourcing
anything. In new Python code, prefer the module:

```python
from ambient_paths import GENERATED, DATASETS, OUTPUTS
```

**Do not commit a literal `/data/...` path.** It will be correct on exactly one
machine. On a new machine, or to override detection, create `env.local.sh` next
to `env.sh` (it is gitignored) with `AMBIENT_BASE=/your/path`.

Lysine's filesystem was renamed `/data` → `/data-local` at some point, which is
why older scripts pointed at a `/data/honjar` that no longer exists. The
indirection above is what keeps that from recurring.

## What is deliberately not in git

**This GitHub repo is public.** It is a fork of `giannisdaras/ambient-omni`, and
anything committed here is world-readable forever. Private notes were pushed
once by accident on 2026-07-31 and had to be purged from history; the rules
below exist so that does not happen again.

- **Datasets, checkpoints, generated samples.** Lysine alone holds ~68 GB of
  datasets, ~354 GB of training outputs and ~4.6 GB of samples. These live under
  `$AMBIENT_BASE` and are *not* replicated between machines — an experiment's
  outputs stay on the machine that ran it. `MACHINES.md` records what is where.
- **Research notes, handoff docs and notebooks.** `context.md`, `HANDOFF.md`,
  `lit_review/` and every `research_notebook_*.ipynb` live in
  `ambient-omni-private-notes/`, outside git. `*.ipynb` is ignored repo-wide;
  do not `git add -f` your way past it.

### Syncing the private notes

Git cannot carry these, and lysine and CSAIL cannot reach each other, so **your
Mac is the hub** for notes the way GitHub is the hub for code:

```bash
./sync.sh notes pull lysine     # cluster -> Mac
./sync.sh notes push csail      # Mac -> cluster
./sync.sh notes status          # file counts on all three
```

Direction is always explicit and nothing is ever deleted — these are additive
unions, not mirrors, so a note that exists on only one machine survives. To
propagate a change everywhere: pull from where you made it, then push to the
other two.

`ambient-omni-private-notes/` exists on exactly the machines you have pushed it
to, and is backed up nowhere. It is the least replicated thing in this project.

## Archived material

`pixel-diffusion/archive/csail/` holds files kept for reference when the two
lines of work were merged:

- `variants/` — CSAIL's versions of files that also exist on lysine with
  different content. Lysine's versions won; these are the losing side, kept so
  nothing was lost.
- `HANDOFF.csail.md` — the CSAIL handoff doc, superseded by the private notes.
- `research_notebook_phase3.outputs-stripped.ipynb` — a second copy of that
  notebook with outputs cleared. The live copy in `notebooks/` retains outputs.

Nothing here is on an active code path. Delete it once you are confident it is
no longer needed.
