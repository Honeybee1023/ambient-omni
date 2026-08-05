#!/bin/bash
# Restores the symlinks to our private research notes.
#
# The notes (literature review, research plans, meeting prep, agent handoff docs)
# are deliberately NOT in this repository, because it is public. They live in a
# canonical directory outside the repo, and what appears here are symlinks.
#
# Why symlinks rather than plain ignored files: ignored files are still deleted
# by `git clean -xdf`, and automated cleanup of untracked files is a normal thing
# for a tool or agent to do. With symlinks, the worst case is that the links are
# removed and the notes themselves are untouched. This script puts the links back.
#
# This script is tracked on purpose: tracked files survive `git clean`, so the
# recovery mechanism cannot be destroyed by the thing it recovers from.
#
#     ./pixel-diffusion/scripts/restore_private_notes.sh

set -u

CANON=${CANON:-/data/honjar/ambient-omni-private-notes}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # pixel-diffusion/

if [ ! -d "$CANON" ]; then
  echo "ERROR: canonical notes directory not found: $CANON" >&2
  echo "Nothing to link to. Check the backups before doing anything else:" >&2
  echo "  ls -d ${CANON%/*}/ambient-omni-private-notes-backup-*" >&2
  exit 1
fi

link() {
  local target=$1 name=$2
  local dest="$HERE/$name"
  if [ -L "$dest" ]; then
    echo "ok       $name (link already present)"
    return
  fi
  if [ -e "$dest" ]; then
    echo "SKIP     $name - a real file/dir is here, refusing to clobber it." >&2
    echo "         Merge it into $CANON by hand, then delete it and re-run." >&2
    return
  fi
  ln -s "$target" "$dest"
  echo "restored $name"
}

link ../../ambient-omni-private-notes/lit_review lit_review
link ../../ambient-omni-private-notes/HANDOFF.md HANDOFF.md
link ../../ambient-omni-private-notes/context.md context.md

echo
echo "Canonical store: $CANON"
find "$CANON" -type f | wc -l | xargs printf "%s files present\n"
