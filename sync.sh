#!/bin/bash
# Keep this checkout in step with the remote.  See SYNC.md.
#
#   ./sync.sh pull                 fast-forward from the remote
#   ./sync.sh push [msg]           commit everything and push
#   ./sync.sh status               where this checkout sits vs the remote
#   ./sync.sh status --all         ...and every other reachable checkout (from the Mac)
#   ./sync.sh notes pull <host>    copy private notes from a cluster to the Mac
#   ./sync.sh notes push <host>    copy private notes from the Mac to a cluster
#   ./sync.sh notes status         show note counts on every machine

set -u

BRANCH_DEFAULT=scaling-law-threshold-search
WANT_NAME="Honeybee1023"
WANT_EMAIL="honjar@mit.edu"

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }

branch() { git rev-parse --abbrev-ref HEAD; }

check_identity() {
    local name email
    name=$(git config user.name || true)
    email=$(git config user.email || true)
    if [ "$name" != "$WANT_NAME" ] || [ "$email" != "$WANT_EMAIL" ]; then
        red "Wrong git identity in this checkout:"
        red "    user.name  = ${name:-<unset>}   (want $WANT_NAME)"
        red "    user.email = ${email:-<unset>}  (want $WANT_EMAIL)"
        echo
        echo "Committing now would mislabel the commit and need a history rewrite to fix."
        echo "Set it with:"
        echo "    git config user.name  \"$WANT_NAME\""
        echo "    git config user.email \"$WANT_EMAIL\""
        return 1
    fi
}

cmd_pull() {
    local b; b=$(branch)
    if [ -n "$(git status --porcelain)" ]; then
        red "Uncommitted changes here -- refusing to pull over them."
        git status --short
        echo
        echo "Commit them first (./sync.sh push), or stash them (git stash)."
        return 1
    fi
    git fetch origin || return 1
    if ! git merge --ff-only "origin/$b"; then
        red "Cannot fast-forward: this checkout has commits the remote does not."
        echo "Push them first (./sync.sh push), or inspect with:"
        echo "    git log --oneline origin/$b..HEAD"
        return 1
    fi
    grn "Up to date with origin/$b."
}

cmd_push() {
    local b msg; b=$(branch)
    check_identity || return 1
    if [ -z "$(git status --porcelain)" ]; then
        ylw "Nothing to commit."
    else
        msg="${1:-}"
        if [ -z "$msg" ]; then
            red "Refusing to invent a commit message."
            echo "Usage: ./sync.sh push \"what changed and why\""
            return 1
        fi
        git add -A || return 1
        git commit -m "$msg" || return 1
    fi
    # Land on top of whatever else arrived, so the remote history stays linear
    # and no one has to resolve a merge commit later.
    git fetch origin || return 1
    if ! git merge --ff-only "origin/$b" 2>/dev/null; then
        ylw "Remote moved on; rebasing onto origin/$b."
        git rebase "origin/$b" || {
            red "Rebase hit a conflict. Resolve, then: git rebase --continue && ./sync.sh push"
            return 1
        }
    fi
    git push origin "$b" || return 1
    grn "Pushed to origin/$b. Run ./sync.sh pull on the other checkouts."
}

cmd_status() {
    local b; b=$(branch)
    git fetch origin >/dev/null 2>&1
    echo "=== $(hostname):$(pwd)"
    echo "    branch:  $b"
    echo "    identity: $(git config user.name || echo '<unset>') <$(git config user.email || echo '<unset>')>"
    echo "    vs origin/$b: $(git rev-list --left-right --count "origin/$b...HEAD" 2>/dev/null | awk '{print $2" ahead, "$1" behind"}')"
    local dirty; dirty=$(git status --porcelain | wc -l | tr -d ' ')
    echo "    uncommitted: $dirty file(s)"

    [ "${1:-}" = "--all" ] || return 0
    echo
    # Only reachable from a machine with ssh access to both clusters (the Mac).
    for spec in "utcs-lysine:/data-local/honjar/ambient-omni" \
                "csail-slurm:/data/scratch/honjar/ambient-omni" \
                "csail-slurm:\$HOME/ambient-omni"; do
        local host=${spec%%:*} path=${spec#*:}
        echo "=== $host:$path"
        ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" "
            cd $path 2>/dev/null || { echo '    unreachable or missing'; exit 0; }
            git fetch origin >/dev/null 2>&1
            b=\$(git rev-parse --abbrev-ref HEAD)
            echo \"    branch:  \$b\"
            echo \"    vs origin/\$b: \$(git rev-list --left-right --count origin/\$b...HEAD 2>/dev/null | awk '{print \$2\" ahead, \"\$1\" behind\"}')\"
            echo \"    uncommitted: \$(git status --porcelain | wc -l | tr -d ' ') file(s)\"
        " 2>/dev/null || echo "    ssh failed"
    done
}

# --- private notes -------------------------------------------------------
# Research notes, handoff docs and notebooks are deliberately NOT in git: the
# GitHub repo is public.  They still need to reach every machine, so the Mac
# acts as the hub the way GitHub does for code.  Lysine and CSAIL cannot ssh to
# each other, so everything relays through here.
#
# Direction is always explicit.  An automatic two-way sync would eventually
# overwrite a note that only existed on one machine, which is the exact failure
# this is meant to prevent.  Nothing is ever deleted: rsync runs without
# --delete, so the union survives.

NOTES_LOCAL="$HOME/ambient-omni-private-notes"

notes_remote_path() {
    case "$1" in
        lysine|utcs-lysine) echo "utcs-lysine:/data-local/honjar/ambient-omni-private-notes" ;;
        csail|csail-slurm)  echo "csail-slurm:\$HOME/ambient-omni-private-notes" ;;
        *) return 1 ;;
    esac
}

cmd_notes() {
    local action="${1:-}" host="${2:-}"
    case "$action" in
        pull|push)
            local spec
            spec=$(notes_remote_path "$host") || {
                red "Unknown machine '$host'. Use: lysine | csail"
                return 1
            }
            mkdir -p "$NOTES_LOCAL"
            if [ "$action" = pull ]; then
                echo "Pulling notes from $host -> $NOTES_LOCAL"
                rsync -avh --no-perms --no-owner --no-group "$spec/" "$NOTES_LOCAL/" || return 1
            else
                echo "Pushing notes from $NOTES_LOCAL -> $host"
                rsync -avh --no-perms --no-owner --no-group "$NOTES_LOCAL/" "$spec/" || return 1
            fi
            grn "Done. Nothing was deleted; this is a union, not a mirror."
            ;;
        status)
            printf '%-14s %s\n' "mac" "$(find "$NOTES_LOCAL" -type f 2>/dev/null | wc -l | tr -d ' ') file(s)"
            for h in lysine csail; do
                local spec path hostname_
                spec=$(notes_remote_path "$h"); hostname_=${spec%%:*}; path=${spec#*:}
                printf '%-14s %s\n' "$h" "$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$hostname_" \
                    "find $path -type f 2>/dev/null | wc -l" 2>/dev/null || echo '?') file(s)"
            done
            ;;
        *)
            red "Usage: ./sync.sh notes {pull|push} {lysine|csail}   or   ./sync.sh notes status"
            return 1
            ;;
    esac
}

case "${1:-}" in
    pull)   cmd_pull ;;
    push)   shift; cmd_push "$@" ;;
    status) shift; cmd_status "$@" ;;
    notes)  shift; cmd_notes "$@" ;;
    *)      sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
