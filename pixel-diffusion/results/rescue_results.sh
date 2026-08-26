#!/bin/bash
# Pull every dynamic-T result off every reachable machine into one place on the
# Mac, and write a consolidated JSON. Explicit and verifiable, unlike the
# monitor's silent scp side-effect (which failed for a whole day unnoticed
# because its errors went to /dev/null).
#
# Safe to re-run: it only adds. Run it whenever a machine is reachable.
SP="$(cd "$(dirname "$0")" && pwd)"
OUT="$SP/rescued_results"
mkdir -p "$OUT"
ok=0; fail=0
for spec in "gdaras@proline.cs.utexas.edu:/var/local/honjar" \
            "csail-slurm:/data/scratch/honjar" \
            "gdaras@lysine.cs.utexas.edu:/data-local/honjar"; do
  host=${spec%%:*}; base=${spec##*:}
  tag=$(echo "$host" | sed 's/.*@//; s/\..*//')
  if ! ssh -o ConnectTimeout=20 -o BatchMode=yes "$host" true 2>/dev/null; then
    echo "SKIP $tag (unreachable)"; fail=$((fail+1)); continue
  fi
  # One tar stream, not one scp per file: 30+ round trips over a ProxyJump is
  # slow and any single failure used to vanish into /dev/null.
  n=$(ssh -o ConnectTimeout=25 "$host" "cd $base/generated 2>/dev/null && ls mind_dyn_*.json fid_dyn_*.json 2>/dev/null | wc -l" 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then
    mkdir -p "$OUT/$tag"
    ssh -o ConnectTimeout=25 "$host" "cd $base/generated && tar cf - mind_dyn_*.json fid_dyn_*.json 2>/dev/null" 2>/dev/null \
      | tar xf - -C "$OUT/$tag" 2>/dev/null
    got=$(ls "$OUT/$tag"/mind_dyn_*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "OK   $tag: $got MIND files"
    ok=$((ok+1))
  else
    echo "OK   $tag: 0 files"
  fi
done
echo "--- machines rescued: $ok, unreachable: $fail ---"
ls "$OUT"/*/mind_dyn_*.json 2>/dev/null | sed 's#.*/mind_dyn_##; s/\.json$//' | sort -u | wc -l | xargs echo "distinct run+seed results held on the Mac:"
