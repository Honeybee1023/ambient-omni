#!/usr/bin/env python3
"""Replace hardcoded per-machine paths with the AMBIENT_BASE indirection.

Both machines lay their tree out identically, so /data/honjar (lysine, from
before the filesystem rename) and /data/scratch/honjar (CSAIL) both collapse to
a single $AMBIENT_BASE.  Run with --apply to write; default is a dry run.
"""
import argparse
import os
import re
import sys

PREFIXES = ("/data/scratch/honjar", "/data-local/honjar", "/data/honjar")
PREFIX_RE = re.compile("|".join(re.escape(p) for p in PREFIXES))

SH_RESOLVER = (
    'AMBIENT_BASE="${AMBIENT_BASE:-$([ -d /data-local/honjar ] '
    '&& echo /data-local/honjar || echo /data/scratch/honjar)}"'
)
PY_RESOLVER = (
    'AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (\n'
    '    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"\n'
    ')'
)

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "venv", "archive"}
# These files *define* the indirection, so rewriting them replaces the literal
# fallbacks with a reference to the very variable they are meant to produce.
SKIP_FILES = {"env.sh", "env.local.sh", "ambient_paths.py", "rewrite_paths.py"}
# Marker both injected resolvers carry.  The resolvers necessarily contain the
# literal fallback paths, so without this guard a second run would rewrite the
# resolver into a reference to itself and blow away the fallbacks.
MIGRATED_MARKER = "Per-machine paths: see env.sh"
SKIP_EXT = {".pyc", ".png", ".jpg", ".jpeg", ".pdf", ".pt", ".pkl", ".npz", ".zip", ".gz", ".ipynb"}

# Quoted single-line string literal that mentions one of the prefixes.
PY_STR_RE = re.compile(
    r"""(?P<pre>[fFrRbB]{0,2})(?P<q>['"])(?P<body>(?:[^'"\\\n]|\\.)*?)(?P=q)"""
)


def rewrite_python(text):
    """Return (new_text, n_subs, warnings)."""
    warnings = []

    def repl(m):
        body = m.group("body")
        if not PREFIX_RE.search(body):
            return m.group(0)
        pre, q = m.group("pre"), m.group("q")
        if "r" in pre.lower():
            warnings.append(f"raw string left alone: {m.group(0)[:70]}")
            return m.group(0)
        if "b" in pre.lower():
            warnings.append(f"bytes literal left alone: {m.group(0)[:70]}")
            return m.group(0)
        is_f = "f" in pre.lower()
        new_body = PREFIX_RE.sub("{AMBIENT_BASE}", body)
        if is_f:
            return f"{pre}{q}{new_body}{q}"
        if "{" in body or "}" in body:
            # Plain string using .format()/braces -- making it an f-string would
            # break those, so concatenate instead.
            rest = PREFIX_RE.sub("", body)
            return f"AMBIENT_BASE + {q}{rest}{q}"
        return f"f{pre}{q}{new_body}{q}"

    new_text, n = PY_STR_RE.subn(repl, text)
    n_subs = len(PREFIX_RE.findall(text)) - len(PREFIX_RE.findall(new_text))
    leftover = PREFIX_RE.findall(new_text)
    if leftover:
        warnings.append(f"{len(leftover)} occurrence(s) outside string literals")
    if n_subs and "AMBIENT_BASE = _os.environ" not in new_text:
        new_text = inject_python_resolver(new_text)
    return new_text, n_subs, warnings


def inject_python_resolver(text):
    """Insert the resolver after the module docstring and any __future__ imports."""
    lines = text.split("\n")
    idx = 0
    # Skip shebang and encoding lines.
    while idx < len(lines) and (lines[idx].startswith("#!") or "coding" in lines[idx][:30]):
        idx += 1
    # Skip a module docstring if present.
    j = idx
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j < len(lines):
        stripped = lines[j].lstrip()
        for quote in ('"""', "'''"):
            if stripped.startswith(quote):
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    idx = j + 1
                else:
                    k = j + 1
                    while k < len(lines) and quote not in lines[k]:
                        k += 1
                    idx = k + 1
                break
    block = [
        "",
        "# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather",
        "# than imported from ambient_paths because these scripts run from varying",
        "# depths and cwds, where an import would need sys.path surgery.",
        "import os as _os",
        PY_RESOLVER,
        "",
    ]
    return "\n".join(lines[:idx] + block + lines[idx:])


def rewrite_shell(text):
    warnings = []
    n_before = len(PREFIX_RE.findall(text))
    if n_before == 0:
        return text, 0, warnings
    # A quoted heredoc delimiter blocks variable expansion, so flag those.
    for m in re.finditer(r"<<-?\s*(['\"])(\w+)\1", text):
        warnings.append(f"quoted heredoc <<{m.group(2)} blocks $AMBIENT_BASE expansion")
    new_text = PREFIX_RE.sub("${AMBIENT_BASE}", text)
    if "AMBIENT_BASE:-" not in new_text:
        new_text = inject_shell_resolver(new_text)
    return new_text, n_before, warnings


def inject_shell_resolver(text):
    """Insert the resolver after the shebang plus any leading comments/set lines."""
    lines = text.split("\n")
    idx = 0
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    # Keep the resolver below the header comment block and any `set -...` lines,
    # so the file still reads top-down.
    last_set = idx
    for i in range(idx, min(len(lines), 80)):
        s = lines[i].strip()
        if s.startswith("set -"):
            last_set = i + 1
        elif s and not s.startswith("#"):
            break
    idx = max(idx, last_set)
    block = [
        "",
        "# Per-machine paths: see env.sh / SYNC.md at the repo root.",
        SH_RESOLVER,
        "",
    ]
    return "\n".join(lines[:idx] + block + lines[idx:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total_files = total_subs = 0
    all_warnings = []
    skipped = []

    for dirpath, dirnames, filenames in os.walk(args.root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn in SKIP_FILES:
                continue
            ext = os.path.splitext(fn)[1]
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, args.root)
            if ext in SKIP_EXT:
                try:
                    with open(path, "rb") as fh:
                        if PREFIX_RE.search(fh.read().decode("utf-8", "ignore")):
                            skipped.append(rel)
                except OSError:
                    pass
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, OSError):
                continue
            if not PREFIX_RE.search(text):
                continue
            if MIGRATED_MARKER in text:
                continue
            if ext == ".py":
                new_text, n, warns = rewrite_python(text)
            elif ext in (".sh", "") and (ext == ".sh" or text.startswith("#!")):
                new_text, n, warns = rewrite_shell(text)
            else:
                skipped.append(rel)
                continue
            if n == 0:
                continue
            total_files += 1
            total_subs += n
            for w in warns:
                all_warnings.append(f"{rel}: {w}")
            print(f"{'WRITE' if args.apply else 'would'} {rel}: {n} path(s)")
            if args.apply:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new_text)

    print(f"\n{total_files} files, {total_subs} path occurrences")
    if skipped:
        print(f"\nNOT rewritten ({len(skipped)}) -- non-code or binary, review by hand:")
        for s in skipped:
            print("   ", s)
    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print("   ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
