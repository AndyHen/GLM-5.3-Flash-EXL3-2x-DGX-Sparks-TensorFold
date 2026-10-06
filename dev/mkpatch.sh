#!/usr/bin/env bash
# dev/mkpatch.sh <name>: .work/tf/src's changes since the "baseline" tag -> patches/<name>.patch, in the -p0 form the
# image applies (paths tensorfold/..., new files included).
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
name="${1:?usage: dev/mkpatch.sh <NNNN-name>}"
cd "$root/.work/tf/src"
git add -N .
git diff --no-prefix baseline -- tensorfold > "$root/patches/$name.patch"
[[ -s "$root/patches/$name.patch" ]] || { echo "no changes since baseline" >&2; rm -f "$root/patches/$name.patch"; exit 1; }
echo "wrote patches/$name.patch ($(grep -c '^+++ ' "$root/patches/$name.patch") files)"
