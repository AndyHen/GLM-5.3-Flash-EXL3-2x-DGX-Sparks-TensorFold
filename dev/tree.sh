#!/usr/bin/env bash
# .work/tf/src: TensorFold's source at TF_VERSION with patches/*.patch applied the way the image's Dockerfile applies
# them (patch -p0 --forward next to the tensorfold package), one commit per patch.
#   dev/tree.sh        every patch; the last one is tagged "baseline"
#   dev/tree.sh 0084   patches numbered below 0084 (tagged "baseline"), then patches/0084-*.patch applied uncommitted,
#                      to edit and regenerate with dev/mkpatch.sh
# The last line printed is the source directory.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/scripts/config.sh" >/dev/null
upto="${1:-}"
work="$root/.work"
mkdir -p "$work"
[[ -d "$work/tensorfold.git" ]] || git clone -q --bare "$TF_REPO" "$work/tensorfold.git"
rm -rf "$work/tf" && mkdir -p "$work/tf"
git --git-dir="$work/tensorfold.git" archive "$TF_VERSION" src | tar -x -C "$work/tf"
cd "$work/tf/src"
git init -q && git config core.autocrlf false && git config user.email dev@local && git config user.name dev
printf '__pycache__/\n*.orig\n*.rej\n' > .git/info/exclude
git add -A && git commit -qm "TensorFold $TF_VERSION"
for p in "$root"/patches/*.patch; do
  n=$(basename "$p"); num=${n%%-*}
  [[ -z "$upto" || "$num" < "$upto" ]] || continue
  patch -p0 --forward -s < "$p" || { echo "patch failed: $n" >&2; exit 1; }
  git add -A && git commit -qm "$n"
done
git tag baseline
if [[ -n "$upto" ]]; then
  for p in "$root"/patches/"$upto"-*.patch; do
    [[ -e "$p" ]] || continue
    patch -p0 --forward -s < "$p" || { echo "patch failed: $(basename "$p")" >&2; exit 1; }
  done
  git add -N .
fi
pwd
