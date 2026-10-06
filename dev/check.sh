#!/usr/bin/env bash
# Everything that can be checked without a Spark: LF endings, every patch onto a fresh TensorFold tree, the patched
# sources compile, the CPU tests and the config defaults.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
if grep -lU $'\r' patches/*.patch start.sh stop.sh start-tp3.sh scripts/*.sh dev/*.sh tools/*.sh 2>/dev/null; then
  echo "CR line endings in the files above" >&2; exit 1
fi
py="$root/.venv/Scripts/python.exe"; [[ -x "$py" ]] || py="$root/.venv/bin/python"
src=$(dev/tree.sh | tail -1)
"$py" -m compileall -q "$src/tensorfold" >/dev/null
dev/fixtures.sh >/dev/null
"$py" -m pytest -q tests/dflash2g
[[ ! -x dev/test_config.sh ]] || dev/test_config.sh
echo "all checks passed"
