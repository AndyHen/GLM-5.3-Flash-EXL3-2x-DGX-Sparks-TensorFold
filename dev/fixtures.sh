#!/usr/bin/env bash
# Small files of the two drafters at their pinned revisions, for the CPU tests (.work/fixtures/{G,incoai}).
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
get() {  # get <repo> <revision> <dir> <file>...
  local repo="$1" rev="$2" dir="$root/.work/fixtures/$3"; shift 3
  mkdir -p "$dir"
  for f in "$@"; do
    [[ -s "$dir/$f" ]] || curl -fsSL "https://huggingface.co/$repo/resolve/$rev/$f" -o "$dir/$f"
  done
}
get canada-quant/GLM-5.3-Flash-DFlash2-G b8ba68b022d612958d1a9c447f294f148a156ed1 G config.json mask_embedding.pt
get incoai/GLM-5.3-Flash-DFlash2 bf582e4eacc1810f76656d1811693ff6c6737d2a incoai config.json
echo "fixtures in $root/.work/fixtures"
