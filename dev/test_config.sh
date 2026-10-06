#!/usr/bin/env bash
# scripts/config.sh's defaults for each drafter (no local.sh / .env influence: a clean environment).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
check() {
  local want="$1"; shift
  local got
  got=$(env -i PATH="$PATH" HOME="$HOME" "$@" bash -c 'source scripts/config.sh >/dev/null
    echo "$DRAFTER $DFLASH2_ID ${DFLASH2_REVISION:0:8} $PARALLEL ${TF_GLM_DFLASH_WINDOW:-unset} $(is_dflash && echo d || echo m)"')
  [[ "$got" == "$want" ]] || { echo "FAIL ($*): got '$got', want '$want'" >&2; exit 1; }
}
check "dflash2g canada-quant/GLM-5.3-Flash-DFlash2-G b8ba68b0 4 2048 d"
check "dflash2 incoai/GLM-5.3-Flash-DFlash2 bf582e4e 4 2048 d" DRAFTER=dflash2
check "mtp canada-quant/GLM-5.3-Flash-DFlash2-G b8ba68b0 1 2048 m" DRAFTER=mtp
check "dflash2g canada-quant/GLM-5.3-Flash-DFlash2-G b8ba68b0 1 0 d" DFLASH_WINDOW=0 PARALLEL=1
check "dflash2g canada-quant/GLM-5.3-Flash-DFlash2-G b8ba68b0 8 2048 d" TP=3
echo "config checks passed"
