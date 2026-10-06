#!/usr/bin/env bash
# Phase 0 of DFlash2-G on this recipe's Sparks: incoai, then G at several windows (PARALLEL=4), then G with full
# attention (PARALLEL=1, 128k window); a report with the go/no-go gate. Run on the head from the repo root with nothing
# else on either Spark's GPU. Each run restarts the server (2-6 min, longer the first time: image build, downloads,
# kernel compiles) and sends ~10 min of requests. RUNS="incoai g-w2048" limits the runs; OUT sets the results folder.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out="${OUT:-$HOME/phase0-results/$(date +%Y%m%d-%H%M)}"
mkdir -p "$out"
want() { [[ -z "${RUNS:-}" || " $RUNS " == *" $1 "* ]]; }
run() {  # run <label> <bench args> -- <VAR=value>...
  local label="$1"; shift
  local args=(); while [[ "$1" != -- ]]; do args+=("$1"); shift; done; shift
  want "$label" || return 0
  echo "== $label: $*  (log: $out/$label.start.log)"
  if ! env "$@" ./start.sh restart > "$out/$label.start.log" 2>&1; then
    echo "   start failed, skipped: tail of the log:"; tail -n 15 "$out/$label.start.log"; return 0
  fi
  docker logs glm53-flash-tf 2>&1 | grep -h "Loaded DFlash mask embedding\|embed_tokens" || true
  tools/phase0_bench.py "$label" --out "$out" "${args[@]}"
}
run incoai   --concurrency 1,4             -- DRAFTER=dflash2  PARALLEL=4
run g-w2048  --concurrency 1,4 --identity  -- DRAFTER=dflash2g PARALLEL=4 DFLASH_WINDOW=2048
run g-w4096  --concurrency 1,4             -- DRAFTER=dflash2g PARALLEL=4 DFLASH_WINDOW=4096
run g-w8192  --concurrency 1,4             -- DRAFTER=dflash2g PARALLEL=4 DFLASH_WINDOW=8192
run g-full   --concurrency 1 --identity    -- DRAFTER=dflash2g PARALLEL=1 DFLASH_WINDOW=0 CONTEXT=131072
./stop.sh >/dev/null 2>&1 || true
tools/phase0_bench.py --report "$out" || true
echo "results: $out/report.md"
