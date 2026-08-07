#!/usr/bin/env bash
# Run all four Movie local-qwen evaluations sequentially.
#
# MAINTAINER SCRIPT — see run_movie_*.sh for per-method details.
#
#   bash kickoff_movie_all.sh            # full runs (est ~35-45 h total)
#   bash kickoff_movie_all.sh --smoke    # all four smokes (est ~2-3 h total)
#
# Order: GPTree first (fails fast on the known misconfiguration), then PI,
# then RRM, then RRF (the ~18 h long pole) last. `set -e` stops the chain on
# the first failure. Each stage is individually restartable — after a kill,
# rerun this script (or just the failed run_movie_*.sh) and completed work
# replays from disk caches/checkpoints.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="${1:-}"

TOTAL_EST="35-45 h"
[[ "$SMOKE" == "--smoke" ]] && TOTAL_EST="2-3 h"
echo "##############################################################"
echo "# Movie x {GPTree, PI, RRM, RRF} on local qwen2.5-coder:14b"
echo "# Total estimated wall-clock: ~${TOTAL_EST}"
echo "# Started: $(date '+%F %H:%M')"
echo "##############################################################"
T0=$(date +%s)

bash "$HERE/run_movie_gptree.sh" $SMOKE
bash "$HERE/run_movie_pi.sh"     $SMOKE
bash "$HERE/run_movie_rrm.sh"    $SMOKE
bash "$HERE/run_movie_rrf.sh"    $SMOKE

ELAPSED_H=$(( ($(date +%s) - T0) / 3600 ))
echo "##############################################################"
echo "# ALL FOUR MOVIE RUNS COMPLETE in ~${ELAPSED_H} h  ($(date '+%F %H:%M'))"
echo "##############################################################"
