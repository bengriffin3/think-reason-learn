#!/usr/bin/env bash
# Launch the Movie RRF (question shortlist) run on local qwen2.5-coder:14b.
#
# MAINTAINER SCRIPT — runs against the PRIVATE repo on the reference machine.
# Override paths via TRL_PRIVATE_REPO / TRL_PY env vars.
#
#   bash run_movie_rrf.sh            # full run  (est 15-21 h — the long pole)
#   bash run_movie_rrf.sh --smoke    # smoke run (est ~20 min, 20 films)
#
# Restart-safe: answered (film, question) pairs are skipped on rerun, and
# replayed calls hit the per-call disk cache for free.
set -euo pipefail

PRIVATE_REPO="${TRL_PRIVATE_REPO:-/Users/Vela/Desktop/VELA/git_repos/think-reason-learn-private}"
PY="${TRL_PY:-/Users/Vela/.venvs/rrf313/bin/python}"
DIR="$PRIVATE_REPO/experiments/movie/local_qwen14b"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"

[[ -f "$PRIVATE_REPO/experiments/movie/records.jsonl" ]] || { echo "FATAL: records.jsonl missing"; exit 1; }
curl -sf "${OPENAI_BASE_URL%/v1}/api/tags" | grep -q "qwen2.5-coder:14b" \
  || { echo "FATAL: Ollama not serving qwen2.5-coder:14b at $OPENAI_BASE_URL"; exit 1; }

if [[ "${1:-}" == "--smoke" ]]; then
  ARGS=(--limit 20 --name smoke); EST_H=1
else
  ARGS=(--name qwen14b); EST_H=18
fi

echo "=============================================================="
echo " Movie / RRF questions / qwen2.5-coder:14b"
echo " ~30,800 (film x question) calls on the full run"
echo " Estimated wall-clock: ~${EST_H} h"
echo " Projected finish:     $(date -v +${EST_H}H '+%a %H:%M' 2>/dev/null || date)"
echo " Output:  $DIR/results/movie_rrf_qwen14b/  (+ movie_rrf_scores_*.jsonl)"
echo " Monitor: bash check_movie_progress.sh"
echo "=============================================================="

mkdir -p "$DIR/results"
caffeinate -i "$PY" "$DIR/run_rrf_local.py" "${ARGS[@]}" 2>&1 | tee -a "$DIR/results/run_movie_rrf.log"
