#!/usr/bin/env bash
# Launch the Movie RRM run on local qwen2.5-coder:14b (Ollama).
#
# MAINTAINER SCRIPT — runs against the PRIVATE repo on the reference machine.
# Override paths via TRL_PRIVATE_REPO / TRL_PY env vars.
#
#   bash run_movie_rrm.sh            # full run  (est 9-12 h)
#   bash run_movie_rrm.sh --smoke    # smoke run (est ~30 min)
#
# Restart-safe: per-call disk cache + rows already in the output JSONL are
# skipped — rerun the same command after a kill. Runs with --no-memory
# (rolling Stage-1 memory disabled) to stay inside qwen's 32k context.
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
  ARGS=(--fit-size 30 --predict-limit 15 --no-memory --name smoke); EST_H=1
else
  ARGS=(--no-memory --name qwen14b); EST_H=10
fi

echo "=============================================================="
echo " Movie / RRM / qwen2.5-coder:14b  (fit 346 train, predict all)"
echo " Estimated wall-clock: ~${EST_H} h"
echo " Projected finish:     $(date -v +${EST_H}H '+%a %H:%M' 2>/dev/null || date)"
echo " Output:  $DIR/results/movie_rrm_qwen14b/  (+ movie_rrm_scores_*.jsonl)"
echo " Monitor: bash check_movie_progress.sh"
echo "=============================================================="

mkdir -p "$DIR/results"
caffeinate -i "$PY" "$DIR/run_rrm_local.py" "${ARGS[@]}" 2>&1 | tee -a "$DIR/results/run_movie_rrm.log"
