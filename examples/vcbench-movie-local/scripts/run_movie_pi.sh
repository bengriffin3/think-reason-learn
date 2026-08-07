#!/usr/bin/env bash
# Launch the Movie PolicyInduction run on local qwen2.5-coder:14b (Ollama).
#
# MAINTAINER SCRIPT — runs against the PRIVATE repo on the reference machine
# to produce the precomputed artifacts this example ships. Students never
# need it. Override paths via TRL_PRIVATE_REPO / TRL_PY env vars.
#
#   bash run_movie_pi.sh            # full run  (est 6-9 h)
#   bash run_movie_pi.sh --smoke    # smoke run (est ~45 min, separate cache)
#
# Restart-safe: per-call disk cache — rerun the same command after a kill.
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
  ARGS=(--fit-size 60 --predict-limit 30 --n-policies 5 --name smoke); EST_H=1
else
  ARGS=(--name qwen14b); EST_H=8
fi

echo "=============================================================="
echo " Movie / PolicyInduction / qwen2.5-coder:14b"
echo " Estimated wall-clock: ~${EST_H} h"
echo " Projected finish:     $(date -v +${EST_H}H '+%a %H:%M' 2>/dev/null || date)"
echo " Output:  $DIR/results/movie_pi_qwen14b/  (+ movie_pi_scores_*.jsonl)"
echo " Monitor: bash check_movie_progress.sh  (from another terminal)"
echo "=============================================================="

mkdir -p "$DIR/results"
caffeinate -i "$PY" "$DIR/run_pi_local.py" "${ARGS[@]}" 2>&1 | tee -a "$DIR/results/run_movie_pi.log"
