#!/usr/bin/env bash
# Launch the Movie GPTree run (fit + predict) on local qwen2.5-coder:14b.
#
# MAINTAINER SCRIPT — runs against the PRIVATE repo on the reference machine.
# Override paths via TRL_PRIVATE_REPO / TRL_PY env vars.
#
#   bash run_movie_gptree.sh            # full run  (est 5-8 h)
#   bash run_movie_gptree.sh --smoke    # smoke run (est ~40 min)
#
# Restart-safe: fit auto-checkpoints per tree node, predict checkpoints every
# 100 films — rerun the same command to resume.
#
# EARLY-FAILURE CHECK (the misconfiguration the first VCBench attempt hit):
# if the fit log shows "Node 1 (... children=0)" and training ends within
# minutes with 1 node, the qwen CODE-question patch didn't take. The runner
# exits with code 2 in that case — do not proceed; investigate.
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
  ARGS=(--sample-size 40 --predict-limit 20 --name smoke); EST_H=1
else
  ARGS=(--name qwen14b); EST_H=7
fi

echo "=============================================================="
echo " Movie / GPTree / qwen2.5-coder:14b  (fit 350 train, predict all)"
echo " Estimated wall-clock: ~${EST_H} h"
echo " Projected finish:     $(date -v +${EST_H}H '+%a %H:%M' 2>/dev/null || date)"
echo " Output:  $DIR/results/movie_gptree_qwen14b/  (+ movie_gptree_scores_*.jsonl)"
echo " Monitor: bash check_movie_progress.sh"
echo "=============================================================="

mkdir -p "$DIR/results"
caffeinate -i "$PY" "$DIR/run_gptree_local.py" "${ARGS[@]}" 2>&1 | tee -a "$DIR/results/run_movie_gptree.log"
