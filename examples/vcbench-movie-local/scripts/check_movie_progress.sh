#!/usr/bin/env bash
# Report progress of the Movie local-qwen runs WITHOUT stopping them.
# Safe to run any time from a second terminal.
#
#   bash check_movie_progress.sh
#
# Reads cache/output line counts and log tails only; touches nothing.
set -uo pipefail

PRIVATE_REPO="${TRL_PRIVATE_REPO:-/Users/Vela/Desktop/VELA/git_repos/think-reason-learn-private}"
R="$PRIVATE_REPO/experiments/movie/local_qwen14b/results"

lines() { [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || echo 0; }
mtime() { [[ -f "$1" ]] && stat -f '%Sm' -t '%H:%M:%S' "$1" || echo "-"; }

pct() {  # pct <count> <target>
  if [[ "$2" -gt 0 ]]; then echo "$(( 100 * $1 / $2 ))%"; else echo "?"; fi
}

echo "Movie local-qwen progress   ($(date '+%F %H:%M:%S'))"
echo "results dir: $R"
echo

# --- PI: ~22,000 cached calls on the full run (fit ~14.7k + predict ~7.3k) ---
PI_CACHE="$R/movie_pi_qwen14b/llm_cache.jsonl"
n=$(lines "$PI_CACHE")
echo "PI      cache calls: $n / ~22000 ($(pct "$n" 22000))   last write: $(mtime "$PI_CACHE")"

# --- RRF: 30,786 (film x question) pairs on the full run ---
RRF_RAW="$R/movie_rrf_qwen14b/raw_responses.jsonl"
n=$(lines "$RRF_RAW")
echo "RRF     pairs done:  $n / 30786 ($(pct "$n" 30786))   last write: $(mtime "$RRF_RAW")"

# --- GPTree: fit nodes from log; predict checkpoint counts films done -------
GPT_LOG="$R/run_movie_gptree.log"
GPT_CKPT="$R/movie_gptree_qwen14b/predict_checkpoint.json"
nodes=0
[[ -f "$GPT_LOG" ]] && nodes=$(grep -c 'Node [0-9]* (id=' "$GPT_LOG" || true)
films=0
if [[ -f "$GPT_CKPT" ]]; then
  films=$(python3 -c "import json;d=json.load(open('$GPT_CKPT'));print(len(d['sample_to_leaf'])+len(d.get('failed',[])))" 2>/dev/null || echo "?")
fi
echo "GPTree  fit nodes logged: $nodes   predict films: $films / 2199   ckpt: $(mtime "$GPT_CKPT")"

# --- RRM: cache ~8-9k calls; scores jsonl rows = films predicted ------------
RRM_CACHE="$R/movie_rrm_qwen14b/llm_cache.jsonl"
RRM_OUT="$R/movie_rrm_scores_qwen14b.jsonl"
n=$(lines "$RRM_CACHE"); m=$(lines "$RRM_OUT")
echo "RRM     cache calls: $n / ~8500 ($(pct "$n" 8500))   films written: $m / 2199   last: $(mtime "$RRM_OUT")"

echo
echo "Targets are approximate (fit-time call counts vary a little)."
echo "A 'last write' more than ~10 min old on the active run usually means a stall —"
echo "check the log tails below and Ollama (curl -s localhost:11434/api/tags)."
echo
for log in run_movie_gptree run_movie_pi run_movie_rrm run_movie_rrf; do
  f="$R/$log.log"
  if [[ -f "$f" ]]; then
    echo "--- tail $log.log ---"
    tail -n 3 "$f"
    echo
  fi
done

# Completed-run metrics, if any exist yet.
for mj in "$R"/movie_pi_qwen14b/movie_pi_metrics.json \
          "$R"/movie_rrf_qwen14b/movie_rrf_metrics.json \
          "$R"/movie_gptree_qwen14b/movie_gptree_metrics.json \
          "$R"/movie_rrm_qwen14b/movie_rrm_metrics.json; do
  [[ -f "$mj" ]] && { echo "--- $(basename "$(dirname "$mj")") metrics ---"; cat "$mj"; echo; }
done
