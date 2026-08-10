#!/usr/bin/env bash
# Run every method (4 reasoning + traditional baseline) on a dataset,
# sequentially, with a total-time banner. Wraps run_{pi,rrf,gptree,rrm}.py
# and run_traditional.py.
#
#   ./run_all.sh --dataset movie              # full run, hours — see the banner
#   ./run_all.sh --dataset movie --smoke      # ~30 min end-to-end check
#   ./run_all.sh --dataset vcbench            # needs $VCBENCH_DATA
#
# Every sub-run resumes from its own cache, so killing this script and
# restarting the identical command picks up where it stopped. `caffeinate -i`
# keeps the machine awake without keeping the display on.
#
# Options:
#   --dataset {movie,vcbench}   which dataset (default movie)
#   --smoke                     tiny slice per method, for validating the setup
#   --out-dir DIR               parent for the per-method output dirs
#   --model NAME                Ollama model (default qwen2.5-coder:14b)
#   --python PATH               interpreter to use
#   --skip-calibrate            don't re-measure timings first
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_ROOT="$(dirname "$HERE")"

DATASET=movie
MODEL="qwen2.5-coder:14b"
PYTHON="${PYTHON:-python3}"
OUT_ROOT=""
SMOKE=""
SKIP_CALIBRATE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --out-dir) OUT_ROOT="$2"; shift 2 ;;
    --smoke) SMOKE="--smoke"; shift ;;
    --skip-calibrate) SKIP_CALIBRATE=1; shift ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$DATASET" in
  movie|vcbench) ;;
  *) echo "--dataset must be movie or vcbench (got '$DATASET')" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Preflight: Ollama has to be up and serving the model, or nothing below works.
# ---------------------------------------------------------------------------
if ! "$PYTHON" - "$MODEL" <<'PY'
import json, sys, urllib.error, urllib.request

wanted = sys.argv[1]
url = "http://localhost:11434/api/tags"
try:
    tags = json.load(urllib.request.urlopen(url, timeout=5))
except (urllib.error.URLError, OSError) as exc:
    sys.exit(f"Ollama is not answering on {url} ({exc}).\n"
             f"Start it with `ollama serve`, then `ollama pull {wanted}`.")
names = [m["name"] for m in tags.get("models", [])]
if wanted not in names:
    sys.exit(f"Ollama is up but does not have {wanted}. Pull it with "
             f"`ollama pull {wanted}`.\nModels present: {', '.join(names) or '(none)'}")
print(f"Ollama is serving {wanted}.")
PY
then
  exit 1
fi

if [[ "$DATASET" == "vcbench" && -z "${VCBENCH_DATA:-}" ]]; then
  echo "VCBench is not publicly downloadable. Request access at https://vcbench.com," >&2
  echo "then point \$VCBENCH_DATA at the CSV. (The Movie path needs no data setup.)" >&2
  exit 1
fi

TIMINGS="$EXAMPLE_ROOT/precomputed/timings.json"
COMMON=(--dataset "$DATASET" --model "$MODEL")
[[ -n "$SMOKE" ]] && COMMON+=("$SMOKE")

cd "$HERE"

# ---------------------------------------------------------------------------
# Calibrate first, so the estimate below comes from this machine.
# ---------------------------------------------------------------------------
if [[ -z "$SKIP_CALIBRATE" && -z "$SMOKE" ]]; then
  echo
  echo "=== calibrating timings on this machine (four short runs) ==="
  caffeinate -i "$PYTHON" calibrate_timings.py --dataset "$DATASET" --model "$MODEL"
fi

if [[ -f "$TIMINGS" ]]; then
  "$PYTHON" - "$TIMINGS" "$DATASET" <<'PY'
import json, sys

timings = json.loads(open(sys.argv[1]).read())
if timings.get("dataset") != sys.argv[2]:
    print(f"(timings.json was measured on {timings.get('dataset')}, not {sys.argv[2]} — "
          f"treat the estimate below as a rough guide)")
def full(entry):
    return entry.get("projected_full_run_s") or entry.get("extrapolated_full_run_s") or 0

total = sum(full(v) for v in timings["methods"].values())
hours = total / 3600
print(f"\nEstimated total for the four reasoning methods on {timings.get('machine')}: "
      f"~{hours:.1f}h")
for name, entry in timings["methods"].items():
    seconds = full(entry)
    if seconds:
        print(f"  {name:8s} ~{seconds / 3600:5.1f}h")
print("Plus run_traditional.py, which is under a minute — it makes no LLM calls.\n")
PY
fi

# ---------------------------------------------------------------------------
# The four reasoning methods, then the traditional baseline.
# ---------------------------------------------------------------------------
STARTED=$(date +%s)
for METHOD in pi rrf gptree rrm; do
  ARGS=("${COMMON[@]}")
  [[ -n "$OUT_ROOT" ]] && ARGS+=(--out-dir "$OUT_ROOT/${METHOD}_${DATASET}")
  echo
  echo "==================== $METHOD · $DATASET ===================="
  caffeinate -i "$PYTHON" "run_${METHOD}.py" "${ARGS[@]}"
done

TRAD_OUT="${OUT_ROOT:-$EXAMPLE_ROOT/results}/traditional_${DATASET}"
echo
echo "==================== traditional · $DATASET ===================="
if [[ "$DATASET" == "movie" ]]; then
  # run_traditional.py takes a records file rather than fetching from
  # HuggingFace, so build one from the same rows the reasoning runs used.
  RECORDS="$TRAD_OUT/records.jsonl"
  mkdir -p "$TRAD_OUT"
  "$PYTHON" - "$RECORDS" <<'PY'
import sys
import _runner_common as rc
import argparse

spec = rc.DATASETS["movie"]
args = argparse.Namespace(dataset="movie", data=None, text_field=None, label_field=None,
                          id_field=None, split_field=None, train_value="train",
                          test_value="test", max_chars=None)
rc.load_frame(args, spec).rename(columns={"text": "summary"}).to_json(
    sys.argv[1], orient="records", lines=True)
print(f"wrote {sys.argv[1]}")
PY
  "$PYTHON" run_traditional.py --input "$RECORDS" --split-field split \
    --text-fields summary --label-col label --id-col id --out-dir "$TRAD_OUT"
else
  "$PYTHON" run_traditional.py --train "$VCBENCH_DATA" --kfold 3 \
    --text-fields anonymised_prose --label-col success --id-col founder_uuid \
    --out-dir "$TRAD_OUT"
fi

echo
echo "All done in $(( ($(date +%s) - STARTED) / 60 )) minutes."
echo "Per-method outputs are under ${OUT_ROOT:-$EXAMPLE_ROOT/results}/."
echo "Compare them with the shipped reference scores in $EXAMPLE_ROOT/precomputed/."
