# VCBench + Movie with a local LLM (qwen2.5-coder:14b via Ollama)

Run all four TRL reasoning-ML methods (PolicyInduction, RRF, GPTree, RRM) on two
real benchmarks with a **free, local** LLM — no paid API keys — then compare them
to a 5-model sklearn baseline and combine both families into an ensemble.

> **Status: skeleton.** Sections marked `TODO(Stage N)` are filled in by later
> stages of this workstream.

---

## What you'll build

1. Four reasoning-ML models scored on **VCBench** (founder-success prediction,
   our anchor dataset) and **Movie** (award-nomination prediction, temporal
   split — where reasoning methods beat traditional ML).
2. A traditional-ML baseline: LR, HistGradientBoosting, RandomForest-250,
   ExtraTrees-250, GaussianNB over `[text-length, MiniLM-L6-v2 embeddings,
   TF-IDF top-400]` features.
3. A combined ensemble: `rank01(traditional_ensemble) + rank01(reasoning_ensemble)`.
4. A time-vs-score chart showing what each extra hour of local compute buys you.

## Hero chart

TODO(Stage 7): `figures/time_vs_score.png` — wall-clock cost vs. ROC-AUC for
every method and ensemble, both datasets.

## Results

TODO(Stage 3/7): results table. Two columns per dataset, per the project's
benchmark-hygiene decisions:

| Method | VCBench public-CV (reproducible in notebooks) | VCBench private target | Movie test (public CV) |
|---|---|---|---|
| PolicyInduction | TODO | TODO | TODO |
| RRF | TODO | TODO | TODO |
| GPTree | TODO | TODO | TODO |
| RRM | TODO | TODO | TODO |
| Reasoning ensemble | TODO | TODO | TODO |
| Traditional best / ensemble | TODO | TODO | TODO |
| **Combined** | TODO | TODO | TODO |

- *Public-CV* numbers are what the notebooks reproduce end-to-end.
- *Private target* (VCBench only) is the held-out test the maintainers score
  against — shown as the number to beat; the labels are never shipped.
- Panel numbers are colleague-grade, not paper-grade: this example teaches the
  methods, it is not a benchmarking paper.

## Prerequisites

- **Ollama** installed and running (`https://ollama.com`).
- The pinned model pulled: `ollama pull qwen2.5-coder:14b`
  (Q4_K_M quantization; captured 2026-08-07 — later re-pulls may differ slightly).
- Python 3.13 with the library + example extras:

  ```bash
  pip install -e ".[examples]"
  ```

## Getting the datasets

Raw data is **not** committed to this repo; fetch it yourself:

- **VCBench** — not publicly downloadable. Request it at
  [vcbench.com](https://vcbench.com) ("Request data" in the sidebar); access is
  granted on request. Ask for the public split: 4,500 founders, ~9% base rate,
  columns `founder_uuid`, `success`, `anonymised_prose`. Save the CSV wherever
  you like and point the example at it:

  ```bash
  export VCBENCH_DATA=~/.trl-data/vcbench/vcbench_final_public.csv
  ```

  The quickstart notebook (`01_quickstart.ipynb`) needs **none** of this — it
  reads the score CSVs in `precomputed/` and nothing else. You only need the raw
  data for the live-run notebooks (02 onward) and the `scripts/run_*.py` runs.
- **Movie** — public HuggingFace dataset, fetched automatically:

  ```python
  from datasets import load_dataset
  ds = load_dataset("Francis2003/Movie-O-Label")
  ```

  Temporal split: train = films before 2013, test = 2013 onward (~67/33,
  base rate ~19%).

## Runtime expectations (read before running anything)

All timings measured on an M2 Mac with Ollama serving `qwen2.5-coder:14b`
serially — your hardware will vary:

- **Movie, all four methods: ~35 h wall-clock total.**
- **VCBench, all four methods: ~70 h wall-clock total.**
- The quickstart notebook (`01_quickstart.ipynb`) uses **precomputed scores**
  and runs in ~5 minutes — start there.
- Every long run is restartable: per-call responses are cached to disk
  (GPTree checkpoints per tree node instead), so a killed run resumes where
  it left off.
- TODO(Stage 6): `scripts/calibrate_timings.py` measures throughput on your
  machine and writes `precomputed/timings.json`.

## Directory tour

| Path | What it is |
|---|---|
| `notebooks/01_quickstart.ipynb` | ~5 min tour using precomputed scores (TODO Stage 4) |
| `notebooks/02_reasoning_methods.ipynb` | PI / RRF / GPTree / RRM walkthrough (TODO Stage 5) |
| `notebooks/03_traditional_baselines.ipynb` | 5-model sklearn suite (TODO Stage 5) |
| `notebooks/04_ensembling.ipynb` | rank-average + combined ensemble (TODO Stage 5) |
| `scripts/run_{pi,rrf,gptree,rrm}.py` | full local runs per method (TODO Stage 6) |
| `scripts/run_traditional.py` | traditional baseline runner (TODO Stage 6) |
| `scripts/run_all.sh` | everything, sequentially (TODO Stage 6) |
| `scripts/calibrate_timings.py` | n=100 throughput calibration → `timings.json` (TODO Stage 6) |
| `scripts/run_movie_*.sh`, `kickoff_movie_all.sh`, `check_movie_progress.sh` | maintainer scripts that produced the shipped Movie artifacts (reference machine only) |
| `src/logprobs_llm_shim.py` | drop-in TRL LLM adapter: chat-completions + logprobs + per-call disk cache, works with Ollama via `OPENAI_BASE_URL` |
| `precomputed/` | per-sample score CSVs from the reference runs (land in Stage 3) |
| `models/` | trained model bundles (land in Stage 3) |
| `figures/` | time-vs-score chart (lands in Stage 7) |

## Why the shim?

TRL's stock OpenAI provider targets the `/v1/responses` endpoint, which
Ollama's OpenAI-compat layer doesn't serve, and it doesn't return token
logprobs (RRM needs them). `src/logprobs_llm_shim.py` is a small drop-in
`LLM` replacement using `chat.completions` with `logprobs=True` and a
JSONL per-call disk cache. Point it at Ollama with:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

## Cloud reference (optional)

The same methods run against cloud LLMs (e.g. Gemini 2.5 Flash) with different
trade-offs — faster wall-clock, per-token cost, API keys required. This example
is local-first; cloud numbers appear in the results table as reference only.
TODO(Stage 5): short appendix.

## Model pinning

`qwen2.5-coder:14b` (Ollama tag, Q4_K_M quantization, captured 2026-08-07).
We pin tag + quantization + capture date rather than a digest — good enough
for a teaching example; expect small score drift if the tag is re-published.
