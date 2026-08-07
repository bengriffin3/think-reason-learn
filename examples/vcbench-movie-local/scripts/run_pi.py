"""Run PolicyInduction end-to-end on a dataset with the local Ollama model.

Fits PI on the training split (policy induction + LR weight training),
predicts the held-out split, and writes per-sample score CSVs plus a
wall-clock sidecar under ``precomputed/``. Uses ``src/logprobs_llm_shim.py``
for chat-completions routing + per-call disk caching (restart-safe).

Implementation lands in Stage 6.
"""
