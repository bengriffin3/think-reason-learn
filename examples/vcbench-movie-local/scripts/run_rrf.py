"""Run RRF (Random Rule Forest) on a dataset with the local Ollama model.

Generates a YES/NO question shortlist from the training split (one cached
LLM call), answers every question per sample, and writes the per-sample
answer matrix plus per-question metrics under ``precomputed/``. Uses
``src/logprobs_llm_shim.py`` for chat-completions routing + per-call disk
caching (restart-safe).

Implementation lands in Stage 6.
"""
