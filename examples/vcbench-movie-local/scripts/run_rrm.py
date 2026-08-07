"""Run RRM (Reasoned Rule Mining) on a dataset with the local Ollama model.

Fits RRM on a stratified training subsample (reason → rule extraction →
policy compile → calibration), then predicts every sample via the 3-vote
ensemble, writing per-sample scores under ``precomputed/``. Uses
``src/logprobs_llm_shim.py`` — RRM needs token logprobs, which the shim
returns from chat-completions — with per-call disk caching (restart-safe).

Implementation lands in Stage 6.
"""
