"""Run GPTree on a dataset with the local Ollama model.

Fits the LLM-question decision tree on a stratified training subsample
(auto-checkpointing per node), then traverses every sample to a leaf
probability, writing per-sample scores under ``precomputed/``. Restart the
same command to resume after a kill.

Implementation lands in Stage 6.
"""
