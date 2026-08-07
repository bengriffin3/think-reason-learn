"""Run the traditional-ML baseline suite on a dataset (no LLM required).

Builds ``[text-length, MiniLM-L6-v2 embedding (384d), TF-IDF top-400]``
features, fits the 5-model suite (LR, HistGradientBoosting, RandomForest-250,
ExtraTrees-250, GaussianNB — class_weight balanced, median imputation),
and writes per-sample probabilities + the rank-average ensemble under
``precomputed/``. Runs in minutes on CPU.

Implementation lands in Stage 6.
"""
