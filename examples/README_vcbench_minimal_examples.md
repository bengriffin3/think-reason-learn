# VCBench Feature Examples

Two minimal example scripts showing different ways to engineer features
from VCBench founder profiles for binary classification.

## Setup

```bash
poetry install
export OPENAI_API_KEY="sk-..."   # needed for LLM-based features
```

## Architecture

```
think_reason_learn/datasets/     <-- parsers, data loading, schema
think_reason_learn/features/     <-- FeatureGenerator, FeatureEvaluator
examples/
  vcbench_lambda_features_minimal.py   <-- Script 1: lambda features (~350 lines)
  vcbench_reasoned_features.py         <-- Script 2: reasoned features (~270 lines)
```

---

## Script 1: Lambda Features (`vcbench_lambda_features_minimal.py`)

Compares three feature-engineering approaches:

| Mode | Description | API key? | Cost |
|------|-------------|----------|------|
| `human` | 15 hand-crafted binary features | No | Free |
| `llm` | LLM-generated Python lambda features | Yes | ~$0.01 |
| `combined` | Both stacked together (default) | Yes | ~$0.01 |

**Key idea**: The LLM *writes code* (lambda expressions) that extracts features.
One LLM call generates all features, then they evaluate locally for free.

```bash
# Human features only (no API key):
python examples/vcbench_lambda_features_minimal.py \
    --input_csv /path/to/vcbench_final_public.csv --feature_mode human

# Combined (default):
python examples/vcbench_lambda_features_minimal.py \
    --input_csv /path/to/vcbench_final_public.csv --n_features 8
```

### Human-engineered features (15 total)

**Education (4):**

| Feature | Logic |
|---------|-------|
| `top_university` | Attended a top-50 QS-ranked university |
| `has_phd` | Holds a PhD or doctorate |
| `has_mba` | Holds an MBA or master's in business/management |
| `stem_degree` | Studied a STEM field (CS, engineering, math, sciences) |

**Career (7):**

| Feature | Logic |
|---------|-------|
| `senior_leadership` | Held a C-suite, founder, or VP role |
| `large_company_exp` | Worked at a company with >= 1000 employees |
| `startup_exp` | Worked at a company with <= 50 employees |
| `long_experience` | Total work experience > 5 years |
| `serial_founder` | Held "founder" roles at 2+ different companies |
| `technical_role` | Held an engineering/developer/CTO role |
| `many_prior_roles` | 4+ prior roles (breadth of experience) |
| `short_tenure_pattern` | Majority of jobs lasted < 2 years |

**Industry fit (1):**

| Feature | Logic |
|---------|-------|
| `industry_match` | A prior job's industry matches the startup's industry |

**Exits (2):**

| Feature | Logic |
|---------|-------|
| `prior_exit` | Involved in at least one prior IPO or acquisition |
| `multiple_exits` | Involved in 2+ prior exits |

---

## Script 2: Reasoned Features (`vcbench_reasoned_features.py`)

Instead of writing code, the LLM *reads each founder's profile* and makes
a subjective judgment. Each founder requires a separate LLM call.

| Feature | Question |
|---------|----------|
| `domain_expertise` | Does this founder have domain expertise relevant to their startup? |
| `serial_entrepreneur` | Does the career trajectory suggest a serial entrepreneur? |
| `strong_network` | Would you expect this founder to have a strong professional network? |

**Key tradeoff**: More nuanced than lambda features (the LLM can reason about
context), but costs 1 LLM call per founder per question instead of 1 total.

```bash
# Default: 100 founders, 3 questions = 300 LLM calls (< $0.02)
python examples/vcbench_reasoned_features.py \
    --input_csv /path/to/vcbench_final_public.csv

# Quick test
python examples/vcbench_reasoned_features.py \
    --input_csv /path/to/vcbench_final_public.csv --max_rows 30
```

---

## Metrics

Both scripts report the same metrics:

- **ROC-AUC** and **PR-AUC** — threshold-independent ranking quality
- **Precision**, **Recall**, **F0.5** — at an F0.5-optimal threshold tuned on the training set
- **Accuracy** — overall correctness

## Comparison: Lambda vs. Reasoned Features

| | Lambda Features | Reasoned Features |
|--|----------------|-------------------|
| LLM calls | 1 (generates code) | N per question (reads each profile) |
| Cost scaling | O(1) | O(N * Q) |
| Feature type | Structured / deterministic | Subjective / holistic |
| Best for | Large datasets, structured fields | Small datasets, nuanced judgments |

**Cost and time warning for reasoned features:** Because the LLM must evaluate
every founder individually, reasoned features are significantly more expensive
and slower than lambda features. With the default 100 founders and 3 questions,
expect ~300 LLM calls (~$0.02, ~30-60s). Scaling to the full VCBench dataset
(~4500 founders) would require ~13,500 calls, which can take several minutes
and cost ~$0.50-1.00 depending on the model. Always start with a small
`--max_rows` value to verify your setup before scaling up.

## Ways to extend

1. **Combine both** — use lambda features for structured signals and reasoned features for subjective ones.
2. **More questions** — add questions to `QUESTIONS` in the reasoned script.
3. **Different models** — try `--model gpt-4.1-mini` for better reasoning quality.
4. **Scale up** — increase `--max_rows` and `--concurrency` for more founders.
5. **Cross-validation** — replace the single split with `StratifiedKFold`.
