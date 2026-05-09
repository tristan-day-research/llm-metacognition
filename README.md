# Fine-Tuning Enhances Latent Metacognitive Capability in Language Models

Code accompanying the submission. Tests the **latent-readout
hypothesis** in Llama-3.1-8B: that linear directions for output-token entropy
and introspectively reported confidence already exist in the pre-trained base
model, and that later training (instruction tuning, then task-specific
fine-tuning) aligns the self-report direction to the entropy direction rather
than installing a self-evaluation mechanism from scratch.

The repo covers two tasks:

- **Explicit Confidence Task (ECT).** The model answers a four-choice MCQ and
  reports confidence on a 1–10 scale.
- **Delegate Game (DG).** The model decides whether to answer or defer to a
  teammate of stated accuracy, using a binary answer/delegate prompt format
  (`delegate_at`).

Each task uses two independent forward passes per item: a **direct-answer
pass** that produces the four-option logits used to estimate uncertainty, and
a **decision pass** that produces the confidence rating or delegation
decision. The decision pass does not see the direct-answer output.

## Repo layout

```
.
├── prompts.py                 # All prompt templates + per-task forward passes.
├── experiment_config.py       # Config for interp_experiments/ runners.
├── finetune_config.py         # Config for fine-tuning + post-hoc evaluation.
│
├── core/                      # Shared library:
│   ├── model_utils.py         #   model loading, LoRA, quantization
│   ├── extraction.py          #   batched activation extraction
│   ├── metrics.py             #   uncertainty metrics (entropy, top_prob, ...)
│   ├── directions.py          #   probe + mean-difference direction finding
│   ├── probes.py              #   linear-probe training and scoring
│   ├── steering.py            #   activation interventions (add / ablate)
│   ├── steering_experiments.py
│   ├── logit_lens.py          #   per-layer unembedding projections
│   └── datasets.py            #   loaders for PopMC / SimpleMC / TriviaMC / ...
│
├── interp_experiments/        # Mech-interp pipeline.
│   └── run_introspection.py   #   activations + directions + probes + logit lens
│
├── analysis/                  # Notebooks consuming the run artifacts.
│   ├── analyze_performance.ipynb     # behavioral results from run_evaluations
│   ├── analyze_activations.ipynb     # direction alignment, probe transfer, logit lens
│   └── analyze_interventions.ipynb   # ablation + steering causality
│
├── finetune/                  # LoRA training (ECT and DG) + evaluation.
│   ├── run_finetuning.py
│   ├── run_evaluations.py
│   ├── create_mixed_dataset.py        # entropy-bin-balanced mix builder
│   ├── create_balanced_dataset.py     # 500-per-bin variant
│   ├── data_handling.py
│   ├── evaluation_metrics.py
│   ├── loss.py                        # ECT + DG losses
│   └── utils.py
│
├── data/                      # MCQ JSONL files (see "Data" below).
└── outputs/                   # Run artifacts (gitignored).
```

## Setup

Tested on a single H100 / A100 (80 GB) with bf16. Smaller cards work via
4-bit / 8-bit loading flags exposed in `experiment_config.py`.

```bash
pip install -r requirements.txt
export HF_TOKEN=<your token>          # for the gated Llama-3.1-8B repos
export WANDB_API_KEY=<your key>       # only if you turn W&B logging on
```

## Data

Three four-choice MCQ datasets, derived from public QA sources:

- `data/PopMC.jsonl` — 14,267 items (PopQA-derived, with one correct option
  and three plausible distractors per row).
- `data/SimpleMC.jsonl` — 500 items (SimpleQA-derived).
- `data/TriviaMC.jsonl` — 2,416 items (TriviaQA-derived).
- `data/mixed_17173_all.jsonl` — concatenation of the three sources, with
  `_train.jsonl` / `_val.jsonl` / `_test.jsonl` splits used for fine-tuning.

Smaller balanced mixes (`mixed_2550_clean`, `mixed_3150_pop_heavy`,
`mixed_11931_max_balanced`, ...) used in earlier configurations also live in
`data/`. They are produced by `finetune/create_mixed_dataset.py` from the
three source files, with stratified entropy-bin coverage (low / medium / high
under teacher entropy) and per-source quotas. See the docstring at the top of
that script for the exact recipe.

## 1 — Fine-tune

Both ECT and DG share the same training script. Pick the task by editing
`finetune_config.ECTConfig.TASK_TYPE` and run:

```bash
python finetune/run_finetuning.py
```

`TASK_TYPE` options:

- `explicit_confidence` — ECT. Trains a soft 1–10 confidence head against an
  entropy-derived target via Gaussian-soft-bin cross-entropy. The target is
  `s_H = 1 - H(p) / log 4` mapped onto the 1–10 grid with width `SIGMA`.
- `delegate_at` — DG, binary. BCE on the (T − A) logit difference against a
  sigmoid-soft delegation target derived from the model's own recorded
  uncertainty (`top_prob` by default), thresholded at the stated teammate
  accuracy. The prompt does not state the threshold.

All other knobs (LoRA rank, learning rate, target metric, teacher mode,
val cadence, ...) live in `ECTConfig` and are documented inline. Two teacher
modes:

- **Frozen teacher** (`USE_RECORDED_RESPONSES=True`) — uses pre-recorded
  per-item entropies from `MCQ_RESULTS_DATA`. Cheaper and more stable.
- **Live teacher** — recomputes entropy on the current model each batch.

Validation runs both `frozen` and `live` targets when
`VAL_RUN_BOTH_FROZEN_AND_LIVE=True`, so both numbers appear in the eval logs.

Outputs:

- `outputs/finetuning/checkpoints/<run-id>/` — LoRA adapter snapshots at the
  step interval set by `CHECKPOINT_STEPS`.
- `outputs/finetuning/logs/<timestamp>_*_evaluation_metrics.jsonl` — periodic
  validation metrics (Spearman against entropy-derived certainty, MCQ
  accuracy, delegation rate, expected team score).
- `outputs/finetuning/logs/<timestamp>_*_print_output.txt` — full run log.

## 2 — Evaluate a checkpoint

Post-hoc evaluation of a single model — base, instruct, or
instruct + a LoRA adapter — across the configured datasets and prompt
families:

```bash
python finetune/run_evaluations.py
```

Eval is also config-driven. The relevant fields on `ECTConfig` are:

- `EVAL_MODEL_TYPE` — `"base"` | `"instruct"` | `"finetuned"`.
- `EVAL_LORA_REPO` — HuggingFace repo (or local PEFT path) for the LoRA
  adapter, used only when `EVAL_MODEL_TYPE == "finetuned"`.
- `EVAL_DATASETS` — list of MCQ JSONLs to evaluate, one at a time.
- `EVAL_RUN_MCQ`, `EVAL_COMPUTE_CONFIDENCE`, `EVAL_RUN_DELEGATE_*` — toggle
  the prompt families (MCQ, self-confidence, delegate AT). All share the
  same fenced prompt layout, so the model sees comparable formats across
  tasks.
- `EVAL_DELEGATE_TEAMMATE_ACCURACY` — accuracy shown to the model in the DG
  setup. Defaults to 0.7.

Outputs (per dataset):

- `outputs/evaluations/<timestamp>_<model>_<dataset>_n<N>.jsonl` — per-sample
  rows + a summary row at the end. The performance notebook reads this.
- `outputs/evaluations/<timestamp>_<model>_<dataset>_n<N>.txt` — human-
  readable run record (config snapshot, example prompts the model actually
  saw, position-bias diagnostics, summary).

To replicate the paper's three-stage comparison, run this script three times
with `EVAL_MODEL_TYPE` set to `base`, then `instruct`, then `finetuned`
(once per ECT adapter and once per DG adapter).

## 3 — Analyze evaluations

```
analysis/analyze_performance.ipynb
```

Reads the `*_evaluation_metrics.jsonl` files and produces the behavioral
tables and plots: Spearman correlations between stated confidence and
entropy-derived certainty (T1, T2), per-dataset breakdowns, ECT-vs-DG
transfer comparisons, calibration curves, expected team score, and
selective-accuracy / AUROC for delegation. Runs locally on the eval logs;
no GPU needed.

## 4 — Collect activations

For the mechanistic results (T1, T3, T4, T5), the introspection runner
collects residual-stream activations on the direct-answer and decision
prompts and saves the artifacts the analysis notebooks consume:

```bash
python interp_experiments/run_introspection.py
```

Defaults live in `experiment_config.IntrospectionExperimentConfig`. The
runner does **one model per invocation** — to populate the three-stage
comparison, run it three times after editing the model fields:

```python
# experiment_config.py — IntrospectionExperimentConfig
BASE_MODEL_NAME = LLAMA_8B_BASE         # base run
BASE_MODEL_NAME = LLAMA_8B_INSTRUCT     # instruct run
BASE_MODEL_NAME = LLAMA_8B_INSTRUCT     # finetuned run; also set
MODEL_NAME      = "<HF adapter id or local PEFT path>"
```

Other knobs:

- `DATASETS` / `NUM_QUESTIONS` — which JSONLs to run and how many items.
- `META_TASKS` — `"confidence"` and/or `"delegate"`. Determines the second
  prompt pass.
- `CONFIDENCE_SCALE` — `"numeric"` (1–10).
- `METRIC` — uncertainty signal the linear probe targets (`entropy`,
  `top_prob`, `margin`, `logit_gap`, `top_logit`); all five are extracted
  regardless, this just selects the saved probe direction.
- `EXTRACT_CONTRAST_DIRECTIONS` — when True, after probe directions the
  runner also computes mean-difference contrast directions (top vs bottom
  `CONTRAST_QUANTILE`) for entropy on direct activations and for stated
  confidence on meta activations. These are the directions used in the
  alignment / steering / ablation analyses.

Per (model × dataset), the runner writes to
`outputs/activations_directions_logitlens/`:

- `*_direct_activations.npz`, `*_meta_activations.npz` — residual-stream
  states from the final token before the model's response, one matrix per
  layer.
- `*_paired_data.json` — per-question metrics, prompts, predicted responses.
- `*_probe_results.json`, `*_*_contrast_directions.npz` — probe weights and
  contrast directions per layer.
- `*_logit_lens.json` — top-K unembedding projections per layer.
- `*_summary.png` — a quick-look figure for sanity checking.

## 5 — Analyze activations

Two notebooks:

```
analysis/analyze_activations.ipynb
```

Reads the artifacts above and produces:

- Probe R² and behavioral Spearman per layer (T1, T2).
- **Cosine alignment** between the entropy and stated-confidence directions
  across stages, plus the per-stage stability comparison that supports the
  "self-report direction moves toward a stable entropy direction" claim
  (T3).
- **Probe transfer**: probes trained on direct-answer states applied to
  decision-pass states, scored across stages (T4).
- Logit-lens projections of both directions across layers and stages.

```
analysis/analyze_interventions.ipynb
```

Reads the same artifacts plus per-layer steering / ablation outputs
produced by `core/steering_experiments.py`, and produces the causality
panels (T5): Δ stated-confidence Spearman vs Δ MCQ accuracy under ablation,
and steering coefficient vs reported confidence.

## Configuration

Two dataclass-style config files hold every defaulted parameter:

- `experiment_config.py` — `IntrospectionExperimentConfig` (mech-interp).
- `finetune_config.py` — `ECTConfig` (training + post-hoc evaluation).

Each runner reads its config at import time. CLI flags on
`run_introspection.py` override per-run; `run_finetuning.py` and
`run_evaluations.py` are config-only — edit the file and rerun.

## Direction-finding methods

| Method      | How                                                                | Strengths                          |
|-------------|--------------------------------------------------------------------|------------------------------------|
| `probe`     | Ridge regression: direction that best predicts the target metric.  | Optimized for prediction R².       |
| `mean_diff` | `mean(top quartile activations) − mean(bottom quartile)`.          | Simple, interpretable, robust.     |

Both are computed per layer and reported side-by-side throughout.

## Uncertainty metrics

| Metric      | Formula            | Higher means    | Linear in logits? |
|-------------|--------------------|-----------------|-------------------|
| `entropy`   | −Σ p log p         | More uncertain  | No                |
| `top_prob`  | max(p)             | More confident  | No                |
| `margin`    | p₁ − p₂            | More confident  | No                |
| `logit_gap` | z₁ − z₂            | More confident  | Yes               |
| `top_logit` | z₁ − mean(z)       | More confident  | Yes               |

The linear-in-logits metrics (`logit_gap`, `top_logit`) tend to be cleaner
targets for linear probes and are often used as DG training targets.

## Reproducing the headline numbers

ECT fine-tune and DG-fine-tune adapters used for the paper were trained on
`data/mixed_17173_all_train.jsonl` with the defaults in `ECTConfig`
(`LEARNING_RATE=1e-5`, `MAX_STEPS=2500`, `BATCH_SIZE=4`,
`LORA_TARGET_MODULES=("q_proj","v_proj","o_proj")`,
`SIGMA=12`, `DELEGATE_TARGET_SOURCE="top_prob"`,
`DELEGATE_TRAIN_TEAMMATE_ACCURACY=0.7`, `DELEGATE_TAU=0.05`). The cross-task
transfer reading (`spearman_neg_entropy_conf` on the ECT prompt for a
DG-trained adapter) is logged automatically by `run_finetuning.py` because
`VAL_RUN_BOTH_FROZEN_AND_LIVE=True` and the validation step always also runs
the confidence prompt regardless of `TASK_TYPE`.
