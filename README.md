# LLM Metacognition

Interpretability and fine-tuning for LLM metacognition: do language models have internal representations of their own uncertainty, and do they actually use them when asked to introspect?

The repo combines two halves of a single research program:

- **[interp_experiments/](interp_experiments/)** — mech-interp experiments on stock and fine-tuned models: identify uncertainty directions in the residual stream, test whether they transfer between direct-answer and meta-judgment (confidence / delegation) prompts, and ablate / steer them to test causal involvement.
- **[finetune/](finetune/)** — LoRA fine-tuning that trains a model's stated confidence to track its own MCQ-answer entropy (the Expected Confidence Task, ECT), plus evaluation scripts.

## Hypothesis

When an LLM is asked "how confident are you?", a model that genuinely **introspects** should access the same internal uncertainty representations it would use to *answer* the underlying question. A model that's just pattern-matching on surface features wouldn't.

The interp half tests this in three steps:

1. **Identify** uncertainty directions in residual-stream activations on direct MCQ.
2. **Transfer**: do the same directions explain variance on meta-judgment activations?
3. **Causality**: do ablation / steering of those directions change meta-judgments?

The finetune half asks: if we explicitly train the model to report calibrated confidence, do the interp results change? Does fine-tuning produce a model that introspects more, or one that just learned a token-level shortcut?

## Repo layout

```
.
├── prompts.py                # All prompt templates (MCQ, stated-confidence,
│                             #   delegate, numeric 1-10, …) used everywhere.
├── experiment_config.py      # Single source of truth for interp_experiments/
│                             #   defaults (one dataclass per runner).
├── finetune_config.py        # Single source of truth for fine-tuning defaults.
├── finetune_prompting.py     # Prompt builders + forward passes specific to
│                             #   fine-tuning (letter-randomization, A-H bins,
│                             #   1-10 numeric scale).
│
├── core/                     # Shared library used by interp + finetune:
│   ├── model_utils.py        #   model loading, quantization
│   ├── extraction.py         #   batched activation extraction
│   ├── metrics.py            #   uncertainty metric computation
│   ├── directions.py         #   probe / mean_diff direction finding
│   ├── probes.py             #   linear probe training + scoring
│   ├── steering.py           #   activation intervention hooks
│   ├── steering_experiments.py
│   ├── logit_lens.py
│   ├── questions.py          #   thin wrapper over core.datasets
│   ├── datasets.py           #   loaders for SimpleMC, TriviaMC, GPQA, MMLU,
│   │                         #     PopMC, TruthfulQA, Garupanese, …
│   └── logres_helpers.py
│
├── interp_experiments/       # Mech-interp runners + the helpers they call.
├── analysis/                 # Post-hoc analysis scripts and notebooks
│                             #   (figures, tables, contrast comparisons).
├── finetune/                 # LoRA training (ECT) + finetune evaluation.
├── data/                     # Question datasets (JSONL).
└── outputs/                  # Run artifacts (gitignored).
```

## Configuration

All runner parameters live in **[experiment_config.py](experiment_config.py)** (interp) and **[finetune_config.py](finetune_config.py)** (fine-tuning). Each runner aliases values from the corresponding dataclass into its own module namespace, so changing a knob in one place propagates everywhere it's used. CLI flags still override per-run.

```python
# experiment_config.py
class AblationCausalityConfig:
    MODEL = LLAMA_8B_INSTRUCT
    INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC"
    DIRECTION_METRIC = "entropy"
    META_TASK = "delegate"
    NUM_QUESTIONS = 100
    ...
```

## Running the interp pipeline

All scripts assume the repo root as the working directory.

**Step 0 — identify uncertainty directions on direct MCQ.**

```bash
python interp_experiments/run_introspection_experiment.py
```

Produces `_direct_activations.npz` + `_paired_data.json` per (model × dataset) under `outputs/`. From those, `run_introspection_probe.py` fits per-layer linear probes onto each uncertainty metric.

**Step 1 — test direct→meta transfer.**

```bash
python interp_experiments/test_meta_transfer.py
```

Loads the direct-task probes and applies them to meta-task activations. Reports per-layer transfer R² and behavioral correlation (stated confidence vs. actual entropy).

**Step 2 — causality (ablation + steering).**

```bash
python interp_experiments/run_ablation_causality.py
python interp_experiments/run_steering_causality.py
```

Both run all layers, both methods (`probe` and `mean_diff`), pooled-null + FDR correction. Ablation tests *necessity* (does removing the direction degrade meta-task calibration?); steering tests *sufficiency* (does adding the direction shift stated confidence?).

**Step 3 — interpret directions.**

```bash
python interp_experiments/act_oracles.py
```

Uses Activation Oracles to interpret what concept a probe direction represents.

There are also driver scripts that sweep multiple model × dataset configurations (e.g. `run_all_stated_conf_ablations.py`, `run_introspection_for_ablation.py`) by mutating module-level globals between runs — this is why the runners' CONFIGURATION blocks alias from `experiment_config.py` rather than referencing the dataclass directly.

## Direction-finding methods

| Method | How | Strengths |
|---|---|---|
| `probe` | Ridge regression: direction that best predicts the target metric | Optimized for prediction R² |
| `mean_diff` | `mean(top quartile activations) − mean(bottom quartile)` | Simple, interpretable, robust |

Both are computed by default and reported side-by-side throughout the pipeline.

## Uncertainty metrics

| Metric | Formula | Higher means | Linear? |
|---|---|---|---|
| `entropy` | −Σ p log p | More uncertain | No |
| `top_prob` | max(p) | More confident | No |
| `margin` | p₁ − p₂ | More confident | No |
| `logit_gap` | z₁ − z₂ | More confident | Yes |
| `top_logit` | z₁ − mean(z) | More confident | Yes |

Linear metrics (`logit_gap`, `top_logit`) tend to be cleaner targets for linear probes.

## Fine-tuning (ECT)

The Expected Confidence Task trains an instruct model (LoRA adapter) so its stated confidence on the 8-bin (or 1–10) scale tracks its own MCQ entropy. Two teacher modes:

- **Dynamic teacher**: entropy computed live from the current model on each batch.
- **Frozen teacher**: pre-recorded MCQ entropies from a JSON file (faster, more stable).

```bash
python finetune/run_finetuning.py \
    --train_data_path data/PopMC_0_difficulty_filtered_train.jsonl \
    --val_data_path data/PopMC_0_difficulty_filtered_val.jsonl \
    --no_use_recorded_responses \
    --val_on_live \
    --confidence_letter_scheme A-H \
    --loss_type gaussian_soft_bin_ce
```

Defaults (LoRA rank/alpha, learning rate, validation cadence, W&B project, …) are in `finetune_config.ECTConfig`.

`CONFIDENCE_FORMAT` selects between the 8-letter scale (`letter_8bin`) and a 1–10 numeric scale (`numeric_1_10`); the prompt builders for both live in [finetune_prompting.py](finetune_prompting.py).

## Datasets

Loaders live in [core/datasets.py](core/datasets.py). Supported: `SimpleMC`, `TriviaMC`, `PopMC`, `PopMC_0_difficulty_filtered`, `GPQA`, `MMLU`, `TruthfulQA`, `Garupanese`, `GarupaneseMC`. Local JSONLs are under [data/](data/).

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=<your token>          # for HuggingFace gated models
export WANDB_API_KEY=<your key>       # only if you use --save_wandb_artifact
```
