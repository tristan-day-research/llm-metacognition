"""
Centralized fine-tuning + evaluation configuration.

Source of truth for parameters shared across `finetune/run_finetuning.py`
(training) and `finetune/run_evaluations.py` (post-hoc evaluation).

Layout:
  1. Module-level path constants
  2. ECTConfig
       a. SHARED        — read by BOTH training and eval; must not drift
       b. TRAINING-ONLY — used solely by run_finetuning.py
       c. EVAL-ONLY     — used solely by run_evaluations.py
  3. FinetuneConfig (legacy umbrella; thin pointer to constants above)

Pattern: run_finetuning.py declares its argparse flags with
`default=ECTConfig.X`, so editing values here changes the defaults across
every entrypoint without touching the scripts. run_evaluations.py is
config-driven (no flags) and reads ECTConfig directly.
"""


from pathlib import Path

from experiment_config import (
    LLAMA_8B_BASE,
    LLAMA_8B_INSTRUCT,
    DEFAULT_ADAPTER,
    DEFAULT_SEED,
    OUTPUTS_DIR,
)


# =============================================================================
# Path constants
# =============================================================================
# All finetune outputs land under outputs/finetune/.
# Was finetune_evals/ + finetune_logs/ at the repo root in the original repo.
FINETUNE_OUTPUTS_DIR = OUTPUTS_DIR / "finetune"
FINETUNE_LOGS_DIR = FINETUNE_OUTPUTS_DIR / "logs"
FINETUNE_CHECKPOINTS_DIR = FINETUNE_OUTPUTS_DIR / "checkpoints"
FINETUNE_EVALS_DIR = FINETUNE_OUTPUTS_DIR / "evals"

# Per-dataset evaluation logs (run_evaluations.py) live here.
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"


# =============================================================================
# ECTConfig — single source of truth for the ECT experiment
# =============================================================================
class ECTConfig:
    """Defaults for the ECT fine-tuning run AND post-hoc evaluation.

    Single source of truth: the SHARED block below is read by both
    run_finetuning.py and run_evaluations.py, so eval cannot silently
    diverge from training on sigma / loss / prompt format / letter schemes.
    """

    # =========================================================================
    # SHARED — read by BOTH training and eval. Must not drift.
    # =========================================================================

    # Soft-label / loss math.
    SIGMA = 10.0  # Gaussian width for entropy → soft-label conversion
    TEMPERATURE = 0.0  # 0 = deterministic argmax; >0 = sampling

    # Loss formulation. Used by training (compute_loss) AND reported by
    # evaluation_metrics as a validation-loss metric.
    LOSS_TYPE = "gaussian_soft_bin_ce"  # or "scalar_confidence_mse"

    # Confidence response format — controls which prompt builder, forward
    # pass, and loss bin spec are used. Self- and other-confidence ALWAYS
    # share this scheme.
    #   "letter_8bin"  — 8 letter tokens (A-H, S-Z, or random) over percentage
    #                    bands matching the prompt text
    #                    "<5%, 5-10%, 10-20%, 20-40%, 40-60%, 60-80%, 80-90%, >90%".
    #                    Soft labels weight each bin by its width since the
    #                    bins aren't uniform.
    #   "numeric_1_5"  — single-digit tokens 1..5. 5 uniform bins of 20% each,
    #                    midpoints at [10, 30, 50, 70, 90].
    #   "numeric_1_10" — single-digit tokens 1..10. 10 uniform bins of 10%
    #                    each, midpoints at [5, 15, 25, …, 95]. Llama 3
    #                    tokenizes "10" as a single token, so this is clean.
    # All three are wired end-to-end: prompt builder → forward pass → loss
    # → evaluation_metrics. Switching this knob is the only change needed.
    CONFIDENCE_FORMAT = "numeric_1_10"

    # MCQ answer letters — ALWAYS used (the model still picks A/B/C/D for the
    # multiple-choice answer regardless of which confidence format you use).
    MCQ_LETTER_SCHEME = "A-D"  # "A-D"/"E-H"/.../"random"
    MCQ_LETTER_RANDOM_SEED = None

    # Confidence letters — ONLY used when CONFIDENCE_FORMAT == "letter_8bin".
    # Harmless to leave set when using a numeric format; the code reads it
    # but ignores it on the numeric path.
    CONFIDENCE_LETTER_SCHEME = "A-H"  # "A-H", "S-Z", or "random"
    CONFIDENCE_LETTER_RANDOM_SEED = None

    # =========================================================================
    # TRAINING — used solely by run_finetuning.py.
    # =========================================================================

    # Model
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DEVICE = "cuda"

    # Data
    # TRAIN_DATA_PATH = "data/PopMC_0_difficulty_filtered_train.jsonl"
    # VAL_DATA_PATH = "data/PopMC_0_difficulty_filtered_val.jsonl"
    # TEST_DATA_PATH = None
    BATCH_SIZE = 4
    MCQ_RESULTS_DATA = None  # path to JSON/JSONL with pre-recorded MCQ entropies

    # Teacher / validation mode
    # USE_RECORDED_RESPONSES: True = frozen teacher (uses MCQ_RESULTS_DATA);
    #                        False = dynamic teacher (recompute logits live).
    # VAL_ON_FROZEN:         True = validate against pre-recorded answers/entropy;
    #                        False = validate against live model.
    USE_RECORDED_RESPONSES = False
    VAL_ON_FROZEN = False

    # LoRA
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training loop
    LEARNING_RATE = 5e-5
    MAX_STEPS = 1_000_000_000  # effectively no cap; rely on early stopping or manual
    LOG_INTERVAL = 20
    VAL_INTERVAL = 100
    LIMIT_VAL_BATCHES = None
    VAL_NUM_SAMPLES = 500
    SHUFFLE_OPTIONS = True
    ENABLE_DATA_LEAKAGE_CHECKS = True
    RANDOMIZE_LETTERS_PER_QUESTION = False

    # Output / checkpointing
    OUTPUT_DIR = FINETUNE_OUTPUTS_DIR / "ect_lora"
    LOGS_DIR = FINETUNE_LOGS_DIR
    CHECKPOINTS_DIR = FINETUNE_CHECKPOINTS_DIR
    CHECKPOINT_STEPS = 500
    SAVE_HF = False
    HF_REPO = None
    SAVE_HF_CHECKPOINTS = False
    HF_CHECKPOINT_REPO = None
    HF_CHECKPOINT_PRIVATE = False

    # Weights & Biases (training)
    WANDB_PROJECT = "llm-metacognition-ect"
    WANDB_RUN_NAME = None
    WANDB_TAGS = None
    WANDB_NOTES = None
    SAVE_WANDB_ARTIFACT = False

    # =========================================================================
    # EVALUATION — used solely by run_evaluations.py.
    # All HOW knobs live in the SHARED block above so eval cannot drift between
    # eval and finetune, since finetuning will be on the data filtered by eval code.
    # The fields below only describe WHAT to evaluate.
    # =========================================================================

    # Which model to evaluate. Exactly one of:
    #   "base"      → raw Llama base (no instruction tuning, no LoRA).
    #                 Uses LLAMA_8B_BASE; prompts emitted in raw-text format
    #                 (no chat tags) because the base model wasn't trained on
    #                 chat-templated input.
    #   "instruct"  → Llama-Instruct (off-the-shelf, no LoRA).
    #                 Uses LLAMA_8B_INSTRUCT; prompts use the chat template.
    #   "finetuned" → Llama-Instruct + the LoRA adapter at EVAL_LORA_REPO.
    EVAL_MODEL_TYPE = "instruct"

    # LoRA adapter — only loaded when EVAL_MODEL_TYPE == "finetuned".
    # Harmless to leave set when evaluating "base" or "instruct"; ignored.
    EVAL_LORA_REPO = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"
    EVAL_MERGE_LORA = False  # if True, merge LoRA into base before eval

    EVAL_DATASETS = (
        "data/PopMC.jsonl",
        # "data/SimpleMC.jsonl",
        # "data/TriviaMC.jsonl",
    )
    EVAL_MAX_SAMPLES = None  # None = full dataset
    EVAL_COMPUTE_CONFIDENCE = True
    EVAL_COMPUTE_OTHER_CONFIDENCE = True
    EVAL_LOG_DIR = EVALUATIONS_DIR

    # Weights & Biases — when True, sends per-sample metrics + summaries to a
    # W&B project so you can compare runs in their UI. Set False for one-off
    # local eval; the JSONL + .txt logs in EVAL_LOG_DIR are produced either way.
    EVAL_USE_WANDB = False
    EVAL_WANDB_PROJECT = "llm-evaluation"


# =============================================================================
# FinetuneConfig — legacy umbrella, points at the constants above
# =============================================================================
class FinetuneConfig:
    """Shared defaults across all finetune entrypoints."""

    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    DEFAULT_ADAPTER = DEFAULT_ADAPTER  # the canonical fine-tuned adapter on HF
    SEED = DEFAULT_SEED
    OUTPUTS_DIR = FINETUNE_OUTPUTS_DIR
    LOGS_DIR = FINETUNE_LOGS_DIR
    CHECKPOINTS_DIR = FINETUNE_CHECKPOINTS_DIR
    EVALS_DIR = FINETUNE_EVALS_DIR
    EVALUATIONS_DIR = EVALUATIONS_DIR
