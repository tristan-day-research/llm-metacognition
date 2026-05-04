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
# All finetune outputs land under outputs/finetuning/.
# Was finetune_evals/ + finetune_logs/ at the repo root in the original repo.
FINETUNE_OUTPUTS_DIR = OUTPUTS_DIR / "finetuning"
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

    # Base model. Single source of truth for which Llama version is being
    # finetuned AND evaluated. Imported from experiment_config so eval and
    # training cannot disagree on Llama-3 vs Llama-3.1, etc.
    # Eval uses LLAMA_8B_BASE only when EVAL_MODEL_TYPE == "base"; for
    # "instruct" and "finetuned" it should use this value.
    MODEL_NAME = LLAMA_8B_INSTRUCT
    DEVICE = "cuda"

    # Soft-label / loss math.
    SIGMA = 7.5 # Gaussian width for entropy → soft-label conversion
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
    # Read by run_evaluations.py and the training prompt builder.
    MCQ_LETTER_SCHEME = "A-D"  # "A-D"/"E-H"/.../"random"
    MCQ_LETTER_RANDOM_SEED = None

    # Confidence letters — ONLY used when CONFIDENCE_FORMAT == "letter_8bin".
    # Harmless to leave set when using a numeric format; the code reads it
    # but ignores it on the numeric path. Read by run_evaluations.py.
    CONFIDENCE_LETTER_SCHEME = "A-H"  # "A-H", "S-Z", or "random"
    CONFIDENCE_LETTER_RANDOM_SEED = None

    # Prompt-construction knobs that affect what the model sees. If training
    # and eval disagree on these, the eval distribution drifts from training.
    SHUFFLE_OPTIONS = True
    RANDOMIZE_LETTERS_PER_QUESTION = False  # uses letters beyond A-D per question

    # Whether to also run the "other" confidence pass during evaluation
    # (i.e. predicting how confident a generic college-educated peer would be).
    # Training (train_step) only ever runs the self pass — this flag toggles
    # whether eval/baseline/test additionally do the other pass.
    COMPUTE_OTHER_CONFIDENCE = False

    # =========================================================================
    # TRAINING — used solely by run_finetuning.py.
    # =========================================================================

    # Data
    TRAIN_DATA_PATH = "data/balanced_metacognition_train.jsonl"
    VAL_DATA_PATH = "data/balanced_metacognition_val.jsonl"
    TEST_DATA_PATH = "data/balanced_metacognition_test.jsonl"
    BATCH_SIZE = 4
    MCQ_RESULTS_DATA = None  # path to JSON/JSONL with pre-recorded MCQ entropies

    # Teacher / validation mode
    # USE_RECORDED_RESPONSES: True = frozen teacher (uses MCQ_RESULTS_DATA);
    #                        False = dynamic teacher (recompute logits live).
    # VAL_ON_FROZEN:         True = validate against pre-recorded answers/entropy;
    #                        False = validate against live model.
    USE_RECORDED_RESPONSES = True
    VAL_ON_FROZEN = True

    # If True, every validation (baseline + periodic + final) runs TWICE:
    # once with val_on_frozen=True and once with val_on_frozen=False. Frozen
    # uses pre-recorded teacher entropy (apples-to-apples with the training
    # target); live recomputes entropy from the current model. Metrics are
    # logged to separate W&B namespaces (val_frozen/* and val_live/*) so you
    # can chart both. Doubles per-validation wall time.
    # When False, only the mode in VAL_ON_FROZEN runs (logs under val/*).
    VAL_RUN_BOTH_FROZEN_AND_LIVE = True

    # LoRA
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ("q_proj", "v_proj", "o_proj")
    # LORA_TARGET_MODULES = ("q_proj", "v_proj")

    # Training loop
    LEARNING_RATE = 1e-6
    MAX_STEPS = 9000
    LOG_INTERVAL = 20
    VAL_INTERVAL = 100
    LIMIT_VAL_BATCHES = None
    VAL_NUM_SAMPLES = 300
    ENABLE_DATA_LEAKAGE_CHECKS = True

    # Gradient clipping (max global L2 norm). Set to None to disable.
    # Prevents the occasional bf16 overflow from poisoning the LoRA weights
    # — without this, training runs sometimes hit NaN around step ~400-500
    # and never recover.
    MAX_GRAD_NORM = 1.0

    # Per-sample loss weight applied to *high-entropy* training samples
    # (entropy >= 2*ln(4)/3 ≈ 0.924, matching the high-bin definition used
    # for the per-bin diagnostics). 1.0 disables the reweighting; values
    # >1 emphasize hard questions, values <1 deemphasize them.
    # Implementation: per-sample loss is computed (reduction='none'),
    # weighted by 1.0 or HIGH_ENTROPY_LOSS_WEIGHT depending on entropy bin,
    # then averaged. Validation loss reporting is NOT reweighted (so val/loss
    # stays comparable across runs).
    HIGH_ENTROPY_LOSS_WEIGHT = 1.3

    # Output / checkpointing
    OUTPUT_DIR = FINETUNE_OUTPUTS_DIR / "ect_lora"
    LOGS_DIR = FINETUNE_LOGS_DIR
    CHECKPOINTS_DIR = FINETUNE_CHECKPOINTS_DIR
    CHECKPOINT_STEPS = 500
    # If False, the local checkpoint dirs (final OUTPUT_DIR + per-step CHECKPOINTS_DIR/...)
    # are deleted after they've been uploaded to W&B / HF, so no local artifacts
    # persist on disk. Local saves still happen briefly because wandb artifact
    # upload reads from a directory; we just clean them up afterwards.
    KEEP_LOCAL_CHECKPOINTS = False
    SAVE_HF = False
    HF_REPO = None
    SAVE_HF_CHECKPOINTS = False
    HF_CHECKPOINT_REPO = None
    HF_CHECKPOINT_PRIVATE = False

    # Weights & Biases (training)
    WANDB_PROJECT = "llm-metacognition-ect"
    WANDB_RUN_NAME = "1-10 bins anchor only, balanced dataset"
    WANDB_TAGS = None
    WANDB_NOTES = None
    SAVE_WANDB_ARTIFACT = True

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
    # EVAL_LORA_REPO = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"
    EVAL_MERGE_LORA = False  # if True, merge LoRA into base before eval

    EVAL_DATASETS = (
        "data/PopMC.jsonl",
        "data/SimpleMC.jsonl",
        "data/TriviaMC.jsonl",
        # 'data/balanced_metacognition_train.jsonl',
        # 'data/balanced_metacognition.jsonl'
    )
    EVAL_MAX_SAMPLES = None  # None = full dataset

    # ----- Per-task toggles ---------------------------------------------------
    # Each toggle controls whether one of the four eval prompt types runs.
    # All four share the same fenced prompt layout so they're directly
    # comparable (see prompts.py: build_*_prompts).
    #
    # EVAL_RUN_MCQ — bare multiple-choice question. Produces the
    #   entropy/accuracy numbers that loss + calibration metrics depend on, so
    #   disabling it leaves those fields zero / nan. Leave True unless you ONLY
    #   want delegate-game numbers.
    EVAL_RUN_MCQ = True
    # Self- and other-confidence prompts. Names kept for back-compat — the
    # rest of the pipeline (evaluation_metrics.run_evaluation) reads these.
    EVAL_COMPUTE_CONFIDENCE = True
    EVAL_COMPUTE_OTHER_CONFIDENCE = False
    # Delegate game — two variants. Either, both, or neither can be enabled.
    #   EVAL_RUN_DELEGATE_ABCDT — single-shot. Model picks one of A/B/C/D
    #     (display letters follow MCQ_LETTER_SCHEME) to answer, or T to
    #     delegate. Same option order as the MCQ pass, so shuffling +
    #     letter-scheme remapping stay wired correctly.
    #   EVAL_RUN_DELEGATE_AT — binary. A = "answer myself", T = "delegate".
    #     The MC options are still shown so the model can judge difficulty,
    #     but it does NOT pick among them.
    EVAL_RUN_DELEGATE_ABCDT = False
    EVAL_RUN_DELEGATE_AT = False
    # Teammate accuracy shown to the model in the delegate-game setup blurb.
    # 0.7 puts the decision boundary near the strong-model accuracy regime.
    EVAL_DELEGATE_TEAMMATE_ACCURACY = 0.7

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
