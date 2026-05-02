"""
Centralized fine-tuning configuration.

Source of truth for parameters of fine-tuning runs under `finetune/`.

Pattern: each script's argparse declares its flags with `default=ECTConfig.X`,
so editing values here changes the defaults across every entrypoint without
touching the scripts themselves. CLI flags still override.
"""


from pathlib import Path

from experiment_config import (
    LLAMA_8B_BASE,
    LLAMA_8B_INSTRUCT,
    DEFAULT_ADAPTER,
    DEFAULT_SEED,
    OUTPUTS_DIR,
)


# All finetune outputs land under outputs/finetune/.
# Was finetune_evals/ + finetune_logs/ at the repo root in the original repo.
FINETUNE_OUTPUTS_DIR = OUTPUTS_DIR / "finetune"
FINETUNE_LOGS_DIR = FINETUNE_OUTPUTS_DIR / "logs"
FINETUNE_CHECKPOINTS_DIR = FINETUNE_OUTPUTS_DIR / "checkpoints"
FINETUNE_EVALS_DIR = FINETUNE_OUTPUTS_DIR / "evals"

# Per-dataset evaluation logs (run_evaluations.py) live here.
EVALUATIONS_DIR = OUTPUTS_DIR / "evaluations"


# ---- run_ECT.py (Expected Confidence Task fine-tuning) -----------------------
class ECTConfig:
    """Defaults for the ECT fine-tuning run (finetune/run_finetuning.py).

    Mirrors the argparse defaults in that script. The script's parser uses
    these as `default=...` so the flags still take precedence at the CLI.
    """

    # Model
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    DEVICE = "cuda"

    # Data
    TRAIN_DATA_PATH = "data/PopMC_0_difficulty_filtered_train.jsonl"
    VAL_DATA_PATH = "data/PopMC_0_difficulty_filtered_val.jsonl"
    TEST_DATA_PATH = None
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
    SIGMA = 10.0  # soft-label distribution width
    TEMPERATURE = 0.0  # 0 = deterministic
    SHUFFLE_OPTIONS = True
    ENABLE_DATA_LEAKAGE_CHECKS = True

    # Loss — used by training (compute_loss) AND reported by evaluation_metrics
    # as a validation-loss metric. Keeping this here means eval can never
    # silently diverge from training on the loss formulation.
    LOSS_TYPE = "gaussian_soft_bin_ce"  # or "scalar_confidence_mse"

    # Confidence response format.
    #   "letter_8bin"  — 8 letter tokens (A-H, S-Z, or random) over percentage
    #                    bands. Uses build_self_confidence_prompts /
    #                    run_confidence_forward_pass in finetune_prompting.py.
    #                    This is the path run_finetuning.py currently trains on.
    #   "numeric_1_10" — single-digit tokens 1..10. Uses
    #                    build_self_confidence_prompts_numeric /
    #                    run_confidence_forward_pass_numeric.
    #                    NOTE: only the prompt + forward pass are wired today;
    #                    loss.py + evaluation_metrics.py still
    #                    assume 8 bins, so training on this format requires a
    #                    follow-up pass to widen those to 10 bins.
    CONFIDENCE_FORMAT = "numeric_1_10"

    # Letter randomization (used when CONFIDENCE_FORMAT == "letter_8bin")
    CONFIDENCE_LETTER_SCHEME = "A-H"  # "A-H", "S-Z", or "random"
    CONFIDENCE_LETTER_RANDOM_SEED = None
    MCQ_LETTER_SCHEME = "A-D"  # "A-D"/"E-H"/.../"random"
    MCQ_LETTER_RANDOM_SEED = None
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

    # Weights & Biases
    WANDB_PROJECT = "llm-metacognition-ect"
    WANDB_RUN_NAME = None
    WANDB_TAGS = None
    WANDB_NOTES = None
    SAVE_WANDB_ARTIFACT = False

    # ---- Post-hoc evaluation (run_evaluations.py) ---------------------------
    # All HOW knobs (sigma, loss, letter scheme, mcq scheme) live above and
    # are shared with training so eval cannot drift. The fields below only
    # describe WHAT to evaluate.
    EVAL_BASE_MODEL = LLAMA_8B_INSTRUCT
    EVAL_LORA_REPO = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"  # "" = base only
    EVAL_MERGE_LORA = False
    EVAL_DATASETS = (
        "data/PopMC.jsonl",
        "data/SimpleMC.jsonl",
        "data/TriviaMC.jsonl",
    )
    EVAL_MAX_SAMPLES = None  # None = full dataset
    EVAL_EVALUATE_BASE_FIRST = True  # base vs finetuned into one JSONL
    EVAL_COMPUTE_CONFIDENCE = True
    EVAL_COMPUTE_OTHER_CONFIDENCE = True
    EVAL_LOG_DIR = EVALUATIONS_DIR
    EVAL_USE_WANDB = False
    EVAL_WANDB_PROJECT = "llm-evaluation"


# ---- shared / cross-cutting --------------------------------------------------
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
