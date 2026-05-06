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

    # Which task to fine-tune on. Eval (run_evaluations.py) is unaffected and
    # always evaluates ALL configured tasks (MCQ + confidence + delegate
    # variants); only the training objective changes here.
    #   "explicit_confidence" — original ECT path. Trains the confidence head
    #                           against soft N-bin labels derived from the
    #                           teacher MCQ pass. Uses CONFIDENCE_FORMAT,
    #                           SIGMA, LOSS_TYPE, FINETUNING_TARGET below.
    #   "delegate_at"         — Delegate Game, binary "A=answer / T=delegate".
    #                           Trains BCE on (T-A) logit difference against a
    #                           sigmoid-soft delegation target derived from
    #                           the model's OWN recorded uncertainty (NOT
    #                           correctness — see DELEGATE_* knobs below).
    #   "delegate_abcdt"      — Delegate Game, single-shot A/B/C/D/T. Same
    #                           soft delegation BCE plus a downweighted CE on
    #                           the recorded model answer for the answer head.
    TASK_TYPE = "delegate_at"

    # Base model. Single source of truth for which Llama version is being
    # finetuned AND evaluated. Imported from experiment_config so eval and
    # training cannot disagree on Llama-3 vs Llama-3.1, etc.
    # Eval uses LLAMA_8B_BASE only when EVAL_MODEL_TYPE == "base"; for
    # "instruct" and "finetuned" it should use this value.
    MODEL_NAME = LLAMA_8B_INSTRUCT
    DEVICE = "cuda"

    # Soft-label / loss math.
    SIGMA = 12 # Gaussian width for entropy → soft-label conversion
    TEMPERATURE = 0.0  # 0 = deterministic argmax; >0 = sampling

    # Loss formulation. Used by training (compute_loss) AND reported by
    # evaluation_metrics as a validation-loss metric.
    LOSS_TYPE = "gaussian_soft_bin_ce"  # or "scalar_confidence_mse"

    # Signal used to build the soft confidence target during training.
    # "entropy"   — normalised MCQ entropy (existing behaviour).
    # "top_logit" — probability of the top MCQ choice, mapped [0.25,1]→[0,100].
    # "logit_gap" — log-prob gap between top and second choice, mapped via tanh.
    FINETUNING_TARGET = "entropy"

    # Confidence response format — controls which prompt builder, forward
    # pass, and loss bin spec are used. Self- and other-confidence ALWAYS
    # share this scheme.
    #   "letter_8bin"  — 8 letter tokens (A-H, S-Z, or random) over percentage
    #                    bands matching the prompt text
    #                    "<5%, 5-10%, 10-20%, 20-40%, 40-60%, 60-80%, 80-90%, >90%".
    #                    Soft labels weight each bin by its width since the
    #                    bins aren't uniform.
    #   "1-5"  — single-digit tokens 1..5. 5 uniform bins of 20% each,
    #            midpoints at [10, 30, 50, 70, 90].
    #   "1-10" — single-digit tokens 1..10. 10 uniform bins of 10%
    #            each, midpoints at [5, 15, 25, …, 95]. Llama 3
    #            tokenizes "10" as a single token, so this is clean.
    # All three are wired end-to-end: prompt builder → forward pass → loss
    # → evaluation_metrics. Switching this knob is the only change needed.
    CONFIDENCE_FORMAT = "1-10"

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

    # ----- Delegate game during training-time validation ----------------------
    # Independent of the EVAL_RUN_DELEGATE_* flags below (which gate
    # run_evaluations.py). When either is True, the validation step runs the
    # delegate prompt(s) on the val set and reports per-variant:
    #     <variant>/delegate_rate         — fraction of samples where the
    #                                        model picked T
    #     <variant>/expected_team_score   — P(answer) * accuracy_when_answered
    #                                        + P(delegate) * teammate_accuracy
    # Logged to W&B under val/* (and val_live/* when the dual mode is on).
    # ABCDT — single-shot. Model picks A/B/C/D to answer, or T to delegate.
    # AT    — binary meta-decision. The team_score for AT requires the bare
    #         MCQ pass, since the model doesn't pick an answer in this prompt:
    #         when AT says "answer myself" we look up the MCQ-pass correctness
    #         on the same sample.
    VAL_RUN_DELEGATE_ABCDT = True
    VAL_RUN_DELEGATE_AT = True
    # TABCD / TA mirror ABCDT / AT but show T (delegate) FIRST in the option
    # block instead of last. Comparing ABCDT-vs-TABCD and AT-vs-TA delegation
    # rates isolates first-position bias as a confound on the delegate decision.
    VAL_RUN_DELEGATE_TABCD = True
    VAL_RUN_DELEGATE_TA = True
    VAL_DELEGATE_TEAMMATE_ACCURACY = 0.7

    # ----- Delegate-game FINE-TUNING knobs ------------------------------------
    # Only consulted when TASK_TYPE in ("delegate_at", "delegate_abcdt").
    # Independent of the VAL_RUN_DELEGATE_* knobs above, which control the
    # validation-side delegate diagnostics for ANY task type.
    #
    # Metacognition-targeted: target is built from the model's OWN recorded
    # MCQ uncertainty. Training the model to delegate based directly on
    # recorded correctness would teach correctness calibration, not
    # metacognition.
    #
    # DELEGATE_TARGET_SOURCE — which recorded uncertainty signal drives the
    #     soft delegation target. See loss.confidence_signal_to_unit().
    #       "top_prob"      → recorded probability of the top MCQ choice.
    #                         Cleanest for the 0.7 threshold below.
    #       "entropy_conf"  → 1 - entropy/ln(4), so 0 = uniform, 1 = peaked.
    #       "prob_gap"      → top_prob - second_prob.
    # DELEGATE_TRAIN_TEAMMATE_ACCURACY — accuracy SHOWN to the model in the
    #     prompt setup AND used as the threshold in the soft target. Keeping
    #     these tied means "the model is asked to delegate iff its self_conf
    #     falls below the teammate's stated accuracy", which is the right
    #     decision rule. Defaults to VAL_DELEGATE_TEAMMATE_ACCURACY so the
    #     train and val delegate setups match by default.
    # DELEGATE_TAU — softness of the sigmoid threshold. Smaller τ → harder
    #     boundary at teammate_accuracy. 0.05 makes the transition span
    #     roughly ±0.1 around the threshold.
    # DELEGATE_ANSWER_LOSS_WEIGHT — only used for TASK_TYPE='delegate_abcdt'.
    #     Weight on the answer-selection CE against the recorded model
    #     answer. The CE is multiplied by (1 - target_delegate) so it
    #     vanishes on samples we want to delegate.
    DELEGATE_TARGET_SOURCE = "top_prob"
    DELEGATE_TRAIN_TEAMMATE_ACCURACY = 0.7
    DELEGATE_TAU = 0.05
    DELEGATE_ANSWER_LOSS_WEIGHT = 0.2

    # =========================================================================
    # TRAINING — used solely by run_finetuning.py.
    # =========================================================================

    # Data
    # TRAIN_DATA_PATH = "data/balanced_metacognition_train.jsonl"
    # VAL_DATA_PATH = "data/balanced_metacognition_val.jsonl"
    # TEST_DATA_PATH = "data/balanced_metacognition_test.jsonl"
    # TRAIN_DATA_PATH = "data/balanced_popMC_train.jsonl"
    # VAL_DATA_PATH = "data/balanced_popMC_val.jsonl"
    # TEST_DATA_PATH = "data/balanced_popMC_test.jsonl"
    # TRAIN_DATA_PATH = "data/mixed_2550_clean_train.jsonl"
    # VAL_DATA_PATH = "data/mixed_2550_clean_val.jsonl"
    # TEST_DATA_PATH = "data/mixed_2550_clean_test.jsonl"
    # TRAIN_DATA_PATH = "data/mixed_3150_pop_heavy_train.jsonl"
    # VAL_DATA_PATH = "data/mixed_3150_pop_heavy_val.jsonl"
    # TEST_DATA_PATH = "data/mixed_3150_pop_heavy_test.jsonl"
    # TRAIN_DATA_PATH = "data/mixed_11931_max_balanced_train.jsonl"
    # VAL_DATA_PATH = "data/mixed_11931_max_balanced_val.jsonl"
    # TEST_DATA_PATH = "data/mixed_11931_max_balanced_test.jsonl"
    TRAIN_DATA_PATH = "data/mixed_17173_all_train.jsonl"
    VAL_DATA_PATH = "data/mixed_17173_all_val.jsonl"
    TEST_DATA_PATH = "data/mixed_17173_all_test.jsonl"
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
    # NOTE: starting smaller for the delegate task (r=8 / α=16) per the
    # delegate-game spec — the model only has to learn a single binary
    # decision, so a fatter adapter risks overfitting noise. For ECT runs
    # bump back to r=16 / α=32.
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ("q_proj", "v_proj", "o_proj")
    # LORA_TARGET_MODULES = ("q_proj", "v_proj")

    # Training loop
    # Delegate-task defaults from the spec. ECT defaults were lr=2e-5,
    # max_steps=2500, val_interval=150, checkpoint_steps=200; restore those
    # when flipping TASK_TYPE back to "explicit_confidence".
    LEARNING_RATE = 2e-5
    MAX_STEPS = 2500
    LOG_INTERVAL = 20
    VAL_INTERVAL = 100
    LIMIT_VAL_BATCHES = None
    VAL_NUM_SAMPLES = 250
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
    HIGH_ENTROPY_LOSS_WEIGHT = 1.0

    # Output / checkpointing
    OUTPUT_DIR = FINETUNE_OUTPUTS_DIR / "ect_lora"
    LOGS_DIR = FINETUNE_LOGS_DIR
    CHECKPOINTS_DIR = FINETUNE_CHECKPOINTS_DIR
    CHECKPOINT_STEPS = 100
    # When True, local checkpoint dirs are preserved on disk after the run.
    # When False, they are deleted after being uploaded to W&B / HF.
    # Default workflow: keep everything local during training, then use
    # finetune/push_checkpoint.py to push only the few checkpoints worth saving.
    KEEP_LOCAL_CHECKPOINTS = True
    SAVE_HF = False
    HF_REPO = None
    # Periodic HF checkpoint pushes during training. Off by default — see
    # finetune/push_checkpoint.py to upload selected local checkpoints
    # post-hoc instead, so HF doesn't fill up with throwaway runs.
    SAVE_HF_CHECKPOINTS = False
    HF_CHECKPOINT_REPO = None
    HF_CHECKPOINT_PRIVATE = False

    # Weights & Biases (training). Single project for both ECT and delegate
    # runs so they're directly comparable in one dashboard. The TASK_TYPE
    # prefix in WANDB_RUN_NAME makes the two cohorts trivially filterable.
    WANDB_PROJECT = "llm-metacognition-ect"
    # Run-name dataset tag, derived from TRAIN_DATA_PATH so swapping datasets
    # above automatically updates the W&B run name.
    # Naming convention:
    #   <prefix>_clean        / <prefix>_pop_heavy   →  <prefix>       (curated mix)
    #   <prefix>_max_balanced                        →  <prefix>_max   (per-source min-bin)
    #   <prefix>_all                                 →  <prefix>_all   (everything, no balancing)
    _DATASET_TAGS = {
        "balanced_metacognition":     "mixed_1494",
        "balanced_popMC":             "PopMC_balanced",
        "mixed_2550_clean":           "mixed_2550",
        "mixed_3150_pop_heavy":       "mixed_3150",
        "mixed_11931_max_balanced":   "mixed_11931_max",
        "mixed_17173_all":            "mixed_17173_all",
    }
    _DATASET_TAG = _DATASET_TAGS.get(
        Path(TRAIN_DATA_PATH).stem.removesuffix("_train"),
        "unknown_dataset",
    )
    # Run-name format depends on task. ECT path stays byte-identical to before
    # so existing dashboards / saved filters keep matching old runs. Delegate
    # runs get a different format that surfaces the delegate-specific knobs
    # (target source, teammate accuracy, tau) and drops the irrelevant
    # FINETUNING_TARGET/sigma fields.
    if TASK_TYPE == "explicit_confidence":
        WANDB_RUN_NAME = f"{_DATASET_TAG}_{FINETUNING_TARGET}_{LEARNING_RATE}_{LORA_TARGET_MODULES}_sigma{SIGMA}_Lora_{LORA_R}_{LORA_ALPHA}"
    else:
        # task_type is the leading tag so W&B's name filter "delegate*" picks
        # up every delegate run regardless of dataset.
        WANDB_RUN_NAME = (
            f"{TASK_TYPE}_{_DATASET_TAG}_{DELEGATE_TARGET_SOURCE}"
            f"_tm{DELEGATE_TRAIN_TEAMMATE_ACCURACY}_tau{DELEGATE_TAU}"
            f"_{LEARNING_RATE}_{LORA_TARGET_MODULES}_Lora_{LORA_R}_{LORA_ALPHA}"
        )
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
    EVAL_COMPUTE_OTHER_CONFIDENCE = True
    # Delegate game — four variants. Any combination can be enabled.
    #   EVAL_RUN_DELEGATE_ABCDT — single-shot. Model picks one of A/B/C/D
    #     (display letters follow MCQ_LETTER_SCHEME) to answer, or T to
    #     delegate. T is the LAST option in the visual block.
    #   EVAL_RUN_DELEGATE_AT    — binary. A = "answer myself", T = "delegate".
    #     The MC options are still shown so the model can judge difficulty,
    #     but it does NOT pick among them. T is the LAST option visually.
    #   EVAL_RUN_DELEGATE_TABCD — mirror of ABCDT with T listed FIRST in the
    #     option block. Used to test for first-position bias inflating the
    #     delegation rate when T is at the end.
    #   EVAL_RUN_DELEGATE_TA    — mirror of AT with T listed FIRST.
    EVAL_RUN_DELEGATE_ABCDT = True
    EVAL_RUN_DELEGATE_AT = True
    EVAL_RUN_DELEGATE_TABCD = True
    EVAL_RUN_DELEGATE_TA = True
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
