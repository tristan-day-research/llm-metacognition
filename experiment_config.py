"""
Centralized experiment configuration.

Single source of truth for the parameters of the active runners under
`interp_experiments/`. Each class below mirrors the CONFIGURATION block
of one runner — edit values here and the runner picks them up at import
time via `from experiment_config import <Cls> as _C`.

Driver scripts can still mutate the runner's module-level alias for one-off
overrides (e.g. swapping `_C.MODEL_NAME` between calls of
`run_introspection_experiment.run_single_experiment`); the alias overwrite
does NOT touch this file.

All paths are relative to the project root.
"""

from pathlib import Path


# =============================================================================
# Shared constants
# =============================================================================

# Backbones we actually load. Everything else has been retired — the project
# is currently scoped to Llama-3.1-8B (base + instruct) ± a LoRA adapter.
LLAMA_8B_BASE = "meta-llama/Llama-3.1-8B"
LLAMA_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"

# When set to a HF repo id (or local PEFT directory), runners that consult
# this constant will load it on top of LLAMA_8B_INSTRUCT for the "finetuned"
# pass. Leaving it None means base/instruct only.
DEFAULT_ADAPTER = None

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_SEED = 42

# Five uncertainty signals derived per-question from the direct (MC) pass.
# All five are extracted on every run; the METRIC knob below just selects
# which one drives the linear probe + saved probe direction.
#   "entropy"   — Shannon entropy over A/B/C/D probabilities. HIGH = uncertain.
#   "top_prob"  — P(argmax). HIGH = confident.
#   "margin"    — P(top) - P(second). HIGH = confident.
#   "logit_gap" — z(top) - z(second), in raw-logit space. HIGH = confident.
#                 Linear probe target: invariant to global logit shifts.
#   "top_logit" — z(top) - mean(z). HIGH = confident.
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]


# =============================================================================
# IntrospectionExperimentConfig
# Used by: interp_experiments/run_introspection_experiment.py
# =============================================================================
class IntrospectionExperimentConfig:
    """Activation + logit-lens + probe collection for one model at a time.

    The runner does ONE model per invocation. To collect base / instruct /
    finetuned, run it three times after editing (BASE_MODEL_NAME, MODEL_NAME):

        base       → BASE_MODEL_NAME = LLAMA_8B_BASE,     MODEL_NAME = LLAMA_8B_BASE
        instruct   → BASE_MODEL_NAME = LLAMA_8B_INSTRUCT, MODEL_NAME = LLAMA_8B_INSTRUCT
        finetuned  → BASE_MODEL_NAME = LLAMA_8B_INSTRUCT, MODEL_NAME = "<HF adapter id>"

    Output filenames already include a model + adapter tag, so the three
    runs don't collide on disk.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    # Backbone to load. Choose LLAMA_8B_BASE or LLAMA_8B_INSTRUCT. Base models
    # bypass the chat template and switch the runner to few-shot prompts.
    BASE_MODEL_NAME = LLAMA_8B_BASE

    # If equal to BASE_MODEL_NAME → no LoRA adapter loaded.
    # Otherwise → HF repo id (or local PEFT path) of the adapter to load on
    # top of BASE_MODEL_NAME for the "finetuned" run. BASE_MODEL_NAME must
    # be the instruct backbone when an adapter is set.
    MODEL_NAME = BASE_MODEL_NAME

    # Memory knobs. Llama-3.1-8B fits on a single ~24 GB GPU in fp16, so
    # both default to False. Flip on if you're running on a smaller card.
    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    # Each entry is either:
    #   1. A path to a .jsonl file on disk, OR
    #   2. A registered dataset NAME recognised by core.datasets.
    #
    # JSONL rows must contain at minimum:
    #     question      : str
    #     options       : {"A": str, "B": str, "C": str, "D": str}
    #     correct_answer: str  (or "correct_letter" — runner accepts either)
    # An "id" / "qid" field is optional; row index is used as a fallback.
    #
    # Registered names (from core.datasets.load_and_format_dataset):
    #   "GPQA", "GPSA", "MMLU", "TruthfulQA", "SimpleQA",
    #   "SimpleMC", "PopMC", "PopMC_0_difficulty_filtered", "TriviaMC",
    #   "Garupanese", "GarupaneseMC".
    DATASETS = ["data/mixed_11931_max_balanced.jsonl"]
    DATASET_NAME = DATASETS[0]  # default for single-run callers; runner sweeps DATASETS

    # Number of questions per dataset. Higher = more reliable probe and
    # contrast directions, more wall time.
    NUM_QUESTIONS_DEFAULT = 800

    # Per-dataset overrides. Key is the dataset name OR the .jsonl filename
    # stem (e.g. "mixed_11931_max_balanced"). Useful when one dataset is
    # smaller than NUM_QUESTIONS_DEFAULT (GPQA only has 447 rows).
    NUM_QUESTIONS_BY_DATASET = {"GPQA": 447}
    NUM_QUESTIONS = NUM_QUESTIONS_BY_DATASET.get(DATASET_NAME, NUM_QUESTIONS_DEFAULT)

    SEED = DEFAULT_SEED

    # ------------------------------------------------------------------
    # Meta tasks
    # ------------------------------------------------------------------

    # Which meta prompt(s) to run alongside the direct MC pass.
    #   "confidence" — model rates how confident it is on a numeric or
    #                  letter scale (see CONFIDENCE_SCALE).
    #   "delegate"   — model either answers (A/B/C/D) or delegates (T) to a
    #                  teammate of stated accuracy TEAMMATE_ACCURACY. With
    #                  CONFIDENCE_SCALE='numeric' / DELEGATE_PROMPT_DESIGN
    #                  ='mc_integrated' this is the single-shot ABCDT prompt.
    META_TASKS = ["confidence"]
    META_TASK = META_TASKS[0]

    # Confidence-task scale. Only consulted when META_TASK == "confidence".
    #   "numeric" — single-digit tokens 1..10 (Llama-3 tokenizes "10" as one token).
    #   "letters" — letter tokens S..Z (legacy 8-bin scale).
    CONFIDENCE_SCALE = "numeric"

    # Delegate-task prompt format. Only consulted when META_TASK == "delegate".
    #   "mc_integrated"  — single-shot A/B/C/D/T prompt. Instruct/finetuned only.
    #   "two_step_digit" — legacy 1-vs-2 binary delegation decision (digits).
    #                      Base models supported via few-shot.
    DELEGATE_PROMPT_DESIGN = "mc_integrated"

    # Teammate accuracy shown to the model in the delegate-task setup, also
    # used as the soft-label threshold above which "answer myself" is preferred.
    TEAMMATE_ACCURACY = 0.7

    # If True AND META_TASK == "delegate", reuse direct-task activations
    # cached from a prior confidence-mode run on the same (model, dataset,
    # seed, NUM_QUESTIONS). Skips one expensive forward pass per question.
    REUSE_DIRECT_FROM_CONFIDENCE = False

    # ------------------------------------------------------------------
    # Few-shot mode (only consulted for BASE models — instruct/finetuned use
    # the chat template instead).
    # ------------------------------------------------------------------

    #   "fixed"              — same handful of exemplars for every question.
    #   "balanced"           — exemplars sampled to balance across confidence
    #                          bins (requires BASE_DELEGATE_POOL_SOURCE for
    #                          delegate).
    #   "deceptive_examples" — exemplars chosen to mislead, used as a control.
    FEW_SHOT_MODE = "fixed"

    # Few-shot mode for the BASE delegate prompt specifically; same options
    # as FEW_SHOT_MODE. Only consulted when META_TASK == "delegate" and the
    # model is a base model.
    BASE_DELEGATE_MODE = "balanced"

    # JSON file with a pre-baked example pool for "balanced" base delegate
    # few-shot. None disables balanced mode. Path is relative to repo root.
    BASE_DELEGATE_POOL_SOURCE = None

    # ------------------------------------------------------------------
    # Probe / direction extraction
    # ------------------------------------------------------------------

    # Which uncertainty metric the linear (Ridge) probe targets. The runner
    # ALWAYS extracts all five metrics from direct activations; this just
    # picks which one drives the introspection probe + saved probe direction.
    # Choices: see AVAILABLE_METRICS at module level.
    METRIC = "entropy"
    AVAILABLE_METRICS = AVAILABLE_METRICS

    # Subset of AVAILABLE_METRICS for which "high value = uncertain" — used
    # to flip sign conventions in the calibration-split analysis. Only
    # entropy increases with uncertainty; the others increase with confidence.
    UNCERTAINTY_METRICS = {"entropy"}

    # Train/test split for the probe. 0.8 = 80% train / 20% test.
    TRAIN_SPLIT = 0.8

    # Ridge regression L2 penalty. 1000.0 is the well-tested default —
    # tighter than sklearn's 1.0 because activations are high-dimensional.
    PROBE_ALPHA = 1000.0

    # PCA dimensionality reduction before Ridge. Probes train and apply
    # faster in PCA space, and for hidden_dim=4096 with N≈800 samples PCA
    # also helps regularise.
    USE_PCA = True
    PCA_COMPONENTS = 100

    # ------------------------------------------------------------------
    # Contrast (mean-difference) direction extraction
    # ------------------------------------------------------------------

    # When True, after the probe-based directions are saved the runner also
    # extracts mean-difference directions per layer for two contrasts:
    #
    #   entropy_contrast            — top vs bottom CONTRAST_QUANTILE of
    #                                 direct-task entropy. Computed on
    #                                 DIRECT activations. Always extracted.
    #
    #   stated_confidence_contrast  — top vs bottom CONTRAST_QUANTILE of soft
    #                                 stated confidence (from the confidence
    #                                 meta prompt). Computed on META
    #                                 activations. Only extracted when
    #                                 META_TASK == "confidence" — there is no
    #                                 stated-confidence signal in delegate runs.
    #
    # Saved as `{directions_prefix}_<name>_contrast_directions.npz`, with
    # one (hidden_dim,) unit-norm vector per layer plus metadata fields.
    EXTRACT_CONTRAST_DIRECTIONS = True

    # Quantile threshold. With 0.25 we contrast the bottom 25% against the
    # top 25% of the signal (i.e. extremes); intermediate questions are
    # dropped from the mean-diff pool.
    CONTRAST_QUANTILE = 0.25

    OUTPUTS_DIR = OUTPUTS_DIR


# =============================================================================
# LogitLensConfig
# Used by: interp_experiments/run_logit_lens.py
# =============================================================================
class LogitLensConfig:
    """Single-question logit-lens diagnostic plot.

    Standalone helper — NOT part of the introspection collection pipeline
    (run_introspection_experiment.py already saves a per-layer logit-lens
    NPZ alongside its activations).
    """

    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    ADAPTER = None  # HF adapter repo id for the finetuned model, or None

    # If True, pull the test question from DATA_FILE by QID. Otherwise use
    # the inline TEST_QUESTION dict.
    USE_DATASET_QUESTION = True
    DATA_FILE = "data/TriviaMC.jsonl"
    QID = "triviamc_146"

    TEST_QUESTION = {
        "question": "What is the capital of France?",
        "options": {"A": "New York", "B": "Tokyo", "C": "Paris", "D": "Denver"},
        "correct_answer": "C",
    }

    # Prompt formatting:
    #   "direct_mc"           — A/B/C/D MC question.
    #   "stated_confidence"   — confidence prompt (S-Z scale).
    #   "other_confidence"    — "what % of college-educated people would know".
    #   "answer_or_delegate"  — legacy 1-vs-2 delegate prompt.
    #   "raw"                 — DEFAULT_PROMPT used verbatim.
    TASK_TYPE = "stated_confidence"
    DEFAULT_PROMPT = "The capital of France is"

    # Which residual stream to project through the lens.
    #   "resid_post" — output of each transformer block (standard "logit lens").
    #   "resid_pre"  — input of each transformer block.
    ACTIVATION_STREAM = "resid_post"

    # LayerNorm convention before unembed.
    #   "final_ln"      — apply the model's final RMSNorm (standard).
    #   "model_default" — same as final_ln on Llama (kept for API symmetry).
    #   "none"          — raw residuals, no LN.
    LN_MODE = "final_ln"

    # Token position to lens. "last" = final real token (left-padded), or
    # an integer index into the prompt.
    TOKEN_POSITION = "last"

    # Top-K tokens to report per layer.
    TOP_K = 20

    # Which layers to lens.
    #   "all"  — every layer.
    #   "auto" — [0, mid, last].
    #   "0,15,31" — comma-separated explicit indices.
    LAYERS_DEFAULT = "all"

    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    OUTPUTS_DIR = OUTPUTS_DIR
