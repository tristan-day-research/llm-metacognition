"""
Centralized experiment configuration.

Single source of truth for the parameters of each experiment runner under
`experiments/`. Each class below mirrors the CONFIGURATION block of one
runner — edit the value here and the runner picks it up.

The runners alias these values into module-level names at import time
(e.g. `MODEL = AblationCausalityConfig.MODEL`) so that:
  1. The rest of each runner's code is untouched.
  2. Driver scripts that mutate `rac.MODEL = ...` between runs still work
     (they overwrite the alias, not this class).

If you want a permanent change for a knob, edit it here. If you want a
one-off override (e.g. `--dry-run`), use a driver script that mutates
the runner's module-level name like `run_all_stated_conf_ablations.py` does.

All paths are relative to the project root.
"""

from pathlib import Path


# ---- shared defaults ----------------------------------------------------------
LLAMA_8B_BASE = "meta-llama/Llama-3.1-8B"
LLAMA_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_ADAPTER = None  # set to a HF adapter repo id when one is published

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_SEED = 42
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]


# ---- run_introspection_experiment.py -----------------------------------------
class IntrospectionExperimentConfig:
    BASE_MODEL_NAME = LLAMA_8B_BASE
    MODEL_NAME = BASE_MODEL_NAME

    DATASETS = ["TriviaMC", "SimpleMC"]
    META_TASKS = ["delegate"]
    DATASET_NAME = DATASETS[0]
    META_TASK = META_TASKS[0]

    NUM_QUESTIONS_DEFAULT = 800
    NUM_QUESTIONS_BY_DATASET = {"GPQA": 447}
    NUM_QUESTIONS = NUM_QUESTIONS_BY_DATASET.get(DATASET_NAME, NUM_QUESTIONS_DEFAULT)

    SEED = DEFAULT_SEED
    LOAD_IN_4BIT = "70B" in BASE_MODEL_NAME
    LOAD_IN_8BIT = False

    FEW_SHOT_MODE = "fixed"
    BASE_DELEGATE_MODE = "balanced"
    BASE_DELEGATE_POOL_SOURCE = None
    TEAMMATE_ACCURACY = 0.7
    DELEGATE_PROMPT_DESIGN = "mc_integrated"
    REUSE_DIRECT_FROM_CONFIDENCE = False
    CONFIDENCE_SCALE = "numeric"

    OUTPUTS_DIR = OUTPUTS_DIR
    TRAIN_SPLIT = 0.8
    PROBE_ALPHA = 1000.0
    USE_PCA = True
    PCA_COMPONENTS = 100

    LOAD_PRETRAINED_PROBE = False
    PRETRAINED_PROBE_PATH = "mc_probe_trained.pkl"

    AVAILABLE_METRICS = AVAILABLE_METRICS
    UNCERTAINTY_METRICS = {"entropy"}
    METRIC = "entropy"


# ---- run_introspection_probe.py ----------------------------------------------
class IntrospectionProbeConfig:
    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    MODEL_NAME = BASE_MODEL_NAME

    DATASETS = ["SimpleMC", "TriviaMC"]
    META_TASKS = ["confidence", "delegate"]
    DATASET_NAME = DATASETS[0]
    META_TASK = META_TASKS[0]

    OUTPUTS_DIR = OUTPUTS_DIR
    AVAILABLE_METRICS = AVAILABLE_METRICS
    METRIC = "logit_gap"


# ---- run_introspection_steering.py -------------------------------------------
class IntrospectionSteeringConfig:
    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    MODEL_NAME = DEFAULT_ADAPTER
    DATASET_NAME = "SimpleMC"

    DIRECTION_TYPE = "entropy"
    AVAILABLE_METRICS = AVAILABLE_METRICS
    METRIC = "entropy"

    D2M_R2_THRESHOLD = 0.20
    D2D_R2_THRESHOLD = D2M_R2_THRESHOLD * 1.5
    META_R2_THRESHOLD = D2M_R2_THRESHOLD

    META_TASK = "confidence"
    OUTPUTS_DIR = OUTPUTS_DIR


# ---- run_introspection_direction_experiment.py -------------------------------
class IntrospectionDirectionConfig:
    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    MODEL_NAME = BASE_MODEL_NAME
    DATASET_NAME = "SimpleMC"
    OUTPUTS_DIR = OUTPUTS_DIR


# ---- run_ablation_causality.py -----------------------------------------------
class AblationCausalityConfig:
    MODEL = LLAMA_8B_INSTRUCT
    ADAPTER = None
    INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC"

    DIRECTION_METRIC = "entropy"
    TARGET_METRIC = "entropy"
    META_TASK = "delegate"

    CONFIDENCE_SCALE = "numeric"
    BASE_CONFIDENCE_FEW_SHOT_MODE = "fixed"
    CONFIDENCE_SIGNAL = "prob"

    NUM_QUESTIONS = 100
    NUM_CONTROLS = 25
    BATCH_SIZE = 8
    SEED = DEFAULT_SEED

    USE_TRANSFER_SPLIT = True
    TRAIN_SPLIT = 0.8
    EXPANDED_BATCH_TARGET = 192

    LAYERS = None
    METHODS = ["mean_diff", "probe"]
    PROBE_POSITIONS = ["final"]

    PRINT_DELTA_DIAGNOSTICS = True
    DELTA_DIAGNOSTIC_TOPK = 5

    BOOTSTRAP_N = 2000
    BOOTSTRAP_SEED = 12345
    BOOTSTRAP_CI_ALPHA = 0.05

    TRANSFER_R2_THRESHOLD = 0.3
    TRANSFER_RESULTS_PATH = None
    NUM_CONTROLS_NONFINAL = 10

    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    OUTPUT_DIR = OUTPUTS_DIR


# ---- run_steering_causality.py -----------------------------------------------
class SteeringCausalityConfig:
    MODEL = LLAMA_8B_INSTRUCT
    ADAPTER = None
    INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC"
    METRIC = "entropy"
    META_TASK = "delegate"

    STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    NUM_QUESTIONS = 100
    NUM_CONTROLS = 25
    BATCH_SIZE = 8
    SEED = DEFAULT_SEED

    USE_TRANSFER_SPLIT = True
    TRAIN_SPLIT = 0.8
    EXPANDED_BATCH_TARGET = 48

    LAYERS = None
    METHODS = None
    PROBE_POSITIONS = ["final"]

    TRANSFER_R2_THRESHOLD = 0.5
    TRANSFER_RESULTS_PATH = None
    NUM_CONTROLS_NONFINAL = 10

    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    OUTPUT_DIR = OUTPUTS_DIR


# ---- run_mc_answer_ablation.py -----------------------------------------------
class MCAnswerAblationConfig:
    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    MODEL_NAME = BASE_MODEL_NAME
    DATASET_NAME = "TriviaMC"

    AVAILABLE_METRICS = AVAILABLE_METRICS
    METRIC = "entropy"
    REVERSE_MODE = False
    META_TASK = "confidence"

    D2M_R2_THRESHOLD = 0.20
    D2D_R2_THRESHOLD = D2M_R2_THRESHOLD * 1.5

    OUTPUTS_DIR = OUTPUTS_DIR

    ABLATION_LAYERS = None
    NUM_QUESTIONS = 500
    NUM_CONTROL_DIRECTIONS = 25
    FDR_ALPHA = 0.05
    FDR_SAFETY_FACTOR = 25
    MIN_CONTROLS_PER_LAYER = 10

    BATCH_SIZE = 1
    VARIANT_BATCH_SIZE = 10
    INTERVENTION_POSITION = "last"

    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    SEED = DEFAULT_SEED


# ---- run_activation_patching.py ----------------------------------------------
class ActivationPatchingConfig:
    BASE_MODEL_NAME = LLAMA_8B_INSTRUCT
    MODEL_NAME = BASE_MODEL_NAME
    DATASET_NAME = "SimpleMC"
    META_TASK = "confidence"

    NUM_PATCH_PAIRS = 100
    PAIRING_METHOD = "extremes"
    BATCH_SIZE = 8

    PATCHING_LAYERS = None
    AVAILABLE_METRICS = AVAILABLE_METRICS
    METRIC = "logit_gap"

    OUTPUTS_DIR = OUTPUTS_DIR
    SEED = DEFAULT_SEED


# ---- run_contrast_intervention.py --------------------------------------------
class ContrastInterventionConfig:
    MODEL = LLAMA_8B_INSTRUCT
    ADAPTER = None

    DIRECTION_A_NAME = "entropy"
    DIRECTION_A_PATH = "outputs/confidence_contrast/Llama-3.1-8B-Instruct_TriviaMC_confidence_contrast_directions.pt"
    DIRECTION_B_NAME = "confidence"
    DIRECTION_B_PATH = "outputs/confidence_contrast/Llama-3.1-8B-Instruct_TriviaMC_mc_entropy_directions.npz"

    NPZ_METHOD = "mean_diff"
    DATASET_PATH = "data/TriviaMC.jsonl"

    NUM_QUESTIONS = 100
    BATCH_SIZE = 8
    SEED = DEFAULT_SEED
    NUM_RANDOM_DIRECTIONS = 2

    USE_TRANSFER_SPLIT = True
    TRAIN_SPLIT = 0.8
    STEERING_MULTIPLIERS = [-3.0, -2.0, 0.0, 2.0, 3.0]

    LAYERS = None
    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    OUTPUT_DIR = OUTPUTS_DIR

    MC_OPTION_STRINGS = [" A", " B", " C", " D"]
    MC_OPTION_LETTERS = ["A", "B", "C", "D"]
    CONF_OPTION_STRINGS = [" S", " T", " U", " V", " W", " X", " Y", " Z"]
    CONF_OPTION_LETTERS = ["S", "T", "U", "V", "W", "X", "Y", "Z"]
    AUTO_DETECT_OPTION_FORMAT = True

    DEBUG_EXAMPLES = 5
    REQUIRE_COVERAGE_THRESHOLD = 0.5


# ---- run_contrastive_direction.py --------------------------------------------
class ContrastiveDirectionConfig:
    BASE_MODEL_NAME = LLAMA_70B_INSTRUCT
    MODEL_NAME = BASE_MODEL_NAME
    DATASET_NAME = "SimpleMC"
    SEED = DEFAULT_SEED

    VALID_METRICS = AVAILABLE_METRICS
    METRIC_HIGHER_IS_CONFIDENT = {
        "entropy": False,
        "top_prob": True,
        "margin": True,
        "logit_gap": True,
        "top_logit": True,
    }
    METRIC_KEY_MAP = {
        "entropy": "direct_entropies",
        "top_prob": "direct_top_probs",
        "margin": "direct_margins",
        "logit_gap": "direct_logit_gaps",
        "top_logit": "direct_top_logits",
    }
    METRIC = "top_logit"
    DIRECTION_TYPES = ["calibration"]
    OUTPUTS_DIR = OUTPUTS_DIR


# ---- run_logit_lens.py -------------------------------------------------------
class LogitLensConfig:
    BASE_MODEL_NAME = LLAMA_70B_INSTRUCT
    ADAPTER = None

    USE_DATASET_QUESTION = True
    DATA_FILE = "data/TriviaMC.jsonl"
    QID = "triviamc_146"

    TEST_QUESTION = {
        "question": "What is the capital of France?",
        "options": {"A": "New York", "B": "Tokyo", "C": "Paris", "D": "Denver"},
        "correct_answer": "C",
    }

    TASK_TYPE = "stated_confidence"
    DEFAULT_PROMPT = "The capital of France is"

    ACTIVATION_STREAM = "resid_post"
    LN_MODE = "final_ln"
    TOKEN_POSITION = "last"
    TOP_K = 20
    LAYERS_DEFAULT = "all"

    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    OUTPUTS_DIR = OUTPUTS_DIR


# ---- run_confidence_tests.py -------------------------------------------------
class ConfidenceTestsConfig:
    MODEL_TYPE = "base"
    FEW_SHOT_MODE = "deceptive_examples"
    RANDOM_FEW_SHOT_SOURCE = "outputs/ECT/Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000_TriviaMC_ect_results.json"

    MODEL_CONFIGS = {
        "base": {"model": LLAMA_8B_BASE, "adapter": None},
        "instruct": {"model": LLAMA_8B_INSTRUCT, "adapter": None},
        "finetuned": {"model": LLAMA_8B_INSTRUCT, "adapter": DEFAULT_ADAPTER},
    }

    DATASET = "TriviaMC"
    NUM_QUESTIONS = 500
    SEED = DEFAULT_SEED
    BATCH_SIZE = 8
    LOAD_IN_4BIT = False
    LOAD_IN_8BIT = False
    OUTPUT_DIR = Path(__file__).parent / "outputs"
