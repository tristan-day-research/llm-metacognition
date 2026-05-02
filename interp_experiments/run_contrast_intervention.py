"""
Causal intervention experiments for contrast directions (entropy vs confidence).

Tests directional ASYMMETRY between:
1. entropy_direction (from entropy contrast on MC questions)
2. confidence_direction (from stated confidence contrast)

Experiments:
A) STEERING: Add +/- alpha * direction at chosen layer
B) ABLATION: Project out direction component at chosen layer

Key question: Does intervening on entropy_direction change stated confidence,
and does intervening on confidence_direction change output entropy?

This tests whether the two directions capture the same or different information.

Usage:
    python run_contrast_intervention.py

Expects contrast directions from confidence_contrast.py:
    outputs/.../Llama-X_TriviaMC_entropy_contrast_directions.pt
    outputs/.../Llama-X_TriviaMC_confidence_contrast_directions.pt
    
And a dataset file (either format works):
    data/TriviaMC.jsonl  (raw: {"question", "correct_answer", "distractors"})
    outputs/Llama-X_TriviaMC_mc_dataset.json  (preprocessed from identify_mc_correlate.py)

Example command:
    python run_contrast_intervention.py  # uses default config below
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, linregress
import warnings

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from core.steering import generate_orthogonal_directions
from core.steering_experiments import (
    SteeringExperimentConfig,
    BatchSteeringHook,
    BatchAblationHook,
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
)
from core.metrics import compute_entropy
from prompts import (
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    format_answer_or_delegate_prompt,
    get_answer_or_delegate_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_direct_prompt,
)

# =============================================================================
# CONFIGURATION — edit values in experiment_config.ContrastInterventionConfig
# =============================================================================
from experiment_config import ContrastInterventionConfig as _C

MODEL = _C.MODEL
ADAPTER = _C.ADAPTER
DIRECTION_A_NAME = _C.DIRECTION_A_NAME
DIRECTION_A_PATH = _C.DIRECTION_A_PATH
DIRECTION_B_NAME = _C.DIRECTION_B_NAME
DIRECTION_B_PATH = _C.DIRECTION_B_PATH
NPZ_METHOD = _C.NPZ_METHOD
DATASET_PATH = _C.DATASET_PATH
NUM_QUESTIONS = _C.NUM_QUESTIONS
BATCH_SIZE = _C.BATCH_SIZE
SEED = _C.SEED
NUM_RANDOM_DIRECTIONS = _C.NUM_RANDOM_DIRECTIONS
USE_TRANSFER_SPLIT = _C.USE_TRANSFER_SPLIT
TRAIN_SPLIT = _C.TRAIN_SPLIT
STEERING_MULTIPLIERS = list(_C.STEERING_MULTIPLIERS)
LAYERS = _C.LAYERS
LOAD_IN_4BIT = _C.LOAD_IN_4BIT
LOAD_IN_8BIT = _C.LOAD_IN_8BIT
OUTPUT_DIR = _C.OUTPUT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

MC_OPTION_STRINGS = list(_C.MC_OPTION_STRINGS)
MC_OPTION_LETTERS = list(_C.MC_OPTION_LETTERS)
CONF_OPTION_STRINGS = list(_C.CONF_OPTION_STRINGS)
CONF_OPTION_LETTERS = list(_C.CONF_OPTION_LETTERS)
AUTO_DETECT_OPTION_FORMAT = _C.AUTO_DETECT_OPTION_FORMAT
DEBUG_EXAMPLES = _C.DEBUG_EXAMPLES
REQUIRE_COVERAGE_THRESHOLD = _C.REQUIRE_COVERAGE_THRESHOLD

# =============================================================================
# OPTION TOKENIZATION HELPERS
# =============================================================================


@dataclass
class OptionTokenInfo:
    """Information about how an option string tokenizes."""
    option_string: str
    token_ids: List[int]
    single_token: bool
    decoded: str  # What this decodes back to (for verification)


def get_option_token_ids(
    option_strings: List[str],
    tokenizer,
    verbose: bool = False
) -> Dict[str, OptionTokenInfo]:
    """
    Get token IDs for each option string, detecting single vs multi-token.
    
    Args:
        option_strings: List of option strings (e.g., [" A", " B", " C", " D"])
        tokenizer: The tokenizer to use
        verbose: Print detailed info
        
    Returns:
        Dict mapping option string to OptionTokenInfo
    """
    result = {}
    
    for opt_str in option_strings:
        token_ids = tokenizer.encode(opt_str, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        single_token = len(token_ids) == 1
        
        result[opt_str] = OptionTokenInfo(
            option_string=opt_str,
            token_ids=token_ids,
            single_token=single_token,
            decoded=decoded
        )
        
        if verbose:
            status = "SINGLE" if single_token else f"MULTI({len(token_ids)})"
            print(f"  '{opt_str}' -> {token_ids} -> '{decoded}' [{status}]")
    
    return result


def check_all_single_token(option_info: Dict[str, OptionTokenInfo]) -> bool:
    """Check if all options tokenize to single tokens."""
    return all(info.single_token for info in option_info.values())


def get_singleton_token_ids(option_info: Dict[str, OptionTokenInfo]) -> List[int]:
    """
    Get single token IDs for all options (only valid if all are single-token).
    
    Raises:
        ValueError: If any option is multi-token
    """
    if not check_all_single_token(option_info):
        multi = [k for k, v in option_info.items() if not v.single_token]
        raise ValueError(f"Cannot get singleton IDs: multi-token options: {multi}")
    
    # Return in the order of the original option_info dict
    return [info.token_ids[0] for info in option_info.values()]


def auto_detect_option_strings(
    candidate_strings_list: List[List[str]],
    tokenizer,
    verbose: bool = True
) -> Tuple[List[str], Dict[str, OptionTokenInfo], bool]:
    """
    Try different option string formats and select the one with most single-token options.
    
    Args:
        candidate_strings_list: List of candidate string lists to try
                               e.g., [[" A", " B", ...], ["A", "B", ...], ["\nA", "\nB", ...]]
        tokenizer: The tokenizer
        verbose: Print detection results
        
    Returns:
        (selected_strings, option_info, all_single_token)
    """
    best_strings = None
    best_info = None
    best_single_count = -1
    
    for candidates in candidate_strings_list:
        info = get_option_token_ids(candidates, tokenizer, verbose=False)
        single_count = sum(1 for v in info.values() if v.single_token)
        
        if single_count > best_single_count:
            best_single_count = single_count
            best_strings = candidates
            best_info = info
    
    all_single = best_single_count == len(best_strings)
    
    if verbose:
        print(f"  Selected format: {best_strings}")
        print(f"  Single-token options: {best_single_count}/{len(best_strings)}")
        for opt_str, info in best_info.items():
            status = "OK" if info.single_token else f"MULTI({len(info.token_ids)})"
            print(f"    '{opt_str}' -> {info.token_ids} [{status}]")
    
    return best_strings, best_info, all_single


def score_options_next_token(
    logits: torch.Tensor,
    option_token_ids: List[int],
) -> np.ndarray:
    """
    Score options using next-token logits (only for single-token options).
    
    Args:
        logits: Logits tensor of shape (vocab_size,) or (batch, vocab_size)
        option_token_ids: List of single token IDs for each option
        
    Returns:
        Probabilities over options, shape (num_options,) or (batch, num_options)
    """
    if logits.dim() == 1:
        option_logits = logits[option_token_ids]
        probs = torch.softmax(option_logits, dim=-1)
        return probs.cpu().numpy()
    else:
        option_logits = logits[:, option_token_ids]
        probs = torch.softmax(option_logits, dim=-1)
        return probs.cpu().numpy()


def score_options_sequence_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_token_sequences: List[List[int]],
) -> np.ndarray:
    """
    Score options by computing log-probability of each sequence continuation.
    
    For multi-token options, we compute the log-prob of the full sequence
    under teacher forcing and softmax across options.
    
    Args:
        model: The language model
        input_ids: Input token IDs, shape (1, seq_len)
        attention_mask: Attention mask, shape (1, seq_len)
        option_token_sequences: List of token ID sequences for each option
        
    Returns:
        Probabilities over options, shape (num_options,)
    """
    log_probs = []
    
    with torch.no_grad():
        for token_seq in option_token_sequences:
            # Append the option sequence to the input
            seq_tensor = torch.tensor([token_seq], device=input_ids.device)
            full_input = torch.cat([input_ids, seq_tensor], dim=1)
            full_mask = torch.cat([
                attention_mask,
                torch.ones(1, len(token_seq), device=attention_mask.device)
            ], dim=1)
            
            # Get logits
            outputs = model(input_ids=full_input, attention_mask=full_mask)
            logits = outputs.logits  # (1, full_seq_len, vocab)
            
            # Compute log prob of each token in the sequence
            total_log_prob = 0.0
            prompt_len = input_ids.shape[1]
            
            for i, target_token in enumerate(token_seq):
                # Position i of the sequence is predicted by logits at position prompt_len - 1 + i
                pos = prompt_len - 1 + i
                token_logits = logits[0, pos]
                log_softmax = torch.log_softmax(token_logits, dim=-1)
                total_log_prob += log_softmax[target_token].item()
            
            log_probs.append(total_log_prob)
    
    # Convert log probs to probabilities via softmax
    log_probs = np.array(log_probs)
    probs = np.exp(log_probs - np.max(log_probs))  # Numerical stability
    probs = probs / probs.sum()
    
    return probs


# =============================================================================
# COVERAGE AND SANITY CHECK UTILITIES
# =============================================================================


def check_greedy_in_options(
    logits: torch.Tensor,
    option_token_ids: List[int],
) -> Tuple[bool, int, int]:
    """
    Check if the greedy (argmax) next token is among the allowed options.
    
    Args:
        logits: Logits for next token, shape (vocab_size,)
        option_token_ids: List of allowed token IDs
        
    Returns:
        (is_covered, greedy_token_id, greedy_rank_in_options)
        - is_covered: True if greedy token is in option_token_ids
        - greedy_token_id: The actual argmax token ID
        - greedy_rank_in_options: If covered, the index in option_token_ids; else -1
    """
    greedy_id = logits.argmax().item()
    
    if greedy_id in option_token_ids:
        rank = option_token_ids.index(greedy_id)
        return True, greedy_id, rank
    else:
        return False, greedy_id, -1


def print_debug_example(
    example_idx: int,
    prompt_type: str,
    logits: torch.Tensor,
    option_token_ids: List[int],
    option_strings: List[str],
    tokenizer,
):
    """Print detailed debug info for a single example."""
    greedy_id = logits.argmax().item()
    greedy_token = tokenizer.decode([greedy_id])
    
    # Get probs for options
    option_logits = logits[option_token_ids]
    probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
    
    # Check coverage
    covered, _, rank = check_greedy_in_options(logits, option_token_ids)
    coverage_str = f"YES (rank={rank})" if covered else f"NO (greedy='{greedy_token}' id={greedy_id})"
    
    print(f"  [{prompt_type}] Example {example_idx}:")
    print(f"    Greedy next token: '{greedy_token}' (id={greedy_id}), In options: {coverage_str}")
    print(f"    Option probs: " + ", ".join(f"{s}:{p:.3f}" for s, p in zip(option_strings, probs)))


# =============================================================================
# DIRECTION LOADING
# =============================================================================


def load_contrast_directions(
    path_a: str,
    path_b: str,
    name_a: str,
    name_b: str,
    npz_method: str = "mean_diff",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], str, str]:
    """
    Load two contrast directions from .pt or .npz files.
    
    Args:
        path_a: Path to first direction file (.pt or .npz)
        path_b: Path to second direction file (.pt or .npz)
        name_a: Label for first direction (e.g., "entropy")
        name_b: Label for second direction (e.g., "confidence")
        npz_method: Method to use from .npz files ("mean_diff" or "probe")
    
    Returns:
        directions_a: {layer: direction_vector}
        directions_b: {layer: direction_vector}
        name_a: Label for first direction
        name_b: Label for second direction
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    
    if not path_a.exists():
        raise FileNotFoundError(f"Direction A not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Direction B not found: {path_b}")
    
    print(f"Loading directions:")
    print(f"  {name_a}: {path_a}")
    print(f"  {name_b}: {path_b}")
    
    # Load tensors (shape: num_layers, hidden_dim) - support both .npz and .pt formats
    def extract_layer_from_npz(data, method: str):
        """Extract layer vectors from npz file for specified method and return as tensor.
        
        Args:
            data: Loaded npz file
            method: Method to extract ("mean_diff" or "probe")
            
        Returns:
            Tensor of shape (num_layers, hidden_dim)
        """
        # Get all keys for the specified method: "{method}_layer_{N}"
        layer_dict = {}
        for key in data.files:
            if key.startswith("_metadata"):
                continue
            # Parse key format: "{method}_layer_{N}" or "{method}_scaler_*"
            parts = key.rsplit("_layer_", 1)
            if len(parts) == 2:
                key_method, layer_str = parts
                if key_method == method:
                    try:
                        layer = int(layer_str)
                        layer_dict[layer] = data[key]
                    except ValueError:
                        continue
        
        if not layer_dict:
            available_methods = set()
            for key in data.files:
                if not key.startswith("_"):
                    parts = key.rsplit("_layer_", 1)
                    if len(parts) == 2:
                        available_methods.add(parts[0])
            raise ValueError(
                f"No directions found for method '{method}'. "
                f"Available methods: {sorted(available_methods)}"
            )
        
        # Sort by layer and stack into tensor
        num_layers = max(layer_dict.keys()) + 1
        vectors = []
        for layer in range(num_layers):
            if layer not in layer_dict:
                raise ValueError(f"Missing layer {layer} for method '{method}'")
            vectors.append(layer_dict[layer])
        
        return torch.from_numpy(np.stack(vectors))
    
    if path_a.suffix == ".npz":
        data_a = np.load(path_a)
        tensor_a = extract_layer_from_npz(data_a, npz_method)
        print(f"  {name_a}: Using '{npz_method}' method from .npz")
    else:
        tensor_a = torch.load(path_a, weights_only=True)
    
    if path_b.suffix == ".npz":
        data_b = np.load(path_b)
        tensor_b = extract_layer_from_npz(data_b, npz_method)
        print(f"  {name_b}: Using '{npz_method}' method from .npz")
    else:
        tensor_b = torch.load(path_b, weights_only=True)
    
    num_layers_a = tensor_a.shape[0]
    num_layers_b = tensor_b.shape[0]
    
    if num_layers_a != num_layers_b:
        raise ValueError(f"Layer count mismatch: {name_a} has {num_layers_a}, {name_b} has {num_layers_b}")
    
    num_layers = num_layers_a
    hidden_dim = tensor_a.shape[1]
    
    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")
    
    # Convert to dict format with normalized directions
    directions_a = {}
    directions_b = {}
    
    for layer in range(num_layers):
        # Normalize to unit vectors
        d_a = tensor_a[layer].numpy()
        norm_a = np.linalg.norm(d_a)
        if norm_a > 0:
            d_a = d_a / norm_a
        directions_a[layer] = d_a
        
        d_b = tensor_b[layer].numpy()
        norm_b = np.linalg.norm(d_b)
        if norm_b > 0:
            d_b = d_b / norm_b
        directions_b[layer] = d_b
    
    return directions_a, directions_b, name_a, name_b


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSONL file.
    
    Supports two formats:
    1. Raw JSONL: {"qid", "question", "correct_answer", "distractors"}
    2. Preprocessed JSON: {"data": [{"question", "options", "correct_idx"}]}
    
    Returns list of questions in standardized format with options as dict.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    questions = []
    option_letters = ['A', 'B', 'C', 'D']
    
    if path.suffix == ".jsonl":
        # Raw JSONL format
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                # Convert to standardized format
                correct_answer = item["correct_answer"]
                distractors = item["distractors"]
                
                # Randomize option order (but track correct letter)
                all_options = [correct_answer] + distractors
                np.random.shuffle(all_options)
                
                # Convert list to dict with letter keys
                options_dict = {}
                correct_letter = None
                for i, opt in enumerate(all_options):
                    letter = option_letters[i]
                    options_dict[letter] = opt
                    if opt == correct_answer:
                        correct_letter = letter
                
                questions.append({
                    "qid": item.get("qid", f"q_{len(questions)}"),
                    "question": item["question"],
                    "options": options_dict,
                    "correct_answer": correct_letter,
                })
    else:
        # Preprocessed JSON format (from identify_mc_correlate.py)
        with open(path, "r") as f:
            data = json.load(f)
        raw_questions = data.get("data", data)
        
        # Ensure options are in dict format
        for q in raw_questions:
            if isinstance(q.get("options"), list):
                # Convert list to dict
                options_dict = {}
                correct_letter = None
                for i, opt in enumerate(q["options"]):
                    letter = option_letters[i]
                    options_dict[letter] = opt
                    if q.get("correct_idx") is not None and i == q["correct_idx"]:
                        correct_letter = letter
                q["options"] = options_dict
                if correct_letter:
                    q["correct_answer"] = correct_letter
        questions = raw_questions
    
    return questions


# =============================================================================
# METRIC COMPUTATION
# =============================================================================


def compute_mc_entropy_from_logits(logits: np.ndarray) -> float:
    """Compute Shannon entropy over MC option logits (A/B/C/D)."""
    # Softmax
    logits = logits - logits.max()  # Numerical stability
    probs = np.exp(logits) / np.exp(logits).sum()
    # Entropy
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_logit_gap(logits: np.ndarray) -> float:
    """Compute gap between top two logits."""
    sorted_logits = np.sort(logits)[::-1]
    return float(sorted_logits[0] - sorted_logits[1])


def compute_mc_max_prob(logits: np.ndarray) -> float:
    """Compute max softmax probability over MC option logits."""
    # Softmax
    logits = logits - logits.max()  # Numerical stability
    probs = np.exp(logits) / np.exp(logits).sum()
    return float(probs.max())


# =============================================================================
# INTERVENTION EXPERIMENTS
# =============================================================================


def run_cross_intervention_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    directions_a: Dict[int, np.ndarray],
    directions_b: Dict[int, np.ndarray],
    name_a: str,
    name_b: str,
    num_controls: int,
    layers: Optional[List[int]],
    multipliers: List[float],
    use_chat_template: bool,
    intervention_type: str = "steering",  # "steering" or "ablation"
    mc_option_info: Optional[Dict[str, OptionTokenInfo]] = None,
    conf_option_info: Optional[Dict[str, OptionTokenInfo]] = None,
) -> Dict:
    """
    Run cross-intervention experiment to test asymmetry.
    
    For each direction (entropy, confidence, random):
    - Apply intervention
    - Measure BOTH stated confidence AND output entropy
    
    This reveals whether the directions capture different information.
    
    Returns:
        Results dict with per-layer, per-direction, per-multiplier metrics
    """
    if layers is None:
        layers = sorted(directions_a.keys())
    
    # Use pre-computed option info or compute now
    if mc_option_info is None:
        mc_option_info = get_option_token_ids(MC_OPTION_STRINGS, tokenizer)
    if conf_option_info is None:
        conf_option_info = get_option_token_ids(CONF_OPTION_STRINGS, tokenizer)
    
    # Extract option strings and token IDs
    mc_option_strings = list(mc_option_info.keys())
    conf_option_strings = list(conf_option_info.keys())
    
    # Check if we can use fast single-token scoring
    mc_single_token = check_all_single_token(mc_option_info)
    conf_single_token = check_all_single_token(conf_option_info)
    
    if mc_single_token:
        mc_token_ids = get_singleton_token_ids(mc_option_info)
    else:
        mc_token_ids = None  # Will need sequence scoring
        warnings.warn("MC options are multi-token - using sequence log-prob scoring (slower)")
    
    if conf_single_token:
        conf_token_ids = get_singleton_token_ids(conf_option_info)
    else:
        conf_token_ids = None
        warnings.warn("Confidence options are multi-token - using sequence log-prob scoring (slower)")
    
    # For display purposes, use the letter labels
    mc_options_display = MC_OPTION_LETTERS
    conf_options_display = CONF_OPTION_LETTERS
    
    # Format prompts for BOTH tasks
    # We need to run each question through both the MC task and confidence task
    mc_prompts = []
    conf_prompts = []
    conf_mappings = []  # For delegate task if used
    
    for q_idx, question in enumerate(questions):
        # MC prompt (direct)
        mc_prompt, _ = format_direct_prompt(question, tokenizer, use_chat_template)
        mc_prompts.append(mc_prompt)
        
        # Confidence prompt (stated confidence)
        conf_prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
        conf_prompts.append(conf_prompt)
        conf_mappings.append(None)  # No mapping for stated confidence
    
    # Pretokenize
    mc_cached = pretokenize_prompts(mc_prompts, tokenizer, DEVICE)
    conf_cached = pretokenize_prompts(conf_prompts, tokenizer, DEVICE)
    
    mc_batches = build_padded_gpu_batches(mc_cached, tokenizer, DEVICE, BATCH_SIZE)
    conf_batches = build_padded_gpu_batches(conf_cached, tokenizer, DEVICE, BATCH_SIZE)
    
    # Precompute direction tensors
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = {}
    
    for layer in layers:
        dir_a = torch.tensor(directions_a[layer], dtype=dtype, device=DEVICE)
        dir_b = torch.tensor(directions_b[layer], dtype=dtype, device=DEVICE)
        
        # Generate random control directions (orthogonal to direction A)
        random_dirs = generate_orthogonal_directions(directions_a[layer], num_controls, seed=SEED + layer)
        random_tensors = [torch.tensor(rd, dtype=dtype, device=DEVICE) for rd in random_dirs]
        
        cached_directions[layer] = {
            name_a: dir_a,
            name_b: dir_b,
            "random": random_tensors,
        }
    
    # Initialize results storage
    results = {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "intervention_type": intervention_type,
        "direction_names": [name_a, name_b],
        "layer_results": {},
        "scoring_mode_mc": "single_token" if mc_single_token else "sequence",
        "scoring_mode_conf": "single_token" if conf_single_token else "sequence",
        "mc_option_strings": mc_option_strings,
        "conf_option_strings": conf_option_strings,
    }
    
    # Coverage tracking
    mc_coverage_count = 0
    conf_coverage_count = 0
    
    # For ablation, we only have one "multiplier" (just ablate)
    if intervention_type == "ablation":
        effective_multipliers = [1.0]  # Placeholder
    else:
        effective_multipliers = [m for m in multipliers if m != 0.0]
    
    nonzero_mult = len(effective_multipliers)
    
    print(f"\nRunning {intervention_type} experiment...")
    print(f"  Layers: {len(layers)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Directions: {name_a}, {name_b}, {num_controls} random")
    if intervention_type == "steering":
        print(f"  Multipliers: {effective_multipliers}")
    
    # Total batches for progress bar (counting each batch for each task)
    num_batches = len(mc_batches)
    num_multipliers = len(effective_multipliers) if intervention_type == "steering" else 1
    # Baseline: 2 tasks * num_batches
    # Per layer, per multiplier: 2 directions * 2 tasks * num_batches + num_controls * 2 tasks * num_batches
    total_batches = (2 * num_batches) + len(layers) * num_multipliers * (2 * 2 * num_batches + num_controls * 2 * num_batches)
    pbar = tqdm(total=total_batches, desc=f"  {intervention_type.capitalize()}")
    current_stage = ""
    
    def update_stage(stage_name):
        """Update progress bar description with current stage."""
        nonlocal current_stage
        if stage_name != current_stage:
            current_stage = stage_name
            pbar.set_description(f"  {intervention_type.capitalize()}: {stage_name}")
    
    # Storage for per-question results
    # Structure: results[layer][direction][multiplier][q_idx] = {mc_entropy, mc_logit_gap, conf_signal}
    for layer in layers:
        results["layer_results"][layer] = {
            f"{name_a}_direction": {m: {} for m in (effective_multipliers if intervention_type == "steering" else [1.0])},
            f"{name_b}_direction": {m: {} for m in (effective_multipliers if intervention_type == "steering" else [1.0])},
            "random_direction": {m: {} for m in (effective_multipliers if intervention_type == "steering" else [1.0])},
            "baseline": {},
        }
    
    def run_task_with_intervention(
        batches,
        token_ids,  # None if multi-token
        option_info,  # OptionTokenInfo dict
        options_display,  # Display labels
        task_name,  # "mc" or "conf"
        direction_name,
        direction_tensor,
        layer,
        multiplier=1.0,
        track_coverage=False,
    ):
        """Run a task (MC or confidence) with intervention and return results."""
        task_results = {}
        coverage_hits = 0
        coverage_total = 0
        
        single_token_mode = token_ids is not None
        
        for batch_indices, batch_inputs in batches:
            B = len(batch_indices)
            
            # Get layer module
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer]
            else:
                layer_module = model.model.layers[layer]
            
            if intervention_type == "steering":
                hook = BatchSteeringHook()
                delta = direction_tensor * multiplier
                delta_batch = delta.unsqueeze(0).expand(B, -1)
                hook.set_delta(delta_batch)
            else:  # ablation
                hook = BatchAblationHook()
                dir_batch = direction_tensor.unsqueeze(0).expand(B, -1)
                hook.set_directions(dir_batch)
            
            hook.register(layer_module)
            
            try:
                with torch.inference_mode():
                    out = model(**batch_inputs)
                    full_logits = out.logits[:, -1, :]  # (B, vocab)
                    
                    if single_token_mode:
                        option_logits = full_logits[:, token_ids]
                        option_logits_np = option_logits.float().cpu().numpy()
                        probs = torch.softmax(option_logits, dim=-1).float().cpu().numpy()
                    else:
                        # Multi-token: need per-example sequence scoring
                        # This is slower, process one at a time
                        probs = []
                        option_logits_np = []
                        option_seqs = [info.token_ids for info in option_info.values()]
                        for i in range(B):
                            # Extract single example
                            single_ids = batch_inputs['input_ids'][i:i+1]
                            single_mask = batch_inputs['attention_mask'][i:i+1]
                            p = score_options_sequence_logprob(model, single_ids, single_mask, option_seqs)
                            probs.append(p)
                            # Create pseudo log-probs for entropy calculation
                            option_logits_np.append(np.log(p + 1e-10))
                        probs = np.array(probs)
                        option_logits_np = np.array(option_logits_np)
                
                for i, q_idx in enumerate(batch_indices):
                    if track_coverage and single_token_mode:
                        covered, greedy_id, _ = check_greedy_in_options(full_logits[i], token_ids)
                        coverage_total += 1
                        if covered:
                            coverage_hits += 1
                    
                    if task_name == "mc":
                        # Compute MC metrics
                        entropy = compute_mc_entropy_from_logits(option_logits_np[i])
                        gap = compute_logit_gap(option_logits_np[i])
                        max_prob = compute_mc_max_prob(option_logits_np[i])
                        task_results[q_idx] = {
                            "mc_entropy": entropy,
                            "mc_logit_gap": gap,
                            "mc_max_prob": max_prob,
                            "mc_pred": options_display[np.argmax(probs[i])],
                        }
                    else:  # confidence task
                        conf = get_stated_confidence_signal(probs[i])
                        task_results[q_idx] = {
                            "conf_signal": float(conf),
                            "conf_pred": options_display[np.argmax(probs[i])],
                        }
            finally:
                hook.remove()
            
            pbar.update(1)
        
        return task_results, coverage_hits, coverage_total
    
    def run_baseline_task(batches, token_ids, option_info, options_display, task_name, print_debug=False):
        """Run task without intervention (baseline)."""
        nonlocal mc_coverage_count, conf_coverage_count
        
        task_results = {}
        debug_printed = 0
        single_token_mode = token_ids is not None
        
        for batch_indices, batch_inputs in batches:
            with torch.inference_mode():
                out = model(**batch_inputs)
                full_logits = out.logits[:, -1, :]  # (B, vocab)
                
                if single_token_mode:
                    option_logits = full_logits[:, token_ids]
                    option_logits_np = option_logits.float().cpu().numpy()
                    probs = torch.softmax(option_logits, dim=-1).float().cpu().numpy()
                else:
                    # Multi-token sequence scoring
                    probs = []
                    option_logits_np = []
                    option_seqs = [info.token_ids for info in option_info.values()]
                    B = len(batch_indices)
                    for i in range(B):
                        single_ids = batch_inputs['input_ids'][i:i+1]
                        single_mask = batch_inputs['attention_mask'][i:i+1]
                        p = score_options_sequence_logprob(model, single_ids, single_mask, option_seqs)
                        probs.append(p)
                        option_logits_np.append(np.log(p + 1e-10))
                    probs = np.array(probs)
                    option_logits_np = np.array(option_logits_np)
            
            for i, q_idx in enumerate(batch_indices):
                # Track coverage for baseline
                if single_token_mode:
                    covered, greedy_id, rank = check_greedy_in_options(full_logits[i], token_ids)
                    if task_name == "mc":
                        if covered:
                            mc_coverage_count += 1
                    else:
                        if covered:
                            conf_coverage_count += 1
                    
                    # Print debug for first few examples
                    if print_debug and debug_printed < DEBUG_EXAMPLES:
                        option_strings = list(option_info.keys())
                        print_debug_example(q_idx, task_name.upper(), full_logits[i], token_ids, option_strings, tokenizer)
                        debug_printed += 1
                
                if task_name == "mc":
                    entropy = compute_mc_entropy_from_logits(option_logits_np[i])
                    gap = compute_logit_gap(option_logits_np[i])
                    max_prob = compute_mc_max_prob(option_logits_np[i])
                    task_results[q_idx] = {
                        "mc_entropy": entropy,
                        "mc_logit_gap": gap,
                        "mc_max_prob": max_prob,
                        "mc_pred": options_display[np.argmax(probs[i])],
                    }
                else:
                    conf = get_stated_confidence_signal(probs[i])
                    task_results[q_idx] = {
                        "conf_signal": float(conf),
                        "conf_pred": options_display[np.argmax(probs[i])],
                    }
            
            pbar.update(1)
        
        return task_results
    
    # Compute baselines (no intervention)
    print(f"\n  Computing baselines (debug output for first {DEBUG_EXAMPLES} examples)...")
    update_stage("Baseline MC")
    baseline_mc = run_baseline_task(
        mc_batches, mc_token_ids, mc_option_info, mc_options_display, "mc", print_debug=True
    )
    
    update_stage("Baseline Conf")
    baseline_conf = run_baseline_task(
        conf_batches, conf_token_ids, conf_option_info, conf_options_display, "conf", print_debug=True
    )
    
    # Report coverage
    mc_coverage_rate = mc_coverage_count / len(questions) if len(questions) > 0 else 0
    conf_coverage_rate = conf_coverage_count / len(questions) if len(questions) > 0 else 0
    
    print(f"\n  Baseline coverage rates:")
    print(f"    MC: {mc_coverage_count}/{len(questions)} = {mc_coverage_rate:.1%}")
    print(f"    Confidence: {conf_coverage_count}/{len(questions)} = {conf_coverage_rate:.1%}")
    
    if mc_coverage_rate < REQUIRE_COVERAGE_THRESHOLD:
        warnings.warn(
            f"MC coverage ({mc_coverage_rate:.1%}) is below threshold ({REQUIRE_COVERAGE_THRESHOLD:.0%}). "
            f"Check option string format - the model may be outputting a different token format."
        )
    if conf_coverage_rate < REQUIRE_COVERAGE_THRESHOLD:
        warnings.warn(
            f"Confidence coverage ({conf_coverage_rate:.1%}) is below threshold ({REQUIRE_COVERAGE_THRESHOLD:.0%}). "
            f"Check option string format."
        )
    
    # Store baseline in all layers (it's the same)
    for layer in layers:
        for q_idx in range(len(questions)):
            results["layer_results"][layer]["baseline"][q_idx] = {
                **baseline_mc[q_idx],
                **baseline_conf[q_idx],
            }
    
    # Run interventions for each layer
    mults_to_test = effective_multipliers if intervention_type == "steering" else [1.0]
    
    for layer_idx, layer in enumerate(layers):
        dir_a = cached_directions[layer][name_a]
        dir_b = cached_directions[layer][name_b]
        random_dirs = cached_directions[layer]["random"]
        
        for mult_idx, mult in enumerate(mults_to_test):
            # Build stage prefix
            if intervention_type == "steering":
                stage_prefix = f"L{layer} α={mult:+.0f}"
            else:
                stage_prefix = f"L{layer}"
            
            # Direction A intervention
            update_stage(f"{stage_prefix} {name_a}")
            mc_results_a, _, _ = run_task_with_intervention(
                mc_batches, mc_token_ids, mc_option_info, mc_options_display, "mc", name_a, dir_a, layer, mult
            )
            conf_results_a, _, _ = run_task_with_intervention(
                conf_batches, conf_token_ids, conf_option_info, conf_options_display, "conf", name_a, dir_a, layer, mult
            )
            
            for q_idx in range(len(questions)):
                results["layer_results"][layer][f"{name_a}_direction"][mult][q_idx] = {
                    **mc_results_a[q_idx],
                    **conf_results_a[q_idx],
                }
            
            # Direction B intervention
            update_stage(f"{stage_prefix} {name_b}")
            mc_results_b, _, _ = run_task_with_intervention(
                mc_batches, mc_token_ids, mc_option_info, mc_options_display, "mc", name_b, dir_b, layer, mult
            )
            conf_results_b, _, _ = run_task_with_intervention(
                conf_batches, conf_token_ids, conf_option_info, conf_options_display, "conf", name_b, dir_b, layer, mult
            )
            
            for q_idx in range(len(questions)):
                results["layer_results"][layer][f"{name_b}_direction"][mult][q_idx] = {
                    **mc_results_b[q_idx],
                    **conf_results_b[q_idx],
                }
            
            # Random direction interventions (average over all)
            random_mc_entropy = [[] for _ in range(len(questions))]
            random_mc_gap = [[] for _ in range(len(questions))]
            random_mc_max_prob = [[] for _ in range(len(questions))]
            random_conf = [[] for _ in range(len(questions))]
            
            for r_idx, r_dir in enumerate(random_dirs):
                update_stage(f"{stage_prefix} rand {r_idx+1}/{num_controls}")
                mc_results_r, _, _ = run_task_with_intervention(
                    mc_batches, mc_token_ids, mc_option_info, mc_options_display, "mc", "random", r_dir, layer, mult
                )
                conf_results_r, _, _ = run_task_with_intervention(
                    conf_batches, conf_token_ids, conf_option_info, conf_options_display, "conf", "random", r_dir, layer, mult
                )
                
                for q_idx in range(len(questions)):
                    random_mc_entropy[q_idx].append(mc_results_r[q_idx]["mc_entropy"])
                    random_mc_gap[q_idx].append(mc_results_r[q_idx]["mc_logit_gap"])
                    random_mc_max_prob[q_idx].append(mc_results_r[q_idx]["mc_max_prob"])
                    random_conf[q_idx].append(conf_results_r[q_idx]["conf_signal"])
            
            # Store averaged random results
            for q_idx in range(len(questions)):
                results["layer_results"][layer]["random_direction"][mult][q_idx] = {
                    "mc_entropy": float(np.mean(random_mc_entropy[q_idx])),
                    "mc_logit_gap": float(np.mean(random_mc_gap[q_idx])),
                    "mc_max_prob": float(np.mean(random_mc_max_prob[q_idx])),
                    "conf_signal": float(np.mean(random_conf[q_idx])),
                }
    
    pbar.close()
    
    # Add coverage metrics to results
    results["mc_coverage"] = mc_coverage_count / len(questions) if len(questions) > 0 else 0
    results["conf_coverage"] = conf_coverage_count / len(questions) if len(questions) > 0 else 0
    
    return results


# =============================================================================
# ANALYSIS
# =============================================================================


def analyze_cross_intervention_results(results: Dict) -> Dict:
    """
    Analyze cross-intervention results for asymmetry.
    
    Key metrics:
    - Effect of direction_a on conf_signal (cross-effect)
    - Effect of direction_b on mc_entropy (cross-effect)
    - Effect of direction_a on mc_entropy (same-modality)
    - Effect of direction_b on conf_signal (same-modality)
    """
    layers = results["layers"]
    intervention_type = results["intervention_type"]
    direction_names = results.get("direction_names", ["entropy", "confidence"])
    name_a, name_b = direction_names
    
    analysis = {
        "intervention_type": intervention_type,
        "layers": layers,
        "num_questions": results["num_questions"],
        "direction_names": direction_names,
        "per_layer": {},
        "summary": {},
    }
    
    if intervention_type == "steering":
        multipliers = results["multipliers"]
        analysis["multipliers"] = multipliers
    
    for layer in layers:
        lr = results["layer_results"][layer]
        baseline = lr["baseline"]
        
        # Get baseline means
        baseline_mc_entropy = np.mean([baseline[q]["mc_entropy"] for q in baseline])
        baseline_conf = np.mean([baseline[q]["conf_signal"] for q in baseline])
        baseline_mc_gap = np.mean([baseline[q]["mc_logit_gap"] for q in baseline])
        baseline_mc_max_prob = np.mean([baseline[q]["mc_max_prob"] for q in baseline])
        
        layer_analysis = {
            "baseline_mc_entropy": float(baseline_mc_entropy),
            "baseline_conf_signal": float(baseline_conf),
            "baseline_mc_logit_gap": float(baseline_mc_gap),
            "baseline_mc_max_prob": float(baseline_mc_max_prob),
        }
        
        if intervention_type == "steering":
            # Compute slopes (effect vs multiplier) using only nonzero multipliers
            for direction in [f"{name_a}_direction", f"{name_b}_direction", "random_direction"]:
                mc_entropy_by_mult = []
                conf_by_mult = []
                mc_gap_by_mult = []
                mc_max_prob_by_mult = []
                
                # Only use nonzero multipliers for fitting
                for mult in [m for m in multipliers if m != 0.0]:
                    if mult in lr[direction]:
                        mc_ents = [lr[direction][mult][q]["mc_entropy"] for q in lr[direction][mult]]
                        confs = [lr[direction][mult][q]["conf_signal"] for q in lr[direction][mult]]
                        mc_gaps = [lr[direction][mult][q]["mc_logit_gap"] for q in lr[direction][mult]]
                        mc_max_probs = [lr[direction][mult][q]["mc_max_prob"] for q in lr[direction][mult]]
                        mc_entropy_by_mult.append((mult, np.mean(mc_ents)))
                        conf_by_mult.append((mult, np.mean(confs)))
                        mc_gap_by_mult.append((mult, np.mean(mc_gaps)))
                        mc_max_prob_by_mult.append((mult, np.mean(mc_max_probs)))
                
                if len(mc_entropy_by_mult) >= 2:
                    mults = np.array([x[0] for x in mc_entropy_by_mult])
                    mc_vals = np.array([x[1] for x in mc_entropy_by_mult])
                    conf_vals = np.array([x[1] for x in conf_by_mult])
                    mc_gap_vals = np.array([x[1] for x in mc_gap_by_mult])
                    mc_max_prob_vals = np.array([x[1] for x in mc_max_prob_by_mult])
                    
                    # Use linregress for slope and R²
                    mc_result = linregress(mults, mc_vals)
                    conf_result = linregress(mults, conf_vals)
                    mc_gap_result = linregress(mults, mc_gap_vals)
                    mc_max_prob_result = linregress(mults, mc_max_prob_vals)
                    
                    mc_slope = mc_result.slope
                    mc_r2 = mc_result.rvalue ** 2
                    conf_slope = conf_result.slope
                    conf_r2 = conf_result.rvalue ** 2
                    mc_gap_slope = mc_gap_result.slope
                    mc_gap_r2 = mc_gap_result.rvalue ** 2
                    mc_max_prob_slope = mc_max_prob_result.slope
                    mc_max_prob_r2 = mc_max_prob_result.rvalue ** 2
                else:
                    mc_slope = 0.0
                    mc_r2 = 0.0
                    conf_slope = 0.0
                    conf_r2 = 0.0
                    mc_gap_slope = 0.0
                    mc_gap_r2 = 0.0
                    mc_max_prob_slope = 0.0
                    mc_max_prob_r2 = 0.0
                
                layer_analysis[f"{direction}_mc_entropy_slope"] = float(mc_slope)
                layer_analysis[f"{direction}_mc_entropy_r2"] = float(mc_r2)
                layer_analysis[f"{direction}_conf_slope"] = float(conf_slope)
                layer_analysis[f"{direction}_conf_r2"] = float(conf_r2)
                layer_analysis[f"{direction}_mc_logit_gap_slope"] = float(mc_gap_slope)
                layer_analysis[f"{direction}_mc_logit_gap_r2"] = float(mc_gap_r2)
                layer_analysis[f"{direction}_mc_max_prob_slope"] = float(mc_max_prob_slope)
                layer_analysis[f"{direction}_mc_max_prob_r2"] = float(mc_max_prob_r2)
                
                # Mean delta at |mult|=2
                if 2.0 in lr[direction] and -2.0 in lr[direction]:
                    mc_delta_pos = np.mean([lr[direction][2.0][q]["mc_entropy"] for q in lr[direction][2.0]]) - baseline_mc_entropy
                    mc_delta_neg = np.mean([lr[direction][-2.0][q]["mc_entropy"] for q in lr[direction][-2.0]]) - baseline_mc_entropy
                    conf_delta_pos = np.mean([lr[direction][2.0][q]["conf_signal"] for q in lr[direction][2.0]]) - baseline_conf
                    conf_delta_neg = np.mean([lr[direction][-2.0][q]["conf_signal"] for q in lr[direction][-2.0]]) - baseline_conf
                    
                    layer_analysis[f"{direction}_mc_entropy_delta_pos2"] = float(mc_delta_pos)
                    layer_analysis[f"{direction}_mc_entropy_delta_neg2"] = float(mc_delta_neg)
                    layer_analysis[f"{direction}_conf_delta_pos2"] = float(conf_delta_pos)
                    layer_analysis[f"{direction}_conf_delta_neg2"] = float(conf_delta_neg)
        
        else:  # ablation
            for direction in [f"{name_a}_direction", f"{name_b}_direction", "random_direction"]:
                mult = 1.0
                if mult in lr[direction]:
                    mc_ents = [lr[direction][mult][q]["mc_entropy"] for q in lr[direction][mult]]
                    confs = [lr[direction][mult][q]["conf_signal"] for q in lr[direction][mult]]
                    mc_gaps = [lr[direction][mult][q]["mc_logit_gap"] for q in lr[direction][mult]]
                    mc_max_probs = [lr[direction][mult][q]["mc_max_prob"] for q in lr[direction][mult]]
                    
                    mc_delta = np.mean(mc_ents) - baseline_mc_entropy
                    conf_delta = np.mean(confs) - baseline_conf
                    mc_gap_delta = np.mean(mc_gaps) - baseline_mc_gap
                    mc_max_prob_delta = np.mean(mc_max_probs) - baseline_mc_max_prob
                    
                    layer_analysis[f"{direction}_mc_entropy_delta"] = float(mc_delta)
                    layer_analysis[f"{direction}_conf_delta"] = float(conf_delta)
                    layer_analysis[f"{direction}_mc_logit_gap_delta"] = float(mc_gap_delta)
                    layer_analysis[f"{direction}_mc_max_prob_delta"] = float(mc_max_prob_delta)
        
        analysis["per_layer"][layer] = layer_analysis
    
    # Summary: find layers with strongest asymmetric effects
    if intervention_type == "steering":
        # Asymmetry = cross-effect / same-modality-effect
        a_cross_effects = [analysis["per_layer"][l].get(f"{name_a}_direction_conf_slope", 0) for l in layers]
        a_same_effects = [analysis["per_layer"][l].get(f"{name_a}_direction_mc_entropy_slope", 0) for l in layers]
        b_cross_effects = [analysis["per_layer"][l].get(f"{name_b}_direction_mc_entropy_slope", 0) for l in layers]
        b_same_effects = [analysis["per_layer"][l].get(f"{name_b}_direction_conf_slope", 0) for l in layers]
        
        best_a_cross_layer = int(layers[np.argmax(np.abs(a_cross_effects))])
        best_b_cross_layer = int(layers[np.argmax(np.abs(b_cross_effects))])
        
        analysis["summary"] = {
            f"best_{name_a}_cross_effect_layer": best_a_cross_layer,
            f"best_{name_a}_cross_effect": float(a_cross_effects[layers.index(best_a_cross_layer)]),
            f"best_{name_b}_cross_effect_layer": best_b_cross_layer,
            f"best_{name_b}_cross_effect": float(b_cross_effects[layers.index(best_b_cross_layer)]),
        }
    
    return analysis


# =============================================================================
# RESULTS SAVING
# =============================================================================


def save_results_to_csv(results: Dict, analysis: Dict, output_path: Path):
    """Save results in tidy CSV format."""
    rows = []
    
    intervention_type = results["intervention_type"]
    layers = results["layers"]
    direction_names = results.get("direction_names", ["entropy", "confidence"])
    name_a, name_b = direction_names
    
    # Get scoring modes and coverage for metadata
    scoring_mode_mc = results.get("scoring_mode_mc", "unknown")
    scoring_mode_conf = results.get("scoring_mode_conf", "unknown")
    mc_coverage = results.get("mc_coverage", None)
    conf_coverage = results.get("conf_coverage", None)
    
    for layer in layers:
        lr = results["layer_results"][layer]
        baseline = lr["baseline"]
        
        # Get R² values for this layer from analysis
        layer_analysis = analysis["per_layer"].get(layer, {})
        
        for q_idx in baseline:
            base_row = {
                "layer": layer,
                "example_id": q_idx,
                "baseline_mc_entropy": baseline[q_idx]["mc_entropy"],
                "baseline_conf_signal": baseline[q_idx]["conf_signal"],
                "baseline_mc_logit_gap": baseline[q_idx]["mc_logit_gap"],
                "baseline_mc_max_prob": baseline[q_idx]["mc_max_prob"],
                "scoring_mode_mc": scoring_mode_mc,
                "scoring_mode_conf": scoring_mode_conf,
            }
            
            for direction in [f"{name_a}_direction", f"{name_b}_direction", "random_direction"]:
                for mult in lr[direction]:
                    if q_idx in lr[direction][mult]:
                        post_data = lr[direction][mult][q_idx]
                        
                        # Get R² for this direction-layer combo (steering only)
                        mc_entropy_r2 = layer_analysis.get(f"{direction}_mc_entropy_r2", None)
                        conf_r2 = layer_analysis.get(f"{direction}_conf_r2", None)
                        mc_gap_r2 = layer_analysis.get(f"{direction}_mc_logit_gap_r2", None)
                        mc_max_prob_r2 = layer_analysis.get(f"{direction}_mc_max_prob_r2", None)
                        
                        row = {
                            **base_row,
                            "intervention_type": intervention_type,
                            "direction": direction.replace("_direction", ""),
                            "alpha": mult if intervention_type == "steering" else "ablate",
                            "post_mc_entropy": post_data["mc_entropy"],
                            "post_conf_signal": post_data["conf_signal"],
                            "post_mc_logit_gap": post_data["mc_logit_gap"],
                            "post_mc_max_prob": post_data["mc_max_prob"],
                            "delta_mc_entropy": post_data["mc_entropy"] - baseline[q_idx]["mc_entropy"],
                            "delta_conf_signal": post_data["conf_signal"] - baseline[q_idx]["conf_signal"],
                            "delta_mc_logit_gap": post_data["mc_logit_gap"] - baseline[q_idx]["mc_logit_gap"],
                            "delta_mc_max_prob": post_data["mc_max_prob"] - baseline[q_idx]["mc_max_prob"],
                            "fit_r2_mc_entropy": mc_entropy_r2,
                            "fit_r2_conf": conf_r2,
                            "fit_r2_mc_gap": mc_gap_r2,
                            "fit_r2_mc_max_prob": mc_max_prob_r2,
                        }
                        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved CSV: {output_path}")
    print(f"    MC coverage: {mc_coverage:.1%}" if mc_coverage else "")
    print(f"    Conf coverage: {conf_coverage:.1%}" if conf_coverage else "")
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_asymmetry_heatmaps(analysis: Dict, output_path: Path):
    """
    Create heatmaps showing cross-intervention effects.
    
    For steering: heatmap of (layer x multiplier) for each metric
    For ablation: bar plots comparing directions
    """
    intervention_type = analysis["intervention_type"]
    layers = analysis["layers"]
    direction_names = analysis.get("direction_names", ["entropy", "confidence"])
    name_a, name_b = direction_names
    
    if intervention_type == "steering":
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Cross-Intervention Effects (Steering) - Slopes with R² values", fontsize=14)
        
        # Panel layout:
        # Row 0: Effect on MC entropy (by dir_a, dir_b, random)
        # Row 1: Effect on stated confidence (by dir_a, dir_b, random)
        
        metrics = [
            (f"{name_a}_direction_mc_entropy_slope", f"{name_a}_direction_mc_entropy_r2", f"{name_a.title()} Dir → MC Entropy"),
            (f"{name_b}_direction_mc_entropy_slope", f"{name_b}_direction_mc_entropy_r2", f"{name_b.title()} Dir → MC Entropy"),
            ("random_direction_mc_entropy_slope", "random_direction_mc_entropy_r2", "Random Dir → MC Entropy"),
            (f"{name_a}_direction_conf_slope", f"{name_a}_direction_conf_r2", f"{name_a.title()} Dir → Conf Signal"),
            (f"{name_b}_direction_conf_slope", f"{name_b}_direction_conf_r2", f"{name_b.title()} Dir → Conf Signal"),
            ("random_direction_conf_slope", "random_direction_conf_r2", "Random Dir → Conf Signal"),
        ]
        
        for idx, (slope_key, r2_key, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = [analysis["per_layer"][l].get(slope_key, 0) for l in layers]
            r2_values = [analysis["per_layer"][l].get(r2_key, 0) for l in layers]
            
            # Color by sign, alpha by R²
            colors = []
            for v, r2 in zip(values, r2_values):
                base_color = 'tab:red' if v < 0 else 'tab:blue'
                colors.append(base_color)
            
            bars = ax.bar(range(len(layers)), values, color=colors)
            
            # Adjust alpha based on R²
            for bar, r2 in zip(bars, r2_values):
                bar.set_alpha(0.3 + 0.7 * r2)  # Range from 0.3 to 1.0
            
            ax.set_xticks(range(0, len(layers), 4))
            ax.set_xticklabels([layers[i] for i in range(0, len(layers), 4)])
            ax.set_xlabel("Layer")
            ax.set_ylabel("Slope (Δmetric / Δα)")
            
            # Add mean R² to title
            mean_r2 = np.mean(r2_values)
            ax.set_title(f"{title}\n(mean R²={mean_r2:.2f})")
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {output_path}")
    
    else:  # ablation
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Cross-Intervention Effects (Ablation)", fontsize=14)
        
        # Panel 0: Effect on MC entropy
        ax0 = axes[0]
        for dir_name, color in [(name_a, "tab:green"), (name_b, "tab:purple"), ("random", "gray")]:
            key = f"{dir_name}_direction_mc_entropy_delta"
            values = [analysis["per_layer"][l].get(key, 0) for l in layers]
            ax0.plot(layers, values, 'o-', label=f"{dir_name} dir", color=color, linewidth=1.5)
        ax0.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax0.set_xlabel("Layer")
        ax0.set_ylabel("ΔMC Entropy (ablated - baseline)")
        ax0.set_title("Effect on MC Entropy")
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        # Panel 1: Effect on confidence signal
        ax1 = axes[1]
        for dir_name, color in [(name_a, "tab:green"), (name_b, "tab:purple"), ("random", "gray")]:
            key = f"{dir_name}_direction_conf_delta"
            values = [analysis["per_layer"][l].get(key, 0) for l in layers]
            ax1.plot(layers, values, 'o-', label=f"{dir_name} dir", color=color, linewidth=1.5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("ΔConf Signal (ablated - baseline)")
        ax1.set_title("Effect on Stated Confidence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {output_path}")


def plot_asymmetry_summary(steering_analysis: Dict, ablation_analysis: Dict, output_path: Path):
    """
    Create summary plot comparing asymmetry between steering and ablation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Intervention Asymmetry Analysis", fontsize=14, fontweight='bold')
    
    layers = steering_analysis["layers"]
    direction_names = steering_analysis.get("direction_names", ["entropy", "confidence"])
    name_a, name_b = direction_names
    
    # Panel 0,0: Steering - direction A effects
    ax = axes[0, 0]
    a_mc = [steering_analysis["per_layer"][l].get(f"{name_a}_direction_mc_entropy_slope", 0) for l in layers]
    a_conf = [steering_analysis["per_layer"][l].get(f"{name_a}_direction_conf_slope", 0) for l in layers]
    ax.plot(layers, a_mc, 'o-', label='→ MC Entropy', color='tab:green', linewidth=1.5)
    ax.plot(layers, a_conf, 's-', label='→ Conf Signal', color='tab:purple', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Slope")
    ax.set_title(f"Steering: {name_a.title()} Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 0,1: Steering - direction B effects
    ax = axes[0, 1]
    b_mc = [steering_analysis["per_layer"][l].get(f"{name_b}_direction_mc_entropy_slope", 0) for l in layers]
    b_conf = [steering_analysis["per_layer"][l].get(f"{name_b}_direction_conf_slope", 0) for l in layers]
    ax.plot(layers, b_mc, 'o-', label='→ MC Entropy', color='tab:green', linewidth=1.5)
    ax.plot(layers, b_conf, 's-', label='→ Conf Signal', color='tab:purple', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Slope")
    ax.set_title(f"Steering: {name_b.title()} Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 1,0: Ablation - direction A effects
    ax = axes[1, 0]
    a_mc_abl = [ablation_analysis["per_layer"][l].get(f"{name_a}_direction_mc_entropy_delta", 0) for l in layers]
    a_conf_abl = [ablation_analysis["per_layer"][l].get(f"{name_a}_direction_conf_delta", 0) for l in layers]
    ax.plot(layers, a_mc_abl, 'o-', label='Δ MC Entropy', color='tab:green', linewidth=1.5)
    ax.plot(layers, a_conf_abl, 's-', label='Δ Conf Signal', color='tab:purple', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta")
    ax.set_title(f"Ablation: {name_a.title()} Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 1,1: Ablation - direction B effects
    ax = axes[1, 1]
    b_mc_abl = [ablation_analysis["per_layer"][l].get(f"{name_b}_direction_mc_entropy_delta", 0) for l in layers]
    b_conf_abl = [ablation_analysis["per_layer"][l].get(f"{name_b}_direction_conf_delta", 0) for l in layers]
    ax.plot(layers, b_mc_abl, 'o-', label='Δ MC Entropy', color='tab:green', linewidth=1.5)
    ax.plot(layers, b_conf_abl, 's-', label='Δ Conf Signal', color='tab:purple', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta")
    ax.set_title(f"Ablation: {name_b.title()} Direction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary plot: {output_path}")


def print_summary(steering_analysis: Dict, ablation_analysis: Dict):
    """Print summary of asymmetry results."""
    print("\n" + "=" * 80)
    print("CROSS-INTERVENTION ASYMMETRY RESULTS")
    print("=" * 80)
    
    direction_names = steering_analysis.get("direction_names", ["entropy", "confidence"])
    name_a, name_b = direction_names
    
    print(f"\nKey question: Do {name_a} and {name_b} directions capture the same information?")
    print("If they're the same, cross-effects should be similar to same-modality effects.")
    print("If different, cross-effects should be weaker or absent.\n")
    
    layers = steering_analysis["layers"]
    
    # Find best layers for interpretation
    print("-" * 80)
    print("STEERING EFFECTS (slope of metric change vs multiplier)")
    print("-" * 80)
    
    # Print table header
    a_short = name_a[0].upper()
    b_short = name_b[0].upper()
    print(f"{'Layer':>6}  {f'{a_short}→MC_ent':>10} {'R²':>5}  {f'{a_short}→Conf':>10} {'R²':>5}  {f'{b_short}→MC_ent':>10} {'R²':>5}  {f'{b_short}→Conf':>10} {'R²':>5}")
    print("-" * 100)
    
    for layer in layers[::4]:  # Every 4th layer for readability
        p = steering_analysis["per_layer"][layer]
        print(f"{layer:>6}  "
              f"{p.get(f'{name_a}_direction_mc_entropy_slope', 0):>+10.4f} "
              f"{p.get(f'{name_a}_direction_mc_entropy_r2', 0):>5.2f}  "
              f"{p.get(f'{name_a}_direction_conf_slope', 0):>+10.4f} "
              f"{p.get(f'{name_a}_direction_conf_r2', 0):>5.2f}  "
              f"{p.get(f'{name_b}_direction_mc_entropy_slope', 0):>+10.4f} "
              f"{p.get(f'{name_b}_direction_mc_entropy_r2', 0):>5.2f}  "
              f"{p.get(f'{name_b}_direction_conf_slope', 0):>+10.4f} "
              f"{p.get(f'{name_b}_direction_conf_r2', 0):>5.2f}")
    
    print("\n" + "-" * 80)
    print("ABLATION EFFECTS (delta from baseline)")
    print("-" * 80)
    
    print(f"{'Layer':>6}  {f'{a_short}→ΔMC_ent':>10}  {f'{a_short}→ΔConf':>10}  {f'{b_short}→ΔMC_ent':>10}  {f'{b_short}→ΔConf':>10}  {'R→ΔMC_ent':>10}  {'R→ΔConf':>10}")
    print("-" * 80)
    
    for layer in layers[::4]:
        p = ablation_analysis["per_layer"][layer]
        print(f"{layer:>6}  "
              f"{p.get(f'{name_a}_direction_mc_entropy_delta', 0):>+10.4f}  "
              f"{p.get(f'{name_a}_direction_conf_delta', 0):>+10.4f}  "
              f"{p.get(f'{name_b}_direction_mc_entropy_delta', 0):>+10.4f}  "
              f"{p.get(f'{name_b}_direction_conf_delta', 0):>+10.4f}  "
              f"{p.get('random_direction_mc_entropy_delta', 0):>+10.4f}  "
              f"{p.get('random_direction_conf_delta', 0):>+10.4f}")
    
    print("\nLegend:")
    print(f"  {a_short} = {name_a.title()} direction, {b_short} = {name_b.title()} direction, R = Random direction")
    print("  MC_ent = MC entropy, Conf = Stated confidence signal")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("CROSS-INTERVENTION ASYMMETRY EXPERIMENT")
    print("=" * 80)
    print(f"\nModel: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Direction A ({DIRECTION_A_NAME}): {DIRECTION_A_PATH}")
    print(f"Direction B ({DIRECTION_B_NAME}): {DIRECTION_B_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Random directions: {NUM_RANDOM_DIRECTIONS}")
    print(f"Multipliers: {STEERING_MULTIPLIERS}")
    
    # Load directions
    print("\nLoading contrast directions...")
    directions_a, directions_b, name_a, name_b = load_contrast_directions(
        DIRECTION_A_PATH, DIRECTION_B_PATH, DIRECTION_A_NAME, DIRECTION_B_NAME, NPZ_METHOD
    )
    num_layers = len(directions_a)
    print(f"  Loaded {num_layers} layers")
    
    # Compute cosine similarity between directions
    print(f"\nCosine similarity between {name_a} and {name_b} directions:")
    cosines = []
    for layer in sorted(directions_a.keys()):
        cos = np.dot(directions_a[layer], directions_b[layer])
        cosines.append(cos)
    print(f"  Mean: {np.mean(cosines):.3f}, Std: {np.std(cosines):.3f}")
    print(f"  Range: [{min(cosines):.3f}, {max(cosines):.3f}]")
    
    # Load dataset
    print("\nLoading dataset...")
    all_data = load_dataset(DATASET_PATH)
    print(f"  Loaded {len(all_data)} questions from {DATASET_PATH}")
    
    if USE_TRANSFER_SPLIT:
        n_total = len(all_data)
        indices = np.arange(n_total)
        _, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)
        questions = [all_data[i] for i in test_idx[:NUM_QUESTIONS]]
        print(f"  Using test split: {len(questions)} questions (from {n_total} total)")
    else:
        questions = all_data[:NUM_QUESTIONS]
        print(f"  Using first {len(questions)} questions")
    
    # Determine layers
    if LAYERS is not None:
        layers = LAYERS
    else:
        layers = sorted(directions_a.keys())
    print(f"  Testing {len(layers)} layers")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, model_num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Chat template: {use_chat_template}")
    print(f"  Device: {DEVICE}")
    
    model_short = get_model_short_name(MODEL)
    base_output = f"{model_short}_contrast_intervention_{name_a}_vs_{name_b}"
    
    # ==========================================================================
    # OPTION TOKENIZATION ANALYSIS (CRITICAL FOR CORRECTNESS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("OPTION TOKENIZATION ANALYSIS")
    print("=" * 80)
    
    # MC options - try multiple formats and pick best
    print("\nMC option tokenization:")
    if AUTO_DETECT_OPTION_FORMAT:
        mc_candidates = [
            MC_OPTION_STRINGS,  # Default: [" A", " B", " C", " D"]
            MC_OPTION_LETTERS,  # Fallback: ["A", "B", "C", "D"]
            ["\nA", "\nB", "\nC", "\nD"],  # Newline prefix
        ]
        mc_option_strings, mc_option_info, mc_all_single = auto_detect_option_strings(
            mc_candidates, tokenizer, verbose=True
        )
    else:
        mc_option_strings = MC_OPTION_STRINGS
        mc_option_info = get_option_token_ids(mc_option_strings, tokenizer, verbose=True)
        mc_all_single = check_all_single_token(mc_option_info)
    
    # Confidence options - try multiple formats and pick best
    print("\nConfidence option tokenization:")
    if AUTO_DETECT_OPTION_FORMAT:
        conf_candidates = [
            CONF_OPTION_STRINGS,  # Default: [" S", " T", ..., " Z"]
            CONF_OPTION_LETTERS,  # Fallback: ["S", "T", ..., "Z"]
        ]
        conf_option_strings, conf_option_info, conf_all_single = auto_detect_option_strings(
            conf_candidates, tokenizer, verbose=True
        )
    else:
        conf_option_strings = CONF_OPTION_STRINGS
        conf_option_info = get_option_token_ids(conf_option_strings, tokenizer, verbose=True)
        conf_all_single = check_all_single_token(conf_option_info)
    
    # Critical check: if not all single-token and sequence scoring not desired, fail
    if not mc_all_single or not conf_all_single:
        print("\n  WARNING: Some options are multi-token.")
        print("  Sequence log-prob scoring will be used (slower but accurate).")
    
    print("=" * 80)
    
    # Run steering experiment
    print("\n" + "=" * 80)
    print("STEERING EXPERIMENT")
    print("=" * 80)
    
    steering_results = run_cross_intervention_experiment(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        directions_a=directions_a,
        directions_b=directions_b,
        name_a=name_a,
        name_b=name_b,
        num_controls=NUM_RANDOM_DIRECTIONS,
        layers=layers,
        multipliers=STEERING_MULTIPLIERS,
        use_chat_template=use_chat_template,
        intervention_type="steering",
        mc_option_info=mc_option_info,
        conf_option_info=conf_option_info,
    )
    
    steering_analysis = analyze_cross_intervention_results(steering_results)
    
    # Save steering results immediately
    print("\n  Saving steering results...")
    csv_path = OUTPUT_DIR / f"{base_output}_steering.csv"
    save_results_to_csv(steering_results, steering_analysis, csv_path)
    plot_asymmetry_heatmaps(steering_analysis, OUTPUT_DIR / f"{base_output}_steering_heatmap.png")
    
    # Run ablation experiment
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT")
    print("=" * 80)
    
    ablation_results = run_cross_intervention_experiment(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        directions_a=directions_a,
        directions_b=directions_b,
        name_a=name_a,
        name_b=name_b,
        num_controls=NUM_RANDOM_DIRECTIONS,
        layers=layers,
        multipliers=[1.0],  # Not used for ablation
        use_chat_template=use_chat_template,
        intervention_type="ablation",
        mc_option_info=mc_option_info,
        conf_option_info=conf_option_info,
    )
    
    ablation_analysis = analyze_cross_intervention_results(ablation_results)
    
    # Save ablation results immediately
    print("\n  Saving ablation results...")
    csv_path = OUTPUT_DIR / f"{base_output}_ablation.csv"
    save_results_to_csv(ablation_results, ablation_analysis, csv_path)
    plot_asymmetry_heatmaps(ablation_analysis, OUTPUT_DIR / f"{base_output}_ablation_heatmap.png")
    
    # Save combined results (requires both experiments)
    print("\n" + "=" * 80)
    print("SAVING COMBINED RESULTS")
    print("=" * 80)
    
    # JSON results
    results_path = OUTPUT_DIR / f"{base_output}_results.json"
    output_json = {
        "config": {
            "model": MODEL,
            "adapter": ADAPTER,
            "direction_a_path": DIRECTION_A_PATH,
            "direction_b_path": DIRECTION_B_PATH,
            "direction_a_name": name_a,
            "direction_b_name": name_b,
            "dataset_path": DATASET_PATH,
            "num_questions": len(questions),
            "num_random_directions": NUM_RANDOM_DIRECTIONS,
            "multipliers": STEERING_MULTIPLIERS,
            "layers": layers,
            "mc_option_strings": mc_option_strings,
            "conf_option_strings": conf_option_strings,
        },
        "coverage": {
            "steering_mc": steering_results.get("mc_coverage", None),
            "steering_conf": steering_results.get("conf_coverage", None),
            "ablation_mc": ablation_results.get("mc_coverage", None),
            "ablation_conf": ablation_results.get("conf_coverage", None),
        },
        "scoring_modes": {
            "mc": steering_results.get("scoring_mode_mc", "unknown"),
            "conf": steering_results.get("scoring_mode_conf", "unknown"),
        },
        "steering": steering_analysis,
        "ablation": ablation_analysis,
    }
    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved JSON: {results_path}")
    
    # Summary plot (needs both experiments)
    plot_asymmetry_summary(steering_analysis, ablation_analysis, OUTPUT_DIR / f"{base_output}_summary.png")
    
    # Print summary
    print_summary(steering_analysis, ablation_analysis)
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  {results_path.name}")
    print(f"  {base_output}_steering.csv")
    print(f"  {base_output}_ablation.csv")
    print(f"  {base_output}_steering_heatmap.png")
    print(f"  {base_output}_ablation_heatmap.png")
    print(f"  {base_output}_summary.png")


def run_self_test():
    """
    Run a quick sanity check with a few examples.
    
    This tests:
    - Option tokenization correctness
    - Coverage rates
    - Basic metric computation
    
    Does NOT run full interventions (that would require directions).
    """
    print("=" * 80)
    print("SELF-TEST: Verifying option tokenization and coverage")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer, model_num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Chat template: {use_chat_template}")
    
    # Test option tokenization
    print("\n" + "-" * 40)
    print("MC OPTION TOKENIZATION")
    print("-" * 40)
    
    mc_candidates = [
        MC_OPTION_STRINGS,
        MC_OPTION_LETTERS,
        ["\nA", "\nB", "\nC", "\nD"],
    ]
    mc_option_strings, mc_option_info, mc_all_single = auto_detect_option_strings(
        mc_candidates, tokenizer, verbose=True
    )
    
    print("\n" + "-" * 40)
    print("CONFIDENCE OPTION TOKENIZATION")
    print("-" * 40)
    
    conf_candidates = [
        CONF_OPTION_STRINGS,
        CONF_OPTION_LETTERS,
    ]
    conf_option_strings, conf_option_info, conf_all_single = auto_detect_option_strings(
        conf_candidates, tokenizer, verbose=True
    )
    
    # Load a few test questions
    print("\n" + "-" * 40)
    print("LOADING TEST QUESTIONS")
    print("-" * 40)
    
    all_data = load_dataset(DATASET_PATH)
    test_questions = all_data[:5]  # Just 5 for testing
    print(f"  Loaded {len(test_questions)} test questions")
    
    # Test MC prompts
    print("\n" + "-" * 40)
    print("TESTING MC PROMPTS")
    print("-" * 40)
    
    if mc_all_single:
        mc_token_ids = get_singleton_token_ids(mc_option_info)
    else:
        mc_token_ids = None
        print("  WARNING: MC options are multi-token, will use sequence scoring")
    
    mc_coverage = 0
    for i, q in enumerate(test_questions):
        mc_prompt, _ = format_direct_prompt(q, tokenizer, use_chat_template)
        
        # Tokenize and get logits
        inputs = tokenizer(mc_prompt, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Check coverage
        if mc_token_ids is not None:
            covered, greedy_id, rank = check_greedy_in_options(logits, mc_token_ids)
            greedy_token = tokenizer.decode([greedy_id])
            
            if covered:
                mc_coverage += 1
            
            # Get option probs
            option_probs = score_options_next_token(logits, mc_token_ids)
            entropy = compute_mc_entropy_from_logits(np.log(option_probs + 1e-10))
            
            print(f"\n  Example {i}: {'COVERED' if covered else 'NOT COVERED'}")
            print(f"    Greedy: '{greedy_token}' (id={greedy_id})")
            print(f"    Option probs: " + ", ".join(f"{s}:{p:.3f}" for s, p in zip(mc_option_strings, option_probs)))
            print(f"    MC entropy: {entropy:.3f}")
    
    mc_coverage_rate = mc_coverage / len(test_questions)
    print(f"\n  MC Coverage: {mc_coverage}/{len(test_questions)} = {mc_coverage_rate:.1%}")
    
    # Test Confidence prompts
    print("\n" + "-" * 40)
    print("TESTING CONFIDENCE PROMPTS")
    print("-" * 40)
    
    if conf_all_single:
        conf_token_ids = get_singleton_token_ids(conf_option_info)
    else:
        conf_token_ids = None
        print("  WARNING: Confidence options are multi-token, will use sequence scoring")
    
    conf_coverage = 0
    for i, q in enumerate(test_questions):
        conf_prompt, _ = format_stated_confidence_prompt(q, tokenizer, use_chat_template)
        
        # Tokenize and get logits
        inputs = tokenizer(conf_prompt, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        if conf_token_ids is not None:
            covered, greedy_id, rank = check_greedy_in_options(logits, conf_token_ids)
            greedy_token = tokenizer.decode([greedy_id])
            
            if covered:
                conf_coverage += 1
            
            option_probs = score_options_next_token(logits, conf_token_ids)
            conf_signal = get_stated_confidence_signal(option_probs)
            
            print(f"\n  Example {i}: {'COVERED' if covered else 'NOT COVERED'}")
            print(f"    Greedy: '{greedy_token}' (id={greedy_id})")
            print(f"    Confidence signal: {conf_signal:.3f}")
    
    conf_coverage_rate = conf_coverage / len(test_questions)
    print(f"\n  Confidence Coverage: {conf_coverage}/{len(test_questions)} = {conf_coverage_rate:.1%}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SELF-TEST SUMMARY")
    print("=" * 80)
    print(f"  MC options: {mc_option_strings}")
    print(f"  MC all single-token: {mc_all_single}")
    print(f"  MC coverage: {mc_coverage_rate:.1%}")
    print(f"  Confidence options: {conf_option_strings}")
    print(f"  Confidence all single-token: {conf_all_single}")
    print(f"  Confidence coverage: {conf_coverage_rate:.1%}")
    
    # Fail loudly if both are multi-token and coverage is bad
    if not mc_all_single and mc_coverage_rate < 0.5:
        raise RuntimeError(
            "MC options are multi-token AND coverage is below 50%. "
            "Check option string format!"
        )
    if not conf_all_single and conf_coverage_rate < 0.5:
        raise RuntimeError(
            "Confidence options are multi-token AND coverage is below 50%. "
            "Check option string format!"
        )
    
    print("\n  Self-test PASSED!")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        run_self_test()
    else:
        main()
