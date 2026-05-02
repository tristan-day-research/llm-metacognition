"""
Analyze and compare probe directions across different experiments.

This script:
1. Loads direction files from various probe experiments:
   - Next-token uncertainty probes (entropy, top_prob, margin, logit_gap, top_logit)
   - MC uncertainty probes (entropy, top_prob, margin, logit_gap, top_logit)
   - Introspection probes (trained on direct MC prompts, tested on meta prompts)
   - Contrastive directions (from confidence_contrast.py)
2. Computes pairwise cosine similarities between directions at each layer
   - This shows whether different methods (probes vs mean-diff, entropy vs confidence)
     find similar directions in activation space
3. Runs logit lens analysis (project directions through unembedding)
   - Shows which tokens each direction "points toward" in vocabulary space
4. Generates visualizations

Direction types and their relationships:
- mc_{metric}_{dataset}: Trained on MC questions to predict uncertainty metric
- introspection_{metric}_{dataset}: Also trained on MC questions (direct prompts) to
  predict the same uncertainty metric. These should be very similar to mc directions
  for the same dataset/metric. The introspection experiment additionally tests whether
  these directions transfer to meta-cognition prompts ("How confident are you...?").
- nexttoken_{metric}: Trained on diverse next-token prediction to predict uncertainty
- entropy_contrast_{dataset}: Mean-diff between high/low entropy activations
- confidence_contrast_{dataset}: Mean-diff between high/low stated confidence activations

Supported file formats:
- .npz files: NumPy arrays (from probe scripts like mc_entropy_probe.py)
- .pt files: PyTorch tensors (from confidence_contrast.py)

Usage:
    python analyze_directions.py                    # Auto-detect directions in outputs/
    python analyze_directions.py --model-only       # Only load model, skip analysis (for debugging)
    python analyze_directions.py --layer 15         # Focus on specific layer
    python analyze_directions.py --skip-logit-lens  # Skip logit lens (faster, no model needed)
    python analyze_directions.py --metric entropy   # Only analyze entropy directions
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import os
import re
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

from core import (
    DEVICE,
    get_model_short_name,
)

# =============================================================================
# Configuration
# =============================================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Use base model
# ADAPTER = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"  # Fine-tuned model

# Output directory - should contain the .npz and .pt direction files
# Also searches subdirectories like confidence_contrast/
OUTPUTS_DIR = Path("outputs/8b_instruct_entropy")
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# Subdirectory patterns to search for .pt contrast files
CONTRAST_SUBDIRS = ["confidence_contrast"]

# Analysis config
TOP_K_TOKENS = 12  # Number of top tokens to show in logit lens heatmaps
TOP_K_TOKENS_SUMMARY = 20  # Number of top tokens to save in JSON and .txt summary
LAYERS_TO_ANALYZE = None  # None = all layers, or list like [10, 15, 20]

# Available uncertainty metrics (same as in probe scripts)
# AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
AVAILABLE_METRICS = ["entropy"]


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}")
    return str(OUTPUTS_DIR / f"{model_short}")


def extract_dataset_from_npz(path: Path) -> Optional[str]:
    """
    Extract dataset name from npz file metadata.

    Returns dataset name if stored in metadata, None otherwise.
    """
    try:
        data = np.load(path)
        if "_metadata_dataset" in data.files:
            return str(data["_metadata_dataset"])
    except Exception:
        pass
    return None


def extract_dataset_from_filename(filename: str, suffix: str) -> Optional[str]:
    """
    Extract dataset name from a direction filename (fallback for old files without metadata).

    Handles patterns like:
    - Llama-3.1-8B-Instruct_SimpleMC_mc_entropy_directions.npz
    - Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000_GPQA_mc_entropy_directions.npz

    The dataset name is the part immediately before the suffix (e.g., _mc_entropy_directions).

    WARNING: This will fail for dataset names containing underscores (e.g., Science_QA).
    Prefer using extract_dataset_from_npz() which reads metadata.
    """
    # Remove .npz extension if present
    if filename.endswith(".npz"):
        filename = filename[:-4]

    # Remove the known suffix
    if not filename.endswith(suffix):
        return None
    filename = filename[:-len(suffix)]

    # Now filename is like:
    # - Llama-3.1-8B-Instruct_SimpleMC
    # - Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000_GPQA

    # The dataset is the last underscore-separated component
    # Split from the right to get the last part
    parts = filename.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]  # The dataset name
    return None


def extract_metric_from_npz(path: Path) -> Optional[str]:
    """
    Extract metric name from npz file metadata.

    Returns metric name if stored in metadata, None otherwise.
    """
    try:
        data = np.load(path)
        if "_metadata_metric" in data.files:
            return str(data["_metadata_metric"])
    except Exception:
        pass
    return None


def find_direction_files(output_dir: Path, model_short: str, metric_filter: Optional[str] = None) -> Dict[str, Path]:
    """
    Find all direction files for a given model.

    Args:
        output_dir: Directory to search
        model_short: Short model name for pattern matching
        metric_filter: If specified, only include files for this metric

    Returns dict mapping direction_type -> path.
    For dataset-specific files (like mc), includes the dataset in the key.
    For metric-specific files, includes the metric in the key.
    """
    direction_files = {}

    # Patterns that are NOT dataset-specific or metric-specific (single file per model)
    simple_patterns = [
        ("introspection_direction", f"{model_short}*_direction_vectors.npz"),
    ]

    for direction_type, pattern in simple_patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Take the most recent if multiple
            direction_files[direction_type] = max(matches, key=lambda p: p.stat().st_mtime)

    # Contrastive directions from compute_contrastive_directions.py:
    # {model}_{dataset}_{metric}_contrastive_{dir_type}_directions.npz
    # where dir_type is "confidence" or "calibration"
    # Also supports old format: {model}_{dataset}_{metric}_contrastive_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        # New format with direction type suffix
        for dir_type in ["confidence", "calibration"]:
            pattern = f"{model_short}*_{metric}_contrastive_{dir_type}_directions.npz"
            for path in output_dir.glob(pattern):
                dataset = extract_dataset_from_npz(path)
                if dataset is None:
                    # Try to extract from filename
                    name = path.name
                    prefix = f"{model_short}_"
                    suffix = f"_{metric}_contrastive_{dir_type}_directions.npz"
                    if name.startswith(prefix) and name.endswith(suffix):
                        dataset = name[len(prefix):-len(suffix)]
                        if "_adapter-" in dataset:
                            parts = dataset.split("_", 1)
                            if len(parts) > 1:
                                dataset = parts[1]

                if dataset:
                    key = f"{dir_type}_{metric}_{dataset}"
                else:
                    key = f"{dir_type}_{metric}"

                if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                    direction_files[key] = path

        # Old format without direction type (backward compatibility)
        contrastive_pattern = f"{model_short}*_{metric}_contrastive_directions.npz"
        for path in output_dir.glob(contrastive_pattern):
            # Skip if this matches the new format (has _confidence_ or _calibration_)
            if "_confidence_directions.npz" in path.name or "_calibration_directions.npz" in path.name:
                continue

            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                # Try to extract from filename: {model}_{dataset}_{metric}_contrastive_directions.npz
                name = path.name
                prefix = f"{model_short}_"
                suffix = f"_{metric}_contrastive_directions.npz"
                if name.startswith(prefix) and name.endswith(suffix):
                    dataset = name[len(prefix):-len(suffix)]
                    if "_adapter-" in dataset:
                        parts = dataset.split("_", 1)
                        if len(parts) > 1:
                            dataset = parts[1]

            if dataset:
                key = f"contrastive_{metric}_{dataset}"
            else:
                key = f"contrastive_{metric}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Introspection direction files from two sources:
    # 1. run_introspection_experiment.py: {model}_{dataset}_introspection[_{task}]_{metric}_directions.npz
    # 2. run_introspection_probe.py: {model}_{dataset}_introspection[_{task}]_{metric}_probe_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        # Helper to extract task from filename
        def extract_task(filename: str, metric: str, has_probe: bool) -> Optional[str]:
            suffix = f"_{metric}_probe_directions\\.npz$" if has_probe else f"_{metric}_directions\\.npz$"
            task_match = re.search(rf"_introspection(?:_([^_]+))?{suffix}", filename)
            return task_match.group(1) if task_match and task_match.group(1) else None

        # 1. Match files from run_introspection_experiment.py (no _probe suffix)
        # Use negative lookahead to exclude _probe_directions files
        experiment_pattern = f"{model_short}*_introspection*_{metric}_directions.npz"
        for path in output_dir.glob(experiment_pattern):
            # Skip if this is actually a _probe_directions file
            if "_probe_directions.npz" in path.name:
                continue

            dataset = extract_dataset_from_npz(path)
            task = extract_task(path.name, metric, has_probe=False)

            if dataset and task:
                key = f"introspection_{task}_{metric}_{dataset}"
            elif dataset:
                key = f"introspection_{metric}_{dataset}"
            elif task:
                key = f"introspection_{task}_{metric}"
            else:
                key = f"introspection_{metric}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

        # 2. Match files from run_introspection_probe.py (_probe suffix)
        probe_pattern = f"{model_short}*_introspection*_{metric}_probe_directions.npz"
        for path in output_dir.glob(probe_pattern):
            dataset = extract_dataset_from_npz(path)
            task = extract_task(path.name, metric, has_probe=True)

            if dataset and task:
                key = f"introspection_probe_{task}_{metric}_{dataset}"
            elif dataset:
                key = f"introspection_probe_{metric}_{dataset}"
            elif task:
                key = f"introspection_probe_{task}_{metric}"
            else:
                key = f"introspection_probe_{metric}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Backward compatibility: old introspection_entropy/probe patterns without dataset
    # These are ONLY for files with the exact pattern {model}_introspection_entropy_directions.npz
    # (no dataset in the name). Skip if we already found dataset-specific introspection files.
    if not metric_filter or metric_filter == "entropy":
        # Only add these if we found NO dataset-specific introspection files
        has_dataset_specific = any(k.startswith("introspection_") and k.count("_") >= 2
                                   for k in direction_files)
        if not has_dataset_specific:
            old_intro_patterns = [
                ("introspection_entropy", f"{model_short}_introspection_entropy_directions.npz"),
                ("introspection_probe", f"{model_short}_introspection_probe_directions.npz"),
            ]
            for key, pattern in old_intro_patterns:
                if key not in direction_files:
                    matches = list(output_dir.glob(pattern))
                    if matches:
                        direction_files[key] = max(matches, key=lambda p: p.stat().st_mtime)

    # Metric-specific nexttoken patterns
    # Pattern: {model}_nexttoken_{metric}_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        nexttoken_pattern = f"{model_short}*_nexttoken_{metric}_directions.npz"
        matches = list(output_dir.glob(nexttoken_pattern))
        if matches:
            path = max(matches, key=lambda p: p.stat().st_mtime)
            direction_files[f"nexttoken_{metric}"] = path

    # Backward compatibility: old nexttoken_entropy_directions.npz format
    if not metric_filter or metric_filter == "entropy":
        old_pattern = f"{model_short}*_nexttoken_entropy_directions.npz"
        old_matches = list(output_dir.glob(old_pattern))
        for path in old_matches:
            # Check if this is NOT a metric-specific file (old format)
            # Old format: model_nexttoken_entropy_directions.npz
            # New format: model_nexttoken_entropy_directions.npz (same name for entropy)
            # We need to check if we already found it via the new pattern
            if "nexttoken_entropy" not in direction_files:
                direction_files["nexttoken_entropy"] = path

    # Dataset-specific and metric-specific MC patterns
    # New pattern: {model}_{dataset}_mc_{metric}_directions.npz
    # Old pattern: {model}_{dataset}_mc_entropy_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        mc_pattern = f"{model_short}*_mc_{metric}_directions.npz"
        mc_matches = list(output_dir.glob(mc_pattern))
        for path in mc_matches:
            # Try to get dataset from metadata first
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                dataset = extract_dataset_from_filename(path.name, f"_mc_{metric}_directions")

            if dataset:
                key = f"mc_{metric}_{dataset}"
            else:
                key = f"mc_{metric}"

            # If we already have this key, keep the most recent
            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Backward compatibility: old mc_entropy_directions.npz format
    if not metric_filter or metric_filter == "entropy":
        old_mc_pattern = f"{model_short}*_mc_entropy_directions.npz"
        old_mc_matches = list(output_dir.glob(old_mc_pattern))
        for path in old_mc_matches:
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                dataset = extract_dataset_from_filename(path.name, "_mc_entropy_directions")

            if dataset:
                key = f"mc_entropy_{dataset}"
            else:
                key = "mc_entropy"

            # Only add if we don't already have this from the new pattern search
            if key not in direction_files:
                direction_files[key] = path

    # ==========================================================================
    # Search for .pt contrast direction files in subdirectories
    # These come from confidence_contrast.py
    # Pattern: {model}_{dataset}_{contrast_type}_contrast_directions.pt
    # ==========================================================================
    for subdir_name in CONTRAST_SUBDIRS:
        subdir = output_dir / subdir_name
        if not subdir.exists():
            continue

        # Search for entropy_contrast and confidence_contrast .pt files
        for contrast_type in ["entropy", "confidence"]:
            pattern = f"{model_short}*_{contrast_type}_contrast_directions.pt"
            pt_matches = list(subdir.glob(pattern))

            for path in pt_matches:
                # Extract dataset from filename
                # Pattern: {model}_{dataset}_{contrast_type}_contrast_directions.pt
                name = path.name
                suffix = f"_{contrast_type}_contrast_directions.pt"
                if name.endswith(suffix):
                    # Remove suffix and extract dataset
                    base = name[:-len(suffix)]
                    # Dataset is the last underscore-separated component before contrast type
                    # e.g., "Llama-3.1-8B-Instruct_adapter-ect_..._TriviaMC" -> "TriviaMC"
                    parts = base.rsplit("_", 1)
                    dataset = parts[-1] if parts else None

                    if dataset:
                        key = f"{contrast_type}_contrast_{dataset}"
                    else:
                        key = f"{contrast_type}_contrast"

                    # If we already have this key, keep the most recent
                    if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                        direction_files[key] = path

    return direction_files


def load_directions(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load directions from a .npz file.

    Args:
        path: Path to .npz file

    Returns dict mapping layer_idx -> {direction_name: direction_vector}

    Handles two key formats:
    - "layer_N_name" or "layer_N" (old format, e.g., from introspection experiments)
    - "name_layer_N" (new format from identify_mc_correlate.py, e.g., "probe_layer_0", "mean_diff_layer_0")
    """
    data = np.load(path)

    directions = defaultdict(dict)

    for key in data.files:
        # Skip metadata keys
        if key.startswith("_metadata"):
            continue

        parts = key.split("_")

        # Try format 1: "layer_N_name" or "layer_N"
        if parts[0] == "layer" and len(parts) >= 2:
            try:
                layer_idx = int(parts[1])
                if len(parts) > 2:
                    direction_name = "_".join(parts[2:])
                else:
                    # For files with just "layer_N" keys, use "probe" as direction name
                    direction_name = "probe"
                directions[layer_idx][direction_name] = data[key]
                continue
            except ValueError:
                pass

        # Try format 2: "name_layer_N" (from identify_mc_correlate.py)
        # Look for "_layer_" in the key
        if "_layer_" in key:
            # Split on "_layer_" to get (name, layer_idx)
            layer_pos = key.rfind("_layer_")
            direction_name = key[:layer_pos]
            try:
                layer_idx = int(key[layer_pos + 7:])  # len("_layer_") == 7
                # Skip scaler keys (probe_scaler_scale_N, probe_scaler_mean_N)
                if "scaler" in direction_name:
                    continue
                directions[layer_idx][direction_name] = data[key]
            except ValueError:
                pass

    return dict(directions)


def load_directions_pt(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load directions from a .pt file (PyTorch tensor).

    These files are saved by confidence_contrast.py and have shape (num_layers, hidden_dim).

    Args:
        path: Path to .pt file

    Returns dict mapping layer_idx -> {direction_name: direction_vector}
    """
    tensor = torch.load(path, map_location="cpu", weights_only=True)

    # Extract direction type from filename
    # e.g., "..._entropy_contrast_directions.pt" -> "mean_diff"
    # e.g., "..._confidence_contrast_directions.pt" -> "mean_diff"
    direction_name = "mean_diff"

    directions = {}
    num_layers = tensor.shape[0]

    for layer_idx in range(num_layers):
        directions[layer_idx] = {direction_name: tensor[layer_idx].numpy()}

    return directions


def load_directions_auto(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load directions from either .npz or .pt file based on extension.

    Args:
        path: Path to direction file

    Returns dict mapping layer_idx -> {direction_name: direction_vector}
    """
    if path.suffix == ".pt":
        return load_directions_pt(path)
    elif path.suffix == ".npz":
        return load_directions(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def compute_cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))


def compute_pairwise_similarities(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layer_idx: int
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise cosine similarities between all direction types at a given layer.

    Returns dict mapping (type1, type2) -> cosine_similarity
    """
    similarities = {}

    # Flatten to get all (source, name) pairs
    direction_items = []
    for source, layers in all_directions.items():
        if layer_idx in layers:
            for name, direction in layers[layer_idx].items():
                full_name = f"{source}/{name}"
                direction_items.append((full_name, direction))

    # Compute pairwise
    for i, (name1, d1) in enumerate(direction_items):
        for j, (name2, d2) in enumerate(direction_items):
            if i <= j:
                sim = compute_cosine_similarity(d1, d2)
                similarities[(name1, name2)] = sim
                similarities[(name2, name1)] = sim

    return similarities


def load_lm_head_and_norm(model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load lm_head weight and final norm weight directly from model files.

    This bypasses the model loading and directly loads just the weights needed
    for logit lens from the safetensors files. Much faster and uses less memory
    than loading the full model.

    Returns:
        Tuple of (lm_head_weight, norm_weight)
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print(f"  Downloading weight index...")

    # Download the index file to find which shards have our weights
    index_file = hf_hub_download(
        repo_id=model_name,
        filename="model.safetensors.index.json",
        token=os.environ.get("HF_TOKEN")
    )

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Find which files contain our weights
    lm_head_file = weight_map.get("lm_head.weight")
    norm_file = weight_map.get("model.norm.weight")

    if not lm_head_file:
        raise ValueError(f"Could not find lm_head.weight in model index")

    # Download and load lm_head
    print(f"  Downloading {lm_head_file}...")
    shard_path = hf_hub_download(
        repo_id=model_name,
        filename=lm_head_file,
        token=os.environ.get("HF_TOKEN")
    )

    print(f"  Loading lm_head weight to {DEVICE}...")
    with safe_open(shard_path, framework="pt", device=DEVICE) as f:
        lm_head_weight = f.get_tensor("lm_head.weight")

    print(f"  Loaded lm_head weight: {lm_head_weight.shape}, dtype: {lm_head_weight.dtype}")

    # Download and load norm weight (may be in same or different shard)
    norm_weight = None
    if norm_file:
        if norm_file != lm_head_file:
            print(f"  Downloading {norm_file}...")
            norm_shard_path = hf_hub_download(
                repo_id=model_name,
                filename=norm_file,
                token=os.environ.get("HF_TOKEN")
            )
        else:
            norm_shard_path = shard_path

        print(f"  Loading norm weight...")
        with safe_open(norm_shard_path, framework="pt", device=DEVICE) as f:
            norm_weight = f.get_tensor("model.norm.weight")
        print(f"  Loaded norm weight: {norm_weight.shape}")
    else:
        print(f"  Warning: Could not find model.norm.weight, skipping normalization")

    return lm_head_weight, norm_weight


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm to a vector."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return weight * x_normed


def clean_token_str(s: str) -> str:
    """Clean token string for display - remove non-ASCII and problematic chars."""
    # Remove non-ASCII characters
    s = re.sub(r'[^\x00-\x7F]+', '', str(s))
    # Remove characters that might trigger MathText parsing
    s = re.sub(r'[\$\^\\]', '', s)
    # Replace newlines and tabs with visible representation
    s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    # Limit length
    if len(s) > 12:
        s = s[:10] + '..'
    return s if s.strip() else repr(s)


def logit_lens_for_layer(
    direction: np.ndarray,
    lm_head_weight: torch.Tensor,
    tokenizer,
    top_k: int = 12,
    norm_weight: Optional[torch.Tensor] = None,
    show_both_poles: bool = True
) -> Dict[str, Tuple[List[str], List[float]]]:
    """
    Project a direction through the unembedding matrix.

    For contrast directions (mean_high - mean_low), this shows:
    - TOP tokens: tokens that become MORE likely when moving toward HIGH end
    - BOTTOM tokens: tokens that become MORE likely when moving toward LOW end

    Args:
        direction: The direction vector to project
        lm_head_weight: The unembedding matrix
        tokenizer: Tokenizer for decoding
        top_k: Number of top/bottom tokens to return
        norm_weight: If provided, apply RMSNorm before unembedding
        show_both_poles: If True, return both top and bottom tokens

    Returns:
        Dict with "top" and optionally "bottom" keys, each containing (tokens, logits)
        Uses raw logits (not softmax) since direction is a difference vector.
    """
    # Project direction through unembedding
    direction_tensor = torch.tensor(
        direction,
        dtype=lm_head_weight.dtype,
        device=lm_head_weight.device
    )

    # Apply RMSNorm if weights provided (matches model's forward pass)
    if norm_weight is not None:
        direction_tensor = rms_norm(direction_tensor, norm_weight)

    logits = direction_tensor @ lm_head_weight.T  # (vocab_size,)

    results = {}

    # Get top-k (positive pole - e.g., HIGH entropy direction)
    top_values, top_indices = torch.topk(logits, top_k)
    top_tokens = tokenizer.batch_decode(top_indices.unsqueeze(-1))
    results["top"] = (top_tokens, top_values.cpu().tolist())

    # Get bottom-k (negative pole - e.g., LOW entropy direction)
    if show_both_poles:
        bottom_values, bottom_indices = torch.topk(logits, top_k, largest=False)
        bottom_tokens = tokenizer.batch_decode(bottom_indices.unsqueeze(-1))
        results["bottom"] = (bottom_tokens, bottom_values.cpu().tolist())

    return results


def analyze_layer(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layer_idx: int,
    lm_head_weight: Optional[torch.Tensor],
    tokenizer,
    top_k: int = 12,
    top_k_summary: int = 20,
    norm_weight: Optional[torch.Tensor] = None
) -> Dict:
    """
    Run full analysis on a single layer.

    Returns dict with similarities and logit lens results.
    
    Args:
        top_k: Number of tokens for heatmap visualization
        top_k_summary: Number of tokens to store in JSON/txt (usually larger)
    """
    results = {
        "layer": layer_idx,
        "similarities": {},
        "logit_lens": {},
    }

    # Compute pairwise similarities
    similarities = compute_pairwise_similarities(all_directions, layer_idx)
    results["similarities"] = {f"{k[0]}__vs__{k[1]}": v for k, v in similarities.items()}

    # Run logit lens on each direction (if weight available)
    # Use top_k_summary for JSON storage (larger set of tokens)
    if lm_head_weight is not None:
        for source, layers in all_directions.items():
            if layer_idx in layers:
                for name, direction in layers[layer_idx].items():
                    full_name = f"{source}/{name}"
                    lens_results = logit_lens_for_layer(direction, lm_head_weight, tokenizer, top_k_summary, norm_weight)
                    # Store both poles
                    results["logit_lens"][full_name] = {
                        "top_tokens": lens_results["top"][0],
                        "top_logits": lens_results["top"][1],
                    }
                    if "bottom" in lens_results:
                        results["logit_lens"][full_name]["bottom_tokens"] = lens_results["bottom"][0]
                        results["logit_lens"][full_name]["bottom_logits"] = lens_results["bottom"][1]

    return results


def generate_logit_lens_summary(
    all_results: Dict[int, Dict],
    output_path: Path,
    model_name: str
):
    """
    Generate a human-readable .txt summary of logit lens results.
    
    For each direction (contrast/probe), shows:
    - Summary: top and bottom token per layer with logit value
    - Full list: top 20 and bottom 20 tokens per layer with logit values
    
    Note: Values are raw logits, not probabilities. For contrast directions,
    positive logits = tokens more likely at HIGH end of the contrast,
    negative logits = tokens more likely at LOW end.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("LOGIT LENS ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Model: {model_name}")
    lines.append(f"Generated: {Path(output_path).name}")
    lines.append("")
    lines.append("NOTE: Values shown are raw logits (not probabilities).")
    lines.append("For contrast directions (high - low):")
    lines.append("  - Positive logits = tokens more likely at HIGH end")
    lines.append("  - Negative logits = tokens more likely at LOW end")
    lines.append("")
    
    # Collect all direction names across all layers
    all_direction_names = set()
    for layer_results in all_results.values():
        if "logit_lens" in layer_results:
            all_direction_names.update(layer_results["logit_lens"].keys())
    
    # Sort layers
    sorted_layers = sorted(all_results.keys())
    
    # Process each direction separately
    for direction_name in sorted(all_direction_names):
        lines.append("=" * 80)
        lines.append(f"DIRECTION: {direction_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # Part 1: Quick summary - top/bottom token per layer
        lines.append("-" * 60)
        lines.append("SUMMARY: Top and Bottom Token per Layer")
        lines.append("-" * 60)
        lines.append(f"{'Layer':<8} {'Top Token':<20} {'Top Logit':<12} {'Bottom Token':<20} {'Bottom Logit':<12}")
        lines.append("-" * 60)
        
        for layer_idx in sorted_layers:
            layer_results = all_results[layer_idx]
            if "logit_lens" not in layer_results:
                continue
            if direction_name not in layer_results["logit_lens"]:
                continue
            
            lens_data = layer_results["logit_lens"][direction_name]
            top_tokens = lens_data.get("top_tokens", [])
            top_logits = lens_data.get("top_logits", [])
            bottom_tokens = lens_data.get("bottom_tokens", [])
            bottom_logits = lens_data.get("bottom_logits", [])
            
            # Get #1 top and bottom
            top_tok = repr(top_tokens[0])[:18] if top_tokens else "N/A"
            top_val = f"{top_logits[0]:.4f}" if top_logits else "N/A"
            bot_tok = repr(bottom_tokens[0])[:18] if bottom_tokens else "N/A"
            bot_val = f"{bottom_logits[0]:.4f}" if bottom_logits else "N/A"
            
            lines.append(f"L{layer_idx:<7} {top_tok:<20} {top_val:<12} {bot_tok:<20} {bot_val:<12}")
        
        lines.append("")
        
        # Part 2: Full token lists per layer
        lines.append("-" * 60)
        lines.append("FULL TOKEN LISTS (Top 20 and Bottom 20 per layer)")
        lines.append("-" * 60)
        
        for layer_idx in sorted_layers:
            layer_results = all_results[layer_idx]
            if "logit_lens" not in layer_results:
                continue
            if direction_name not in layer_results["logit_lens"]:
                continue
            
            lens_data = layer_results["logit_lens"][direction_name]
            top_tokens = lens_data.get("top_tokens", [])
            top_logits = lens_data.get("top_logits", [])
            bottom_tokens = lens_data.get("bottom_tokens", [])
            bottom_logits = lens_data.get("bottom_logits", [])
            
            lines.append("")
            lines.append(f"=== Layer {layer_idx} ===")
            
            # Top tokens
            lines.append(f"  TOP {len(top_tokens)} tokens (HIGH end of direction):")
            for i, (tok, logit) in enumerate(zip(top_tokens, top_logits)):
                lines.append(f"    {i+1:>2}. {logit:>8.4f}  {repr(tok)}")
            
            # Bottom tokens
            if bottom_tokens:
                lines.append(f"  BOTTOM {len(bottom_tokens)} tokens (LOW end of direction):")
                for i, (tok, logit) in enumerate(zip(bottom_tokens, bottom_logits)):
                    lines.append(f"    {i+1:>2}. {logit:>8.4f}  {repr(tok)}")
        
        lines.append("")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Saved logit lens summary to {output_path}")


def get_model_display_name() -> str:
    """Get a human-readable model name for plot titles."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if ADAPTER:
        return f"{model_short} + Fine-tuned Adapter"
    else:
        return f"{model_short} (Instruct, no adapter)"


def plot_logit_lens_heatmap(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layers: List[int],
    direction_source: str,
    direction_name: str,
    lm_head_weight: torch.Tensor,
    tokenizer,
    output_path: Path,
    top_k: int = 12,
    norm_weight: Optional[torch.Tensor] = None
):
    """
    Plot heatmap showing BOTH poles of a direction.

    For contrast directions (e.g., entropy_contrast = high - low):
    - TOP panel: tokens associated with HIGH end (positive logits)
    - BOTTOM panel: tokens associated with LOW end (negative logits)

    Rows = layers, Columns = top-k tokens
    Cell values = raw logits, annotations = token strings
    """
    top_token_data = []
    top_logit_data = []
    bottom_token_data = []
    bottom_logit_data = []

    for layer_idx in layers:
        if direction_source in all_directions and layer_idx in all_directions[direction_source]:
            if direction_name in all_directions[direction_source][layer_idx]:
                direction = all_directions[direction_source][layer_idx][direction_name]
                lens_results = logit_lens_for_layer(direction, lm_head_weight, tokenizer, top_k, norm_weight)
                top_token_data.append(lens_results["top"][0])
                top_logit_data.append(lens_results["top"][1])
                if "bottom" in lens_results:
                    bottom_token_data.append(lens_results["bottom"][0])
                    bottom_logit_data.append(lens_results["bottom"][1])
            else:
                top_token_data.append([''] * top_k)
                top_logit_data.append([0.0] * top_k)
                bottom_token_data.append([''] * top_k)
                bottom_logit_data.append([0.0] * top_k)
        else:
            top_token_data.append([''] * top_k)
            top_logit_data.append([0.0] * top_k)
            bottom_token_data.append([''] * top_k)
            bottom_logit_data.append([0.0] * top_k)

    if not top_logit_data:
        print(f"No data for {direction_source}/{direction_name}")
        return

    # Convert to arrays
    top_logit_array = np.array(top_logit_data)
    top_token_labels = np.array(top_token_data)
    bottom_logit_array = np.array(bottom_logit_data) if bottom_logit_data else None
    bottom_token_labels = np.array(bottom_token_data) if bottom_token_data else None

    # Clean token labels for display
    cleaned_top_tokens = np.vectorize(clean_token_str)(top_token_labels)
    cleaned_bottom_tokens = np.vectorize(clean_token_str)(bottom_token_labels) if bottom_token_labels is not None else None

    # Determine pole labels based on direction source
    if "entropy" in direction_source.lower():
        high_label = "HIGH Entropy (Uncertain)"
        low_label = "LOW Entropy (Confident)"
    elif "confidence" in direction_source.lower():
        high_label = "HIGH Stated Confidence"
        low_label = "LOW Stated Confidence"
    else:
        high_label = "Positive Direction"
        low_label = "Negative Direction"

    # Plot - two panels if we have both poles
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig_height = max(10, len(layers) * 0.3)

    if cleaned_bottom_tokens is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, fig_height))

        # Top panel - HIGH end (positive logits)
        sns.heatmap(
            top_logit_array,
            annot=cleaned_top_tokens,
            fmt='',
            cmap="Reds",
            xticklabels=False,
            yticklabels=[f"L{l}" for l in layers],
            ax=ax1,
            cbar_kws={'label': 'Logit Value'}
        )
        ax1.set_title(f"{high_label}\n(tokens that become MORE likely)")
        ax1.set_xlabel(f"Top {top_k} Tokens")
        ax1.set_ylabel("Layer")

        # Bottom panel - LOW end (negative logits)
        # Note: bottom logits are negative, so we negate for color intensity
        sns.heatmap(
            -bottom_logit_array,  # Negate so more negative = higher intensity
            annot=cleaned_bottom_tokens,
            fmt='',
            cmap="Blues",
            xticklabels=False,
            yticklabels=[f"L{l}" for l in layers],
            ax=ax2,
            cbar_kws={'label': '-Logit Value (more negative = stronger)'}
        )
        ax2.set_title(f"{low_label}\n(tokens that become MORE likely)")
        ax2.set_xlabel(f"Top {top_k} Tokens")
        ax2.set_ylabel("Layer")

        model_name = get_model_display_name()
        fig.suptitle(f"Logit Lens: {direction_source}/{direction_name}\n{model_name}\nBOTH POLES of the contrast direction", fontsize=14, y=1.02)
    else:
        # Single panel if we only have top
        fig, ax = plt.subplots(figsize=(14, fig_height))
        sns.heatmap(
            top_logit_array,
            annot=cleaned_top_tokens,
            fmt='',
            cmap="Reds",
            xticklabels=False,
            yticklabels=[f"L{l}" for l in layers],
            ax=ax,
            cbar_kws={'label': 'Logit Value'}
        )
        model_name = get_model_display_name()
        ax.set_title(f"Logit Lens: {direction_source}/{direction_name}\n{model_name}\n(Top {top_k} tokens per layer)")
        ax.set_xlabel(f"Top {top_k} Tokens")
        ax.set_ylabel("Layer")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved logit lens heatmap to {output_path}")
    plt.close()


def plot_similarity_across_layers(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layers: List[int],
    output_path: Path
):
    """
    Plot how cosine similarity between direction types evolves across layers.
    """
    # Get all unique direction pairs
    all_names = set()
    for source, layer_data in all_directions.items():
        for layer_idx, directions in layer_data.items():
            for name in directions.keys():
                all_names.add(f"{source}/{name}")

    all_names = sorted(all_names)

    if len(all_names) < 2:
        # Skip silently - only one direction type
        return

    # Compute similarities for each layer
    pair_data = defaultdict(list)

    for layer_idx in layers:
        sims = compute_pairwise_similarities(all_directions, layer_idx)
        for (n1, n2), sim in sims.items():
            if n1 < n2:  # Avoid duplicates
                pair_data[(n1, n2)].append((layer_idx, sim))

    # Plot
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(12, 6))

    for (n1, n2), data in pair_data.items():
        if data:
            xs, ys = zip(*sorted(data))
            # Use source names for clearer labels
            # n1, n2 are like "source/direction_name"
            src1, dir1 = n1.split('/', 1)
            src2, dir2 = n2.split('/', 1)
            # Shorten source names for readability
            src1_short = src1.replace("_entropy", "").replace("introspection_", "intro_")
            src2_short = src2.replace("_entropy", "").replace("introspection_", "intro_")
            label = f"{src1_short} vs {src2_short}"
            ax.plot(xs, ys, 'o-', label=label, alpha=0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    model_name = get_model_display_name()
    ax.set_title(f"Direction Similarity Across Layers\n{model_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved similarity-across-layers plot to {output_path}")
    plt.close()


def plot_token_intersection(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layers: List[int],
    lm_head_weight: torch.Tensor,
    tokenizer,
    output_path: Path,
    top_k: int = 12,
    norm_weight: Optional[torch.Tensor] = None
):
    """
    Plot tokens that appear in multiple direction types at each layer.

    Shows ALL 4 combinations:
    - HIGH entropy ∩ LOW confidence (both indicate uncertainty - SHOULD overlap)
    - LOW entropy ∩ HIGH confidence (both indicate confidence - SHOULD overlap)
    - HIGH entropy ∩ HIGH confidence (MISMATCH - entropy says uncertain, confidence says certain)
    - LOW entropy ∩ LOW confidence (MISMATCH - entropy says certain, confidence says uncertain)
    """
    # Collect tokens for each direction at each layer
    # Structure: {layer: {direction_key: {pole: set(tokens)}}}
    layer_tokens: Dict[int, Dict[str, Dict[str, set]]] = defaultdict(lambda: defaultdict(dict))

    for source, layers_dict in all_directions.items():
        for layer_idx in layers:
            if layer_idx not in layers_dict:
                continue
            for dir_name, direction in layers_dict[layer_idx].items():
                lens_results = logit_lens_for_layer(
                    direction, lm_head_weight, tokenizer, top_k, norm_weight
                )
                key = f"{source}/{dir_name}"
                layer_tokens[layer_idx][key]["top"] = set(lens_results["top"][0])
                if "bottom" in lens_results:
                    layer_tokens[layer_idx][key]["bottom"] = set(lens_results["bottom"][0])

    # Find entropy and confidence keys
    entropy_keys = [k for k in layer_tokens[layers[0]].keys() if "entropy" in k.lower()]
    confidence_keys = [k for k in layer_tokens[layers[0]].keys() if "confidence" in k.lower()]

    if not entropy_keys or not confidence_keys:
        print("Need both entropy and confidence contrasts for intersection plot")
        return

    entropy_key = entropy_keys[0]
    confidence_key = confidence_keys[0]

    # Collect all 4 intersection combinations
    # Aligned (should overlap):
    high_ent_low_conf = []   # Both say "uncertain"
    low_ent_high_conf = []   # Both say "confident"
    # Mismatched:
    high_ent_high_conf = []  # Entropy=uncertain, Confidence=confident
    low_ent_low_conf = []    # Entropy=confident, Confidence=uncertain

    for layer_idx in layers:
        if layer_idx not in layer_tokens:
            high_ent_low_conf.append(set())
            low_ent_high_conf.append(set())
            high_ent_high_conf.append(set())
            low_ent_low_conf.append(set())
            continue

        entropy_data = layer_tokens[layer_idx].get(entropy_key, {})
        confidence_data = layer_tokens[layer_idx].get(confidence_key, {})

        high_entropy = entropy_data.get("top", set())
        low_entropy = entropy_data.get("bottom", set())
        high_confidence = confidence_data.get("top", set())
        low_confidence = confidence_data.get("bottom", set())

        # All 4 combinations
        high_ent_low_conf.append(high_entropy & low_confidence)
        low_ent_high_conf.append(low_entropy & high_confidence)
        high_ent_high_conf.append(high_entropy & high_confidence)
        low_ent_low_conf.append(low_entropy & low_confidence)

    # Create visualization - 2x2 grid
    model_name = get_model_display_name()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig_height = max(10, len(layers) * 0.25)
    fig, axes = plt.subplots(2, 2, figsize=(24, fig_height * 2))

    def plot_intersection(ax, intersections, title, color, layers):
        counts = [len(s) for s in intersections]
        ax.barh(range(len(layers)), counts, color=color, alpha=0.7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Number of overlapping tokens")
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()

        # Add token labels
        for i, (count, tokens) in enumerate(zip(counts, intersections)):
            if count > 0:
                token_str = ", ".join(clean_token_str(t) for t in sorted(tokens)[:6])
                if len(tokens) > 6:
                    token_str += f" +{len(tokens)-6}"
                ax.text(count + 0.1, i, token_str, va='center', fontsize=6)

        return np.mean(counts)

    # Top row: ALIGNED combinations (should overlap)
    avg1 = plot_intersection(
        axes[0, 0], high_ent_low_conf,
        f"ALIGNED: High Entropy ∩ Low Confidence\n(Both = UNCERTAIN)",
        'purple', layers
    )
    avg2 = plot_intersection(
        axes[0, 1], low_ent_high_conf,
        f"ALIGNED: Low Entropy ∩ High Confidence\n(Both = CONFIDENT)",
        'green', layers
    )

    # Bottom row: MISMATCHED combinations
    avg3 = plot_intersection(
        axes[1, 0], high_ent_high_conf,
        f"MISMATCH: High Entropy ∩ High Confidence\n(Entropy=uncertain, Confidence=confident)",
        'orange', layers
    )
    avg4 = plot_intersection(
        axes[1, 1], low_ent_low_conf,
        f"MISMATCH: Low Entropy ∩ Low Confidence\n(Entropy=confident, Confidence=uncertain)",
        'red', layers
    )

    fig.suptitle(
        f"Token Intersection: ALL 4 Combinations of Entropy vs Confidence\n"
        f"{model_name}\n"
        f"(Top {top_k} tokens from each direction)",
        fontsize=14, y=1.02
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved token intersection plot to {output_path}")
    plt.close()

    # Print summary
    print(f"\n  Token Intersection Summary (avg tokens/layer):")
    print(f"    ALIGNED - High Entropy ∩ Low Confidence:  {avg1:.1f}")
    print(f"    ALIGNED - Low Entropy ∩ High Confidence:  {avg2:.1f}")
    print(f"    MISMATCH - High Entropy ∩ High Confidence: {avg3:.1f}")
    print(f"    MISMATCH - Low Entropy ∩ Low Confidence:  {avg4:.1f}")
    print(f"  If contrasts are similar, ALIGNED should be higher than MISMATCH.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare probe directions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze only entropy contrast (logit lens on just this direction)
  python analyze_directions.py --types entropy_contrast

  # Analyze only confidence contrast
  python analyze_directions.py --types confidence_contrast

  # Analyze both contrasts (compare them to each other)
  python analyze_directions.py --types entropy_contrast,confidence_contrast

  # Analyze MC probe directions only
  python analyze_directions.py --types mc_entropy

  # Compare probe to contrast
  python analyze_directions.py --types mc_entropy,entropy_contrast

  # List available direction types without running analysis
  python analyze_directions.py --list-types
        """
    )
    parser.add_argument("--model-only", action="store_true",
                        help="Only load model, skip analysis")
    parser.add_argument("--layer", type=int, default=None,
                        help="Focus on specific layer")
    parser.add_argument("--metric", type=str, default=None, choices=AVAILABLE_METRICS,
                        help="Only analyze directions for this metric")
    parser.add_argument("--types", type=str, default=None,
                        help="Comma-separated list of direction types to analyze "
                             "(e.g., 'entropy_contrast,confidence_contrast'). "
                             "Use --list-types to see available types.")
    parser.add_argument("--list-types", action="store_true",
                        help="List available direction types and exit")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--skip-logit-lens", action="store_true",
                        help="Skip logit lens analysis (no model loading needed)")
    args = parser.parse_args()

    # Find direction files first (needed for --list-types)
    model_short = get_model_short_name(BASE_MODEL_NAME)
    direction_files = find_direction_files(OUTPUTS_DIR, model_short, metric_filter=args.metric)

    # Handle --list-types: show available types and exit
    if args.list_types:
        print("=" * 60)
        print("AVAILABLE DIRECTION TYPES")
        print("=" * 60)
        print(f"Output directory: {OUTPUTS_DIR}")
        print(f"Model pattern: {model_short}*")
        print()
        if direction_files:
            print("Found direction types:")
            for name, path in sorted(direction_files.items()):
                print(f"  {name}")
                print(f"      -> {path.name}")
            print()
            print("Use --types to select which to analyze, e.g.:")
            print("  --types entropy_contrast")
            print("  --types entropy_contrast,confidence_contrast")
            print("  --types mc_entropy,entropy_contrast")
        else:
            print("No direction files found.")
        return

    # Print configuration
    print("=" * 60)
    print("DIRECTION ANALYSIS CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Adapter: {ADAPTER if ADAPTER else 'None (base model)'}")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Contrast subdirs to search: {CONTRAST_SUBDIRS}")
    if args.metric:
        print(f"Metric filter: {args.metric}")
    if args.types:
        print(f"Direction type filter: {args.types}")
    print("=" * 60)

    if not direction_files:
        print(f"\nNo direction files found in {OUTPUTS_DIR} for model {model_short}")
        if args.metric:
            print(f"  (filtered by metric: {args.metric})")
        print("\nRun one of these scripts first to generate direction files:")
        print("  - mc_entropy_probe.py          (.npz - trained probe directions)")
        print("  - confidence_contrast.py       (.pt  - mean-diff contrast directions)")
        print("  - nexttoken_entropy_probe.py   (.npz - next-token probe directions)")
        print("  - run_introspection_experiment.py")
        print(f"\nSearched in: {OUTPUTS_DIR}")
        print(f"Also searched subdirs: {CONTRAST_SUBDIRS}")
        return

    # Filter by --types if specified
    if args.types:
        type_filters = [t.strip() for t in args.types.split(",")]
        filtered_files = {}
        for name, path in direction_files.items():
            # Check if any filter matches the start of the direction name
            for type_filter in type_filters:
                if name.startswith(type_filter):
                    filtered_files[name] = path
                    break
        
        if not filtered_files:
            print(f"\nNo direction files match --types={args.types}")
            print("\nAvailable types:")
            for name in sorted(direction_files.keys()):
                print(f"  {name}")
            return
        
        direction_files = filtered_files

    print(f"\nAnalyzing {len(direction_files)} direction file(s):")
    for name, path in direction_files.items():
        print(f"  {name}: {path.name}")

    # Load all directions (supports both .npz and .pt files)
    all_directions = {}
    for source, path in direction_files.items():
        all_directions[source] = load_directions_auto(path)
        print(f"  Loaded {source}: {len(all_directions[source])} layers ({path.suffix})")

    # Determine layers to analyze
    all_layers = set()
    for layers_dict in all_directions.values():
        all_layers.update(layers_dict.keys())
    all_layers = sorted(all_layers)

    if args.layer is not None:
        layers_to_analyze = [args.layer]
    elif LAYERS_TO_ANALYZE is not None:
        layers_to_analyze = LAYERS_TO_ANALYZE
    else:
        layers_to_analyze = all_layers

    print(f"\nLayers available: {len(all_layers)} layers")
    print(f"Layers to analyze: {len(layers_to_analyze)} layers")

    # Load tokenizer and lm_head weight for logit lens (unless skipped)
    tokenizer = None
    lm_head_weight = None
    norm_weight = None

    if not args.skip_logit_lens:
        from transformers import AutoTokenizer
        from dotenv import load_dotenv

        load_dotenv()

        print(f"\nLoading tokenizer: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            token=os.environ.get("HF_TOKEN")
        )

        if args.model_only:
            print("Tokenizer loaded. Exiting (--model-only specified)")
            return

        # Load lm_head weight and norm weight directly (much faster than loading full model)
        print(f"\nLoading lm_head and norm weights for logit lens...")
        lm_head_weight, norm_weight = load_lm_head_and_norm(BASE_MODEL_NAME)
    else:
        print("\nSkipping logit lens analysis (--skip-logit-lens)")

    # Run analysis
    output_prefix = get_output_prefix()
    all_results = {}

    for layer_idx in tqdm(layers_to_analyze, desc="Analyzing layers"):
        results = analyze_layer(
            all_directions, layer_idx, lm_head_weight, tokenizer,
            top_k=TOP_K_TOKENS, top_k_summary=TOP_K_TOKENS_SUMMARY, norm_weight=norm_weight
        )
        all_results[layer_idx] = results

    # Save results
    results_path = Path(f"{output_prefix}_direction_analysis.json")

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\nSaved analysis results to {results_path}")

    # Generate human-readable .txt summary of logit lens results
    if lm_head_weight is not None:
        summary_path = Path(f"{output_prefix}_logit_lens_summary.txt")
        generate_logit_lens_summary(all_results, summary_path, get_model_display_name())

    # Generate plots
    if not args.no_plots:
        # Check if we have multiple direction types for similarity plots
        num_direction_types = sum(
            1 for source in all_directions.values()
            for layer_dirs in source.values()
            for _ in layer_dirs.keys()
        ) // len(all_layers) if all_layers else 0

        if num_direction_types >= 2:
            # Similarity across layers
            plot_similarity_across_layers(
                all_directions, all_layers,
                Path(f"{output_prefix}_direction_similarity_across_layers.png")
            )
        else:
            print("\nOnly one direction type found - skipping similarity plot")

        # Logit lens heatmaps for each direction type (if we have weights)
        if lm_head_weight is not None:
            print(f"\nGenerating logit lens heatmaps for {len(all_directions)} direction source(s)...")
            for source, layers_dict in all_directions.items():
                # Get direction names from first available layer
                first_layer = next(iter(layers_dict.keys()))
                direction_names = list(layers_dict[first_layer].keys())
                print(f"  Source: {source}")
                print(f"    Directions: {direction_names}")
                for direction_name in direction_names:
                    # Parse source key to generate clean filename
                    # Source formats:
                    #   - mc_entropy_TriviaMC (probe: dir_type=mc, metric=entropy, dataset=TriviaMC)
                    #   - entropy_contrast_TriviaMC (contrast: dir_type=entropy_contrast, dataset=TriviaMC)
                    #   - confidence_contrast_TriviaMC (contrast: dir_type=confidence_contrast, dataset=TriviaMC)

                    # Handle contrast directions specially (they have "_contrast_" in the name)
                    if "_contrast_" in source:
                        # Format: {type}_contrast_{dataset}
                        parts = source.split("_contrast_")
                        contrast_type = parts[0]  # "entropy" or "confidence"
                        dataset = parts[1] if len(parts) > 1 else ""
                        dir_type = f"{contrast_type}_contrast"

                        if dataset:
                            base = f"{output_prefix}_{dataset}_{dir_type}_logit_lens"
                        else:
                            base = f"{output_prefix}_{dir_type}_logit_lens"
                    else:
                        # Handle probe directions: {dir_type}_{metric}_{dataset}
                        source_parts = source.split("_")

                        # Find metric in source (it's one of AVAILABLE_METRICS)
                        metric_idx = None
                        for i, part in enumerate(source_parts):
                            if part in AVAILABLE_METRICS:
                                metric_idx = i
                                break

                        if metric_idx is not None:
                            dir_type = "_".join(source_parts[:metric_idx])
                            metric = source_parts[metric_idx]
                            dataset = "_".join(source_parts[metric_idx + 1:]) if metric_idx + 1 < len(source_parts) else ""

                            if dataset:
                                base = f"{output_prefix}_{dataset}_{dir_type}_{metric}_logit_lens"
                            else:
                                base = f"{output_prefix}_{dir_type}_{metric}_logit_lens"
                        else:
                            # Fallback
                            base = f"{output_prefix}_logit_lens_{source}"

                    # Add direction_name suffix if it adds info (skip for mean_diff on contrasts, probe on probes)
                    if direction_name == "mean_diff" and "_contrast" in source:
                        filename = f"{base}.png"
                    elif direction_name == "probe":
                        filename = f"{base}.png"
                    elif direction_name in source:
                        filename = f"{base}.png"
                    else:
                        filename = f"{base}_{direction_name}.png"

                    print(f"    -> Plotting: {Path(filename).name}")
                    plot_logit_lens_heatmap(
                        all_directions, all_layers, source, direction_name,
                        lm_head_weight, tokenizer,
                        Path(filename),
                        top_k=TOP_K_TOKENS,
                        norm_weight=norm_weight
                    )

            # Token intersection plot (if we have both entropy and confidence contrasts)
            has_entropy = any("entropy" in k.lower() for k in all_directions.keys())
            has_confidence = any("confidence" in k.lower() for k in all_directions.keys())

            if has_entropy and has_confidence:
                print("\nGenerating token intersection plot (entropy vs confidence)...")
                plot_token_intersection(
                    all_directions, all_layers,
                    lm_head_weight, tokenizer,
                    Path(f"{output_prefix}_token_intersection.png"),
                    top_k=TOP_K_TOKENS,
                    norm_weight=norm_weight
                )
            else:
                print("\nSkipping intersection plot (need both entropy and confidence contrasts)")

    print("\n" + "="*80)
    print("DIRECTION ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
