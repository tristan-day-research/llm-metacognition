"""
Analyze shared vs unique components of MC entropy directions for introspection.

This script tests whether the model uses shared or unique MC uncertainty signals
when performing meta-judgment (answering "how confident are you?").

Approach:
1. Load MC entropy directions from multiple datasets (SimpleMC, TriviaMC, GPQA)
2. Decompose into:
   - Shared component: Average of normalized MC directions (what's common)
   - Unique components: Residuals after projecting out shared
3. For a specified dataset:
   - Load direct activations (from mc_entropy_probe.py)
   - Load meta activations (from run_introspection_experiment.py)
4. Train 1D probes along shared/unique directions on direct activations
5. Test whether those probes TRANSFER to meta activations

Key question: When the model judges its own confidence, is it using:
- The shared MC uncertainty signal? (general uncertainty representation)
- Dataset-specific unique signals? (domain-specific calibration)

Usage:
    python analyze_shared_unique.py --dataset SimpleMC
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from core import (
    DEVICE,
    get_model_short_name,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Analysis config
TRAIN_SPLIT = 0.8
SEED = 42

np.random.seed(SEED)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
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

    # The dataset is the last underscore-separated component
    parts = filename.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]
    return None


def find_mc_direction_files(output_dir: Path, model_short: str) -> Dict[str, Path]:
    """
    Find all MC/introspection direction files for a given model.

    Searches for:
    - mc_entropy_probe.py: {model}_{dataset}_mc_entropy_directions.npz
    - run_introspection_experiment.py: {model}_{dataset}_introspection_{metric}_directions.npz

    Returns dict mapping dataset_name -> path.
    """
    mc_files = {}

    # Pattern 1: mc_entropy_probe.py output
    mc_pattern = f"{model_short}*_mc_entropy_directions.npz"
    mc_matches = list(output_dir.glob(mc_pattern))

    for path in mc_matches:
        dataset = extract_dataset_from_npz(path)
        if dataset is None:
            dataset = extract_dataset_from_filename(path.name, "_mc_entropy_directions")

        if dataset:
            if dataset not in mc_files or path.stat().st_mtime > mc_files[dataset].stat().st_mtime:
                mc_files[dataset] = path

    # Pattern 2: run_introspection_experiment.py output
    # Matches: {model}_{dataset}_introspection_{metric}_directions.npz
    introspection_pattern = f"{model_short}*_introspection_*_directions.npz"
    introspection_matches = list(output_dir.glob(introspection_pattern))

    for path in introspection_matches:
        dataset = extract_dataset_from_npz(path)
        if dataset is None:
            # Extract dataset from filename: {model}_{dataset}_introspection_{metric}_directions.npz
            # Try to extract by finding _introspection_ and taking what's before it
            filename = path.stem  # Remove .npz
            if "_introspection_" in filename:
                prefix = filename.split("_introspection_")[0]
                # Dataset is the last component before _introspection
                parts = prefix.rsplit("_", 1)
                if len(parts) == 2:
                    dataset = parts[1]

        if dataset:
            # Only add if we don't already have this dataset, or if this file is newer
            if dataset not in mc_files or path.stat().st_mtime > mc_files[dataset].stat().st_mtime:
                mc_files[dataset] = path

    return mc_files


def find_mc_activations_file(output_dir: Path, model_short: str, dataset: str) -> Optional[Path]:
    """
    Find direct activations file for a specific dataset.

    Searches for:
    - mc_entropy_probe.py: {model}_{dataset}_mc_activations.npz
    - run_introspection_experiment.py: {model}_{dataset}_introspection_direct_activations.npz

    Returns most recently modified match.
    """
    candidates = []

    # Pattern 1: mc_entropy_probe.py output
    mc_pattern = f"{model_short}*_{dataset}_mc_activations.npz"
    candidates.extend(output_dir.glob(mc_pattern))

    # Pattern 2: run_introspection_experiment.py output
    introspection_pattern = f"{model_short}*_{dataset}_introspection_direct_activations.npz"
    candidates.extend(output_dir.glob(introspection_pattern))

    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def find_introspection_files(output_dir: Path, model_short: str, dataset: str) -> Dict[str, Path]:
    """
    Find introspection experiment files for a specific dataset.

    Returns dict with keys: 'meta_activations', 'paired_data'
    (Direct activations come from mc_entropy_probe.py instead)
    """
    files = {}

    # Meta activations (from run_introspection_experiment.py)
    meta_pattern = f"{model_short}*_{dataset}_introspection_meta_activations.npz"
    meta_matches = list(output_dir.glob(meta_pattern))
    if meta_matches:
        files['meta_activations'] = max(meta_matches, key=lambda p: p.stat().st_mtime)

    # Paired data (for confidence values)
    paired_pattern = f"{model_short}*_{dataset}_introspection_paired_data.json"
    paired_matches = list(output_dir.glob(paired_pattern))
    if paired_matches:
        files['paired_data'] = max(paired_matches, key=lambda p: p.stat().st_mtime)

    return files


def load_mc_directions(path: Path) -> Dict[int, np.ndarray]:
    """
    Load MC entropy directions from a .npz file.

    Returns dict mapping layer_idx -> direction_vector (normalized).
    """
    data = np.load(path)
    directions = {}

    for key in data.files:
        parts = key.split("_")
        if parts[0] == "layer" and len(parts) >= 2:
            try:
                layer_idx = int(parts[1])
                direction = data[key]
                direction = direction / np.linalg.norm(direction)
                directions[layer_idx] = direction
            except ValueError:
                continue

    return directions


def load_mc_activations(mc_activations_path: Path) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load direct activations and entropies from mc_entropy_probe.py output.

    Returns:
        direct_activations: Dict[layer_idx -> (n_samples, hidden_dim)]
        entropies: np.ndarray of shape (n_samples,)
    """
    data = np.load(mc_activations_path)
    activations = {}
    entropies = None

    for key in data.files:
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            activations[layer_idx] = data[key]
        elif key == "entropies":
            entropies = data[key]

    if entropies is None:
        raise ValueError(f"No entropies found in {mc_activations_path}")

    return activations, entropies


def load_meta_activations(
    meta_activations_path: Path,
    paired_data_path: Path
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load meta activations from run_introspection_experiment.py output.

    Returns:
        meta_activations: Dict[layer_idx -> (n_samples, hidden_dim)]
        direct_entropies: np.ndarray of shape (n_samples,) - from paired data
        stated_confidences: np.ndarray of shape (n_samples,)
    """
    # Load meta activations
    meta_data = np.load(meta_activations_path)
    meta_activations = {}
    for key in meta_data.files:
        if key.startswith("layer_"):
            layer_idx = int(key.split("_")[1])
            meta_activations[layer_idx] = meta_data[key]

    # Load paired data for entropies and confidences
    with open(paired_data_path, 'r') as f:
        paired_data = json.load(f)

    direct_entropies = np.array(paired_data["direct_entropies"])

    # Convert meta responses to confidence values
    META_RANGE_MIDPOINTS = {
        "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
        "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
    }

    meta_task = paired_data.get("config", {}).get("meta_task", "confidence")

    if meta_task == "delegate":
        meta_probs = paired_data["meta_probs"]
        meta_mappings = paired_data.get("meta_mappings", [None] * len(meta_probs))
        stated_confidences = []
        for i, (probs, mapping) in enumerate(zip(meta_probs, meta_mappings)):
            if mapping is not None and mapping.get("1") == "Answer":
                stated_confidences.append(probs[0])
            else:
                stated_confidences.append(probs[1] if len(probs) > 1 else probs[0])
        stated_confidences = np.array(stated_confidences)
    else:
        meta_responses = paired_data["meta_responses"]
        stated_confidences = np.array([
            META_RANGE_MIDPOINTS.get(r, 0.5) for r in meta_responses
        ])

    return meta_activations, direct_entropies, stated_confidences


def compute_shared_component(
    mc_directions: Dict[str, Dict[int, np.ndarray]]
) -> Dict[int, np.ndarray]:
    """
    Compute shared component across MC directions at each layer.

    The shared component is the average of the normalized MC directions,
    then re-normalized.
    """
    all_layers = set()
    for directions in mc_directions.values():
        all_layers.update(directions.keys())

    shared = {}
    for layer_idx in all_layers:
        layer_directions = []
        for dataset, directions in mc_directions.items():
            if layer_idx in directions:
                layer_directions.append(directions[layer_idx])

        if len(layer_directions) < 2:
            continue

        stacked = np.stack(layer_directions, axis=0)
        mean_direction = stacked.mean(axis=0)
        shared[layer_idx] = mean_direction / np.linalg.norm(mean_direction)

    return shared


def compute_unique_components(
    mc_directions: Dict[str, Dict[int, np.ndarray]],
    shared_component: Dict[int, np.ndarray]
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Compute unique component for each MC direction by removing the shared component.

    unique = direction - (direction . shared) * shared
    """
    unique = {}

    for dataset, directions in mc_directions.items():
        unique[dataset] = {}
        for layer_idx, direction in directions.items():
            if layer_idx not in shared_component:
                continue

            shared = shared_component[layer_idx]
            projection = np.dot(direction, shared)
            residual = direction - projection * shared

            residual_norm = np.linalg.norm(residual)
            if residual_norm > 1e-8:
                unique[dataset][layer_idx] = residual / residual_norm
            else:
                unique[dataset][layer_idx] = None

    return unique


def train_1d_probe_along_direction(
    activations: np.ndarray,
    targets: np.ndarray,
    direction: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> Dict:
    """
    Train a 1D probe along a fixed direction.

    Instead of learning the direction, we project activations onto the given direction
    and fit a simple linear mapping: predicted_entropy = a * projection + b

    This tests whether the pre-specified direction contains entropy information.
    """
    # Project activations onto direction
    projections = activations @ direction  # (n_samples,)

    # Split
    proj_train = projections[train_idx].reshape(-1, 1)
    proj_test = projections[test_idx].reshape(-1, 1)
    y_train = targets[train_idx]
    y_test = targets[test_idx]

    # Standardize projections
    scaler = StandardScaler()
    proj_train_scaled = scaler.fit_transform(proj_train)
    proj_test_scaled = scaler.transform(proj_test)

    # Simple linear regression (Ridge with small alpha for numerical stability)
    probe = Ridge(alpha=1.0)
    probe.fit(proj_train_scaled, y_train)

    # Evaluate
    y_pred_train = probe.predict(proj_train_scaled)
    y_pred_test = probe.predict(proj_test_scaled)

    return {
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "scaler": scaler,
        "probe": probe,
    }


def apply_1d_probe_to_new_data(
    activations: np.ndarray,
    targets: np.ndarray,
    direction: np.ndarray,
    test_idx: np.ndarray,
    scaler: StandardScaler,
    probe: Ridge,
    use_separate_scaling: bool = True
) -> Dict:
    """
    Apply a trained 1D probe to new data (meta activations).

    If use_separate_scaling=True, standardize new data using its own statistics
    (fixes distribution shift between direct and meta activations).
    """
    # Project onto direction
    projections = activations @ direction
    proj_test = projections[test_idx].reshape(-1, 1)
    y_test = targets[test_idx]

    if use_separate_scaling:
        # Standardize using new data's own statistics
        new_scaler = StandardScaler()
        proj_test_scaled = new_scaler.fit_transform(proj_test)
    else:
        # Use original scaler (may cause distribution shift issues)
        proj_test_scaled = scaler.transform(proj_test)

    y_pred = probe.predict(proj_test_scaled)

    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "predictions": y_pred,
    }


def run_transfer_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    shared_component: Dict[int, np.ndarray],
    unique_component: Dict[int, np.ndarray],
    dataset_name: str
) -> Dict:
    """
    Run the transfer analysis for shared vs unique directions.

    For each layer:
    1. Train 1D probe along shared direction on direct activations → entropy
    2. Test transfer to meta activations
    3. Train 1D probe along unique direction (for this dataset only) on direct → entropy
    4. Test transfer to meta activations
    5. Compare: which transfers better?

    Only tests the unique direction for the specified dataset (not other datasets'
    unique directions, which would be meaningless on this dataset's activations).
    """
    # Same train/test split for all
    n_samples = len(direct_entropies)
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, train_size=TRAIN_SPLIT, random_state=SEED
    )

    results = {
        "layers": [],
        "shared": {
            "direct_r2": [],
            "meta_r2": [],
            "transfer_ratio": [],
        },
        "unique": {
            "direct_r2": [],
            "meta_r2": [],
            "transfer_ratio": [],
        },
        "dataset": dataset_name,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
    }

    layers = sorted(shared_component.keys())

    for layer_idx in tqdm(layers, desc="Running transfer analysis"):
        if layer_idx not in direct_activations or layer_idx not in meta_activations:
            continue

        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = direct_entropies

        results["layers"].append(layer_idx)

        # ===== SHARED DIRECTION =====
        shared_dir = shared_component[layer_idx]

        # Train on direct
        shared_direct_result = train_1d_probe_along_direction(
            X_direct, y, shared_dir, train_idx, test_idx
        )

        # Transfer to meta
        shared_meta_result = apply_1d_probe_to_new_data(
            X_meta, y, shared_dir, test_idx,
            shared_direct_result["scaler"],
            shared_direct_result["probe"],
            use_separate_scaling=True
        )

        direct_r2 = shared_direct_result["test_r2"]
        meta_r2 = shared_meta_result["r2"]
        transfer = meta_r2 / max(direct_r2, 0.001) if direct_r2 > 0 else 0.0

        results["shared"]["direct_r2"].append(direct_r2)
        results["shared"]["meta_r2"].append(meta_r2)
        results["shared"]["transfer_ratio"].append(transfer)

        # ===== UNIQUE DIRECTION (for this dataset only) =====
        if layer_idx not in unique_component or unique_component[layer_idx] is None:
            results["unique"]["direct_r2"].append(np.nan)
            results["unique"]["meta_r2"].append(np.nan)
            results["unique"]["transfer_ratio"].append(np.nan)
            continue

        unique_dir = unique_component[layer_idx]

        # Train on direct
        unique_direct_result = train_1d_probe_along_direction(
            X_direct, y, unique_dir, train_idx, test_idx
        )

        # Transfer to meta
        unique_meta_result = apply_1d_probe_to_new_data(
            X_meta, y, unique_dir, test_idx,
            unique_direct_result["scaler"],
            unique_direct_result["probe"],
            use_separate_scaling=True
        )

        direct_r2 = unique_direct_result["test_r2"]
        meta_r2 = unique_meta_result["r2"]
        transfer = meta_r2 / max(direct_r2, 0.001) if direct_r2 > 0 else 0.0

        results["unique"]["direct_r2"].append(direct_r2)
        results["unique"]["meta_r2"].append(meta_r2)
        results["unique"]["transfer_ratio"].append(transfer)

    return results


def compute_decomposition_stats(
    mc_directions: Dict[str, Dict[int, np.ndarray]],
    shared_component: Dict[int, np.ndarray],
    unique_components: Dict[str, Dict[int, np.ndarray]]
) -> Dict:
    """
    Compute statistics about the shared/unique decomposition.
    """
    results = {
        "layers": [],
        "shared_variance_explained": {},
        "original_similarities": {},
        "unique_similarities": {},
    }

    datasets = list(mc_directions.keys())
    layers = sorted(shared_component.keys())

    for dataset in datasets:
        results["shared_variance_explained"][dataset] = []

    for d1, d2 in [(datasets[i], datasets[j])
                   for i in range(len(datasets))
                   for j in range(i+1, len(datasets))]:
        results["original_similarities"][(d1, d2)] = []
        results["unique_similarities"][(d1, d2)] = []

    for layer_idx in layers:
        results["layers"].append(layer_idx)
        shared = shared_component[layer_idx]

        for dataset, directions in mc_directions.items():
            if layer_idx in directions:
                proj = np.dot(directions[layer_idx], shared)
                var_explained = proj ** 2
                results["shared_variance_explained"][dataset].append(float(var_explained))
            else:
                results["shared_variance_explained"][dataset].append(np.nan)

        for d1, d2 in results["original_similarities"].keys():
            if layer_idx in mc_directions.get(d1, {}) and layer_idx in mc_directions.get(d2, {}):
                orig_sim = np.dot(mc_directions[d1][layer_idx], mc_directions[d2][layer_idx])
                results["original_similarities"][(d1, d2)].append(float(orig_sim))
            else:
                results["original_similarities"][(d1, d2)].append(np.nan)

            u1 = unique_components.get(d1, {}).get(layer_idx)
            u2 = unique_components.get(d2, {}).get(layer_idx)
            if u1 is not None and u2 is not None:
                unique_sim = np.dot(u1, u2)
                results["unique_similarities"][(d1, d2)].append(float(unique_sim))
            else:
                results["unique_similarities"][(d1, d2)].append(np.nan)

    return results


def plot_transfer_analysis(
    transfer_results: Dict,
    decomp_stats: Dict,
    output_path: str
):
    """Plot transfer analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = transfer_results["layers"]
    dataset = transfer_results["dataset"]

    # 1. Direct R² (how well do directions predict entropy on direct data?)
    ax = axes[0, 0]
    ax.plot(layers, transfer_results["shared"]["direct_r2"],
            'o-', label="Shared", linewidth=2, markersize=4, color='blue')
    ax.plot(layers, transfer_results["unique"]["direct_r2"],
            's--', label=f"Unique ({dataset})", linewidth=2, markersize=4, color='orange')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (Direct → Entropy)")
    ax.set_title("Direct Prediction: How Well Does Each Direction Predict Entropy?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Meta R² (transfer test - can we predict entropy from meta activations?)
    ax = axes[0, 1]
    ax.plot(layers, transfer_results["shared"]["meta_r2"],
            'o-', label="Shared", linewidth=2, markersize=4, color='blue')
    ax.plot(layers, transfer_results["unique"]["meta_r2"],
            's--', label=f"Unique ({dataset})", linewidth=2, markersize=4, color='orange')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² (Meta → Entropy)")
    ax.set_title("Transfer Test: Does Direction Transfer to Meta-Judgment?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Transfer ratio (meta_r2 / direct_r2)
    ax = axes[1, 0]
    ax.plot(layers, transfer_results["shared"]["transfer_ratio"],
            'o-', label="Shared", linewidth=2, markersize=4, color='blue')
    ax.plot(layers, transfer_results["unique"]["transfer_ratio"],
            's--', label=f"Unique ({dataset})", linewidth=2, markersize=4, color='orange')
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect transfer')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Transfer Ratio (Meta R² / Direct R²)")
    ax.set_title("Transfer Ratio: How Well Does Each Direction Transfer?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 2.0)

    # 4. Variance explained by shared component
    ax = axes[1, 1]
    for ds, var_exp in decomp_stats["shared_variance_explained"].items():
        ax.plot(decomp_stats["layers"], var_exp, 'o-', label=ds, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Explained (shared)")
    ax.set_title("Shared Component Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved transfer analysis plot to {output_path}")
    plt.close()


def plot_decomposition_only(
    decomp_stats: Dict,
    output_path: str
):
    """Plot decomposition analysis only (no transfer data)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = decomp_stats["layers"]

    # 1. Variance explained by shared component
    ax = axes[0]
    for dataset, var_exp in decomp_stats["shared_variance_explained"].items():
        ax.plot(layers, var_exp, 'o-', label=dataset, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Variance Explained (shared)")
    ax.set_title("Shared Component Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 2. Original vs Unique pairwise similarities
    ax = axes[1]
    for (d1, d2), orig_sims in decomp_stats["original_similarities"].items():
        label = f"{d1}-{d2}"
        ax.plot(layers, orig_sims, 'o-', label=f"Orig: {label}", alpha=0.7)
    for (d1, d2), unique_sims in decomp_stats["unique_similarities"].items():
        label = f"{d1}-{d2}"
        ax.plot(layers, unique_sims, 's--', label=f"Unique: {label}", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Pairwise Similarities: Original vs Unique")
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved decomposition plot to {output_path}")
    plt.close()


def print_summary(
    mc_directions: Dict[str, Dict[int, np.ndarray]],
    decomp_stats: Dict,
    transfer_results: Optional[Dict]
):
    """Print summary of analysis."""
    print("\n" + "=" * 70)
    print("SHARED VS UNIQUE TRANSFER ANALYSIS")
    print("=" * 70)

    datasets = list(mc_directions.keys())
    print(f"\nMC datasets found: {datasets}")
    print(f"Number of layers: {len(decomp_stats['layers'])}")

    # Variance explained
    print("\n--- Variance Explained by Shared Component ---")
    for dataset, var_exp in decomp_stats["shared_variance_explained"].items():
        valid = [v for v in var_exp if not np.isnan(v)]
        if valid:
            print(f"  {dataset}: mean={np.mean(valid):.3f}, max={np.max(valid):.3f}")

    all_var = []
    for var_exp in decomp_stats["shared_variance_explained"].values():
        all_var.extend([v for v in var_exp if not np.isnan(v)])
    if all_var:
        mean_shared = np.mean(all_var)
        print(f"\n  Overall mean variance in shared: {mean_shared:.1%}")
        print(f"  Overall mean variance in unique: {1 - mean_shared:.1%}")

    if transfer_results:
        dataset = transfer_results["dataset"]
        print(f"\n--- Transfer Analysis Results (Dataset: {dataset}) ---")
        layers = transfer_results["layers"]

        # Best layers for shared
        shared_direct = transfer_results["shared"]["direct_r2"]
        shared_meta = transfer_results["shared"]["meta_r2"]
        shared_transfer = transfer_results["shared"]["transfer_ratio"]

        best_direct_idx = np.argmax(shared_direct)
        best_meta_idx = np.argmax(shared_meta)

        print(f"\n  Shared Component:")
        print(f"    Best direct R²: {shared_direct[best_direct_idx]:.4f} (layer {layers[best_direct_idx]})")
        print(f"    Best meta R²:   {shared_meta[best_meta_idx]:.4f} (layer {layers[best_meta_idx]})")
        print(f"    Transfer ratio at best direct: {shared_transfer[best_direct_idx]:.2%}")

        # Best for unique (only one dataset now)
        unique_direct = [v for v in transfer_results["unique"]["direct_r2"] if not np.isnan(v)]
        unique_meta = [v for v in transfer_results["unique"]["meta_r2"] if not np.isnan(v)]
        unique_transfer = transfer_results["unique"]["transfer_ratio"]

        if unique_direct and unique_meta:
            best_unique_direct_idx = np.nanargmax(transfer_results["unique"]["direct_r2"])
            best_unique_meta_idx = np.nanargmax(transfer_results["unique"]["meta_r2"])

            print(f"\n  Unique Component ({dataset}):")
            print(f"    Best direct R²: {max(unique_direct):.4f} (layer {layers[best_unique_direct_idx]})")
            print(f"    Best meta R²:   {max(unique_meta):.4f} (layer {layers[best_unique_meta_idx]})")
            print(f"    Transfer ratio at best direct: {unique_transfer[best_unique_direct_idx]:.2%}")

        # Key finding
        print("\n--- Key Finding ---")
        best_shared_meta = max(shared_meta)
        best_unique_meta = max(unique_meta) if unique_meta else 0.0

        if unique_meta:
            if best_shared_meta > best_unique_meta:
                print(f"  Shared component transfers BETTER to meta ({best_shared_meta:.4f} vs {best_unique_meta:.4f})")
                print("  -> Model uses GENERAL uncertainty signal for meta-judgment")
            else:
                print(f"  Unique component transfers BETTER to meta ({best_unique_meta:.4f} vs {best_shared_meta:.4f})")
                print("  -> Model uses DOMAIN-SPECIFIC signals for meta-judgment")
        else:
            print(f"  Shared component best meta R²: {best_shared_meta:.4f}")
            print("  (No valid unique component results to compare)")


def main():
    parser = argparse.ArgumentParser(description="Analyze shared vs unique MC direction transfer")
    parser.add_argument("--dataset", type=str, default="SimpleMC",
                        help="Dataset to use for transfer analysis (default: SimpleMC)")
    args = parser.parse_args()

    model_short = get_model_short_name(BASE_MODEL_NAME)
    output_prefix = get_output_prefix()

    print("=" * 70)
    print("SHARED VS UNIQUE MC DIRECTION TRANSFER ANALYSIS")
    print("=" * 70)
    print(f"Model: {BASE_MODEL_NAME}")
    print(f"Dataset: {args.dataset}")
    print(f"Output prefix: {output_prefix}")

    # Find MC direction files
    mc_files = find_mc_direction_files(OUTPUTS_DIR, model_short)

    if len(mc_files) < 2:
        print(f"\nError: Need at least 2 MC direction files, found {len(mc_files)}")
        if mc_files:
            print(f"  Found: {list(mc_files.keys())}")
        print("\nRun mc_entropy_probe.py on multiple datasets first:")
        print("  - SimpleMC")
        print("  - TriviaMC")
        print("  - GPQA")
        return

    # Check that the specified dataset has MC directions
    if args.dataset not in mc_files:
        print(f"\nError: No MC directions found for dataset '{args.dataset}'")
        print(f"  Available datasets: {list(mc_files.keys())}")
        return

    print(f"\nFound {len(mc_files)} MC direction files:")
    for dataset, path in mc_files.items():
        print(f"  {dataset}: {path}")

    # Load MC directions
    print("\nLoading MC directions...")
    mc_directions = {}
    for dataset, path in mc_files.items():
        mc_directions[dataset] = load_mc_directions(path)
        print(f"  {dataset}: {len(mc_directions[dataset])} layers")

    # Compute shared component
    print("\nComputing shared component...")
    shared_component = compute_shared_component(mc_directions)
    print(f"  Shared component computed for {len(shared_component)} layers")

    # Compute unique components
    print("\nComputing unique components...")
    unique_components = compute_unique_components(mc_directions, shared_component)
    for dataset in unique_components:
        valid = sum(1 for d in unique_components[dataset].values() if d is not None)
        print(f"  {dataset}: {valid} valid unique directions")

    # Compute decomposition statistics
    print("\nComputing decomposition statistics...")
    decomp_stats = compute_decomposition_stats(mc_directions, shared_component, unique_components)

    # Save decomposed directions
    directions_data = {}
    for layer_idx, direction in shared_component.items():
        directions_data[f"layer_{layer_idx}_shared"] = direction
    for dataset, unique_dirs in unique_components.items():
        for layer_idx, direction in unique_dirs.items():
            if direction is not None:
                directions_data[f"layer_{layer_idx}_unique_{dataset}"] = direction

    directions_path = f"{output_prefix}_shared_unique_directions.npz"
    np.savez_compressed(directions_path, **directions_data)
    print(f"Saved decomposed directions to {directions_path}")

    # ===== TRANSFER ANALYSIS =====
    # Find direct activations (from mc_entropy_probe.py)
    mc_activations_path = find_mc_activations_file(OUTPUTS_DIR, model_short, args.dataset)
    if mc_activations_path is None:
        print(f"\nError: No MC activations found for dataset '{args.dataset}'")
        print("  Run mc_entropy_probe.py first.")
        return

    # Find meta activations (from run_introspection_experiment.py)
    intro_files = find_introspection_files(OUTPUTS_DIR, model_short, args.dataset)
    if 'meta_activations' not in intro_files or 'paired_data' not in intro_files:
        print(f"\nError: No introspection data found for dataset '{args.dataset}'")
        print("  Run run_introspection_experiment.py first.")
        print("  Missing files:")
        if 'meta_activations' not in intro_files:
            print("    - meta_activations")
        if 'paired_data' not in intro_files:
            print("    - paired_data")
        return

    # Load direct activations from mc_entropy_probe.py
    print(f"\nLoading direct activations from mc_entropy_probe.py...")
    direct_acts, direct_entropies = load_mc_activations(mc_activations_path)
    print(f"  Direct activations: {len(direct_acts)} layers, {direct_acts[0].shape[0]} samples")
    print(f"  Direct entropies: range [{direct_entropies.min():.2f}, {direct_entropies.max():.2f}]")

    # Load meta activations from run_introspection_experiment.py
    print(f"\nLoading meta activations from run_introspection_experiment.py...")
    meta_acts, meta_entropies, stated_confs = load_meta_activations(
        intro_files['meta_activations'],
        intro_files['paired_data']
    )
    print(f"  Meta activations: {len(meta_acts)} layers, {meta_acts[list(meta_acts.keys())[0]].shape[0]} samples")

    # Note: direct_entropies from mc_entropy_probe has more samples than meta_activations
    # We'll use the entropies from paired_data which matches meta_activations length
    n_meta = meta_acts[list(meta_acts.keys())[0]].shape[0]
    n_direct = direct_acts[0].shape[0]

    if n_meta != n_direct:
        print(f"\n  Note: Sample count mismatch (direct={n_direct}, meta={n_meta})")
        print(f"  Using first {n_meta} samples from direct activations to match meta.")
        # Truncate direct activations to match meta
        direct_acts = {k: v[:n_meta] for k, v in direct_acts.items()}
        direct_entropies = meta_entropies  # Use entropies from paired data

    # Run transfer analysis
    print("\nRunning transfer analysis...")
    transfer_results = run_transfer_analysis(
        direct_acts, meta_acts, direct_entropies,
        shared_component, unique_components[args.dataset],
        args.dataset
    )

    # Save transfer results
    transfer_path = f"{output_prefix}_{args.dataset}_shared_unique_transfer.json"
    transfer_save = {
        "layers": transfer_results["layers"],
        "shared": transfer_results["shared"],
        "unique": transfer_results["unique"],
        "dataset": transfer_results["dataset"],
        "config": {
            "train_split": TRAIN_SPLIT,
            "seed": SEED,
        }
    }
    with open(transfer_path, 'w') as f:
        json.dump(transfer_save, f, indent=2)
    print(f"Saved transfer results to {transfer_path}")

    # Plot transfer analysis
    plot_transfer_analysis(
        transfer_results,
        decomp_stats,
        f"{output_prefix}_{args.dataset}_shared_unique_transfer.png"
    )

    # Save decomposition stats
    decomp_save = {
        "layers": decomp_stats["layers"],
        "shared_variance_explained": decomp_stats["shared_variance_explained"],
        "original_similarities": {
            f"{d1}_{d2}": v for (d1, d2), v in decomp_stats["original_similarities"].items()
        },
        "unique_similarities": {
            f"{d1}_{d2}": v for (d1, d2), v in decomp_stats["unique_similarities"].items()
        },
    }
    decomp_path = f"{output_prefix}_shared_unique_stats.json"
    with open(decomp_path, 'w') as f:
        json.dump(decomp_save, f, indent=2)
    print(f"Saved decomposition stats to {decomp_path}")

    # Print summary
    print_summary(mc_directions, decomp_stats, transfer_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
