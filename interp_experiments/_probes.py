"""
Probe training + direction extraction for the introspection pipeline.

Three families of directions are produced from the same activations:
  1. *Probe* direction — Ridge regression coefficient that predicts the
     selected uncertainty METRIC from the residual stream, mapped back to
     the original activation space (`extract_direction`).
  2. *MC-answer* direction — first principal component of a 4-class
     LogisticRegression (`extract_mc_answer_direction`).
  3. *Contrast* direction — mean(activations[high-tail]) − mean(activations[low-tail])
     of a scalar signal, percentile-binned (`compute_contrast_directions`).

Static knobs (`TRAIN_SPLIT`, `PROBE_ALPHA`, `USE_PCA`, `PCA_COMPONENTS`,
`SEED`) come from `IntrospectionExperimentConfig`. Pure numpy/sklearn —
no GPU, no model state.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiment_config import IntrospectionExperimentConfig as _C


# =============================================================================
# Direction extraction (single-vector summaries of trained probes)
# =============================================================================

def extract_direction(
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge,
) -> np.ndarray:
    """Map Ridge coefficients back to the original activation space, unit-normed."""
    coef = probe.coef_
    if pca is not None:
        direction_scaled = pca.components_.T @ coef
    else:
        direction_scaled = coef
    direction_original = direction_scaled / scaler.scale_
    return direction_original / np.linalg.norm(direction_original)


def extract_mc_answer_direction(
    scaler: StandardScaler,
    pca: PCA,
    clf: LogisticRegression,
) -> np.ndarray:
    """First PC of the 4-class LogisticRegression coefficients, mapped back to activation space.

    `clf.coef_` has shape (n_classes, n_pca_components) — taking PC1 of those
    rows captures the dominant axis of variation distinguishing the MC
    classes, then we project back to the original space.
    """
    coef_pca = PCA(n_components=1)
    coef_pca.fit(clf.coef_)
    pc1 = coef_pca.components_[0]
    direction_scaled = pca.components_.T @ pc1
    direction_original = direction_scaled / scaler.scale_
    return direction_original / np.linalg.norm(direction_original)


def compute_contrast_directions(
    activations: Dict[int, np.ndarray],
    signal: np.ndarray,
    percent: float = 25.0,
    samples_per_bin: Optional[int] = None,
    seed: int = 0,
) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray]]:
    """Per-layer mean-difference direction between top and bottom groups of `signal`.

    Splits questions by percentile (default: top 25% vs bottom 25%, middle
    50% dropped) and at each layer takes
        mean(activations[high]) - mean(activations[low]),
    normalised to unit length.

    Args:
        activations: {layer_idx: (n_questions, hidden_dim)} float per layer.
        signal: (n_questions,) scalar per question (entropy, soft stated
            confidence, etc). NaNs must be filtered by the caller.
        percent: tail width in percent, in (0, 50). 25 → bottom 25% vs top 25%.
        samples_per_bin: if set, randomly subsample to this many per tail
            before averaging (deterministic via `seed`).
        seed: RNG seed for the optional subsample step.

    Returns:
        directions: {layer_idx: (hidden_dim,) unit-norm contrast vector}
        meta: dict of scalar np.arrays with keys "percent", "low_threshold",
            "high_threshold", "n_low", "n_high", "n_low_used",
            "n_high_used", "samples_per_bin", "seed". Suitable for direct
            splatting into `np.savez_compressed(..., **meta)`.
    """
    if not (0 < percent < 50):
        raise ValueError(f"percent must be in (0, 50), got {percent}")

    signal = np.asarray(signal, dtype=np.float64)
    quantile = percent / 100.0
    lo_thresh = float(np.quantile(signal, quantile))
    hi_thresh = float(np.quantile(signal, 1.0 - quantile))
    low_idx = np.where(signal <= lo_thresh)[0]
    high_idx = np.where(signal >= hi_thresh)[0]
    n_low = int(low_idx.size)
    n_high = int(high_idx.size)
    if n_low == 0 or n_high == 0:
        raise ValueError(
            f"Empty contrast group(s): n_low={n_low}, n_high={n_high} "
            f"(percent={percent}, signal range "
            f"[{float(signal.min()):.3f}, {float(signal.max()):.3f}])"
        )

    if samples_per_bin is not None and samples_per_bin > 0:
        rng = np.random.default_rng(seed)
        if low_idx.size > samples_per_bin:
            low_idx = rng.choice(low_idx, size=samples_per_bin, replace=False)
        if high_idx.size > samples_per_bin:
            high_idx = rng.choice(high_idx, size=samples_per_bin, replace=False)
    n_low_used = int(low_idx.size)
    n_high_used = int(high_idx.size)

    directions: Dict[int, np.ndarray] = {}
    for layer_idx in sorted(activations.keys()):
        acts = activations[layer_idx]
        diff = acts[high_idx].mean(axis=0) - acts[low_idx].mean(axis=0)
        norm = float(np.linalg.norm(diff))
        directions[layer_idx] = diff if norm < 1e-10 else (diff / norm)

    meta = {
        "percent": np.array(percent, dtype=np.float32),
        "low_threshold": np.array(lo_thresh, dtype=np.float32),
        "high_threshold": np.array(hi_thresh, dtype=np.float32),
        "n_low": np.array(n_low, dtype=np.int32),
        "n_high": np.array(n_high, dtype=np.int32),
        "n_low_used": np.array(n_low_used, dtype=np.int32),
        "n_high_used": np.array(n_high_used, dtype=np.int32),
        "samples_per_bin": np.array(
            -1 if samples_per_bin is None else int(samples_per_bin), dtype=np.int32,
        ),
        "seed": np.array(int(seed), dtype=np.int32),
    }
    return directions, meta


# =============================================================================
# Ridge regression probe (entropy / generic uncertainty metric)
# =============================================================================

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    return_components: bool = False,
) -> Dict:
    """Train a Ridge linear probe against `y` from activations.

    Reads `_C.PROBE_ALPHA`, `_C.USE_PCA`, `_C.PCA_COMPONENTS` for the
    standardize → (PCA →) Ridge pipeline.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = None
    if _C.USE_PCA:
        n_components = min(_C.PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    probe = Ridge(alpha=_C.PROBE_ALPHA)
    probe.fit(X_train_final, y_train)

    y_pred_train = probe.predict(X_train_final)
    y_pred_test = probe.predict(X_test_final)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_pearson, _ = pearsonr(y_test, y_pred_test)
    test_spearman, _ = spearmanr(y_test, y_pred_test)

    result = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "test_pearson": test_pearson,
        "test_spearman": test_spearman,
        "predictions": y_pred_test,
        "pca_variance_explained": pca.explained_variance_ratio_.sum() if _C.USE_PCA else None,
    }

    if return_components:
        result["scaler"] = scaler
        result["pca"] = pca
        result["probe"] = probe

    return result


def apply_trained_probe(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge,
) -> Dict:
    """Apply a trained probe to new activations using the original scaler (shared scaling)."""
    X_scaled = scaler.transform(X)
    X_final = pca.transform(X_scaled) if pca is not None else X_scaled
    y_pred = probe.predict(X_final)
    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "pearson": pearsonr(y, y_pred)[0],
        "spearman": spearmanr(y, y_pred)[0],
        "predictions": y_pred,
    }


def apply_probe_centering_only(
    X_meta: np.ndarray,
    y_test: np.ndarray,
    direct_scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge,
) -> Dict:
    """Apply probe with mean-shift correction but the direct scaler's variance.

    Fixes the per-prompt offset (intercept) without distorting the geometry
    (angles) of the activation space — useful for testing whether the
    uncertainty direction is structurally identical across direct/meta
    prompts but lives at a shifted absolute position.
    """
    meta_mean = np.mean(X_meta, axis=0)
    X_centered = X_meta - meta_mean
    X_scaled = X_centered / direct_scaler.scale_
    X_final = pca.transform(X_scaled) if pca is not None else X_scaled
    y_pred = probe.predict(X_final)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "pearson": pearsonr(y_test, y_pred)[0],
        "spearman": spearmanr(y_test, y_pred)[0],
        "predictions": y_pred,
    }


def apply_probe_with_separate_scaling(
    X: np.ndarray,
    y: np.ndarray,
    pca: Optional[PCA],
    probe: Ridge,
) -> Dict:
    """Apply probe with a separately-fit scaler — domain-adaptation upper bound."""
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X)
    X_final = pca.transform(X_scaled) if pca is not None else X_scaled
    y_pred = probe.predict(X_final)
    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "pearson": pearsonr(y, y_pred)[0],
        "spearman": spearmanr(y, y_pred)[0],
        "predictions": y_pred,
    }


# =============================================================================
# Layer sweeps
# =============================================================================

def run_introspection_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    pretrained_probe_path: Optional[str] = None,
    extract_directions: bool = True,
):
    """Per-layer Ridge probe sweep: direct→direct, direct→meta (3 ways), shuffled, meta→meta.

    Returns (results, test_idx, directions, probe_components).
    """
    print(f"\nRunning introspection analysis across {len(direct_activations)} layers...")

    n_questions = len(direct_entropies)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(
        indices, train_size=_C.TRAIN_SPLIT, random_state=_C.SEED,
    )

    print(f"Train set: {len(train_idx)} questions, Test set: {len(test_idx)} questions")

    results: Dict[int, Dict] = {}
    directions = {} if extract_directions else None
    probe_components: Dict[int, Dict] = {}

    for layer_idx in tqdm(sorted(direct_activations.keys()), desc="Training probes"):
        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = direct_entropies

        X_direct_train = X_direct[train_idx]
        X_direct_test = X_direct[test_idx]
        X_meta_test = X_meta[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        direct_results = train_probe(
            X_direct_train, y_train, X_direct_test, y_test, return_components=True,
        )
        meta_results_shared = apply_trained_probe(
            X_meta_test, y_test,
            direct_results["scaler"], direct_results["pca"], direct_results["probe"],
        )
        meta_results_centered = apply_probe_centering_only(
            X_meta_test, y_test,
            direct_results["scaler"], direct_results["pca"], direct_results["probe"],
        )
        meta_results_separate = apply_probe_with_separate_scaling(
            X_meta_test, y_test,
            direct_results["pca"], direct_results["probe"],
        )

        shuffled_y_train = y_train.copy()
        np.random.shuffle(shuffled_y_train)
        shuffled_results = train_probe(
            X_direct_train, shuffled_y_train, X_direct_test, y_test, return_components=False,
        )

        meta_to_meta_results = train_probe(
            X_meta[train_idx], y_train, X_meta_test, y_test, return_components=False,
        )

        if extract_directions:
            directions[layer_idx] = extract_direction(
                direct_results["scaler"], direct_results["pca"], direct_results["probe"],
            )

        probe_components[layer_idx] = {
            "scaler": direct_results["scaler"],
            "pca": direct_results["pca"],
            "probe": direct_results["probe"],
        }

        results[layer_idx] = {
            "direct_to_direct": {
                "train_r2": direct_results["train_r2"],
                "test_r2": direct_results["test_r2"],
                "test_pearson": direct_results["test_pearson"],
                "test_mae": direct_results["test_mae"],
                "predictions": direct_results["predictions"].tolist(),
            },
            "direct_to_meta": {
                "r2": meta_results_shared["r2"],
                "pearson": meta_results_shared["pearson"],
                "mae": meta_results_shared["mae"],
                "predictions": meta_results_shared["predictions"].tolist(),
            },
            "direct_to_meta_centered": {
                "r2": meta_results_centered["r2"],
                "pearson": meta_results_centered["pearson"],
                "mae": meta_results_centered["mae"],
                "predictions": meta_results_centered["predictions"].tolist(),
            },
            "direct_to_meta_fixed": {
                "r2": meta_results_separate["r2"],
                "pearson": meta_results_separate["pearson"],
                "mae": meta_results_separate["mae"],
                "predictions": meta_results_separate["predictions"].tolist(),
            },
            "shuffled_baseline": {
                "r2": shuffled_results["test_r2"],
                "mae": shuffled_results["test_mae"],
            },
            "meta_to_meta": {
                "train_r2": meta_to_meta_results["train_r2"],
                "test_r2": meta_to_meta_results["test_r2"],
            },
            "pca_variance_explained": direct_results["pca_variance_explained"],
        }

    return results, test_idx, directions, probe_components


def run_mc_answer_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    model_predicted_answer: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict, Dict, Dict[int, np.ndarray]]:
    """Per-layer multiclass LogisticRegression on direct→{predicted MC class}.

    Returns:
        results: {layer_idx: {d2d_accuracy, d2m_*_accuracy, shuffled_accuracy, ...}}
        mc_probe_components: {layer_idx: {"scaler", "pca", "clf"}}
        mc_directions: {layer_idx: (hidden_dim,) unit-norm direction}
    """
    print(f"\nRunning MC answer probe analysis across {len(direct_activations)} layers...")

    results: Dict[int, Dict] = {}
    mc_probe_components: Dict[int, Dict] = {}
    mc_directions: Dict[int, np.ndarray] = {}

    for layer_idx in tqdm(sorted(direct_activations.keys()), desc="Training MC answer probes"):
        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = model_predicted_answer

        X_direct_train = X_direct[train_idx]
        X_direct_test = X_direct[test_idx]
        X_meta_test = X_meta[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_direct_train_scaled = scaler.fit_transform(X_direct_train)
        X_direct_test_scaled = scaler.transform(X_direct_test)

        pca = PCA(n_components=min(256, X_direct_train_scaled.shape[1], X_direct_train_scaled.shape[0]))
        X_direct_train_pca = pca.fit_transform(X_direct_train_scaled)
        X_direct_test_pca = pca.transform(X_direct_test_scaled)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_direct_train_pca, y_train)

        d2d_accuracy = clf.score(X_direct_test_pca, y_test)
        d2d_predictions = clf.predict(X_direct_test_pca)

        # Transfer 1: separate scaling (upper bound)
        meta_scaler_sep = StandardScaler()
        X_meta_test_sep = meta_scaler_sep.fit_transform(X_meta_test)
        X_meta_test_sep_pca = pca.transform(X_meta_test_sep)
        d2m_separate_accuracy = clf.score(X_meta_test_sep_pca, y_test)

        # Transfer 2: centered scaling (rigorous)
        meta_mean = np.mean(X_meta_test, axis=0)
        X_meta_centered = X_meta_test - meta_mean
        X_meta_test_cen = X_meta_centered / scaler.scale_
        X_meta_test_cen_pca = pca.transform(X_meta_test_cen)
        d2m_centered_accuracy = clf.score(X_meta_test_cen_pca, y_test)

        shuffled_y_train = y_train.copy()
        np.random.shuffle(shuffled_y_train)
        clf_shuffled = LogisticRegression(max_iter=1000, random_state=42)
        clf_shuffled.fit(X_direct_train_pca, shuffled_y_train)
        shuffled_accuracy = clf_shuffled.score(X_direct_test_pca, y_test)

        results[layer_idx] = {
            "d2d_accuracy": d2d_accuracy,
            "d2m_accuracy": d2m_separate_accuracy,
            "d2m_separate_accuracy": d2m_separate_accuracy,
            "d2m_centered_accuracy": d2m_centered_accuracy,
            "shuffled_accuracy": shuffled_accuracy,
            "d2d_predictions": d2d_predictions.tolist(),
        }

        mc_probe_components[layer_idx] = {"scaler": scaler, "pca": pca, "clf": clf}
        mc_directions[layer_idx] = extract_mc_answer_direction(scaler, pca, clf)

    return results, mc_probe_components, mc_directions


def apply_probes_to_other(
    other_activations: Dict[int, np.ndarray],
    entropy_probe_components: Dict[int, Dict],
    mc_probe_components: Dict[int, Dict],
    direct_entropy: np.ndarray,
    model_predicted_answer: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """Apply trained direct→entropy and direct→MC probes to other-confidence activations.

    Reuses probe components from `run_introspection_analysis` /
    `run_mc_answer_analysis` so this is a transfer-only test (no fresh
    training). Useful as a control: if direct-trained probes transfer
    equally well to other-confidence activations, the model encodes
    uncertainty similarly regardless of the meta task framing.
    """
    print("\nApplying trained probes to other-confidence activations...")

    y_entropy = direct_entropy[test_idx]
    y_mc = model_predicted_answer[test_idx]

    results: Dict[int, Dict] = {}
    for layer_idx in tqdm(sorted(other_activations.keys()), desc="Testing D→M(Other) transfer"):
        X_other = other_activations[layer_idx][test_idx]

        ent_comps = entropy_probe_components[layer_idx]
        other_mean = np.mean(X_other, axis=0)
        X_other_scaled = (X_other - other_mean) / ent_comps["scaler"].scale_
        X_other_pca = ent_comps["pca"].transform(X_other_scaled)
        entropy_preds = ent_comps["probe"].predict(X_other_pca)
        entropy_r2 = r2_score(y_entropy, entropy_preds)

        mc_comps = mc_probe_components[layer_idx]
        X_other_mc_scaled = (X_other - np.mean(X_other, axis=0)) / mc_comps["scaler"].scale_
        X_other_mc_pca = mc_comps["pca"].transform(X_other_mc_scaled)
        mc_acc = mc_comps["clf"].score(X_other_mc_pca, y_mc)

        results[layer_idx] = {
            "d2m_other_entropy_r2": entropy_r2,
            "d2m_other_mc_accuracy": mc_acc,
        }

    return results
