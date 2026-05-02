"""
Direction finding methods for identifying internal correlates.

Two fundamentally different approaches:
1. probe: Linear regression to find direction that best predicts target metric
2. mean_diff: mean(high_metric_samples) - mean(low_metric_samples)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# Numerical stability helpers
# =============================================================================
# Many saved activation tensors are float16. StandardScaler.transform preserves dtype,
# so small feature stds can yield huge standardized values that overflow float16 and/or
# poison downstream PCA/Ridge. We force float32 scaling and floor tiny scales.
MIN_SCALE = 1e-6  # tune if needed

def _as_float32(X: np.ndarray) -> np.ndarray:
    """Return X as float32 (copy only if needed)."""
    return np.asarray(X, dtype=np.float32)

def _safe_scale(scale: np.ndarray, min_scale: float = MIN_SCALE) -> np.ndarray:
    """Floor per-feature scales and sanitize non-finite values."""
    scale = np.asarray(scale, dtype=np.float32)
    scale = np.where(np.isfinite(scale), scale, min_scale)
    return np.maximum(scale, min_scale)


def _sanitize_r2(r2: float) -> float:
    """Ensure R² is finite and not above 1 (tiny numerical overshoots are clamped)."""
    if not np.isfinite(r2):
        return float('nan')
    if r2 > 1.0:
        return 1.0
    return float(r2)



def probe_direction(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1000.0,
    use_pca: bool = True,
    pca_components: int = 100,
    bootstrap_splits: List[Tuple[np.ndarray, np.ndarray]] = None,
    return_scaler: bool = False,
    return_probe: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Find direction via Ridge regression probe.

    Args:
        X: (n_samples, hidden_dim) activation matrix
        y: (n_samples,) target metric values
        alpha: Ridge regularization strength
        use_pca: Whether to use PCA dimensionality reduction
        pca_components: Number of PCA components
        bootstrap_splits: Pre-generated list of (train_idx, test_idx) tuples for bootstrap.
                         If provided, computes confidence intervals. If None, fits on all data.
        return_scaler: If True, include scaler scale/mean in info dict for transfer tests
        return_probe: If True, include full probe pipeline (scaler, pca, ridge) for transfer tests

    Returns:
        direction: Normalized direction vector (hidden_dim,)
        info: Dict with r2, mae, correlation, and fit details (with std if bootstrap)
              If return_scaler=True, also includes 'scaler_scale' and 'scaler_mean'
              If return_probe=True, also includes 'scaler', 'pca', 'ridge' objects
    """
    from sklearn.metrics import mean_absolute_error

    def _fit_and_eval(X_train, y_train, X_test, y_test, return_scaler_info=False, return_probe_objects=False):
        # Force float32 to avoid float16 overflow during scaling
        X_train = _as_float32(X_train)
        X_test = _as_float32(X_test)
        # Fit probe and return metrics and direction.
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Prevent division by zero: clip scale values to minimum threshold
        # This handles features with near-zero variance
        # Use MIN_SCALE as minimum to avoid numerical issues with very small values
        min_scale = MIN_SCALE
        scaler.scale_ = _safe_scale(scaler.scale_, min_scale)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if use_pca:
            n_comp = min(pca_components, X_train.shape[0], X_train.shape[1])
            pca_model = PCA(n_components=n_comp)
            X_train_final = pca_model.fit_transform(X_train_scaled)
            X_test_final = pca_model.transform(X_test_scaled)
            variance_explained = float(pca_model.explained_variance_ratio_.sum())
        else:
            pca_model = None
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
            variance_explained = None

        probe = Ridge(alpha=alpha)
        probe.fit(X_train_final, y_train)

        y_pred = probe.predict(X_test_final)
        r2 = _sanitize_r2(r2_score(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)

        # Extract direction in original space
        if pca_model is not None:
            direction = pca_model.inverse_transform(probe.coef_.reshape(1, -1)).flatten()
            direction = direction / scaler.scale_
        else:
            direction = probe.coef_.copy()
            direction = direction / scaler.scale_

        direction = direction / np.linalg.norm(direction)

        if return_probe_objects:
            return r2, mae, corr, direction, variance_explained, scaler, pca_model, probe
        if return_scaler_info:
            return r2, mae, corr, direction, variance_explained, scaler.scale_, scaler.mean_
        return r2, mae, corr, direction, variance_explained

    if bootstrap_splits is not None and len(bootstrap_splits) > 0:
        # Use pre-generated bootstrap splits
        test_r2s, test_maes, test_corrs = [], [], []

        for train_idx, test_idx in bootstrap_splits:
            r2, mae, corr, _, _ = _fit_and_eval(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            test_r2s.append(r2)
            test_maes.append(mae)
            test_corrs.append(corr)

        # Final direction from first split's training data (canonical)
        train_idx = bootstrap_splits[0][0]
        if return_probe:
            _, _, _, direction, variance_explained, scaler_obj, pca_obj, ridge_obj = _fit_and_eval(
                X[train_idx], y[train_idx], X[train_idx], y[train_idx], return_probe_objects=True
            )
        elif return_scaler:
            _, _, _, direction, variance_explained, scaler_scale, scaler_mean = _fit_and_eval(
                X[train_idx], y[train_idx], X[train_idx], y[train_idx], return_scaler_info=True
            )
        else:
            _, _, _, direction, variance_explained = _fit_and_eval(
                X[train_idx], y[train_idx], X[train_idx], y[train_idx]
            )

        info = {
            "r2": float(np.mean(test_r2s)),
            "r2_std": float(np.std(test_r2s)),
            "mae": float(np.mean(test_maes)),
            "mae_std": float(np.std(test_maes)),
            "corr": float(np.mean(test_corrs)),
            "corr_std": float(np.std(test_corrs)),
            "pca_variance_explained": variance_explained,
            "alpha": alpha,
            "n_components": pca_components if use_pca else X.shape[1],
            "n_bootstrap": len(bootstrap_splits),
        }
        if return_probe:
            info["scaler"] = scaler_obj
            info["pca"] = pca_obj
            info["ridge"] = ridge_obj
            info["scaler_scale"] = scaler_obj.scale_
            info["scaler_mean"] = scaler_obj.mean_
        elif return_scaler:
            info["scaler_scale"] = scaler_scale
            info["scaler_mean"] = scaler_mean
    else:
        # No bootstrap - fit on all data
        scaler = StandardScaler()
        X = _as_float32(X)
        scaler.fit(X)

        # Prevent division by zero: clip scale values to minimum threshold
        min_scale = MIN_SCALE
        scaler.scale_ = _safe_scale(scaler.scale_, min_scale)

        X_scaled = scaler.transform(X)

        if use_pca:
            n_components = min(pca_components, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_components)
            X_final = pca.fit_transform(X_scaled)
            variance_explained = float(pca.explained_variance_ratio_.sum())
        else:
            pca = None
            X_final = X_scaled
            variance_explained = None

        probe = Ridge(alpha=alpha)
        probe.fit(X_final, y)

        y_pred = probe.predict(X_final)
        r2 = _sanitize_r2(r2_score(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        corr, _ = pearsonr(y, y_pred)

        if pca is not None:
            direction = pca.inverse_transform(probe.coef_.reshape(1, -1)).flatten()
            direction = direction / scaler.scale_
        else:
            direction = probe.coef_.copy()
            direction = direction / scaler.scale_

        direction = direction / np.linalg.norm(direction)

        info = {
            "r2": float(r2),
            "mae": float(mae),
            "corr": float(corr),
            "pca_variance_explained": variance_explained,
            "alpha": alpha,
            "n_components": pca_components if use_pca else X.shape[1],
        }
        if return_probe:
            info["scaler"] = scaler
            info["pca"] = pca
            info["ridge"] = probe
            info["scaler_scale"] = scaler.scale_
            info["scaler_mean"] = scaler.mean_
        elif return_scaler:
            info["scaler_scale"] = scaler.scale_
            info["scaler_mean"] = scaler.mean_

    return direction, info


def mean_diff_direction(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.25,
    bootstrap_splits: List[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Find direction via mean difference between high and low metric samples.

    Direction = mean(top_quantile) - mean(bottom_quantile)

    Args:
        X: (n_samples, hidden_dim) activation matrix
        y: (n_samples,) target metric values
        quantile: Fraction of samples to use for each group (top and bottom)
        bootstrap_splits: Pre-generated list of (train_idx, test_idx) tuples for bootstrap.
                         If provided, computes confidence intervals. If None, fits on all data.

    Returns:
        direction: Normalized direction vector (hidden_dim,)
        info: Dict with group statistics and fit metrics (with std if bootstrap)
    """
    def _fit_and_eval(X_data, y_data, X_test, y_test):
        """Compute mean_diff direction on data, evaluate on test."""
        n = len(y_data)
        n_group = max(1, int(n * quantile))

        sorted_idx = np.argsort(y_data)
        low_idx = sorted_idx[:n_group]
        high_idx = sorted_idx[-n_group:]

        mean_low = X_data[low_idx].mean(axis=0)
        mean_high = X_data[high_idx].mean(axis=0)

        direction = mean_high - mean_low
        magnitude = np.linalg.norm(direction)
        direction = direction / (magnitude + 1e-10)

        # Evaluate on test set
        projections = X_test @ direction
        corr, _ = pearsonr(y_test, projections)
        r2 = float(corr ** 2)

        # Compute MAE
        proj_mean, y_mean = projections.mean(), y_test.mean()
        proj_std, y_std = projections.std(), y_test.std()
        if proj_std > 0:
            y_pred = y_mean + (projections - proj_mean) * (y_std / proj_std) * np.sign(corr)
            mae = float(np.abs(y_test - y_pred).mean())
        else:
            mae = float(np.abs(y_test - y_mean).mean())

        return r2, mae, corr, direction, magnitude, n_group

    if bootstrap_splits is not None and len(bootstrap_splits) > 0:
        # Bootstrap for confidence intervals
        test_r2s, test_maes, test_corrs = [], [], []

        for train_idx, test_idx in bootstrap_splits:
            r2, mae, corr, _, _, _ = _fit_and_eval(
                X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            test_r2s.append(r2)
            test_maes.append(mae)
            test_corrs.append(corr)

        # Final direction from first split's training data (canonical)
        train_idx = bootstrap_splits[0][0]
        _, _, _, direction, magnitude, n_group = _fit_and_eval(
            X[train_idx], y[train_idx], X[train_idx], y[train_idx]
        )

        info = {
            "r2": float(np.mean(test_r2s)),
            "r2_std": float(np.std(test_r2s)),
            "mae": float(np.mean(test_maes)),
            "mae_std": float(np.std(test_maes)),
            "corr": float(np.mean(test_corrs)),
            "corr_std": float(np.std(test_corrs)),
            "quantile": quantile,
            "n_group": n_group,
            "direction_magnitude": float(magnitude),
            "n_bootstrap": len(bootstrap_splits),
        }
    else:
        # No bootstrap - fit on all data
        n = len(y)
        n_group = max(1, int(n * quantile))

        sorted_idx = np.argsort(y)
        low_idx = sorted_idx[:n_group]
        high_idx = sorted_idx[-n_group:]

        mean_low = X[low_idx].mean(axis=0)
        mean_high = X[high_idx].mean(axis=0)

        direction = mean_high - mean_low
        magnitude = np.linalg.norm(direction)
        direction = direction / (magnitude + 1e-10)

        projections = X @ direction
        corr, _ = pearsonr(y, projections)
        r2 = float(corr ** 2)

        proj_mean, y_mean = projections.mean(), y.mean()
        proj_std, y_std = projections.std(), y.std()
        if proj_std > 0:
            y_pred = y_mean + (projections - proj_mean) * (y_std / proj_std) * np.sign(corr)
            mae = float(np.abs(y - y_pred).mean())
        else:
            mae = float(np.abs(y - y_mean).mean())

        info = {
            "r2": r2,
            "mae": mae,
            "corr": float(corr),
            "quantile": quantile,
            "n_low": n_group,
            "n_high": n_group,
            "metric_mean_low": float(y[low_idx].mean()),
            "metric_mean_high": float(y[high_idx].mean()),
            "direction_magnitude": float(magnitude),
        }

    return direction, info


def _process_layer(
    layer: int,
    X: np.ndarray,
    y: np.ndarray,
    methods: List[str],
    probe_alpha: float,
    probe_pca_components: int,
    bootstrap_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    mean_diff_quantile: float,
    return_scaler: bool = False,
    return_probe: bool = False
) -> Tuple[int, Dict[str, np.ndarray], Dict[str, Dict]]:
    """Process a single layer - used for parallel execution."""
    layer_directions = {}
    layer_fits = {}

    for method in methods:
        if method == "probe":
            direction, info = probe_direction(
                X, y,
                alpha=probe_alpha,
                pca_components=probe_pca_components,
                bootstrap_splits=bootstrap_splits,
                return_scaler=return_scaler,
                return_probe=return_probe,
            )
        elif method == "mean_diff":
            direction, info = mean_diff_direction(
                X, y,
                quantile=mean_diff_quantile,
                bootstrap_splits=bootstrap_splits,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        layer_directions[method] = direction
        layer_fits[method] = info

    return layer, layer_directions, layer_fits


def find_directions(
    activations_by_layer: Dict[int, np.ndarray],
    target_values: np.ndarray,
    methods: List[str] = None,
    probe_alpha: float = 1000.0,
    probe_pca_components: int = 100,
    probe_n_bootstrap: int = 0,
    probe_train_split: float = 0.8,
    mean_diff_quantile: float = 0.25,
    seed: int = 42,
    n_jobs: int = -1,
    return_scaler: bool = False,
    return_probe: bool = False
) -> Dict:
    """
    Find directions using multiple methods across all layers.

    Args:
        activations_by_layer: {layer_idx: (n_samples, hidden_dim)}
        target_values: (n_samples,) metric values to predict
        methods: Which methods to use. Default: ["probe", "mean_diff"]
        probe_alpha: Ridge regularization for probe method
        probe_pca_components: PCA components for probe method
        probe_n_bootstrap: Bootstrap iterations for probe (0 = no bootstrap)
        probe_train_split: Train/test split ratio for bootstrap
        mean_diff_quantile: Quantile for mean_diff method
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        return_scaler: If True, include scaler info in fits dict (for transfer tests)
        return_probe: If True, include full probe objects (scaler, pca, ridge) for transfer tests.
                     NOTE: must use n_jobs=1 since sklearn objects can't be pickled across processes.

    Returns:
        {
            "directions": {method: {layer: direction_vector}},
            "fits": {method: {layer: {"r2": float, "corr": float, ...}}},
            "comparison": {layer: {"cosine_sim": float, ...}}
        }
    """
    from joblib import Parallel, delayed

    if methods is None:
        methods = ["probe", "mean_diff"]

    layers = sorted(activations_by_layer.keys())
    y = np.asarray(target_values)
    n_samples = len(y)

    # Pre-generate bootstrap splits ONCE (shared across all layers)
    bootstrap_splits = None
    if probe_n_bootstrap > 0 and "probe" in methods:
        rng = np.random.RandomState(seed)
        bootstrap_splits = []
        for _ in range(probe_n_bootstrap):
            indices = np.arange(n_samples)
            rng.shuffle(indices)
            split_idx = int(n_samples * probe_train_split)
            bootstrap_splits.append((indices[:split_idx].copy(), indices[split_idx:].copy()))

    # Run layers in parallel (or sequential with progress bar)
    # Note: return_probe requires n_jobs=1 since sklearn objects can't be pickled
    if return_probe and n_jobs != 1:
        print("Warning: return_probe=True requires n_jobs=1, forcing sequential processing")
        n_jobs = 1

    if n_jobs == 1:
        # Sequential with tqdm progress
        from tqdm import tqdm
        layer_results = []
        for layer in tqdm(layers, desc="Processing layers"):
            result = _process_layer(
                layer,
                activations_by_layer[layer],
                y,
                methods,
                probe_alpha,
                probe_pca_components,
                bootstrap_splits,
                mean_diff_quantile,
                return_scaler,
                return_probe
            )
            layer_results.append(result)
    else:
        # Parallel processing
        layer_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_layer)(
                layer,
                activations_by_layer[layer],
                y,
                methods,
                probe_alpha,
                probe_pca_components,
                bootstrap_splits,
                mean_diff_quantile,
                return_scaler,
                return_probe
            )
            for layer in layers
        )

    # Collect results
    results = {
        "directions": {m: {} for m in methods},
        "fits": {m: {} for m in methods},
        "comparison": {},
    }

    for layer, layer_directions, layer_fits in layer_results:
        for method in methods:
            results["directions"][method][layer] = layer_directions[method]
            results["fits"][method][layer] = layer_fits[method]

        # Compare methods at this layer
        if len(methods) == 2:
            d1, d2 = layer_directions[methods[0]], layer_directions[methods[1]]
            cosine_sim = float(np.dot(d1, d2))
            results["comparison"][layer] = {
                "cosine_sim": cosine_sim,
                "methods": methods,
            }

    return results


def apply_direction(
    activations: np.ndarray,
    direction: np.ndarray
) -> np.ndarray:
    """
    Project activations onto direction to get scalar predictions.

    Args:
        activations: (n_samples, hidden_dim) or (hidden_dim,)
        direction: (hidden_dim,) normalized direction vector

    Returns:
        Projections: (n_samples,) or scalar
    """
    return np.dot(activations, direction)


def apply_probe_shared(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: Optional[PCA],
    ridge: Ridge
) -> Dict:
    """
    Apply a pre-trained probe to new data using the original scaler.

    This is the "Shared Scaler" test - strictest transfer test.
    Uses the exact same scaling (mean AND variance) as training.
    Usually fails due to prompt offset shifting the mean.
    """
    from sklearn.metrics import r2_score, mean_absolute_error

    # Defensive clipping (scaler should already be clipped, but be safe)
    X = _as_float32(X)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    safe_scale = _safe_scale(scaler.scale_)
    X_scaled = (X - mean) / safe_scale

    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    y_pred = ridge.predict(X_final)

    r2 = _sanitize_r2(r2_score(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    pearson, _ = pearsonr(y, y_pred)

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson": float(pearson),
        "predictions": y_pred
    }


def apply_probe_centered(
    X_meta: np.ndarray,
    y: np.ndarray,
    direct_scaler: StandardScaler,
    pca: Optional[PCA],
    ridge: Ridge
) -> Dict:
    """
    Apply probe with Mean-Shift correction but Shared Variance.

    This is the "Centered Scaler" test - rigorous transfer test.
    Centers meta data using its OWN mean, but scales using DIRECT variance.
    Tests: "Is the geometry (direction) preserved despite offset shift?"
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    # Center meta using its own mean (but keep direct variance)
    X_meta = _as_float32(X_meta)
    meta_mean = np.mean(X_meta, axis=0, dtype=np.float32)
    X_centered = X_meta - meta_mean

    safe_scale = _safe_scale(direct_scaler.scale_)
    X_scaled = X_centered / safe_scale

    # Apply PCA and probe
    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    y_pred = ridge.predict(X_final)

    r2 = _sanitize_r2(r2_score(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    pearson, _ = pearsonr(y, y_pred)

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson": float(pearson),
        "predictions": y_pred
    }


def apply_probe_separate(
    X: np.ndarray,
    y: np.ndarray,
    pca: Optional[PCA],
    ridge: Ridge
) -> Dict:
    """
    Apply a pre-trained probe to new data with SEPARATE standardization.

    This is the "Separate Scaler" test - upper bound / domain adaptation.
    Standardizes meta activations using their own statistics (Mean=0, Var=1).
    """
    from sklearn.metrics import r2_score, mean_absolute_error

    # Standardize using meta's own statistics
    X = _as_float32(X)
    new_scaler = StandardScaler()
    new_scaler.fit(X)
    # Floor tiny stds and sanitize non-finite values
    new_scaler.scale_ = _safe_scale(new_scaler.scale_)
    mean = np.asarray(new_scaler.mean_, dtype=np.float32)
    scale = _safe_scale(new_scaler.scale_)
    X_scaled = (X - mean) / scale

    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    y_pred = ridge.predict(X_final)

    r2 = _sanitize_r2(r2_score(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    pearson, _ = pearsonr(y, y_pred)

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson": float(pearson),
        "predictions": y_pred
    }


def evaluate_transfer(
    activations: np.ndarray,
    direction: np.ndarray,
    target_values: np.ndarray,
    scaler_scale: np.ndarray = None,
    scaling: str = "none",
    n_bootstrap: int = 0,
    seed: int = 42
) -> Dict:
    """
    Evaluate how well a direction predicts targets on new data.

    NOTE: This uses direction vectors only. For rigorous transfer tests with
    full probe pipeline, use apply_probe_centered() or apply_probe_separate().

    Args:
        activations: (n_samples, hidden_dim)
        direction: (hidden_dim,) direction found on training data
        target_values: (n_samples,) ground truth values
        scaler_scale: (hidden_dim,) std from training StandardScaler (for centered scaling)
        scaling: "none", "separate", or "centered"
            - "none": Raw projection (current behavior)
            - "separate": Center and scale using meta data's own mean/std
            - "centered": Center using meta mean, scale using direct's std (rigorous transfer)
        n_bootstrap: If > 0, compute bootstrap confidence intervals
        seed: Random seed for bootstrap

    Returns:
        Dict with r2, correlation, and predictions (with std if bootstrap)
    """
    # Apply scaling
    if scaling == "centered":
        if scaler_scale is None:
            raise ValueError("scaler_scale required for centered scaling")
        meta_mean = activations.mean(axis=0)
        X_scaled = (activations - meta_mean) / scaler_scale
        projections = X_scaled @ direction
    elif scaling == "separate":
        meta_mean = activations.mean(axis=0)
        meta_std = activations.std(axis=0)
        meta_std = np.where(meta_std > 0, meta_std, 1.0)  # Avoid div by zero
        X_scaled = (activations - meta_mean) / meta_std
        projections = X_scaled @ direction
    else:  # "none"
        projections = apply_direction(activations, direction)

    corr, p_value = pearsonr(target_values, projections)
    r2 = float(corr ** 2)

    result = {
        "r2": r2,
        "corr": float(corr),
        "corr_pvalue": float(p_value),
        "projections": projections,
    }

    # Bootstrap confidence intervals
    if n_bootstrap > 0:
        rng = np.random.RandomState(seed)
        n_samples = len(target_values)
        bootstrap_r2s = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, n_samples, replace=True)
            boot_corr, _ = pearsonr(target_values[idx], projections[idx])
            bootstrap_r2s.append(float(boot_corr ** 2))

        result["r2_std"] = float(np.std(bootstrap_r2s))
        result["r2_ci_low"] = float(np.percentile(bootstrap_r2s, 2.5))
        result["r2_ci_high"] = float(np.percentile(bootstrap_r2s, 97.5))

    return result

def apply_mean_diff_transfer(
    X: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Apply a mean-difference direction to activations.

    This is intentionally simple: it returns the raw 1D projection scores.
    Any calibration (e.g., mapping scores to a metric scale) should be done
    by the caller, since mean-diff directions do not come with a trained
    regression head.

    Args:
        X: (n_samples, hidden_dim) activations
        direction: (hidden_dim,) direction vector

    Returns:
        scores: (n_samples,) raw projection scores = X @ direction
    """
    X = _as_float32(X)
    d = np.asarray(direction, dtype=np.float32)
    return X @ d
