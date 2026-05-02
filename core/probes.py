"""
Linear probe training and evaluation utilities.

Provides:
- Ridge regression probe training with optional PCA
- Transfer testing (train on X, test on Y)
- Permutation testing for significance
"""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Default probe configuration
DEFAULT_PROBE_CONFIG = {
    "alpha": 1000.0,       # Ridge regularization
    "use_pca": True,       # Whether to apply PCA
    "pca_components": 100, # Number of PCA components
    "train_split": 0.8,    # Train/test split ratio
}


class LinearProbe:
    """
    Linear probe for predicting a target from activations.

    Supports PCA dimensionality reduction and Ridge regularization.
    """

    def __init__(
        self,
        alpha: float = 1000.0,
        use_pca: bool = True,
        pca_components: int = 100
    ):
        self.alpha = alpha
        self.use_pca = use_pca
        self.pca_components = pca_components

        self.scaler = StandardScaler()
        self.pca = None
        self.probe = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """
        Fit the probe on training data.

        Args:
            X: (n_samples, n_features) activation matrix
            y: (n_samples,) target values

        Returns:
            self
        """
        X_scaled = self.scaler.fit_transform(X)

        if self.use_pca:
            n_components = min(self.pca_components, X.shape[0], X.shape[1])
            self.pca = PCA(n_components=n_components)
            X_final = self.pca.fit_transform(X_scaled)
        else:
            X_final = X_scaled

        self.probe.fit(X_final, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values from activations."""
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if self.pca is not None:
            X_final = self.pca.transform(X_scaled)
        else:
            X_final = X_scaled

        return self.probe.predict(X_final)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate probe on test data.

        Returns dict with r2, mae, and predictions.
        """
        y_pred = self.predict(X)
        return {
            "r2": r2_score(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "predictions": y_pred
        }

    @property
    def pca_variance_explained(self) -> Optional[float]:
        """Return total variance explained by PCA, if used."""
        if self.pca is not None:
            return float(self.pca.explained_variance_ratio_.sum())
        return None

    def get_direction(self, in_pca_space: bool = False) -> np.ndarray:
        """
        Extract the direction vector from a fitted probe.

        The direction indicates what the probe is "looking for" in activation space.
        For entropy probes, moving along this direction should correspond to
        changing predicted entropy.

        Args:
            in_pca_space: If True, return direction in PCA space (if PCA was used).
                         Otherwise, return in original activation space.

        Returns:
            Normalized direction vector
        """
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")

        if in_pca_space or self.pca is None:
            direction = self.probe.coef_.copy()
        else:
            # Project back to original space through PCA
            pca_direction = self.probe.coef_
            direction = self.pca.inverse_transform(pca_direction.reshape(1, -1)).flatten()
            # Undo standardization scaling
            direction = direction / self.scaler.scale_

        return direction / np.linalg.norm(direction)


def train_and_evaluate_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1000.0,
    use_pca: bool = True,
    pca_components: int = 100
) -> Tuple[LinearProbe, Dict]:
    """
    Train a probe and evaluate on train and test sets.

    Returns:
        Tuple of (fitted probe, results dict)
    """
    probe = LinearProbe(alpha=alpha, use_pca=use_pca, pca_components=pca_components)
    probe.fit(X_train, y_train)

    train_results = probe.evaluate(X_train, y_train)
    test_results = probe.evaluate(X_test, y_test)

    return probe, {
        "train_r2": train_results["r2"],
        "train_mae": train_results["mae"],
        "test_r2": test_results["r2"],
        "test_mae": test_results["mae"],
        "predictions": test_results["predictions"],
        "pca_variance_explained": probe.pca_variance_explained
    }


def test_transfer(
    probe: LinearProbe,
    X_transfer: np.ndarray,
    y_transfer: np.ndarray
) -> Dict:
    """
    Test a trained probe on transfer data (e.g., meta activations).

    Args:
        probe: Fitted LinearProbe
        X_transfer: Transfer activations
        y_transfer: Ground truth targets

    Returns:
        Dict with r2, mae, predictions
    """
    return probe.evaluate(X_transfer, y_transfer)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 1000.0,
    use_pca: bool = True,
    pca_components: int = 100,
    train_split: float = 0.8,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run permutation test to assess significance of probe performance.

    Trains probe on real data, then on n_permutations shuffled versions
    to compute empirical p-value.

    Args:
        X: Activation matrix
        y: Target values
        n_permutations: Number of permutation trials
        alpha, use_pca, pca_components: Probe config
        train_split: Fraction of data for training
        seed: Random seed
        verbose: Show progress bar

    Returns:
        Dict with:
            - true_r2: R² on real data
            - null_r2s: Array of R² values from permutations
            - p_value: Fraction of null R²s >= true R²
            - mean_null: Mean of null distribution
            - std_null: Std of null distribution
    """
    rng = np.random.RandomState(seed)

    # Split data
    n = len(y)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, train_size=train_split, random_state=seed)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train on real data
    probe, results = train_and_evaluate_probe(
        X_train, y_train, X_test, y_test,
        alpha=alpha, use_pca=use_pca, pca_components=pca_components
    )
    true_r2 = results["test_r2"]

    # Permutation tests
    null_r2s = []
    iterator = range(n_permutations)
    if verbose:
        iterator = tqdm(iterator, desc="Permutation test")

    for _ in iterator:
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)

        y_train_shuf = y_shuffled[train_idx]
        y_test_shuf = y_shuffled[test_idx]

        _, perm_results = train_and_evaluate_probe(
            X_train, y_train_shuf, X_test, y_test_shuf,
            alpha=alpha, use_pca=use_pca, pca_components=pca_components
        )
        null_r2s.append(perm_results["test_r2"])

    null_r2s = np.array(null_r2s)
    p_value = (null_r2s >= true_r2).mean()

    return {
        "true_r2": true_r2,
        "null_r2s": null_r2s,
        "p_value": p_value,
        "mean_null": null_r2s.mean(),
        "std_null": null_r2s.std()
    }


def run_layer_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    train_split: float = 0.8,
    seed: int = 42,
    alpha: float = 1000.0,
    use_pca: bool = True,
    pca_components: int = 100
) -> Tuple[Dict, np.ndarray]:
    """
    Run introspection analysis across all layers.

    For each layer:
    1. Train probe on direct activations -> direct entropy
    2. Test on held-out direct data (sanity check)
    3. Test on meta activations -> direct entropy (introspection test)
    4. Train on meta -> meta (signal existence check)

    Args:
        direct_activations: {layer_idx: (n_questions, hidden_dim)}
        meta_activations: {layer_idx: (n_questions, hidden_dim)}
        direct_entropies: (n_questions,) array
        train_split, seed: Data split config
        alpha, use_pca, pca_components: Probe config

    Returns:
        Tuple of (results dict by layer, test indices)
    """
    n_questions = len(direct_entropies)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(indices, train_size=train_split, random_state=seed)

    results = {}

    for layer_idx in tqdm(sorted(direct_activations.keys()), desc="Analyzing layers"):
        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = direct_entropies

        # Split
        X_direct_train = X_direct[train_idx]
        X_direct_test = X_direct[test_idx]
        X_meta_test = X_meta[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # 1. Train on direct, test on direct (sanity check)
        probe, direct_results = train_and_evaluate_probe(
            X_direct_train, y_train, X_direct_test, y_test,
            alpha=alpha, use_pca=use_pca, pca_components=pca_components
        )

        # 2. Apply direct-trained probe to meta (introspection test)
        meta_results = test_transfer(probe, X_meta_test, y_test)

        # 3. Shuffled baseline
        y_shuffled = y_test.copy()
        np.random.shuffle(y_shuffled)
        shuffled_results = test_transfer(probe, X_meta_test, y_shuffled)

        # 4. Train on meta, test on meta (signal existence)
        _, meta_to_meta = train_and_evaluate_probe(
            X_meta[train_idx], y_train, X_meta_test, y_test,
            alpha=alpha, use_pca=use_pca, pca_components=pca_components
        )

        results[layer_idx] = {
            "direct_to_direct": {
                "train_r2": direct_results["train_r2"],
                "test_r2": direct_results["test_r2"],
                "train_mae": direct_results["train_mae"],
                "test_mae": direct_results["test_mae"],
            },
            "direct_to_meta": {
                "r2": meta_results["r2"],
                "mae": meta_results["mae"],
                "predictions": meta_results["predictions"].tolist(),
            },
            "shuffled_baseline": {
                "r2": shuffled_results["r2"],
                "mae": shuffled_results["mae"],
            },
            "meta_to_meta": {
                "train_r2": meta_to_meta["train_r2"],
                "test_r2": meta_to_meta["test_r2"],
            },
            "pca_variance_explained": direct_results["pca_variance_explained"]
        }

    return results, test_idx


# ============================================================================
# INTROSPECTION MAPPING PROBE
# ============================================================================

def compute_introspection_scores(
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray
) -> np.ndarray:
    """
    Compute introspection scores: alignment between entropy and stated confidence.

    Score = -entropy_z * confidence_z
    Positive when: low entropy + high confidence, OR high entropy + low confidence
    (i.e., when the model is well-calibrated)

    Args:
        direct_entropies: (n,) array of entropy values
        stated_confidences: (n,) array of confidence values (0-1 scale)

    Returns:
        (n,) array of introspection scores
    """
    entropy_z = (direct_entropies - direct_entropies.mean()) / direct_entropies.std()
    confidence_z = (stated_confidences - stated_confidences.mean()) / stated_confidences.std()
    return -entropy_z * confidence_z


def train_introspection_mapping_probe(
    meta_activations: np.ndarray,
    introspection_scores: np.ndarray,
    alpha: float = 1000.0,
    use_pca: bool = False,
    pca_components: int = 100
) -> Tuple[LinearProbe, np.ndarray]:
    """
    Train a probe to predict introspection score from meta activations.

    This finds the direction along which activations vary with alignment
    between stated confidence and actual entropy.

    Args:
        meta_activations: (n_samples, hidden_dim) activation matrix
        introspection_scores: (n_samples,) introspection scores
        alpha: Ridge regularization
        use_pca: Whether to use PCA (default False to get full-dim direction)
        pca_components: PCA components if used

    Returns:
        Tuple of (fitted probe, direction vector in original activation space)
    """
    probe = LinearProbe(alpha=alpha, use_pca=use_pca, pca_components=pca_components)
    probe.fit(meta_activations, introspection_scores)

    # Extract direction in original space
    if use_pca and probe.pca is not None:
        # Project probe weights back through PCA
        pca_direction = probe.probe.coef_
        direction = probe.pca.inverse_transform(pca_direction.reshape(1, -1)).flatten()
        # Unscale
        direction = direction / probe.scaler.scale_
    else:
        # Direct weights, unscaled
        direction = probe.probe.coef_ / probe.scaler.scale_

    # Normalize
    direction = direction / np.linalg.norm(direction)

    return probe, direction


def compute_contrastive_direction(
    meta_activations: np.ndarray,
    introspection_scores: np.ndarray,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25
) -> np.ndarray:
    """
    Compute direction via contrastive mean difference.

    Alternative to probe-based direction: take mean of high-alignment examples
    minus mean of low-alignment examples.

    Args:
        meta_activations: (n_samples, hidden_dim)
        introspection_scores: (n_samples,)
        top_quantile: Fraction of samples to use for high-alignment group
        bottom_quantile: Fraction of samples to use for low-alignment group

    Returns:
        Normalized direction vector (hidden_dim,)
    """
    n = len(introspection_scores)
    sorted_idx = np.argsort(introspection_scores)

    # Bottom quantile (low alignment / miscalibrated)
    n_bottom = int(n * bottom_quantile)
    bottom_idx = sorted_idx[:n_bottom]

    # Top quantile (high alignment / well-calibrated)
    n_top = int(n * top_quantile)
    top_idx = sorted_idx[-n_top:]

    # Contrastive direction: well-calibrated - miscalibrated
    mean_top = meta_activations[top_idx].mean(axis=0)
    mean_bottom = meta_activations[bottom_idx].mean(axis=0)

    direction = mean_top - mean_bottom
    direction = direction / np.linalg.norm(direction)

    return direction


def extract_probe_direction(
    probe: LinearProbe,
    in_pca_space: bool = False
) -> np.ndarray:
    """
    Extract the direction vector from a fitted probe.

    Args:
        probe: Fitted LinearProbe
        in_pca_space: If True, return direction in PCA space (if PCA was used)

    Returns:
        Normalized direction vector
    """
    if not probe.is_fitted:
        raise RuntimeError("Probe not fitted")

    if in_pca_space or probe.pca is None:
        direction = probe.probe.coef_.copy()
    else:
        # Project back to original space
        pca_direction = probe.probe.coef_
        direction = probe.pca.inverse_transform(pca_direction.reshape(1, -1)).flatten()
        direction = direction / probe.scaler.scale_

    return direction / np.linalg.norm(direction)


def run_introspection_mapping_analysis(
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    train_split: float = 0.8,
    seed: int = 42,
    alpha: float = 1000.0,
    n_permutations: int = 100
) -> Tuple[Dict, Dict[int, np.ndarray]]:
    """
    Run introspection mapping analysis across all layers.

    For each layer, trains a probe to predict introspection score
    from meta activations and extracts the direction.

    Args:
        meta_activations: {layer_idx: (n_questions, hidden_dim)}
        direct_entropies: (n_questions,) entropy values
        stated_confidences: (n_questions,) confidence values (0-1)
        train_split, seed: Data split config
        alpha: Ridge regularization
        n_permutations: Permutation tests per layer

    Returns:
        Tuple of:
            - results: Dict with probe performance by layer
            - directions: {layer_idx: direction_vector}
    """
    introspection_scores = compute_introspection_scores(direct_entropies, stated_confidences)

    n_questions = len(introspection_scores)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(indices, train_size=train_split, random_state=seed)

    results = {}
    directions = {}

    for layer_idx in tqdm(sorted(meta_activations.keys()), desc="Introspection mapping"):
        X = meta_activations[layer_idx]
        y = introspection_scores

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train probe (no PCA to get full direction)
        probe, direction = train_introspection_mapping_probe(
            X_train, y_train, alpha=alpha, use_pca=False
        )
        directions[layer_idx] = direction

        # Evaluate
        train_eval = probe.evaluate(X_train, y_train)
        test_eval = probe.evaluate(X_test, y_test)

        # Also compute contrastive direction for comparison
        contrastive_dir = compute_contrastive_direction(X, y)
        cosine_sim = np.dot(direction, contrastive_dir)

        # Quick permutation test
        rng = np.random.RandomState(seed)
        null_r2s = []
        for _ in range(n_permutations):
            y_shuf = y.copy()
            rng.shuffle(y_shuf)
            p, _ = train_introspection_mapping_probe(X_train, y_shuf[train_idx], alpha=alpha, use_pca=False)
            null_r2s.append(p.evaluate(X_test, y_shuf[test_idx])["r2"])

        null_r2s = np.array(null_r2s)
        p_value = (null_r2s >= test_eval["r2"]).mean()

        results[layer_idx] = {
            "train_r2": train_eval["r2"],
            "test_r2": test_eval["r2"],
            "train_mae": train_eval["mae"],
            "test_mae": test_eval["mae"],
            "p_value": p_value,
            "significant_p05": p_value < 0.05,
            "probe_contrastive_cosine": float(cosine_sim),
        }

    return results, directions


# ============================================================================
# CLUSTER-BASED DIRECTION COMPUTATION
# ============================================================================


def compute_cluster_centroids(
    activations: np.ndarray,
    metric_values: np.ndarray,
    n_clusters: int = 3,
    method: str = "quantile"
) -> Dict:
    """
    Compute cluster centroids for activations grouped by metric value.

    This supports cluster-based interventions where uncertainty may be encoded
    categorically rather than continuously (e.g., low/mid/high as discrete states).

    Args:
        activations: (n_samples, hidden_dim) activation matrix
        metric_values: (n_samples,) the uncertainty metric to cluster by
        n_clusters: Number of clusters (e.g., 3 for low/mid/high)
        method: Clustering method:
            - "quantile": Group by metric value percentiles (always uses metric)
            - "kmeans": Cluster in activation space (may not align with metric)

    Returns:
        Dict with:
            - centroids: (n_clusters, hidden_dim) cluster centers
            - labels: (n_samples,) cluster assignment per sample
            - boundaries: metric value boundaries for quantile method
            - cluster_sizes: (n_clusters,) number of samples per cluster
            - cluster_metric_means: (n_clusters,) mean metric value per cluster
    """
    n_samples = len(metric_values)

    if method == "quantile":
        # Group by metric percentiles
        percentiles = np.linspace(0, 100, n_clusters + 1)
        boundaries = np.percentile(metric_values, percentiles)

        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_clusters):
            if i == n_clusters - 1:
                # Last cluster includes the max
                mask = (metric_values >= boundaries[i]) & (metric_values <= boundaries[i + 1])
            else:
                mask = (metric_values >= boundaries[i]) & (metric_values < boundaries[i + 1])
            labels[mask] = i

        # Compute centroids
        centroids = np.zeros((n_clusters, activations.shape[1]))
        for i in range(n_clusters):
            cluster_mask = labels == i
            if cluster_mask.sum() > 0:
                centroids[i] = activations[cluster_mask].mean(axis=0)

    elif method == "kmeans":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activations)
        centroids = kmeans.cluster_centers_

        # Sort clusters by mean metric value so cluster 0 = lowest metric
        cluster_metric_means = [metric_values[labels == i].mean() for i in range(n_clusters)]
        sorted_order = np.argsort(cluster_metric_means)

        # Remap labels and centroids to sorted order
        label_map = {old: new for new, old in enumerate(sorted_order)}
        labels = np.array([label_map[l] for l in labels])
        centroids = centroids[sorted_order]

        # Boundaries not meaningful for kmeans, but compute approximate ones
        boundaries = np.array([metric_values[labels == i].min() for i in range(n_clusters)] +
                              [metric_values.max()])

    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantile' or 'kmeans'.")

    # Compute cluster statistics
    cluster_sizes = np.array([np.sum(labels == i) for i in range(n_clusters)])
    cluster_metric_means = np.array([metric_values[labels == i].mean() for i in range(n_clusters)])

    return {
        "centroids": centroids,
        "labels": labels,
        "boundaries": boundaries,
        "cluster_sizes": cluster_sizes,
        "cluster_metric_means": cluster_metric_means,
        "n_clusters": n_clusters,
        "method": method,
    }


def compute_cluster_directions(
    centroids: np.ndarray,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute pairwise directions between cluster centroids.

    These directions can be used for steering: moving along "low_to_high"
    should shift the model's behavior from low-metric to high-metric.

    Args:
        centroids: (n_clusters, hidden_dim) cluster centers, ordered by metric value
                  (cluster 0 = lowest metric, cluster -1 = highest)
        normalize: Whether to normalize directions to unit length

    Returns:
        Dict mapping direction name to direction vector:
            - "low_to_high": From lowest to highest cluster
            - "low_to_mid": From lowest to middle cluster (if 3+ clusters)
            - "mid_to_high": From middle to highest cluster (if 3+ clusters)
            Plus additional pairwise directions for larger n_clusters
    """
    n_clusters = len(centroids)
    directions = {}

    # Primary direction: lowest to highest
    direction = centroids[-1] - centroids[0]
    if normalize:
        direction = direction / (np.linalg.norm(direction) + 1e-10)
    directions["low_to_high"] = direction

    # Additional directions for 3+ clusters
    if n_clusters >= 3:
        mid_idx = n_clusters // 2

        # Low to mid
        direction = centroids[mid_idx] - centroids[0]
        if normalize:
            direction = direction / (np.linalg.norm(direction) + 1e-10)
        directions["low_to_mid"] = direction

        # Mid to high
        direction = centroids[-1] - centroids[mid_idx]
        if normalize:
            direction = direction / (np.linalg.norm(direction) + 1e-10)
        directions["mid_to_high"] = direction

    # For completeness, add all pairwise directions (useful for analysis)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            direction = centroids[j] - centroids[i]
            if normalize:
                direction = direction / (np.linalg.norm(direction) + 1e-10)
            directions[f"cluster_{i}_to_{j}"] = direction

    return directions


def compute_caa_direction(
    activations: np.ndarray,
    metric_values: np.ndarray,
    high_quantile: float = 0.25,
    low_quantile: float = 0.25
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Contrastive Activation Addition (CAA) direction.

    Direction = mean(high_metric_activations) - mean(low_metric_activations)

    Unlike probe directions (which are optimized for prediction), CAA captures
    the empirical difference between high and low metric examples.

    Args:
        activations: (n_samples, hidden_dim) activation matrix
        metric_values: (n_samples,) metric values to contrast
        high_quantile: Top fraction of samples to use for high group
        low_quantile: Bottom fraction of samples to use for low group

    Returns:
        Tuple of:
            - direction: Normalized CAA direction vector
            - info: Dict with group statistics
    """
    n = len(metric_values)
    sorted_idx = np.argsort(metric_values)

    # Low group (bottom quantile)
    n_low = int(n * low_quantile)
    low_idx = sorted_idx[:n_low]

    # High group (top quantile)
    n_high = int(n * high_quantile)
    high_idx = sorted_idx[-n_high:]

    # Compute means
    mean_low = activations[low_idx].mean(axis=0)
    mean_high = activations[high_idx].mean(axis=0)

    # CAA direction
    direction = mean_high - mean_low
    direction_norm = np.linalg.norm(direction)
    direction = direction / (direction_norm + 1e-10)

    info = {
        "n_low": n_low,
        "n_high": n_high,
        "metric_mean_low": float(metric_values[low_idx].mean()),
        "metric_mean_high": float(metric_values[high_idx].mean()),
        "metric_std_low": float(metric_values[low_idx].std()),
        "metric_std_high": float(metric_values[high_idx].std()),
        "direction_magnitude": float(direction_norm),
        "high_quantile": high_quantile,
        "low_quantile": low_quantile,
    }

    return direction, info


def compare_directions(
    directions: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute pairwise cosine similarities between direction vectors.

    Useful for comparing probe directions, CAA directions, and cluster directions
    to see if they identify similar representational axes.

    Args:
        directions: Dict mapping direction name to direction vector

    Returns:
        Dict mapping "name1_vs_name2" to cosine similarity
    """
    names = list(directions.keys())
    similarities = {}

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i < j:
                d1 = directions[name1]
                d2 = directions[name2]
                # Normalize just in case
                d1 = d1 / (np.linalg.norm(d1) + 1e-10)
                d2 = d2 / (np.linalg.norm(d2) + 1e-10)
                sim = float(np.dot(d1, d2))
                similarities[f"{name1}_vs_{name2}"] = sim

    return similarities
