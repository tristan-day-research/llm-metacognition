"""
Steering and ablation utilities for activation intervention experiments.

Provides:
- SteeringHook: Add scaled direction to activations
- AblationHook: Zero out projection onto direction
- Utilities for generating control directions
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from contextlib import contextmanager


class SteeringHook:
    """
    Hook that adds a scaled direction vector to layer activations.

    Usage:
        hook = SteeringHook(direction, multiplier=2.0)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        # ... run forward pass ...
        handle.remove()
    """

    def __init__(
        self,
        direction,
        multiplier: float = 1.0,
        position: str = "last"
    ):
        """
        Args:
            direction: (hidden_dim,) direction vector to add (will be normalized).
                       Can be numpy array or torch tensor.
            multiplier: Scalar to multiply direction by
            position: Which token position to steer:
                - "last": Only the last token
                - "all": All tokens
                - int: Specific position index
        """
        # Convert to tensor if needed, then normalize
        if isinstance(direction, np.ndarray):
            direction = torch.tensor(direction, dtype=torch.float32)
        else:
            direction = direction.float().cpu()
        # Ensure normalized so multiplier has consistent meaning across directions
        self.direction = direction / direction.norm()
        self.multiplier = multiplier
        self.position = position
        self._device_set = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Move direction to same device/dtype as activations
        if not self._device_set:
            self.direction = self.direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            self._device_set = True

        # Determine which positions to steer
        if self.position == "last":
            # Only steer the last token
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.multiplier * self.direction
        elif self.position == "all":
            # Steer all tokens
            hidden_states = hidden_states + self.multiplier * self.direction
        elif isinstance(self.position, int):
            # Steer specific position
            hidden_states[:, self.position, :] = hidden_states[:, self.position, :] + self.multiplier * self.direction

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


class AblationHook:
    """
    Hook that removes the component of activations along a direction.

    Projects out the direction: x' = x - (x · d) * d

    Usage:
        hook = AblationHook(direction)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        # ... run forward pass ...
        handle.remove()
    """

    def __init__(
        self,
        direction,
        position: str = "last"
    ):
        """
        Args:
            direction: (hidden_dim,) direction to ablate (will be normalized).
                       Can be numpy array or torch tensor.
            position: Which token position to ablate ("last", "all", or int)
        """
        # Convert to tensor if needed, then normalize
        if isinstance(direction, np.ndarray):
            direction = torch.tensor(direction, dtype=torch.float32)
        else:
            direction = direction.float().cpu()
        # Ensure normalized
        self.direction = direction / direction.norm()
        self.position = position
        self._device_set = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if not self._device_set:
            self.direction = self.direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            self._device_set = True

        if self.position == "last":
            # Project out direction from last token
            last_hidden = hidden_states[:, -1, :]  # (batch, hidden)
            proj = (last_hidden @ self.direction).unsqueeze(-1) * self.direction  # (batch, hidden)
            hidden_states[:, -1, :] = last_hidden - proj
        elif self.position == "all":
            # Project out from all tokens
            proj = (hidden_states @ self.direction).unsqueeze(-1) * self.direction
            hidden_states = hidden_states - proj
        elif isinstance(self.position, int):
            pos_hidden = hidden_states[:, self.position, :]
            proj = (pos_hidden @ self.direction).unsqueeze(-1) * self.direction
            hidden_states[:, self.position, :] = pos_hidden - proj

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


def generate_orthogonal_directions(
    direction: np.ndarray,
    n_directions: int,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Generate random directions orthogonal to the given direction.

    Useful for control conditions in steering experiments.

    Args:
        direction: (hidden_dim,) reference direction
        n_directions: Number of orthogonal directions to generate
        seed: Random seed

    Returns:
        List of n_directions orthogonal unit vectors
    """
    rng = np.random.RandomState(seed)
    hidden_dim = len(direction)

    # Normalize reference
    direction = direction / np.linalg.norm(direction)

    orthogonal_dirs = []
    for i in range(n_directions):
        # Generate random vector
        random_vec = rng.randn(hidden_dim)

        # Remove component along reference direction
        random_vec = random_vec - np.dot(random_vec, direction) * direction

        # Remove components along previously generated directions
        for prev_dir in orthogonal_dirs:
            random_vec = random_vec - np.dot(random_vec, prev_dir) * prev_dir

        # Normalize
        random_vec = random_vec / np.linalg.norm(random_vec)
        orthogonal_dirs.append(random_vec)

    return orthogonal_dirs


@contextmanager
def steering_context(
    model,
    layer_idx: int,
    direction: np.ndarray,
    multiplier: float = 1.0,
    position: str = "last"
):
    """
    Context manager for steering a single layer.

    Usage:
        with steering_context(model, layer_idx=15, direction=d, multiplier=2.0):
            output = model(input_ids)
    """
    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    hook = SteeringHook(direction, multiplier, position)
    handle = layers[layer_idx].register_forward_hook(hook)

    try:
        yield hook
    finally:
        handle.remove()


@contextmanager
def ablation_context(
    model,
    layer_idx: int,
    direction: np.ndarray,
    position: str = "last"
):
    """
    Context manager for ablating a direction from a single layer.

    Usage:
        with ablation_context(model, layer_idx=15, direction=d):
            output = model(input_ids)
    """
    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    hook = AblationHook(direction, position)
    handle = layers[layer_idx].register_forward_hook(hook)

    try:
        yield hook
    finally:
        handle.remove()


@contextmanager
def multi_layer_steering_context(
    model,
    layer_indices: List[int],
    directions: Dict[int, np.ndarray],
    multiplier: float = 1.0,
    position: str = "last"
):
    """
    Context manager for steering multiple layers simultaneously.

    Args:
        model: The transformer model
        layer_indices: List of layer indices to steer
        directions: Dict mapping layer_idx to direction vector
        multiplier: Steering strength
        position: Token position to steer

    Usage:
        with multi_layer_steering_context(model, [10, 15, 20], directions, multiplier=2.0):
            output = model(input_ids)
    """
    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    handles = []
    for layer_idx in layer_indices:
        if layer_idx not in directions:
            raise ValueError(f"No direction provided for layer {layer_idx}")
        hook = SteeringHook(directions[layer_idx], multiplier, position)
        handle = layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def compute_projection_magnitude(
    activations: np.ndarray,
    direction: np.ndarray
) -> np.ndarray:
    """
    Compute magnitude of activation projection onto direction.

    Args:
        activations: (n_samples, hidden_dim)
        direction: (hidden_dim,) normalized direction

    Returns:
        (n_samples,) projection magnitudes
    """
    direction = direction / np.linalg.norm(direction)
    return activations @ direction


def measure_steering_effect(
    activations_before: np.ndarray,
    activations_after: np.ndarray,
    direction: np.ndarray
) -> Dict:
    """
    Measure the effect of steering on activations.

    Args:
        activations_before: (n_samples, hidden_dim) before steering
        activations_after: (n_samples, hidden_dim) after steering
        direction: (hidden_dim,) steering direction

    Returns:
        Dict with projection changes and overall shift statistics
    """
    direction = direction / np.linalg.norm(direction)

    proj_before = activations_before @ direction
    proj_after = activations_after @ direction
    proj_change = proj_after - proj_before

    # Overall L2 change
    l2_change = np.linalg.norm(activations_after - activations_before, axis=1)

    # Fraction of change in direction vs orthogonal
    total_change = activations_after - activations_before
    change_in_direction = (total_change @ direction)[:, np.newaxis] * direction
    change_orthogonal = total_change - change_in_direction

    frac_in_direction = np.linalg.norm(change_in_direction, axis=1) / (l2_change + 1e-10)

    return {
        "projection_before_mean": float(proj_before.mean()),
        "projection_after_mean": float(proj_after.mean()),
        "projection_change_mean": float(proj_change.mean()),
        "projection_change_std": float(proj_change.std()),
        "l2_change_mean": float(l2_change.mean()),
        "fraction_in_direction_mean": float(frac_in_direction.mean()),
    }


# =============================================================================
# ACTIVATION PATCHING
# =============================================================================


class PatchingHook:
    """
    Hook that replaces activations with pre-computed target activations.

    Unlike steering (which adds a direction), patching swaps in the full
    activation pattern from a different example. This tests whether the
    complete activation (not just a 1D projection) causally determines behavior.

    Usage:
        # Load target activations from cache
        target_act = cached_activations[target_idx]  # (hidden_dim,)

        hook = PatchingHook(target_act, position="last")
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        # ... run source question through model ...
        handle.remove()
        # The model now saw target's activations at layer_idx
    """

    def __init__(
        self,
        target_activations: torch.Tensor,
        position: str = "last",
        last_token_indices: Optional[torch.Tensor] = None
    ):
        """
        Args:
            target_activations: Activations to inject.
                - For single example: (hidden_dim,)
                - For batch: (batch_size, hidden_dim)
            position: Which token position to patch:
                - "last": Last non-padding token (requires last_token_indices for padded batches)
                - "all": All tokens (broadcasts target to all positions)
                - int: Specific position index
            last_token_indices: For padded batches with position="last", the index of
                the last real token per batch item. Shape: (batch_size,)
        """
        # Handle numpy arrays
        if isinstance(target_activations, np.ndarray):
            target_activations = torch.tensor(target_activations, dtype=torch.float32)
        else:
            target_activations = target_activations.float().cpu()

        self.target_activations = target_activations
        self.position = position
        self.last_token_indices = last_token_indices
        self._device_set = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Move target to same device/dtype
        if not self._device_set:
            self.target_activations = self.target_activations.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            if self.last_token_indices is not None:
                self.last_token_indices = self.last_token_indices.to(hidden_states.device)
            self._device_set = True

        batch_size = hidden_states.shape[0]

        # Ensure target has batch dimension
        if self.target_activations.dim() == 1:
            # Single activation, broadcast to batch
            target = self.target_activations.unsqueeze(0).expand(batch_size, -1)
        else:
            target = self.target_activations

        if self.position == "last":
            if self.last_token_indices is not None:
                # Variable last positions (padded batch)
                for i in range(batch_size):
                    last_idx = self.last_token_indices[i].item()
                    hidden_states[i, last_idx, :] = target[i]
            else:
                # Assume last position is -1 (no padding)
                hidden_states[:, -1, :] = target
        elif self.position == "all":
            # Replace all positions with target (broadcasts)
            hidden_states[:, :, :] = target.unsqueeze(1)
        elif isinstance(self.position, int):
            hidden_states[:, self.position, :] = target

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


class BatchPatchingHook:
    """
    Hook for efficient batched patching with per-example targets.

    When testing many source→target pairs, this hook allows running multiple
    pairs in a single forward pass by specifying different target activations
    for each batch element.

    Usage:
        # Prepare batch: sources are questions, targets are cached activations
        source_prompts = [q1, q2, q3, ...]
        target_activations = torch.stack([cached[t1], cached[t2], cached[t3], ...])

        hook = BatchPatchingHook(target_activations)  # (batch, hidden)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        outputs = model(tokenized_sources)
        handle.remove()
    """

    def __init__(
        self,
        target_activations: torch.Tensor,
        position: str = "last",
        last_token_indices: Optional[torch.Tensor] = None
    ):
        """
        Args:
            target_activations: (batch_size, hidden_dim) - one target per batch element
            position: "last", "all", or int
            last_token_indices: (batch_size,) last token index for each example
        """
        if isinstance(target_activations, np.ndarray):
            target_activations = torch.tensor(target_activations, dtype=torch.float32)
        else:
            target_activations = target_activations.float().cpu()

        if target_activations.dim() != 2:
            raise ValueError(f"Expected 2D target_activations (batch, hidden), got shape {target_activations.shape}")

        self.target_activations = target_activations
        self.position = position
        self.last_token_indices = last_token_indices
        self._device_set = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        if not self._device_set:
            self.target_activations = self.target_activations.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            if self.last_token_indices is not None:
                self.last_token_indices = self.last_token_indices.to(hidden_states.device)
            self._device_set = True

        batch_size = hidden_states.shape[0]

        if batch_size != self.target_activations.shape[0]:
            raise ValueError(
                f"Batch size mismatch: hidden_states has {batch_size}, "
                f"target_activations has {self.target_activations.shape[0]}"
            )

        if self.position == "last":
            if self.last_token_indices is not None:
                for i in range(batch_size):
                    last_idx = self.last_token_indices[i].item()
                    hidden_states[i, last_idx, :] = self.target_activations[i]
            else:
                hidden_states[:, -1, :] = self.target_activations
        elif self.position == "all":
            hidden_states[:, :, :] = self.target_activations.unsqueeze(1)
        elif isinstance(self.position, int):
            hidden_states[:, self.position, :] = self.target_activations

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


@contextmanager
def patching_context(
    model,
    layer_idx: int,
    target_activations: torch.Tensor,
    position: str = "last",
    last_token_indices: Optional[torch.Tensor] = None
):
    """
    Context manager for activation patching at a single layer.

    Usage:
        with patching_context(model, layer_idx=15, target_activations=cached_act):
            output = model(source_input_ids)
            # output reflects source question with target's layer-15 activations
    """
    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    hook = PatchingHook(target_activations, position, last_token_indices)
    handle = layers[layer_idx].register_forward_hook(hook)

    try:
        yield hook
    finally:
        handle.remove()


@contextmanager
def batch_patching_context(
    model,
    layer_idx: int,
    target_activations: torch.Tensor,
    position: str = "last",
    last_token_indices: Optional[torch.Tensor] = None
):
    """
    Context manager for batched activation patching.

    Each batch element gets its own target activation injected.

    Usage:
        targets = torch.stack([cached[t1], cached[t2], ...])  # (batch, hidden)
        with batch_patching_context(model, layer_idx=15, target_activations=targets):
            output = model(source_batch_input_ids)
    """
    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    hook = BatchPatchingHook(target_activations, position, last_token_indices)
    handle = layers[layer_idx].register_forward_hook(hook)

    try:
        yield hook
    finally:
        handle.remove()
