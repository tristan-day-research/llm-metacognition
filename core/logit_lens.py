"""
Logit Lens: Unembed intermediate residual stream activations to vocabulary logits.

This module implements the "logit lens" technique for mechanistic interpretability,
which allows us to see what tokens a model would predict at each layer by projecting
the residual stream through the unembedding matrix.

WHAT IS VANILLA LOGIT LENS?
--------------------------
At the final layer, a transformer converts the residual stream vector to logits via:
    logits = LayerNorm(residual) @ W_U + b_U

The logit lens applies this same transformation to intermediate layers to peek at
what the model "thinks" at each stage of processing. This reveals how predictions
evolve through the network.

LN_MODE OPTIONS:
---------------
- "none": Raw residual stream, no normalization. 
    logits = residual @ W_U + b_U
    Fast but may produce meaningless logits since the residual stream scale
    evolves through layers.

- "final_ln": Apply the model's final LayerNorm before unembedding.
    logits = final_ln(residual) @ W_U + b_U  
    Standard approach that makes logits comparable across layers.

- "model_default": For TransformerLens, use the built-in ln_final handling.
    For HuggingFace, equivalent to "final_ln".

ACTIVATION STREAM OPTIONS:
-------------------------
- "resid_post": Residual stream AFTER layer L's attention+MLP (most common)
- "resid_pre": Residual stream BEFORE layer L's attention (less common)

Note: In HuggingFace models, we capture the output of each transformer block,
which is resid_post. Getting resid_pre requires additional hook placement.

FUTURE EXTENSION - DIRECTION UNEMBEDDING:
----------------------------------------
This module is designed to support unembedding arbitrary direction vectors
(e.g., mean-diff or probe weights) to see their effect on token logits:
    delta_logits = unembed_direction_to_delta_logits(model, direction, ln_mode)

Since directions represent differences/changes, we typically use ln_mode="none"
to avoid the nonlinear LayerNorm interaction.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass


@dataclass
class LogitLensConfig:
    """Configuration for logit lens analysis."""
    activation_stream: str = "resid_post"  # "resid_pre" or "resid_post"
    ln_mode: str = "final_ln"  # "none", "final_ln", or "model_default"
    token_position: Union[str, int] = "last"  # "last" or integer index
    top_k: int = 20  # Number of top/bottom tokens to report


def get_unembedding_matrix(model) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract the unembedding matrix and bias from a HuggingFace model.
    
    Returns:
        W_U: (d_model, vocab_size) unembedding weight matrix
        b_U: (vocab_size,) bias, or None if model has no lm_head bias
    """
    # Handle PEFT/adapter models
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
    else:
        base = model
    
    # Get lm_head (unembedding layer)
    lm_head = base.lm_head
    
    # Weight: (vocab_size, d_model) in HuggingFace, transpose to (d_model, vocab_size)
    W_U = lm_head.weight.T  # Now (d_model, vocab_size)
    
    # Bias: may or may not exist
    b_U = lm_head.bias if hasattr(lm_head, 'bias') and lm_head.bias is not None else None
    
    return W_U, b_U


def get_final_layernorm(model) -> torch.nn.Module:
    """
    Get the final LayerNorm from a HuggingFace model.
    
    Returns:
        The final LayerNorm module (model.model.norm for Llama-style models)
    """
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
    else:
        base = model
    
    # Llama, Mistral, etc. use model.model.norm
    if hasattr(base, 'model') and hasattr(base.model, 'norm'):
        return base.model.norm
    
    # GPT-2 style uses transformer.ln_f
    if hasattr(base, 'transformer') and hasattr(base.transformer, 'ln_f'):
        return base.transformer.ln_f
    
    raise AttributeError(
        "Could not find final LayerNorm. Supported architectures: "
        "Llama (model.model.norm), GPT-2 (transformer.ln_f)"
    )


def unembed_vector_to_logits(
    model,
    vector: torch.Tensor,
    ln_mode: str = "final_ln",
    W_U: Optional[torch.Tensor] = None,
    b_U: Optional[torch.Tensor] = None,
    final_ln: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Map a (d_model,) vector to (vocab_size,) logits via the unembedding matrix.
    
    This is the central utility function that can be used for:
    1. Vanilla logit lens (unembed residual stream activations)
    2. Direction unembedding (unembed mean-diff or probe weight vectors)
    
    Args:
        model: HuggingFace model (used to get W_U, b_U, final_ln if not provided)
        vector: (d_model,) tensor to unembed
        ln_mode: How to handle LayerNorm:
            - "none": No normalization (use for direction vectors)
            - "final_ln": Apply model's final LayerNorm (standard for logit lens)
            - "model_default": Same as "final_ln" for HuggingFace
        W_U: Optional pre-cached unembedding matrix (d_model, vocab_size)
        b_U: Optional pre-cached unembedding bias (vocab_size,)
        final_ln: Optional pre-cached final LayerNorm module
    
    Returns:
        logits: (vocab_size,) tensor of logits
    """
    # Get unembedding components if not provided
    if W_U is None or b_U is None:
        W_U_model, b_U_model = get_unembedding_matrix(model)
        W_U = W_U if W_U is not None else W_U_model
        b_U = b_U if b_U is not None else b_U_model
    
    # Ensure vector is on same device as W_U
    vector = vector.to(W_U.device)
    
    # Apply LayerNorm if requested
    if ln_mode in ("final_ln", "model_default"):
        if final_ln is None:
            final_ln = get_final_layernorm(model)
        # LayerNorm expects at least 2D, add batch dim and remove
        vector = final_ln(vector.unsqueeze(0)).squeeze(0)
    elif ln_mode == "none":
        pass  # No normalization
    else:
        raise ValueError(f"Unknown ln_mode: {ln_mode}. Use 'none', 'final_ln', or 'model_default'")
    
    # Compute logits: vector @ W_U + b_U
    logits = vector @ W_U
    if b_U is not None:
        logits = logits + b_U
    
    return logits


def unembed_direction_to_delta_logits(
    model,
    direction_vector: torch.Tensor,
    ln_mode: str = "none",
) -> torch.Tensor:
    """
    Map a direction vector (e.g., mean-diff or probe weights) to delta logits.
    
    This is a convenience wrapper around unembed_vector_to_logits specifically
    for direction vectors. The key difference from vanilla logit lens:
    - Directions represent *changes* in the residual stream
    - LayerNorm is typically not applied (ln_mode="none") to avoid nonlinear effects
    - The output represents how much each token's logit would change
    
    Args:
        model: HuggingFace model
        direction_vector: (d_model,) direction vector (normalized or unnormalized)
        ln_mode: How to handle LayerNorm. Default "none" for directions.
            Use "final_ln" only if you want to see the effect after normalization.
    
    Returns:
        delta_logits: (vocab_size,) change in logits from moving in this direction
    """
    return unembed_vector_to_logits(model, direction_vector, ln_mode=ln_mode)


def topk_tokens(
    tokenizer,
    logits: torch.Tensor,
    k: int = 20
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top-k highest and lowest logit tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer for decoding token IDs
        logits: (vocab_size,) tensor of logits
        k: Number of top/bottom tokens to return
    
    Returns:
        Dict with:
            "top_positive": List of {token_str, token_id, logit} for highest logits
            "top_negative": List of {token_str, token_id, logit} for lowest logits
    """
    logits_np = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits
    
    # Top k highest logits
    top_indices = np.argsort(logits_np)[-k:][::-1]  # Descending
    top_positive = []
    for idx in top_indices:
        token_str = tokenizer.decode([idx])
        top_positive.append({
            "token_str": repr(token_str),  # repr to show whitespace/special chars
            "token_id": int(idx),
            "logit": float(logits_np[idx])
        })
    
    # Top k lowest (most negative) logits
    bottom_indices = np.argsort(logits_np)[:k]  # Ascending (most negative first)
    top_negative = []
    for idx in bottom_indices:
        token_str = tokenizer.decode([idx])
        top_negative.append({
            "token_str": repr(token_str),
            "token_id": int(idx),
            "logit": float(logits_np[idx])
        })
    
    return {
        "top_positive": top_positive,
        "top_negative": top_negative
    }


def select_token_position(
    residuals: torch.Tensor,
    token_position: Union[str, int]
) -> torch.Tensor:
    """
    Select a single token's activation from the residual stream.
    
    Args:
        residuals: (seq_len, d_model) residual activations for a sequence
        token_position: "last" for final token, or integer index
    
    Returns:
        (d_model,) activation vector for the selected token
    """
    if token_position == "last":
        return residuals[-1]
    elif isinstance(token_position, int):
        return residuals[token_position]
    else:
        raise ValueError(f"token_position must be 'last' or int, got {token_position}")


class LogitLensAnalyzer:
    """
    Analyzer for running logit lens on HuggingFace models.
    
    This class handles:
    1. Running forward passes with activation caching
    2. Extracting residual stream activations at specified layers
    3. Unembedding activations to logits
    4. Generating reports with top tokens per layer
    """
    
    def __init__(self, model, tokenizer, num_layers: int):
        """
        Initialize the analyzer.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer  
            num_layers: Number of transformer layers
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        
        # Pre-cache unembedding components for efficiency
        self.W_U, self.b_U = get_unembedding_matrix(model)
        self.final_ln = get_final_layernorm(model)
        
        # Activation storage
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
    
    def _make_hook(self, layer_idx: int, stream: str):
        """Create a forward hook that captures residual stream activations."""
        def hook(module, input, output):
            if stream == "resid_post":
                # Output of the layer is resid_post
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self._activations[layer_idx] = hidden_states.detach()
            elif stream == "resid_pre":
                # Input to the layer is resid_pre
                if isinstance(input, tuple):
                    hidden_states = input[0]
                else:
                    hidden_states = input
                self._activations[layer_idx] = hidden_states.detach()
        return hook
    
    def _register_hooks(self, layers: List[int], stream: str):
        """Register hooks on specified layers."""
        if hasattr(self.model, 'get_base_model'):
            base = self.model.get_base_model()
            model_layers = base.model.layers
        else:
            model_layers = self.model.model.layers
        
        for layer_idx in layers:
            if layer_idx >= len(model_layers):
                raise ValueError(f"Layer {layer_idx} out of range (model has {len(model_layers)} layers)")
            hook = self._make_hook(layer_idx, stream)
            handle = model_layers[layer_idx].register_forward_hook(hook)
            self._hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
    
    def run_and_cache(
        self,
        prompt: str,
        layers: List[int],
        activation_stream: str = "resid_post"
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Run forward pass and cache activations at specified layers.
        
        Args:
            prompt: Input text
            layers: Which layers to cache activations for
            activation_stream: "resid_pre" or "resid_post"
        
        Returns:
            tokens: (seq_len,) input token IDs
            final_logits: (vocab_size,) logits from final layer
            cache: {layer_idx: (seq_len, d_model)} cached activations
        """
        self._activations = {}
        self._register_hooks(layers, activation_stream)
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            
            # Get final logits at last position
            final_logits = outputs.logits[0, -1, :]
            
            # Copy activations (squeeze batch dimension)
            cache = {
                layer_idx: acts[0].clone()  # (seq_len, d_model)
                for layer_idx, acts in self._activations.items()
            }
            
            return input_ids[0], final_logits, cache
            
        finally:
            self._remove_hooks()
    
    def get_residual_activations(
        self,
        cache: Dict[int, torch.Tensor],
        layer: int
    ) -> torch.Tensor:
        """
        Get residual stream activations for a layer from cache.
        
        Args:
            cache: Activation cache from run_and_cache
            layer: Layer index
        
        Returns:
            (seq_len, d_model) residual activations
        """
        if layer not in cache:
            raise KeyError(f"Layer {layer} not in cache. Available: {list(cache.keys())}")
        return cache[layer]
    
    def logit_lens_at_layer(
        self,
        cache: Dict[int, torch.Tensor],
        layer: int,
        token_position: Union[str, int],
        ln_mode: str,
        k: int = 20
    ) -> Dict[str, Any]:
        """
        Run logit lens analysis at a single layer.
        
        Args:
            cache: Activation cache from run_and_cache
            layer: Layer to analyze
            token_position: Which token position ("last" or int)
            ln_mode: LayerNorm mode ("none", "final_ln", "model_default")
            k: Number of top tokens to return
        
        Returns:
            Dict with layer analysis results
        """
        # Get residual activations
        residuals = self.get_residual_activations(cache, layer)
        
        # Select token
        token_vec = select_token_position(residuals, token_position)
        
        # Unembed to logits
        layer_logits = unembed_vector_to_logits(
            self.model,
            token_vec,
            ln_mode=ln_mode,
            W_U=self.W_U,
            b_U=self.b_U,
            final_ln=self.final_ln
        )
        
        # Get top tokens
        tokens = topk_tokens(self.tokenizer, layer_logits, k=k)
        
        return {
            "layer": layer,
            "top_tokens": tokens,
            "logits_mean": float(layer_logits.mean()),
            "logits_std": float(layer_logits.std()),
            "logits_max": float(layer_logits.max()),
            "logits_min": float(layer_logits.min()),
        }
    
    def logit_lens_report(
        self,
        prompt: str,
        layers: List[int],
        config: LogitLensConfig
    ) -> Dict[str, Any]:
        """
        Generate a full logit lens report for a prompt.
        
        Args:
            prompt: Input text
            layers: List of layers to analyze
            config: LogitLensConfig with analysis settings
        
        Returns:
            JSON-serializable dict with full report
        """
        # Run forward pass and cache
        tokens, final_logits, cache = self.run_and_cache(
            prompt, layers, config.activation_stream
        )
        
        # Get token strings for context
        token_strs = [repr(self.tokenizer.decode([t])) for t in tokens.tolist()]
        
        # Analyze each layer
        layer_results = {}
        for layer in layers:
            layer_results[str(layer)] = self.logit_lens_at_layer(
                cache, layer, config.token_position, config.ln_mode, config.top_k
            )
        
        # Get final output tokens for comparison
        final_tokens = topk_tokens(self.tokenizer, final_logits, k=config.top_k)
        
        return {
            "prompt": prompt,
            "config": {
                "activation_stream": config.activation_stream,
                "ln_mode": config.ln_mode,
                "token_position": config.token_position,
                "top_k": config.top_k,
            },
            "input_tokens": token_strs,
            "num_tokens": len(tokens),
            "layers_analyzed": layers,
            "layer_results": layer_results,
            "final_output": {
                "top_tokens": final_tokens,
            }
        }


def auto_select_layers(num_layers: int) -> List[int]:
    """
    Auto-select representative layers: [0, mid, last].
    
    Args:
        num_layers: Total number of layers in model
    
    Returns:
        List of layer indices
    """
    mid = num_layers // 2
    last = num_layers - 1
    return [0, mid, last]


def print_logit_lens_summary(report: Dict[str, Any], top_n: int = 10):
    """
    Print a human-readable summary of logit lens results.
    
    Args:
        report: Output from LogitLensAnalyzer.logit_lens_report
        top_n: Number of top tokens to show per layer
    """
    print("\n" + "=" * 80)
    print("LOGIT LENS REPORT")
    print("=" * 80)
    print(f"Prompt: {report['prompt'][:100]}..." if len(report['prompt']) > 100 else f"Prompt: {report['prompt']}")
    print(f"Tokens: {report['num_tokens']}")
    print(f"Config: stream={report['config']['activation_stream']}, "
          f"ln_mode={report['config']['ln_mode']}, "
          f"position={report['config']['token_position']}")
    print()
    
    for layer_str, layer_data in report['layer_results'].items():
        print(f"\n--- Layer {layer_str} ---")
        print(f"Logits: mean={layer_data['logits_mean']:.2f}, "
              f"std={layer_data['logits_std']:.2f}, "
              f"range=[{layer_data['logits_min']:.2f}, {layer_data['logits_max']:.2f}]")
        print(f"Top {top_n} tokens (highest logit):")
        for i, tok in enumerate(layer_data['top_tokens']['top_positive'][:top_n]):
            print(f"  {i+1:2d}. {tok['token_str']:<20} logit={tok['logit']:8.3f}")
    
    print(f"\n--- Final Output (actual model output) ---")
    final_tokens = report['final_output']['top_tokens']['top_positive'][:top_n]
    for i, tok in enumerate(final_tokens):
        print(f"  {i+1:2d}. {tok['token_str']:<20} logit={tok['logit']:8.3f}")
    
    print("\n" + "=" * 80)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    """
    Minimal smoke test for the logit lens module.
    
    Loads a small model and runs logit lens on "The capital of France is"
    at layers [0, mid, last] using the last token position.
    
    Usage:
        python -m core.logit_lens
    
    Note: This is a smoke test only. For full functionality, use run_logit_lens.py
    """
    import sys
    sys.path.insert(0, str(__file__).rsplit("/core/", 1)[0])
    
    from core.model_utils import load_model_and_tokenizer, DEVICE
    
    # Use a smaller model for smoke test if 70B is too large
    # Change this to a model you have available
    SMOKE_TEST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Small model for quick test
    SMOKE_TEST_PROMPT = "The capital of France is"
    
    print(f"=== LOGIT LENS SMOKE TEST ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {SMOKE_TEST_MODEL}")
    print(f"Prompt: {SMOKE_TEST_PROMPT}")
    print()
    
    # Load model
    print("Loading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(SMOKE_TEST_MODEL)
    print(f"Model has {num_layers} layers")
    
    # Select layers: [0, mid, last]
    layers = auto_select_layers(num_layers)
    print(f"Analyzing layers: {layers}")
    
    # Create analyzer and run
    config = LogitLensConfig(
        activation_stream="resid_post",
        ln_mode="final_ln",
        token_position="last",
        top_k=20
    )
    
    analyzer = LogitLensAnalyzer(model, tokenizer, num_layers)
    report = analyzer.logit_lens_report(SMOKE_TEST_PROMPT, layers, config)
    
    # Print results
    print_logit_lens_summary(report, top_n=10)
    
    print("\n=== SMOKE TEST PASSED ===")
    print("The logit lens module is working correctly.")
