"""
Core utilities for introspection experiments.

Modules:
- model_utils: Model loading, naming, device detection
- extraction: BatchedExtractor for activation/logit extraction
- metrics: Uncertainty metric computation (entropy, logit_gap, etc.)
- directions: Direction finding methods (probe, mean_diff)
- probes: Linear probe training, transfer testing, permutation tests
- questions: Question loading, hashing, consistency verification
- steering: Steering and ablation hooks for activation intervention
- steering_experiments: Experiment runners and statistical analysis
- logit_lens: Logit lens for unembedding residual stream to vocabulary
"""

from .model_utils import (
    DEVICE,
    is_base_model,
    has_chat_template,
    get_model_short_name,
    get_run_name,
    load_model_and_tokenizer,
    should_use_chat_template,
)

from .extraction import (
    compute_entropy_from_probs,
    BatchedExtractor,
)

from .metrics import (
    compute_entropy,
    compute_metrics_single,
    compute_mc_metrics,
    compute_nexttoken_metrics,
    METRIC_INFO,
    metric_sign_for_confidence,
)

from .directions import (
    probe_direction,
    mean_diff_direction,
    find_directions,
    apply_direction,
    apply_probe_shared,
    apply_probe_centered,
    apply_probe_separate,
    evaluate_transfer,
)

from .probes import (
    LinearProbe,
    train_and_evaluate_probe,
    test_transfer,
    permutation_test,
    run_layer_analysis,
    # Introspection mapping
    compute_introspection_scores,
    train_introspection_mapping_probe,
    compute_contrastive_direction,
    extract_probe_direction,
    run_introspection_mapping_analysis,
)

from .questions import (
    load_questions,
    get_question_hash,
    save_question_set,
    load_question_set,
    verify_question_consistency,
    format_direct_prompt,
    split_questions,
)

from .steering import (
    SteeringHook,
    AblationHook,
    generate_orthogonal_directions,
    steering_context,
    ablation_context,
    multi_layer_steering_context,
    compute_projection_magnitude,
    measure_steering_effect,
)

from .steering_experiments import (
    SteeringExperimentConfig,
    # KV cache utilities
    extract_cache_tensors,
    create_fresh_cache,
    get_kv_cache,
    # Batch hooks
    BatchSteeringHook,
    BatchAblationHook,
    # Tokenization
    pretokenize_prompts,
    build_padded_gpu_batches,
    # Direction prep
    precompute_direction_tensors,
    # Experiment runners
    run_steering_experiment,
    run_ablation_experiment,
    # Analysis
    compute_correlation,
    get_expected_slope_sign,
    analyze_steering_results,
    analyze_ablation_results,
    # Printing
    print_steering_summary,
    print_ablation_summary,
)

from .logit_lens import (
    LogitLensConfig,
    LogitLensAnalyzer,
    # Core utilities
    get_unembedding_matrix,
    get_final_layernorm,
    unembed_vector_to_logits,
    unembed_direction_to_delta_logits,
    topk_tokens,
    select_token_position,
    auto_select_layers,
    print_logit_lens_summary,
)

__all__ = [
    # model_utils
    "DEVICE",
    "is_base_model",
    "has_chat_template",
    "get_model_short_name",
    "get_run_name",
    "load_model_and_tokenizer",
    "should_use_chat_template",
    # extraction
    "compute_entropy_from_probs",
    "BatchedExtractor",
    # metrics
    "compute_entropy",
    "compute_metrics_single",
    "compute_mc_metrics",
    "compute_nexttoken_metrics",
    "METRIC_INFO",
    "metric_sign_for_confidence",
    # directions
    "probe_direction",
    "mean_diff_direction",
    "find_directions",
    "apply_direction",
    "apply_probe_shared",
    "apply_probe_centered",
    "apply_probe_separate",
    "evaluate_transfer",
    # probes
    "LinearProbe",
    "train_and_evaluate_probe",
    "test_transfer",
    "permutation_test",
    "run_layer_analysis",
    # introspection mapping
    "compute_introspection_scores",
    "train_introspection_mapping_probe",
    "compute_contrastive_direction",
    "extract_probe_direction",
    "run_introspection_mapping_analysis",
    # questions
    "load_questions",
    "get_question_hash",
    "save_question_set",
    "load_question_set",
    "verify_question_consistency",
    "format_direct_prompt",
    "split_questions",
    # steering
    "SteeringHook",
    "AblationHook",
    "generate_orthogonal_directions",
    "steering_context",
    "ablation_context",
    "multi_layer_steering_context",
    "compute_projection_magnitude",
    "measure_steering_effect",
    # steering_experiments
    "SteeringExperimentConfig",
    "extract_cache_tensors",
    "create_fresh_cache",
    "get_kv_cache",
    "BatchSteeringHook",
    "BatchAblationHook",
    "pretokenize_prompts",
    "build_padded_gpu_batches",
    "precompute_direction_tensors",
    "run_steering_experiment",
    "run_ablation_experiment",
    "compute_correlation",
    "get_expected_slope_sign",
    "analyze_steering_results",
    "analyze_ablation_results",
    "print_steering_summary",
    "print_ablation_summary",
    # logit_lens
    "LogitLensConfig",
    "LogitLensAnalyzer",
    "get_unembedding_matrix",
    "get_final_layernorm",
    "unembed_vector_to_logits",
    "unembed_direction_to_delta_logits",
    "topk_tokens",
    "select_token_position",
    "auto_select_layers",
    "print_logit_lens_summary",
]
